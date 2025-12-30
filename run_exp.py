import os
import torch
import torch.nn as nn
import logging
import random
import numpy as np
import hydra
from typing import List, Dict, Any, Union
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.func import functional_call, grad, jvp
from torch.utils.data import DataLoader

from peft import get_peft_model, LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear
from accelerate import Accelerator

from utils import (
    train_text_to_text_model,
    initialize_text_to_text_model,
    transform_dataset,
    merge_llama,
)
import wandb
from data import DATASET_MAP

log = logging.getLogger(__name__)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def find_all_linear_modules(model) -> List[str]:
    linear_cls = torch.nn.Linear
    output_layer_names = ["lm_head", "embed_tokens"]
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(
            [output_layer in name for output_layer in output_layer_names]
        ):
            module_names.add(name.split(".")[-1])
    return list(module_names)

def reinit_lora_modules(name, module, init_config, peft_conf, **kwargs):
    """
    SVD 스케일을 안정화한 Hessian LoRA 초기화
    """
    hvp_provider = kwargs["hvp_provider"]
    r = module.lora_A.default.weight.shape[0]

    # HVP 계산 (평균 Hessian 정보 추출)
    Hv = hvp_provider(module, name)
    
    # Hessian의 에너지를 추출하기 위해 M 구성
    M = -Hv.float()
    q = min(128, min(M.shape))
    U, S, V = torch.svd_lowrank(M, q=q, niter=16)
    V = V.T 

    # 스케일 안정화: 특이값 S의 루트를 양쪽에 배분하여 초기 Loss 폭주 방지
    S_sqrt = torch.sqrt(S[:r])
    B = U[:, :r] * S_sqrt.unsqueeze(0)      # (out, r)
    A = S_sqrt.unsqueeze(1) * V[:r, :]      # (r, in)

    scaling_factor = 1.0
    if hasattr(module, "scaling") and isinstance(module.scaling, dict) and "default" in module.scaling:
        scaling_factor = module.scaling["default"]

    scale_mode = getattr(init_config, "scale", "default")
    if scale_mode == "gd":
        # gd 모드에서는 scaling_factor의 영향을 중립화하기 위해 조정
        A = A / (scaling_factor ** 0.5)
        B = B / (scaling_factor ** 0.5)

    # 파라미터 업데이트
    module.lora_B.default.weight.data.copy_(B.contiguous())
    module.lora_A.default.weight.data.copy_(A.contiguous())

def reinit_lora(model, init_config, peft_conf, **kwargs):
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing LoRA (Hessian)",
        total=len(list(model.named_modules())),
    ):
        if isinstance(module, LoraLinear):
            reinit_lora_modules(name, module, init_config, peft_conf, **kwargs)
    return model

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_exp(cfg: DictConfig):
    assert cfg.init.mode == "hessian"
    log.info(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    accelerator = Accelerator()
    model_name = cfg.model.name
    model_type = cfg.model.type
    dataset_name = cfg.dataset_name
    dataset_func = DATASET_MAP[dataset_name]

    use_peft = cfg.peft.use_peft
    lora_r = cfg.peft.lora_r
    lora_target_modules = cfg.peft.lora_target_modules

    # ---- wandb 초기화 ----
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "lora_r": lora_r,
        "init_mode": cfg.init.mode,
        "init_scale": cfg.init.scale,
    }
    wandb.init(project=cfg.wandb.project + "_" + dataset_name, config=config)

    # ---- 데이터 로드 및 HVP용 서브셋 준비 ----
    train_set, val_set, _ = dataset_func()
    base_model_for_hvp, tokenizer = initialize_text_to_text_model(
        model_name, model_type, cfg.model.bf16, use_peft=False
    )
    base_model_for_hvp = base_model_for_hvp.to("cuda").eval()

    # HVP 배치 샘플링
    num_hvp_samples = cfg.init.bsz * cfg.init.iters
    if isinstance(train_set, list):
        temp_set = train_set[:num_hvp_samples]
    else:
        temp_set = train_set.select(range(num_hvp_samples))

    temp_set = transform_dataset(model_type, tokenizer, temp_set, cfg.init.max_length)
    temp_loader = DataLoader(temp_set, batch_size=1, shuffle=False)
    
    modules_hvp = dict(base_model_for_hvp.named_modules())
    params0 = dict(base_model_for_hvp.named_parameters())
    buffers0 = dict(base_model_for_hvp.named_buffers())

    def hvp_provider(lora_module, name):
      cleaned_name = name
      prefixes_to_remove = ["base_model.model.", "base_model.", "model.model."]
      for prefix in prefixes_to_remove:
        if cleaned_name.startswith(prefix):
          cleaned_name = cleaned_name[len(prefix):]

      candidates = [cleaned_name, cleaned_name.replace("model.", "")]
        
      base_layer = None
      picked_name = None
      for cand in candidates:
        if cand in modules_hvp:
          base_layer = modules_hvp[cand]
          picked_name = cand
          break
            
      if base_layer is None:
        for k, v in modules_hvp.items():
          if k in name or name in k:
            base_layer = v
            picked_name = k
            break

      if base_layer is None:
        sample_keys = list(modules_hvp.keys())[:5]
        raise KeyError(
          f"[HVP] Cannot find module for '{name}'. \n"
          f"Tried candidates: {candidates}. \n"
          f"Available keys in HVP model (sample): {sample_keys}"
        )

      W = base_layer.weight
      v = torch.empty_like(W).bernoulli_(0.5).mul_(2).add_(-1) # Rademacher
        
      w_name = None
      for n, p in base_model_for_hvp.named_parameters():
        if p is W:
          w_name = n
          break
        
      if w_name is None:
        raise RuntimeError(f"Could not find parameter name for {picked_name}")

      # 2. 여러 배치를 순회하며 평균 HVP 계산
      total_Hv = None
      count = 0
      hvp_iters = getattr(cfg.init, "iters", 4)

      loader_iter = iter(temp_loader)
      for _ in range(hvp_iters):
        try:
          batch = next(loader_iter)
        except StopIteration:
          break
            
        batch = {k: v.to(base_model_for_hvp.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        def loss_fn(W_local):
          params = {**params0, w_name: W_local}
          out = functional_call(base_model_for_hvp, (params, buffers0), (), batch)
          return out.loss

        grad_fn = grad(loss_fn)
        _, Hv = jvp(grad_fn, (W,), (v,))
            
        if total_Hv is None:
          total_Hv = Hv.detach()
        else:
          total_Hv += Hv.detach()
        count += 1

      return total_Hv / count if count > 0 else torch.zeros_like(W)

    # ---- 3) 실제 학습용 모델 구성 (LoRA 적용) ----
    base_model_train, _ = initialize_text_to_text_model(
        model_name, model_type, cfg.model.bf16, use_peft=False
    )
    
    if lora_target_modules == "all":
        lora_target_modules = find_all_linear_modules(base_model_train)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=cfg.peft.lora_alpha,
        lora_dropout=cfg.peft.lora_dropout,
        target_modules=lora_target_modules,
        use_rslora=cfg.peft.use_rslora,
    )

    model = get_peft_model(base_model_train, peft_config)
    model = model.to("cuda")

    # ---- 4) Hessian Re-initialization 실행 ----
    model = reinit_lora(model, cfg.init, cfg.peft, hvp_provider=hvp_provider)
    base_model_for_hvp.cpu() # 메모리 확보
    torch.cuda.empty_cache()

    # ---- 5) 학습 시작 ----
    model = train_text_to_text_model(
        f"{cfg.wandb.project}/{wandb.run.name}",
        train_set, val_set, model, tokenizer, model_type,
        num_train_epochs=cfg.model.epochs,
        per_device_batch_size=cfg.model.per_device_batch_size,
        real_batch_size=cfg.model.real_batch_size,
        bf16=cfg.model.bf16,
        learning_rate=cfg.model.learning_rate,
        seed=cfg.seed,
        num_process=accelerator.num_processes,
    )

    wandb.finish()

if __name__ == "__main__":
    run_exp()
