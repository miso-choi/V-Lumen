"""Model construction and checkpoint I/O.

Checkpoint layout (a single torch file):

    format        "vlumen-v1"
    config        VLumenConfig.to_dict()
    stage         "midtrain" | "sft"
    step          optimizer steps of that stage
    quantizer     frozen W_proj + codebook            (settings with V-LUMEN)
    vlumen_state  VLumenModule state per fusion layer  (settings with V-LUMEN)
    tch_state     PRH projections + DRH prior          (settings with V-LUMEN)
    lora_state    PEFT LoRA weights                    (LoRA settings)
    moe_state     MoE adapter weights                  (MoE settings)
    optimizer     only in mid-training checkpoints (for resuming)
"""

from __future__ import annotations

from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from .codebook import FrozenCodebookQuantizer, load_lavit_codebook
from .config import VLumenConfig
from .model import VLumenVLM

FORMAT = "vlumen-v1"


def load_backbone(model_id: str, device: str):
    from transformers import Qwen3VLForConditionalGeneration
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
    )
    for p in base.parameters():
        p.requires_grad_(False)
    return base


def load_processor(model_id: str):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id)


def build_model(cfg: VLumenConfig, base, device: str, codebook: Optional[torch.Tensor] = None) -> VLumenVLM:
    """Wrap `base` with the modules of `cfg` (fresh weights). LoRA is applied last."""
    quantizer = None
    if cfg.has_vlumen:
        codebook = load_lavit_codebook() if codebook is None else codebook
        quantizer = FrozenCodebookQuantizer(cfg.hidden_dim, codebook).to(device).to(torch.bfloat16)
    model = VLumenVLM(base, cfg, quantizer)
    if model.vlumen_modules is not None:
        model.vlumen_modules = model.vlumen_modules.to(device).to(torch.float32)
    if model.prh_proj_q is not None:
        model.prh_proj_q = model.prh_proj_q.to(device).to(torch.float32)
        model.prh_proj_k = model.prh_proj_k.to(device).to(torch.float32)
    if model.drh_ema_prior is not None:
        model.drh_ema_prior = model.drh_ema_prior.to(device)
    if model.moe_modules is not None:
        model.moe_modules = model.moe_modules.to(device).to(torch.float32)
    if cfg.has_lora:
        model.model = get_peft_model(model.model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
        ))
    return model


def state_dict_for_checkpoint(model: VLumenVLM) -> dict:
    cfg = model.config
    out = {}
    if cfg.has_vlumen:
        out["quantizer"] = model.quantizer.state_dict()
        out["vlumen_state"] = model.vlumen_modules.state_dict()
        tch = {}
        if model.prh_proj_q is not None:
            tch["prh_proj_q"] = model.prh_proj_q.state_dict()
            tch["prh_proj_k"] = model.prh_proj_k.state_dict()
        if model.drh_ema_prior is not None:
            tch["drh_ema_prior"] = model.drh_ema_prior.detach().cpu()
        out["tch_state"] = tch
    if cfg.has_lora:
        out["lora_state"] = get_peft_model_state_dict(model.model)
    if cfg.has_moe:
        out["moe_state"] = model.moe_modules.state_dict()
    return out


def save_checkpoint(path: str, model: VLumenVLM, stage: str, step: int, optimizer=None) -> None:
    ckpt = {"format": FORMAT, "config": model.config.to_dict(), "stage": stage, "step": step}
    ckpt.update(state_dict_for_checkpoint(model))
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, path)


def read_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} checkpoint (format={ckpt.get('format')!r})")
    return ckpt


def load_weights(model: VLumenVLM, ckpt: dict) -> None:
    """Load trained weights into a model built from ckpt['config'] (strict)."""
    cfg = model.config
    if cfg.has_vlumen:
        model.quantizer.load_checkpoint_state(ckpt["quantizer"])
        model.vlumen_modules.load_state_dict(ckpt["vlumen_state"], strict=True)
        tch = ckpt.get("tch_state", {})
        if model.prh_proj_q is not None:
            model.prh_proj_q.load_state_dict(tch["prh_proj_q"])
            model.prh_proj_k.load_state_dict(tch["prh_proj_k"])
        if model.drh_ema_prior is not None:
            model.drh_ema_prior.copy_(tch["drh_ema_prior"].to(model.drh_ema_prior))
    if cfg.has_moe:
        model.moe_modules.load_state_dict(ckpt["moe_state"], strict=True)
    if cfg.has_lora:
        result = set_peft_model_state_dict(model.model, ckpt["lora_state"])
        unexpected = getattr(result, "unexpected_keys", [])
        if unexpected:
            raise KeyError(f"unexpected LoRA keys in checkpoint: {unexpected[:5]}")


def load_model(path: str, device: str):
    """Build the backbone + modules described by a checkpoint and load its weights."""
    ckpt = read_checkpoint(path)
    cfg = VLumenConfig.from_dict(ckpt["config"])
    base = load_backbone(cfg.model_id, device)
    # The checkpoint already holds the codebook, so LaVIT does not need to be downloaded.
    codebook = ckpt["quantizer"]["codebook"].float() if cfg.has_vlumen else None
    model = build_model(cfg, base, device, codebook=codebook)
    load_weights(model, ckpt)
    model.eval()
    return model, cfg, ckpt
