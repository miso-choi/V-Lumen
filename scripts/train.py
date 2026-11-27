#!/usr/bin/env python3
"""Two-stage training of V-LUMEN and the adapter baselines (paper §3.1, App. E).

  # Stage 1: mid-training from the frozen backbone
  python scripts/train.py --stage midtrain --setting vlumen --steps 10000 --out ckpt/vlumen_seed1/midtrain.pt

  # Stage 2: SFT, initialised from the stage-1 checkpoint
  python scripts/train.py --stage sft --init ckpt/vlumen_seed1/midtrain.pt --steps 2000 --out ckpt/vlumen_seed1/sft.pt

The backbone is always frozen; only V-LUMEN (memory tables + fusion) and/or the
LoRA / MoE adapter are trained, with an answer-only next-token loss.
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from vlumen import budget
from vlumen.checkpoint import build_model, load_backbone, load_processor, load_weights, read_checkpoint, save_checkpoint
from vlumen.codebook import load_lavit_codebook
from vlumen.config import TRAIN_TARGETS, VLumenConfig
from vlumen.data.benchmarks import FewShotDataset
from vlumen.data.collate import make_answer_only_collate
from vlumen.data.mixture import MixtureDataset

# Stage-specific optimisation settings.
STAGE = {
    #            memory-table lr multiplier, cosine floor (x lr), data cap per source, held-out fraction
    "midtrain": dict(table_lr_mult=5.0, min_lr_ratio=0.1, cap=6000, val_fraction=0.0),
    "sft":      dict(table_lr_mult=0.1, min_lr_ratio=0.05, cap=4000, val_fraction=0.1),
}
WEIGHT_DECAY = 0.1
ADAPTER_CLIP = 1.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=list(STAGE))
    p.add_argument("--setting", choices=TRAIN_TARGETS, help="midtrain only (SFT takes it from --init)")
    p.add_argument("--init", help="sft: mid-training checkpoint to start from")
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=None, help="default: 4 (midtrain) / 8 (sft)")
    p.add_argument("--grad_accum", type=int, default=1, help="micro-batches per step (effective batch = batch_size x grad_accum)")
    p.add_argument("--lr", type=float, default=None, help="default: 4e-4 (midtrain) / 1e-4 (sft)")
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data_cap", type=int, default=None, help="rows per training source (default: 6000 / 4000)")
    p.add_argument("--few_shot_cap", type=int, default=200, help="few-shot examples per reasoning benchmark")
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    # Model (midtrain only; SFT inherits everything from --init)
    p.add_argument("--budget_scale", type=float, default=0.5, help="k in V-LUMEN(1) + adapter(k)")
    p.add_argument("--fusion_layers", default="2,8")
    p.add_argument("--memory_dim", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--rows_per_cell", type=int, default=81920)
    p.add_argument("--no_axial_pooling", action="store_true")
    p.add_argument("--no_coord_bias", action="store_true")
    p.add_argument("--no_prh", action="store_true")
    p.add_argument("--no_drh", action="store_true")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_max_rank", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.stage == "midtrain" and not a.setting:
        p.error("--setting is required for --stage midtrain")
    if a.stage == "sft" and not a.init:
        p.error("--init is required for --stage sft")
    a.batch_size = a.batch_size or (4 if a.stage == "midtrain" else 8)
    a.lr = a.lr or (4e-4 if a.stage == "midtrain" else 1e-4)
    a.data_cap = a.data_cap or STAGE[a.stage]["cap"]
    return a


# ──────────────────────────────────────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────────────────────────────────────

def midtrain_config(a, base) -> VLumenConfig:
    cfg = VLumenConfig(
        train_target=a.setting,
        hidden_dim=base.config.text_config.hidden_size,
        image_token_id=base.config.image_token_id,
        memory_dim=a.memory_dim,
        n_heads=a.n_heads,
        rows_per_cell=a.rows_per_cell,
        fusion_layers=[int(x) for x in a.fusion_layers.split(",")],
        use_axial_pooling=not a.no_axial_pooling,
        use_coord_bias=not a.no_coord_bias,
        use_prh=not a.no_prh,
        use_drh=not a.no_drh,
        budget_scale=a.budget_scale,
    )
    cfg.vlumen_params = budget.vlumen_params(cfg)
    k = cfg.budget_scale
    if cfg.has_lora:
        cfg.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        inout = budget.lora_inout_sum(base, cfg.lora_target_modules)
        scale = k if cfg.has_vlumen else 1.0 + k
        cfg.lora_rank, cfg.lora_alpha = budget.lora_rank_and_alpha(cfg.vlumen_params, scale, inout, a.lora_max_rank)
        cfg.lora_dropout = a.lora_dropout
        print(f"[budget] LoRA({scale:g}): rank={cfg.lora_rank} alpha={cfg.lora_alpha} "
              f"params={cfg.lora_rank * inout:,} (P_v={cfg.vlumen_params:,})")
    if cfg.has_moe:
        shape = budget.moe_shape_with_vlumen(cfg) if cfg.has_vlumen else budget.moe_shape_only(cfg)
        cfg.moe_num_experts, cfg.moe_expert_hidden_dim, cfg.moe_top_k = shape.num_experts, shape.expert_hidden_dim, shape.top_k
        print(f"[budget] MoE({'k' if cfg.has_vlumen else '1+k'}): experts={shape.num_experts} hidden={shape.expert_hidden_dim} "
              f"top_k={shape.top_k} accessible={shape.accessible:,} activated/token={shape.activated:,}")
    return cfg


def build(a):
    if a.stage == "midtrain":
        base = load_backbone(VLumenConfig().model_id, a.device)
        cfg = midtrain_config(a, base)
        model = build_model(cfg, base, a.device, codebook=load_lavit_codebook() if cfg.has_vlumen else None)
        if cfg.has_vlumen:
            assert model.vlumen_param_count() == cfg.vlumen_params
    else:
        ckpt = read_checkpoint(a.init)
        if ckpt["stage"] != "midtrain":
            raise ValueError(f"--init must be a mid-training checkpoint (got stage={ckpt['stage']})")
        cfg = VLumenConfig.from_dict(ckpt["config"])
        base = load_backbone(cfg.model_id, a.device)
        codebook = ckpt["quantizer"]["codebook"].float() if cfg.has_vlumen else None
        model = build_model(cfg, base, a.device, codebook=codebook)
        load_weights(model, ckpt)
        print(f"[init] {a.init} (step {ckpt['step']})")
    return model, cfg


def param_groups(model, cfg, lr: float, table_lr_mult: float):
    """V-LUMEN: memory rows / decayed projections / non-decayed norm & scale; adapters: one decayed group."""
    groups, vlumen_split = [], None
    if cfg.has_vlumen:
        tables, decay, no_decay = [], [], []
        for module in model.vlumen_modules:
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if "tables" in name:
                    tables.append(p)
                elif name.endswith("residual_scale") or name.startswith("norm."):
                    no_decay.append(p)
                else:
                    decay.append(p)
        groups += [
            {"params": tables, "lr": lr * table_lr_mult, "weight_decay": 0.0},
            {"params": decay, "lr": lr, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
        vlumen_split = (len(tables), len(decay) + len(no_decay))
    adapter = []
    if cfg.has_lora:
        adapter = [p for p in model.model.parameters() if p.requires_grad]
    elif cfg.has_moe:
        adapter = list(model.moe_modules.parameters())
    if adapter:
        groups.append({"params": adapter, "lr": lr, "weight_decay": WEIGHT_DECAY})
    return groups, adapter, vlumen_split


def clip_gradients(model, cfg, adapter_params):
    """Per V-LUMEN module: memory rows and the rest clipped separately (max norm sqrt(memory_dim / 128)).
    Adapter parameters: one global clip at 1.0."""
    if cfg.has_vlumen:
        max_norm = math.sqrt(max(1, cfg.memory_dim) / 128.0)
        for module in model.vlumen_modules:
            tables, rest = [], []
            for name, p in module.named_parameters():
                if p.requires_grad and p.grad is not None:
                    (tables if "tables" in name else rest).append(p)
            if rest:
                torch.nn.utils.clip_grad_norm_(rest, max_norm=max_norm)
            if tables:
                torch.nn.utils.clip_grad_norm_(tables, max_norm=max_norm)
    if adapter_params:
        torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=ADAPTER_CLIP)


# ──────────────────────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    st = STAGE[a.stage]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    model, cfg = build(a)
    processor = load_processor(cfg.model_id)
    print(f"[train] stage={a.stage} setting={cfg.train_target} fusion_layers={cfg.fusion_layers} "
          f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    data = ConcatDataset([MixtureDataset(a.stage, cap=a.data_cap), FewShotDataset(a.few_shot_cap)])
    if st["val_fraction"] > 0:
        n_val = max(1, int(len(data) * st["val_fraction"]))
        data, _ = random_split(data, [len(data) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    print(f"[train] training samples: {len(data):,}")
    loader = DataLoader(data, batch_size=a.batch_size, shuffle=True, num_workers=0,
                        collate_fn=make_answer_only_collate(processor, a.max_len, a.device),
                        generator=torch.Generator().manual_seed(a.seed))

    groups, adapter_params, _ = param_groups(model, cfg, a.lr, st["table_lr_mult"])
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.steps, eta_min=a.lr * st["min_lr_ratio"])

    model.train()
    model.model.eval()   # frozen backbone (also keeps LoRA dropout inactive)

    def next_batch(it):
        while True:
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            if (b["labels"] != -100).any():
                return b, it

    it, step, t_start = iter(loader), 0, time.time()
    log = {"loss": 0.0, "lb": 0.0, "gate": 0.0, "n": 0}
    consecutive_oom = 0
    while step < a.steps:
        batch, it = next_batch(it)
        try:
            optimizer.zero_grad()
            loss_sum = lb_sum = 0.0
            for micro in range(a.grad_accum):
                if micro > 0:
                    batch, it = next_batch(it)
                out = model(**batch)
                (out.loss / a.grad_accum).backward()
                loss_sum += out.loss.item()
                lb = model.load_balance_loss()
                lb_sum += lb.item() if lb is not None else 0.0
                del out
            clip_gradients(model, cfg, adapter_params)
            optimizer.step()
            scheduler.step()
            consecutive_oom = 0
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            consecutive_oom += 1
            print(f"  OOM at step {step}, batch skipped")
            if consecutive_oom >= 20:
                raise RuntimeError("20 consecutive OOMs: lower --batch_size and raise --grad_accum")
            continue
        step += 1

        log["loss"] += loss_sum / a.grad_accum
        log["lb"] += lb_sum / a.grad_accum
        if cfg.has_vlumen:
            log["gate"] += sum(m.last_gate_mean for m in model.vlumen_modules) / len(model.vlumen_modules)
        log["n"] += 1
        if step % a.log_every == 0 or step == a.steps:
            n = log["n"]
            elapsed = time.time() - t_start
            msg = (f"step {step:6d}/{a.steps}  loss {log['loss'] / n:.4f}  lr {scheduler.get_last_lr()[0]:.2e}"
                   f"  elapsed {elapsed / 60:.1f}m  eta {elapsed / step * (a.steps - step) / 60:.1f}m")
            if cfg.has_vlumen:
                scale = sum(float(m.residual_scale) for m in model.vlumen_modules) / len(model.vlumen_modules)
                msg += f"  gate {log['gate'] / n:.3f}  res_scale {scale:.3f}"
            if cfg.has_moe:
                msg += f"  moe_lb {log['lb'] / n:.4f}"
            print(msg, flush=True)
            log = {"loss": 0.0, "lb": 0.0, "gate": 0.0, "n": 0}
        if a.save_every and step % a.save_every == 0 and step < a.steps:
            save_checkpoint(str(Path(a.out).with_name(f"{Path(a.out).stem}_step{step}.pt")), model, a.stage, step)

    save_checkpoint(a.out, model, a.stage, step)
    print(f"[train] saved {a.out}  ({(time.time() - t_start) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
