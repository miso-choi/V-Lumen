"""Parameter-budget matching for the LoRA and MoE baselines (paper App. G).

Notation: V-LUMEN(1) is the V-LUMEN memory with P_v trainable parameters and
adapter(k) is an adapter with k * P_v parameters (k = budget_scale = 0.5).

  LoRA + V-LUMEN : LoRA(k)       rank = floor(k * P_v / sum(in + out))
  LoRA-only      : LoRA(1 + k)   rank = floor((1 + k) * P_v / sum(in + out))
  MoE + V-LUMEN  : MoE(k)        accessible ~ k * P_v, activated/token ~ V-LUMEN's own activated cost
  MoE-only       : MoE(1 + k)    accessible and activated/token matched to V-LUMEN(1) + MoE(k)

The MoE structure (experts, hidden size, top-k) is found by grid search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import VLumenConfig
from .memory_table import cell_table_sizes

LORA_ALPHA_REF = 16           # alpha at the reference budget scale
LORA_ALPHA_REF_SCALE = 0.25   # alpha / rank is kept ~ constant around this budget


# ──────────────────────────────────────────────────────────────────────────────
#  V-LUMEN parameter counts
# ──────────────────────────────────────────────────────────────────────────────

def vlumen_params_per_layer(cfg: VLumenConfig) -> int:
    """Trainable parameters of one VLumenModule (memory rows + gating/fusion)."""
    D, c, n_cells = cfg.hidden_dim, cfg.cell_dim, cfg.n_cells
    rows = sum(sum(s) for s in cell_table_sizes(cfg.rows_per_cell, len(cfg.ngram_orders), cfg.n_heads))
    return rows * c + D * n_cells + n_cells * c * D + 3 * D + D + 1


def vlumen_params(cfg: VLumenConfig) -> int:
    return len(cfg.fusion_layers) * vlumen_params_per_layer(cfg)


def vlumen_activated_params_per_token(cfg: VLumenConfig) -> int:
    """Per-token activated parameters of V-LUMEN summed over fusion layers:
    gathered memory scalars + gate + value projections + depthwise conv + RMSNorm."""
    D, c, n_cells = cfg.hidden_dim, cfg.cell_dim, cfg.n_cells
    per_layer = cfg.memory_dim + D * n_cells + n_cells * c * D + 3 * D + D
    return len(cfg.fusion_layers) * per_layer


# ──────────────────────────────────────────────────────────────────────────────
#  LoRA
# ──────────────────────────────────────────────────────────────────────────────

def lora_inout_sum(model: torch.nn.Module, target_modules: list[str]) -> int:
    total = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and any(name.endswith(t) for t in target_modules):
            total += mod.in_features + mod.out_features
    if total == 0:
        raise ValueError(f"No LoRA target modules matched: {target_modules}")
    return total


def lora_rank_and_alpha(p_vlumen: int, scale: float, inout_sum: int, max_rank: int) -> tuple[int, int]:
    """LoRA rank for a budget of scale * P_v, and alpha with alpha / rank kept ~ constant."""
    def rank_for(s):
        return max(1, min(max_rank, max(1, int(p_vlumen * s)) // inout_sum))
    rank = rank_for(scale)
    ref_rank = rank_for(LORA_ALPHA_REF_SCALE)
    alpha = max(1, int(round(LORA_ALPHA_REF * rank / max(1, ref_rank))))
    return rank, alpha


# ──────────────────────────────────────────────────────────────────────────────
#  MoE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MoEShape:
    num_experts: int
    expert_hidden_dim: int
    top_k: int
    accessible: int
    activated: int


def moe_accessible_params(D: int, n_layers: int, experts: int, hidden: int, shared: int = 0) -> int:
    return n_layers * (D * experts + (experts + shared) * 3 * D * hidden + D)


def moe_activated_params(D: int, n_layers: int, experts: int, hidden: int, top_k: int, shared: int = 0) -> int:
    return n_layers * (D * experts + (top_k + shared) * 3 * D * hidden + D)


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, value)
    return max(multiple, int(math.ceil(max(1, value) / multiple) * multiple))


def _hidden_candidates(raw: list[float], multiple: int, max_hidden: int) -> list[int]:
    vals = set()
    for r in raw:
        if not math.isfinite(r) or r <= 0:
            continue
        center = max(1, int(round(r / max(1, multiple))))
        for delta in range(-3, 4):
            vals.add(_round_up((center + delta) * max(1, multiple), multiple))
    return sorted(v for v in vals if 1 <= v <= max_hidden)


def search_moe_shape(
    D: int,
    n_layers: int,
    target_accessible: int,
    target_activated: int,
    max_experts: int = 256,
    max_top_k: int = 4,
    max_hidden: int = 4096,
    hidden_multiple: int = 8,
    shared: int = 0,
) -> MoEShape:
    """Grid search over (experts, top_k, hidden) minimising the summed relative error
    to the accessible and per-token activated parameter targets."""
    best = None
    for experts in range(1, max_experts + 1):
        for top_k in range(1, min(experts, max_top_k) + 1):
            base = n_layers * (D * experts + D)
            h_acc = (target_accessible - base) / max(1, n_layers * (experts + shared) * 3 * D)
            h_act = (target_activated - base) / max(1, n_layers * (top_k + shared) * 3 * D)
            for hidden in _hidden_candidates([h_acc, h_act, (h_acc + h_act) / 2.0], hidden_multiple, max_hidden):
                acc = moe_accessible_params(D, n_layers, experts, hidden, shared)
                act = moe_activated_params(D, n_layers, experts, hidden, top_k, shared)
                e_acc = abs(acc - target_accessible) / max(1, target_accessible)
                e_act = abs(act - target_activated) / max(1, target_activated)
                cand = (e_acc + e_act, max(e_acc, e_act), acc, act, experts, hidden, top_k)
                if best is None or cand < best:
                    best = cand
    if best is None:
        raise ValueError("No MoE shape found; increase the search bounds.")
    _, _, acc, act, experts, hidden, top_k = best
    return MoEShape(experts, hidden, top_k, acc, act)


def moe_shape_with_vlumen(cfg: VLumenConfig) -> MoEShape:
    """MoE(k) inside MoE + V-LUMEN: k * P_v accessible, V-LUMEN's activated cost per token."""
    n_layers = len(cfg.fusion_layers)
    return search_moe_shape(
        cfg.hidden_dim, n_layers,
        target_accessible=max(1, int(vlumen_params(cfg) * cfg.budget_scale)),
        target_activated=max(1, vlumen_activated_params_per_token(cfg)),
    )


def moe_shape_only(cfg: VLumenConfig) -> MoEShape:
    """MoE(1 + k): matched to V-LUMEN(1) + MoE(k) in accessible and activated parameters."""
    own = moe_shape_with_vlumen(cfg)
    return search_moe_shape(
        cfg.hidden_dim, len(cfg.fusion_layers),
        target_accessible=vlumen_params(cfg) + own.accessible,
        target_activated=vlumen_activated_params_per_token(cfg) + own.activated,
    )
