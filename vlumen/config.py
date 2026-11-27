"""V-LUMEN configuration.

A single dataclass describes every setting in the paper (V-LUMEN-only, LoRA-only,
LoRA + V-LUMEN, MoE-only, MoE + V-LUMEN). It is stored in every checkpoint, so
evaluation rebuilds exactly the model that was trained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import List

IMAGE_TOKEN_TYPE = 0
TEXT_TOKEN_TYPE = 1

TRAIN_TARGETS = ("vlumen", "lora", "lora_vlumen", "moe", "moe_vlumen")


@dataclass
class VLumenConfig:
    # ── Setting ──────────────────────────────────────────────────────────────
    # vlumen      : V-LUMEN on the frozen backbone (V-LUMEN-only)
    # lora        : LoRA adapter only (LoRA-only)
    # lora_vlumen : LoRA(k) + V-LUMEN(1)
    # moe         : routed MoE adapter only (MoE-only)
    # moe_vlumen  : MoE(k) + V-LUMEN(1)
    train_target: str = "vlumen"

    # ── Backbone (Qwen3-VL-2B-Instruct) ─────────────────────────────────────
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    hidden_dim: int = 2048
    image_token_id: int = 151655
    # Token that closes the user turn (<|im_end|>); TCH reads h_txt right before it.
    query_end_token_id: int = 151645

    # ── Frozen codebook (§2.2) ──────────────────────────────────────────────
    visual_vocab_size: int = 16384        # LaVIT codebook rows (V)

    # ── Memory table (§2.5) ─────────────────────────────────────────────────
    memory_dim: int = 512                 # concatenated width of all memory cells
    n_heads: int = 8                      # hash heads per n-gram order
    ngram_orders: List[int] = field(default_factory=lambda: [2, 3])
    rows_per_cell: int = 81920            # lower bound; each cell gets a distinct prime >= this
    memory_init_std: float = 0.02

    # Decoder layers after which the retrieved memory is fused (Eq. 1).
    fusion_layers: List[int] = field(default_factory=lambda: [2, 8])
    residual_scale_init: float = 0.75

    # ── Spatial Aggregation-based Hashing (SAH, §2.3) ───────────────────────
    visual_grid: int = 8                  # 224x224 image -> 8x8 = 64 visual tokens
    use_axial_pooling: bool = True
    axial_mix_weight: float = 0.5         # beta in Eq. (4)
    use_coord_bias: bool = True
    coord_bins: int = 4                   # 4x4 coordinate buckets

    # ── Text-Conditioned Hashing (TCH, §2.4) ────────────────────────────────
    use_prh: bool = True                  # Projected Relevance Hashing
    use_drh: bool = True                  # Debiased Relevance Hashing
    prh_proj_dim: int = 64                # P
    tch_n_bins: int = 8                   # K rank buckets
    tch_ensemble_eta: float = 0.5         # eta: weight of PRH in the PRH + DRH ensemble
    drh_ema_beta: float = 0.99
    drh_lambda: float = 1.0

    # ── LoRA adapter (LoRA-only / LoRA + V-LUMEN) ───────────────────────────
    lora_rank: int = 0
    lora_alpha: int = 0
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=list)

    # ── MoE adapter (MoE-only / MoE + V-LUMEN) ──────────────────────────────
    moe_num_experts: int = 0
    moe_expert_hidden_dim: int = 0
    moe_top_k: int = 0
    moe_num_shared_experts: int = 0
    moe_load_balance_alpha: float = 1e-2

    # ── Bookkeeping (filled in by the training script) ──────────────────────
    budget_scale: float = 0.5             # k in V-LUMEN(1) + adapter(k)
    vlumen_params: int = 0                # trainable V-LUMEN params used for budget matching

    @property
    def has_vlumen(self) -> bool:
        return self.train_target in ("vlumen", "lora_vlumen", "moe_vlumen")

    @property
    def has_lora(self) -> bool:
        return self.train_target in ("lora", "lora_vlumen")

    @property
    def has_moe(self) -> bool:
        return self.train_target in ("moe", "moe_vlumen")

    @property
    def use_tch(self) -> bool:
        return self.use_prh or self.use_drh

    @property
    def n_cells(self) -> int:
        return len(self.ngram_orders) * self.n_heads

    @property
    def cell_dim(self) -> int:
        return self.memory_dim // self.n_cells

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VLumenConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"Unknown VLumenConfig fields in checkpoint: {unknown}")
        return cls(**d)

    def __post_init__(self):
        if self.train_target not in TRAIN_TARGETS:
            raise ValueError(f"train_target must be one of {TRAIN_TARGETS}, got {self.train_target!r}")
        if self.memory_dim % self.n_cells != 0:
            raise ValueError(
                f"memory_dim={self.memory_dim} must be divisible by "
                f"len(ngram_orders) x n_heads = {self.n_cells}"
            )
