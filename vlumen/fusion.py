"""Per-layer V-LUMEN module: memory lookup + Multi-Head Conditional Gating (paper §2.5, App. D.6).

For each fusion layer l and token t (image positions only):

    e_{o,h}  = row read from memory cell (o, h)                   (cell_dim,)
    a_{o,h}  = sigmoid(W_gate h_{l,t})_{o,h}                       one gate per cell
    u        = sum_{o,h} a_{o,h} * W_{o,h} e_{o,h}                 (D,)
    u        = u + DepthwiseConv1d_k3(u)                           local smoothing
    h_{l,t} <- h_{l,t} + s * RMSNorm(u)                            gated residual, learnable s
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .config import IMAGE_TOKEN_TYPE, VLumenConfig
from .memory_table import MultiHeadMemoryTable


class VLumenModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int = 512,
        n_heads: int = 8,
        ngram_orders: Optional[List[int]] = None,
        rows_per_cell: int = 81920,
        memory_init_std: float = 0.02,
        residual_scale_init: float = 0.75,
        visual_grid: int = 8,
        use_axial_pooling: bool = True,
        axial_mix_weight: float = 0.5,
        use_coord_bias: bool = True,
        coord_bins: int = 4,
        use_tch: bool = True,
    ):
        super().__init__()
        ngram_orders = ngram_orders or [2, 3]
        self.n_cells = len(ngram_orders) * n_heads
        if memory_dim % self.n_cells != 0:
            raise ValueError(f"memory_dim={memory_dim} must be divisible by n_cells={self.n_cells}")
        self.cell_dim = memory_dim // self.n_cells
        self.use_tch = use_tch

        self.memory = MultiHeadMemoryTable(
            memory_dim=memory_dim,
            n_heads=n_heads,
            ngram_orders=ngram_orders,
            rows_per_cell=rows_per_cell,
            init_std=memory_init_std,
            visual_grid=visual_grid,
            use_axial_pooling=use_axial_pooling,
            axial_mix_weight=axial_mix_weight,
            use_coord_bias=use_coord_bias,
            coord_bins=coord_bins,
            use_text_salt=use_tch,
        )
        # Multi-Head Conditional Gating: one gate logit per memory cell from h.
        self.gate_proj = nn.Linear(hidden_dim, self.n_cells, bias=False)
        self.value_projs = nn.ModuleList(
            [nn.Linear(self.cell_dim, hidden_dim, bias=False) for _ in range(self.n_cells)]
        )
        self.short_conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False
        )
        nn.init.zeros_(self.short_conv.weight)
        self.norm = nn.RMSNorm(hidden_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

        self.last_gate_mean: float = 0.0   # logging only

    @classmethod
    def from_config(cls, cfg: VLumenConfig) -> "VLumenModule":
        return cls(
            hidden_dim=cfg.hidden_dim,
            memory_dim=cfg.memory_dim,
            n_heads=cfg.n_heads,
            ngram_orders=cfg.ngram_orders,
            rows_per_cell=cfg.rows_per_cell,
            memory_init_std=cfg.memory_init_std,
            residual_scale_init=cfg.residual_scale_init,
            visual_grid=cfg.visual_grid,
            use_axial_pooling=cfg.use_axial_pooling,
            axial_mix_weight=cfg.axial_mix_weight,
            use_coord_bias=cfg.use_coord_bias,
            coord_bins=cfg.coord_bins,
            use_tch=cfg.use_tch,
        )

    def forward(
        self,
        h: torch.Tensor,                       # (B, T, D) hidden state after the layer
        token_type_ids: torch.Tensor,          # (B, T) 0 = image, 1 = text
        visual_ids: torch.Tensor,              # (B, T) codebook IDs (0 at text positions)
        text_bins: Optional[torch.Tensor] = None,  # (B, G, G) TCH bins
    ) -> torch.Tensor:
        e = self.memory(visual_ids, token_type_ids, text_bins if self.use_tch else None)
        img = (token_type_ids == IMAGE_TOKEN_TYPE).unsqueeze(-1).to(h.dtype)
        e = e * img                                            # memory is read at image positions only

        gate_logits = self.gate_proj(h.to(self.gate_proj.weight.dtype))
        gates = torch.sigmoid(gate_logits)
        self.last_gate_mean = gates.detach().mean().item()

        out = torch.zeros_like(h)
        for c, v_proj in enumerate(self.value_projs):
            v = v_proj(e[..., c * self.cell_dim:(c + 1) * self.cell_dim])
            out = out + gates[..., c:c + 1] * v
        out = out + self.short_conv(out.transpose(1, 2)).transpose(1, 2)
        out = self.norm(out)
        return (h + self.residual_scale.to(out.dtype) * out).to(h.dtype)
