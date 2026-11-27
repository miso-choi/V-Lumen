"""Routed MoE adapter used by the MoE-only and MoE + V-LUMEN settings."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_dim: int, expert_hidden_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, expert_hidden_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim, expert_hidden_dim, bias=False)
        self.down_proj = nn.Linear(expert_hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEAdapter(nn.Module):
    """Top-k routed SwiGLU experts (+ optional shared experts), residual + RMSNorm output.

    A Switch-style load-balancing loss E * sum_i f_i p_i is stored after every forward.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        expert_hidden_dim: int,
        top_k: int = 1,
        num_shared_experts: int = 0,
    ):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(hidden_dim, expert_hidden_dim) for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList(
            [SwiGLUExpert(hidden_dim, expert_hidden_dim) for _ in range(num_shared_experts)]
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.last_load_balance_loss: Optional[torch.Tensor] = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        dtype_in = h.dtype
        x = h.to(self.router.weight.dtype)
        B, T, D = x.shape
        flat = x.reshape(B * T, D)

        probs = torch.softmax(self.router(flat), dim=-1)
        top_w, top_i = torch.topk(probs, k=self.top_k, dim=-1)
        with torch.no_grad():
            dispatch = torch.zeros_like(probs)
            dispatch.scatter_(1, top_i, 1.0)
            frac_tokens = dispatch.mean(dim=0)
        self.last_load_balance_loss = float(self.num_experts) * torch.sum(frac_tokens * probs.mean(dim=0))

        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            mask = top_i == expert_id
            if not mask.any():
                continue
            token_idx, slot_idx = torch.nonzero(mask, as_tuple=True)
            out.index_add_(0, token_idx, expert(flat[token_idx]) * top_w[token_idx, slot_idx].unsqueeze(-1))
        if self.num_shared_experts > 0:
            shared = torch.zeros_like(flat)
            for expert in self.shared_experts:
                shared = shared + expert(flat)
            out = out + shared / float(self.num_shared_experts)

        out = self.norm(out).view(B, T, D)
        return (h + out).to(dtype_in)
