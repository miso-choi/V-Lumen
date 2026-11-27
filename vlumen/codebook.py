"""Discretization via a frozen codebook (paper §2.2).

Visual hidden states are projected by a frozen random matrix W_proj and mapped to
the index of the nearest (cosine) row of the pretrained LaVIT codebook. Neither
W_proj nor the codebook is trained; both are stored in every checkpoint so the
visual codes (and therefore the memory addresses) are identical across
mid-training, SFT and evaluation.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LAVIT_REPO = "rain1011/LaVIT-7B-v2"
LAVIT_ENCODER_FILE = "visual_tokenizer/tokenizer_encoder.bin"


def load_lavit_codebook() -> torch.Tensor:
    """Return the LaVIT codebook (16384 x 32).

    Set LAVIT_PATH to a local LaVIT-7B-v2 snapshot directory to skip the download.
    """
    local = os.environ.get("LAVIT_PATH")
    if local:
        path = Path(local) / LAVIT_ENCODER_FILE
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(LAVIT_REPO, LAVIT_ENCODER_FILE))
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state["quantize.embedding.weight"]


class FrozenCodebookQuantizer(nn.Module):
    """x (N, D_hidden) -> codebook index (N,) via a frozen random projection."""

    def __init__(self, in_dim: int, codebook: torch.Tensor, proj_seed: int = 0):
        super().__init__()
        _, code_dim = codebook.shape
        self.proj = nn.Linear(in_dim, code_dim, bias=False)
        g = torch.Generator().manual_seed(proj_seed)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(code_dim, in_dim, generator=g) / math.sqrt(in_dim))
        self.proj.weight.requires_grad_(False)
        self.register_buffer("codebook", F.normalize(codebook.float(), dim=-1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.proj(x.to(self.proj.weight.dtype)), dim=-1)
        return (z.to(self.codebook.dtype) @ self.codebook.T).argmax(dim=-1)

    def load_checkpoint_state(self, state: dict) -> None:
        """Strictly restore W_proj and the codebook saved in a checkpoint."""
        saved, built = tuple(state["codebook"].shape), tuple(self.codebook.shape)
        if saved != built:
            raise ValueError(f"checkpoint codebook {saved} != built codebook {built}")
        self.load_state_dict(state)
