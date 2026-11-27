"""Multi-head memory table with spatially aggregated, text-conditioned hash keys.

Paper §2.3-2.5. For every n-gram order n and hash head h there is one memory cell:
an embedding table with a distinct prime number of rows. A visual token's key is
built from its codebook ID, then

  1. Spatial Aggregation-based Hashing (SAH, §2.3)
       axial pooling : blend the row / column summary of the 8x8 code grid into the key
       coordinate bias: XOR a coarse 4x4 position bucket into the key
  2. Text-Conditioned Hashing (TCH, §2.4)
       XOR a per-patch text-relevance bin (PRH / DRH) into the key

and each cell maps the key to a row index with its own hash function. The
retrieved rows of all cells are concatenated into a (memory_dim,) vector.
Keys are integers and are never trained; only the table rows are.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .config import IMAGE_TOKEN_TYPE

_STRIP_HASH_PRIME = 131071      # 2^17 - 1, rolling hash for strips without a repeated code
_COORD_SALT = 1_000_003
_TEXT_SALT = 2_000_003


# ──────────────────────────────────────────────────────────────────────────────
#  Prime table sizes
# ──────────────────────────────────────────────────────────────────────────────

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _next_primes(base: int, count: int) -> List[int]:
    """`count` consecutive primes >= base."""
    out, n = [], max(base, 2)
    while len(out) < count:
        if _is_prime(n):
            out.append(n)
        n += 1
    return out


def cell_table_sizes(rows_per_cell: int, n_orders: int, n_heads: int) -> List[List[int]]:
    """Distinct prime row counts for every (order, head) memory cell."""
    sizes, base = [], rows_per_cell
    for _ in range(n_orders):
        primes = _next_primes(base, n_heads)
        sizes.append(primes)
        base = max(primes) + 1
    return sizes


# ──────────────────────────────────────────────────────────────────────────────
#  Spatial Aggregation-based Hashing (§2.3) and the TCH salt (§2.4)
# ──────────────────────────────────────────────────────────────────────────────

def _strip_hash(ids: torch.Tensor) -> int:
    """Order-independent rolling hash of a 1-D strip of codebook IDs (sorted first)."""
    h = 0
    for x in ids.sort()[0]:
        h = (h * _STRIP_HASH_PRIME + int(x.item())) & 0x7FFF_FFFF
    return h


def axial_pool(code_grid: torch.Tensor, key_grid: torch.Tensor, mix_weight: float) -> torch.Tensor:
    """Blend the row/column summary of the code grid into the key grid (Eq. 3-4).

    A strip's summary is its modal codebook ID if some ID repeats, otherwise a
    rolling hash of its sorted IDs. The result is an integer hash key.
    """
    H, W = code_grid.shape
    row_key = torch.empty_like(code_grid)
    col_key = torch.empty_like(code_grid)
    for r in range(H):
        vals, counts = torch.unique(code_grid[r, :], return_counts=True)
        if counts.max().item() >= 2:
            row_key[r, :] = vals[counts.argmax()].item()
        else:
            row_key[r, :] = _strip_hash(code_grid[r, :])
    for c in range(W):
        vals, counts = torch.unique(code_grid[:, c], return_counts=True)
        if counts.max().item() >= 2:
            col_key[:, c] = vals[counts.argmax()].item()
        else:
            col_key[:, c] = _strip_hash(code_grid[:, c])
    axial = ((row_key + col_key) // 2).clamp_min(0).long()
    w = float(mix_weight)
    mixed = (1.0 - w) * key_grid.float() + w * axial.float()
    return mixed.round().clamp_min(0).long()


def coord_bias(key_grid: torch.Tensor, coord_bins: int) -> torch.Tensor:
    """XOR a coarse (coord_bins x coord_bins) position bucket into the key grid (Eq. 5)."""
    H, W = key_grid.shape
    rr = torch.arange(H, device=key_grid.device).view(-1, 1).expand(H, W)
    cc = torch.arange(W, device=key_grid.device).view(1, -1).expand(H, W)
    bin_r = (rr.long() * coord_bins) // max(H, 1)
    bin_c = (cc.long() * coord_bins) // max(W, 1)
    bucket = bin_r * coord_bins + bin_c
    return (key_grid.long() ^ (bucket * _COORD_SALT)).clamp_min(0)


def text_salt(key_grid: torch.Tensor, text_bins: torch.Tensor) -> torch.Tensor:
    """XOR the per-patch text-relevance bin into the key grid (Eq. 11)."""
    if text_bins.shape != key_grid.shape:
        raise ValueError(f"text_bins {tuple(text_bins.shape)} != key grid {tuple(key_grid.shape)}")
    return (key_grid.long() ^ (text_bins.long() * _TEXT_SALT)).clamp_min(0)


# ──────────────────────────────────────────────────────────────────────────────
#  Memory table
# ──────────────────────────────────────────────────────────────────────────────

class MultiHeadMemoryTable(nn.Module):
    """n_orders x n_heads memory cells, each an embedding table of prime size.

    forward(visual_ids, token_type_ids, text_bins) -> (B, T, memory_dim)

    Images whose token block is exactly visual_grid x visual_grid use the SAH /
    TCH key. Any other sample (e.g. several images) falls back to a plain
    polynomial rolling hash over the code sequence.
    """

    def __init__(
        self,
        memory_dim: int = 512,
        n_heads: int = 8,
        ngram_orders: Optional[List[int]] = None,
        rows_per_cell: int = 81920,
        init_std: float = 0.02,
        visual_grid: int = 8,
        use_axial_pooling: bool = True,
        axial_mix_weight: float = 0.5,
        use_coord_bias: bool = True,
        coord_bins: int = 4,
        use_text_salt: bool = True,
    ):
        super().__init__()
        self.orders = sorted(ngram_orders or [2, 3])
        self.n_orders = len(self.orders)
        self.n_heads = n_heads
        self.memory_dim = memory_dim
        if memory_dim % (n_heads * self.n_orders) != 0:
            raise ValueError(f"memory_dim={memory_dim} must be divisible by n_heads x n_orders")
        self.cell_dim = memory_dim // (n_heads * self.n_orders)
        self.visual_grid = visual_grid
        self.use_axial_pooling = use_axial_pooling
        self.axial_mix_weight = float(axial_mix_weight)
        self.use_coord_bias = use_coord_bias
        self.coord_bins = int(coord_bins)
        self.use_text_salt = use_text_salt

        self.table_sizes = cell_table_sizes(rows_per_cell, self.n_orders, n_heads)
        self.tables = nn.ModuleList()
        for o in range(self.n_orders):
            for h in range(n_heads):
                emb = nn.Embedding(self.table_sizes[o][h], self.cell_dim)
                nn.init.normal_(emb.weight, std=init_std)
                self.tables.append(emb)

        # Random, fixed hash coefficients per (order, head); saved with the table.
        for o, n in enumerate(self.orders):
            self.register_buffer(f"coeff_{o}", torch.randint(1, 1_000_000, (n_heads, n), dtype=torch.long))

    @property
    def uses_spatial_key(self) -> bool:
        return self.use_axial_pooling or self.use_coord_bias or self.use_text_salt

    # ── hashing ──────────────────────────────────────────────────────────────
    def _rolling_hash(self, seq: torch.Tensor, o: int) -> torch.Tensor:
        """Plain polynomial rolling n-gram hash over the code sequence -> (B, T, n_heads)."""
        n = self.orders[o]
        coeff = getattr(self, f"coeff_{o}")
        B, T = seq.shape
        raw = torch.zeros(B, T, self.n_heads, dtype=torch.long, device=seq.device)
        for i in range(n):
            shift = n - 1 - i
            if shift == 0:
                tok = seq.long()
            else:
                tok = torch.roll(seq.long(), shifts=shift, dims=1)
                tok[:, :shift] = 0
            raw = raw + tok.unsqueeze(-1) * coeff[:, i].view(1, 1, -1)
        out = torch.zeros_like(raw)
        for h in range(self.n_heads):
            out[:, :, h] = raw[:, :, h] % self.table_sizes[o][h]
        out[:, : n - 1, :] = 0
        return out

    def _grid_rows(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        """(B,) True where the sample has exactly visual_grid^2 image tokens."""
        expected = self.visual_grid * self.visual_grid
        return (token_type_ids == IMAGE_TOKEN_TYPE).sum(dim=1) == expected

    def _spatial_keys(self, seq, token_type_ids, valid, text_bins) -> torch.Tensor:
        """Per-token integer keys; image tokens of grid-complete samples get the SAH/TCH key."""
        keys = seq.long().clone()
        g = self.visual_grid
        for b in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            img_pos = torch.nonzero(token_type_ids[b] == IMAGE_TOKEN_TYPE, as_tuple=False).flatten()
            grid = keys[b, img_pos].view(g, g)
            if self.use_axial_pooling:
                grid = axial_pool(seq[b, img_pos].long().view(g, g), grid, self.axial_mix_weight)
            if self.use_coord_bias:
                grid = coord_bias(grid, self.coord_bins)
            if text_bins is not None:
                grid = text_salt(grid, text_bins[b])
            keys[b, img_pos] = grid.view(-1)
        return keys

    def _spatial_hash(self, seq, o, token_type_ids, text_bins) -> torch.Tensor:
        """Per-cell hash of the spatial key; samples without a full grid keep the rolling hash."""
        out = self._rolling_hash(seq, o)
        valid = self._grid_rows(token_type_ids)
        keys = self._spatial_keys(seq, token_type_ids, valid, text_bins)
        n = self.orders[o]
        coeff = getattr(self, f"coeff_{o}")
        for b in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            raw = torch.zeros(seq.shape[1], self.n_heads, dtype=torch.long, device=seq.device)
            for i in range(n):
                raw = raw + (keys[b] + i).unsqueeze(-1) * coeff[:, i].view(1, -1)
            for h in range(self.n_heads):
                out[b, :, h] = raw[:, h] % self.table_sizes[o][h]
        return out

    def addresses(self, visual_ids, token_type_ids, text_bins=None) -> List[torch.Tensor]:
        """Row index of every cell, one (B, T, n_heads) tensor per order."""
        if not self.use_text_salt:
            text_bins = None
        spatial = self.use_axial_pooling or self.use_coord_bias or text_bins is not None
        return [
            self._spatial_hash(visual_ids, o, token_type_ids, text_bins) if spatial
            else self._rolling_hash(visual_ids, o)
            for o in range(self.n_orders)
        ]

    def gather(self, addresses: List[torch.Tensor]) -> torch.Tensor:
        parts = []
        for o, hash_ids in enumerate(addresses):
            heads = []
            for h in range(self.n_heads):
                p = self.table_sizes[o][h]
                heads.append(self.tables[o * self.n_heads + h](hash_ids[:, :, h].clamp(0, p - 1)))
            parts.append(torch.cat(heads, dim=-1))
        return torch.cat(parts, dim=-1)

    def forward(self, visual_ids, token_type_ids, text_bins=None) -> torch.Tensor:
        if visual_ids.dim() == 1:
            visual_ids = visual_ids.unsqueeze(0)
        return self.gather(self.addresses(visual_ids, token_type_ids, text_bins))
