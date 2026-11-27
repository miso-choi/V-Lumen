"""Hook-based wrapper that attaches V-LUMEN (and/or an MoE adapter) to a frozen Qwen3-VL.

Hooks registered on the backbone's decoder layers (prefill only; decode steps with
T == 1 reuse the memory already fused into the KV cache):

  layer 0, pre-hook          frozen codebook: visual hidden states -> codebook IDs
  first fusion layer, pre    Text-Conditioned Hashing: per-patch relevance bins (PRH / DRH)
  every fusion layer, post   V-LUMEN module: memory lookup + gated residual fusion
  every fusion layer, post   MoE adapter (MoE-only / MoE + V-LUMEN), after V-LUMEN

V-LUMEN hooks are active only while `self._ctx` is populated, i.e. inside
`forward()` or the `vlumen_context()` manager used for generation. Outside of it
the backbone runs unmodified, which is how the vanilla baseline is evaluated.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .codebook import FrozenCodebookQuantizer
from .config import IMAGE_TOKEN_TYPE, VLumenConfig
from .fusion import VLumenModule
from .moe import MoEAdapter

# ──────────────────────────────────────────────────────────────────────────────
#  Text-Conditioned Hashing (§2.4, App. D.1-D.2)
# ──────────────────────────────────────────────────────────────────────────────

def rank_discretize(scores: torch.Tensor, n_bins: int) -> torch.Tensor:
    """(B, N) scores -> (B, N) rank bucket in [0, n_bins)."""
    _, N = scores.shape
    ranks = scores.argsort(dim=-1).argsort(dim=-1)
    return (ranks.long() * int(n_bins) // max(N, 1)).clamp_(0, int(n_bins) - 1)


def query_end_positions(text_ids: torch.Tensor, is_img: torch.Tensor, end_token_id: int) -> torch.Tensor:
    """(B,) position of the last question token (right before the first end token
    after the image block), or -1. Identical at training and generation time."""
    B, _ = text_ids.shape
    pos = torch.full((B,), -1, dtype=torch.long, device=text_ids.device)
    for b in range(B):
        img = torch.nonzero(is_img[b], as_tuple=False).flatten()
        start = int(img[-1]) + 1 if img.numel() else 0
        ends = torch.nonzero(text_ids[b, start:] == end_token_id, as_tuple=False).flatten()
        if ends.numel() and int(ends[0]) > 0:
            pos[b] = start + int(ends[0]) - 1
    return pos


def text_relevance_bins(
    hidden: torch.Tensor,                    # (B, T, D) input of the first fusion layer
    is_img: torch.Tensor,                    # (B, T)
    text_pos: torch.Tensor,                  # (B,) h_txt position, -1 if none
    cfg: VLumenConfig,
    prh_q: Optional[nn.Linear],
    prh_k: Optional[nn.Linear],
    drh_prior: Optional[torch.Tensor],       # (G*G,)
) -> Optional[torch.Tensor]:
    """(B, G, G) per-patch TCH bins, or None when no sample has a full image grid."""
    B, _, D = hidden.shape
    g = cfg.visual_grid
    n_patches = g * g
    valid = is_img.sum(dim=1) == n_patches
    if not valid.any():
        return None

    h_text = hidden[torch.arange(B, device=hidden.device), text_pos.clamp_min(0)]   # (B, D)
    h_vis = torch.zeros(B, n_patches, D, device=hidden.device, dtype=hidden.dtype)
    for b in torch.nonzero(valid, as_tuple=False).flatten().tolist():
        h_vis[b] = hidden[b, torch.nonzero(is_img[b], as_tuple=False).flatten()]

    score_prh = score_drh = None
    if cfg.use_prh:
        q = prh_q(h_text.to(prh_q.weight.dtype))                         # (B, P)
        k = prh_k(h_vis.to(prh_k.weight.dtype))                          # (B, N, P)
        score_prh = (q.unsqueeze(1) * k).sum(-1) / (q.shape[-1] ** 0.5)
    if cfg.use_drh:
        raw = (h_text.float().unsqueeze(1) * h_vis.float()).sum(-1) / (D ** 0.5)
        prob = torch.softmax(raw, dim=-1)
        score_drh = prob - cfg.drh_lambda * drh_prior.to(device=prob.device, dtype=prob.dtype).unsqueeze(0)

    if score_prh is not None and score_drh is not None:
        eta = float(cfg.tch_ensemble_eta)
        def _norm(x):
            return x / (x.std(dim=-1, keepdim=True) + 1e-6)
        scores = eta * _norm(score_prh.detach().float()) + (1.0 - eta) * _norm(score_drh.float())
    elif score_prh is not None:
        scores = score_prh.detach().float()
    else:
        scores = score_drh.float()

    bins = rank_discretize(scores, cfg.tch_n_bins)
    bins[text_pos < 0] = 0
    return bins.view(B, g, g)


# ──────────────────────────────────────────────────────────────────────────────
#  Wrapper
# ──────────────────────────────────────────────────────────────────────────────

def _layers(backbone) -> nn.ModuleList:
    """Decoder layers of a (possibly PEFT-wrapped) Qwen3-VL model."""
    return backbone.model.language_model.layers


class VLumenVLM(nn.Module):
    """Frozen backbone + V-LUMEN modules and/or MoE adapters at `cfg.fusion_layers`.

    LoRA, when used, is applied to `self.model` with PEFT *after* construction;
    the hooks live on the decoder-layer objects and keep working.
    """

    def __init__(self, base_model, cfg: VLumenConfig, quantizer: Optional[FrozenCodebookQuantizer] = None):
        super().__init__()
        self.model = base_model
        self.config = cfg
        self.quantizer = quantizer
        self.moe_enabled = True
        self._ctx: Dict = {}
        self._hooks: List = []
        layers = _layers(base_model)

        self.vlumen_modules: Optional[nn.ModuleList] = None
        self.prh_proj_q = self.prh_proj_k = None
        if cfg.has_vlumen:
            if quantizer is None:
                raise ValueError("a FrozenCodebookQuantizer is required when V-LUMEN is enabled")
            # PRH projections: random and frozen (no gradient reaches them through
            # rank bucketing + XOR + integer gather).
            if cfg.use_prh:
                self.prh_proj_q = nn.Linear(cfg.hidden_dim, cfg.prh_proj_dim, bias=False)
                self.prh_proj_k = nn.Linear(cfg.hidden_dim, cfg.prh_proj_dim, bias=False)
                for proj in (self.prh_proj_q, self.prh_proj_k):
                    nn.init.normal_(proj.weight, std=0.02)
                    proj.weight.requires_grad_(False)
            if cfg.use_drh:
                n_patches = cfg.visual_grid * cfg.visual_grid
                # DRH prior: EMA of the mean text-to-patch attention (uniform at init).
                self.register_buffer("drh_ema_prior", torch.full((n_patches,), 1.0 / n_patches))
            else:
                self.drh_ema_prior = None
            self.vlumen_modules = nn.ModuleList([VLumenModule.from_config(cfg) for _ in cfg.fusion_layers])

            self._hooks.append(layers[0].register_forward_pre_hook(self._codebook_hook, with_kwargs=True))
            if cfg.use_tch:
                self._hooks.append(
                    layers[min(cfg.fusion_layers)].register_forward_pre_hook(self._tch_hook, with_kwargs=True)
                )
            for module, layer_id in zip(self.vlumen_modules, cfg.fusion_layers):
                self._hooks.append(layers[layer_id].register_forward_hook(self._make_fusion_hook(module)))
        else:
            self.drh_ema_prior = None

        self.moe_modules: Optional[nn.ModuleList] = None
        if cfg.has_moe:
            self.moe_modules = nn.ModuleList([
                MoEAdapter(cfg.hidden_dim, cfg.moe_num_experts, cfg.moe_expert_hidden_dim,
                           cfg.moe_top_k, cfg.moe_num_shared_experts)
                for _ in cfg.fusion_layers
            ])
            for module, layer_id in zip(self.moe_modules, cfg.fusion_layers):
                self._hooks.append(layers[layer_id].register_forward_hook(self._make_moe_hook(module)))

    # ── hooks ────────────────────────────────────────────────────────────────
    @staticmethod
    def _hidden_from(args, kwargs):
        return args[0] if args else kwargs.get("hidden_states")

    def _codebook_hook(self, module, args, kwargs):
        if not self._ctx:
            return
        hidden = self._hidden_from(args, kwargs)
        if hidden is None or hidden.shape[1] == 1:
            return
        B, T, _ = hidden.shape
        is_img = self._ctx["token_type_ids"] == IMAGE_TOKEN_TYPE
        visual_ids = torch.zeros(B, T, dtype=torch.long, device=hidden.device)
        if is_img.any():
            visual_ids[is_img] = self.quantizer(hidden[is_img])
        self._ctx["visual_ids"] = visual_ids

    def _tch_hook(self, module, args, kwargs):
        if not self._ctx or "visual_ids" not in self._ctx:
            return
        hidden = self._hidden_from(args, kwargs)
        if hidden is None or hidden.shape[1] == 1:
            return
        cfg = self.config
        is_img = self._ctx["token_type_ids"] == IMAGE_TOKEN_TYPE
        text_pos = query_end_positions(self._ctx["text_ids"], is_img, cfg.query_end_token_id)
        bins = text_relevance_bins(hidden, is_img, text_pos, cfg,
                                   self.prh_proj_q, self.prh_proj_k, self.drh_ema_prior)
        if bins is not None:
            self._ctx["text_bins"] = bins

        # DRH prior update (training only), after the bins of this batch are computed.
        if cfg.use_drh and self.training:
            with torch.no_grad():
                n_patches = cfg.visual_grid * cfg.visual_grid
                rows = torch.nonzero((is_img.sum(dim=1) == n_patches) & (text_pos >= 0),
                                     as_tuple=False).flatten().tolist()
                if rows:
                    D = hidden.shape[-1]
                    ht = torch.stack([hidden[b, text_pos[b]].float() for b in rows])
                    hv = torch.stack([hidden[b][is_img[b]].float() for b in rows])
                    prob = torch.softmax((ht.unsqueeze(1) * hv).sum(-1) / (D ** 0.5), dim=-1).mean(dim=0)
                    if self.drh_ema_prior.device != prob.device:
                        self.drh_ema_prior = self.drh_ema_prior.to(prob.device)
                    beta = float(cfg.drh_ema_beta)
                    self.drh_ema_prior.mul_(beta).add_((1.0 - beta) * prob.to(self.drh_ema_prior.dtype))

    def _make_fusion_hook(self, vlumen_module: VLumenModule):
        def hook(module, inputs, output):
            if not self._ctx or "visual_ids" not in self._ctx:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] == 1:
                return output
            new_h = vlumen_module(hidden, self._ctx["token_type_ids"], self._ctx["visual_ids"],
                                  text_bins=self._ctx.get("text_bins"))
            return (new_h,) + output[1:] if isinstance(output, tuple) else new_h
        return hook

    def _make_moe_hook(self, moe_module: MoEAdapter):
        def hook(module, inputs, output):
            if not self.moe_enabled:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] == 1:
                return output
            new_h = moe_module(hidden)
            return (new_h,) + output[1:] if isinstance(output, tuple) else new_h
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── context / forward ────────────────────────────────────────────────────
    def _make_ctx(self, input_ids: torch.Tensor) -> Dict:
        is_img = input_ids == self.config.image_token_id
        return {
            "token_type_ids": torch.where(is_img, torch.zeros_like(input_ids), torch.ones_like(input_ids)),
            "text_ids": input_ids.masked_fill(is_img, 0),
        }

    @contextlib.contextmanager
    def vlumen_context(self, input_ids: torch.Tensor):
        """Activate the V-LUMEN hooks for calls to `self.model` (e.g. `self.model.generate`)."""
        self._ctx = self._make_ctx(input_ids)
        try:
            yield
        finally:
            self._ctx = {}

    def set_moe_enabled(self, enabled: bool):
        self.moe_enabled = bool(enabled)

    def load_balance_loss(self) -> Optional[torch.Tensor]:
        if self.moe_modules is None:
            return None
        losses = [m.last_load_balance_loss for m in self.moe_modules if m.last_load_balance_loss is not None]
        return torch.stack(losses).mean() if losses else None

    def forward(self, input_ids: torch.Tensor, **kwargs):
        if self.moe_modules is not None:
            for m in self.moe_modules:
                m.last_load_balance_loss = None
        with self.vlumen_context(input_ids):
            outputs = self.model(input_ids=input_ids, **kwargs)
        if self.moe_modules is None or not self.moe_enabled:
            return outputs
        lb_loss = self.load_balance_loss()
        if lb_loss is not None and getattr(outputs, "loss", None) is not None:
            ce_loss = outputs.loss
            outputs.ce_loss = ce_loss
            outputs.loss = ce_loss + ce_loss.new_tensor(float(self.config.moe_load_balance_alpha)) * lb_loss.to(ce_loss.dtype)
        return outputs

    # ── parameters ───────────────────────────────────────────────────────────
    def vlumen_param_count(self) -> int:
        if self.vlumen_modules is None:
            return 0
        return sum(p.numel() for p in self.vlumen_modules.parameters() if p.requires_grad)
