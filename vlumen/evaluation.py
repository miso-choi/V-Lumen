"""Greedy generation and relaxed-accuracy scoring (paper §3.1, "Evaluations").

Every checkpoint is scored together with the vanilla backbone in the same pass:
  vanilla : backbone only (V-LUMEN hooks inactive, LoRA disabled, MoE disabled)
  model   : the trained setting
"""

from __future__ import annotations

import contextlib
import re

import torch
from qwen_vl_utils import process_vision_info

from .model import VLumenVLM


def build_inputs(sample: dict, processor, device: str):
    msgs = [{"role": "user", "content": (
        [{"type": "image", "image": img} for img in sample["images"]]
        + [{"type": "text", "text": sample["question"]}]
    )}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, *_ = process_vision_info(msgs)
    return processor(text=text, images=images, return_tensors="pt").to(device)


@torch.no_grad()
def generate(model: VLumenVLM, inputs, max_new_tokens: int, vanilla: bool = False) -> torch.Tensor:
    backbone = model.model
    if vanilla:
        adapters_off = backbone.disable_adapter() if hasattr(backbone, "disable_adapter") else contextlib.nullcontext()
        model.set_moe_enabled(False)
        try:
            with adapters_off:
                out = backbone.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            model.set_moe_enabled(True)
    else:
        model.set_moe_enabled(True)
        with model.vlumen_context(inputs["input_ids"]):
            out = backbone.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return out[:, inputs["input_ids"].shape[1]:]


# ── scoring ──────────────────────────────────────────────────────────────────

def _number(text: str):
    m = re.search(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return float(m.group().replace(",", "")) if m else None


def relaxed_accuracy(pred: str, gt: str, tol: float = 0.05) -> bool:
    """Numbers: within 5% relative error. Otherwise: exact match after normalising
    case, punctuation and whitespace."""
    p, g = _number(pred), _number(gt)
    if p is not None and g is not None:
        return abs(p - g) / max(abs(g), 1.0) <= tol

    def norm(s):
        s = re.sub(r"[^\w\s]", "", s.lower().strip())
        return " ".join(s.split())
    return norm(pred) == norm(gt)


def normalize_prediction(sample: dict, pred: str) -> str:
    """MMMU / MMMU-Pro are multiple choice: "A. ...", "(A) ..." -> "A"."""
    pred = str(pred).strip()
    if str(sample.get("source", "")).lower().startswith("mmmu/mmmu"):
        m = re.match(r"^\s*[\(\[]?\s*([A-Z])\s*[\)\]\.:,-]?(?:\s+.*)?$", pred, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return pred


def normalize_answer(gt: str) -> str:
    """Unwrap single-element list strings such as "['3']"."""
    if isinstance(gt, str) and gt.startswith("[") and gt.endswith("]"):
        import ast
        try:
            parsed = ast.literal_eval(gt)
            if isinstance(parsed, list) and len(parsed) == 1:
                gt = str(parsed[0])
        except Exception:
            pass
    return gt.strip()


def evaluate(model: VLumenVLM, processor, samples: list[dict], max_new_tokens: int, device: str,
             with_vanilla: bool = True, log_every: int = 50) -> tuple[list[dict], dict]:
    results, n_ok, n_ok_vanilla = [], 0, 0
    for i, sample in enumerate(samples):
        try:
            inputs = build_inputs(sample, processor, device)
            gt = normalize_answer(sample["answer"])
            row = {"gt": gt}
            if with_vanilla:
                pred = processor.batch_decode(generate(model, inputs, max_new_tokens, vanilla=True),
                                              skip_special_tokens=True)[0].strip()
                row["vanilla"] = pred
                row["ok_vanilla"] = relaxed_accuracy(normalize_prediction(sample, pred), gt)
            pred = processor.batch_decode(generate(model, inputs, max_new_tokens),
                                          skip_special_tokens=True)[0].strip()
            row["pred"] = pred
            row["ok"] = relaxed_accuracy(normalize_prediction(sample, pred), gt)
        except Exception as e:   # e.g. an image the processor cannot handle
            print(f"  sample {i} skipped: {e}")
            continue
        results.append(row)
        n_ok += row["ok"]
        n_ok_vanilla += row.get("ok_vanilla", 0)
        if (i + 1) % log_every == 0:
            msg = f"  {i + 1}/{len(samples)}  acc={n_ok / len(results) * 100:.2f}%"
            if with_vanilla:
                msg += f"  vanilla={n_ok_vanilla / len(results) * 100:.2f}%"
            print(msg, flush=True)

    n = max(len(results), 1)
    metrics = {"n": len(results), "acc": n_ok / n}
    if with_vanilla:
        metrics["vanilla_acc"] = n_ok_vanilla / n
        metrics["delta"] = metrics["acc"] - metrics["vanilla_acc"]
    return results, metrics
