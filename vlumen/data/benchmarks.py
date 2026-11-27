"""Evaluation benchmarks and the few-shot split (paper §3.1).

Reasoning benchmarks (Tab. 2): ChartMuseum, FRIEDA, VisRes, MMMU, MMMU-Pro.
For each, a fixed split seed assigns ~10% of the examples to a few-shot training
pool (used in both training stages) and the rest to evaluation; evaluation uses
up to the first 1,000 held-out examples.

Zero-shot benchmarks (Tab. 3): TextVQA, OCR-VQA, VizWiz-VQA, evaluated on the
same held-out partition. They are never used for training.

All images are resized to 224 x 224.
"""

from __future__ import annotations

import ast
import math
import random
from collections import Counter
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = (224, 224)
REASONING_BENCHMARKS = ("chartmuseum", "frieda", "visres", "mmmu", "mmmu_pro")
ZERO_SHOT_BENCHMARKS = ("textvqa", "ocrvqa", "vizwiz")
BENCHMARKS = REASONING_BENCHMARKS + ZERO_SHOT_BENCHMARKS

FEW_SHOT_RATIO = 0.1
SPLIT_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
#  Split helpers
# ──────────────────────────────────────────────────────────────────────────────

def _in_split(idx: int, split: str, ratio: float, seed: int) -> bool:
    """Per-example split used by every benchmark except MMMU-Pro."""
    if split == "all":
        return True
    in_train = random.Random(seed + idx).random() < ratio
    if split == "train":
        return in_train
    if split == "eval":
        return not in_train
    raise ValueError(f"unknown split {split!r}")


def _exact_split(n: int, split: str, ratio: float, seed: int) -> set[int] | None:
    """Exact-size shuffled split (MMMU-Pro)."""
    if split == "all":
        return None
    shuffled = list(range(n))
    random.Random(seed).shuffle(shuffled)
    train = set(shuffled[: math.floor(n * ratio)])
    if split == "train":
        return train
    if split == "eval":
        return set(range(n)) - train
    raise ValueError(f"unknown split {split!r}")


def _resize(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


def _answer_text(ans) -> str:
    if isinstance(ans, list) and len(ans) == 1:
        return str(ans[0]).strip()
    return str(ans).strip()


def _sample(images, question, answer, source):
    return {"images": images, "question": question, "answer": answer, "source": source}


def _mcq(question: str, options) -> str:
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except Exception:
            options = []
    if not isinstance(options, list) or not options:
        return question
    lines = [question.strip(), "Options:"] + [f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(options)]
    return "\n".join(lines).strip()


# ──────────────────────────────────────────────────────────────────────────────
#  Loaders: (n, split, ratio, seed) -> list of samples, in dataset order
# ──────────────────────────────────────────────────────────────────────────────

def load_chartmuseum(n, split, ratio, seed):
    ds = load_dataset("lytang/ChartMuseum", split="test")
    root = Path(snapshot_download("lytang/ChartMuseum", repo_type="dataset", allow_patterns=["images/*"]))
    out = []
    for idx, row in enumerate(ds):
        if _in_split(idx, split, ratio, seed):
            img = _resize(Image.open(root / row["image"]))
            out.append(_sample([img], row["question"], _answer_text(row["answer"]), "chartmuseum"))
    return out if n is None else out[:n]


def load_frieda(n, split, ratio, seed):
    ds = load_dataset("knowledge-computing/FRIEDA", split="data")
    out = []
    for idx, row in enumerate(ds):
        if not _in_split(idx, split, ratio, seed):
            continue
        images = []
        for p in row.get("images") or []:
            try:
                images.append(_resize(Image.open(hf_hub_download(
                    "knowledge-computing/FRIEDA", p, repo_type="dataset"))))
            except Exception:
                continue
        if images:
            out.append(_sample(images, row["question_text"], _answer_text(row["expected_answer"]), "frieda"))
    return out if n is None else out[:n]


_VISRES_CONFIGS = [
    "level_1_global_occlusion_50", "level_1_global_occlusion_70", "level_1_global_occlusion_80",
    "level_1_edges", "level_1_location_random_sampling", "level_1_brightness", "level_1_blur",
    "level_1_rotation", "level_1_rotation_random_sampling", "level_1_edges_random_sampling",
    "level_1_location", "level_2_uniform_count", "level_2_count_progression",
    "level_2_uniform_orientation", "level_2_count_2_same_1_diff", "level_2_orientation_2same_1diff",
    "level_2_uniform_color", "level_2_count_arithmetic", "level_2_count_minmax",
    "level_2_orientation_3_diff", "level_2_color_2same_1diff",
    "level_2_color_3_diff", "level_3_independent_color_object_rientation",  # sic: subset name on the Hub
    "level_3_coupled_color_orientation",
    "level_3_coupled_color_count", "level_3_spiral_color_orientation",
]


def load_visres(n, split, ratio, seed, configs_per_level: int = 4):
    available = set(get_dataset_config_names("tiiuae/visres_bench"))
    configs = [c for c in _VISRES_CONFIGS if c in available]
    # Use `configs_per_level` randomly chosen subsets of each difficulty level.
    rng = random.Random(seed)
    selected = []
    for level in ("level_1", "level_2", "level_3"):
        pool = [c for c in configs if c.startswith(level)]
        rng.shuffle(pool)
        selected.extend(pool[:configs_per_level])
    out, seen = [], 0
    for cfg in selected:
        for row in load_dataset("tiiuae/visres_bench", cfg, split="test"):
            idx, seen = seen, seen + 1
            if not _in_split(idx, split, ratio, seed):
                continue
            imgs = [_resize(im) for im in (row.get("images") or []) if isinstance(im, Image.Image)]
            if not imgs:
                continue
            q = row.get("guided_question") or row.get("generic_question") or row.get("question") or ""
            out.append(_sample(imgs, q, _answer_text(row["answer"]), f"visres:{cfg}"))
            if n is not None and len(out) >= n:
                return out
    return out if n is None else out[:n]


def _mmmu_images(row):
    return [_resize(row[f"image_{i}"]) for i in range(1, 8) if isinstance(row.get(f"image_{i}"), Image.Image)]


def load_mmmu(n, split, ratio, seed):
    out, seen = [], 0
    for cfg in sorted(get_dataset_config_names("MMMU/MMMU")):
        for row in load_dataset("MMMU/MMMU", cfg, split="validation"):
            idx, seen = seen, seen + 1
            if not _in_split(idx, split, ratio, seed):
                continue
            images = _mmmu_images(row)
            if images:
                q = _mcq(row.get("question", ""), row.get("options"))
                out.append(_sample(images, q, _answer_text(row.get("answer", "")), f"MMMU/MMMU:{cfg}"))
    return out if n is None else out[:n]


def load_mmmu_pro(n, split, ratio, seed):
    cfg = "standard (10 options)"
    ds = load_dataset("MMMU/MMMU_Pro", cfg, split="test")
    keep = _exact_split(len(ds), split, ratio, seed)
    out = []
    for idx in range(len(ds)):
        if keep is not None and idx not in keep:
            continue
        row = ds[idx]
        images = _mmmu_images(row)
        if not images:
            continue
        q = _mcq(row.get("question", ""), row.get("options"))
        out.append(_sample(images, q, _answer_text(row.get("answer", "")), f"MMMU/MMMU_Pro:{cfg}"))
        if n is not None and len(out) >= n:
            break
    return out


def _load_vqa(repo, hf_split, source, n, split, ratio, seed):
    """VQA-style sets with several annotator answers; the most common one is the target."""
    out = []
    for idx, row in enumerate(load_dataset(repo, split=hf_split)):
        if not _in_split(idx, split, ratio, seed):
            continue
        img, answers = row.get("image"), row.get("answers", [])
        if not isinstance(img, Image.Image) or not answers:
            continue
        gt = Counter(answers).most_common(1)[0][0]
        out.append(_sample([_resize(img)], row.get("question", ""), _answer_text(gt), source))
        if n is not None and len(out) >= n:
            break
    return out


def load_textvqa(n, split, ratio, seed):
    return _load_vqa("lmms-lab/TextVQA", "validation", "textvqa", n, split, ratio, seed)


def load_vizwiz(n, split, ratio, seed):
    return _load_vqa("lmms-lab-encoder/VizWiz-VQA", "val", "vizwiz", n, split, ratio, seed)


def load_ocrvqa(n, split, ratio, seed):
    """One sample per (image, question) pair; the split runs over the flattened questions."""
    out, flat = [], 0
    for row in load_dataset("howard-hou/OCR-VQA", split="validation"):
        img, questions, answers = row.get("image"), row.get("questions") or [], row.get("answers") or []
        if not isinstance(img, Image.Image) or not questions or not answers:
            continue
        image = None
        for q, a in zip(questions, answers):
            keep = _in_split(flat, split, ratio, seed)
            flat += 1
            if not keep or not str(a).strip():
                continue
            if image is None:
                image = _resize(img)
            out.append(_sample([image], q, _answer_text(a), "ocrvqa"))
            if n is not None and len(out) >= n:
                return out
    return out


_LOADERS = {
    "chartmuseum": load_chartmuseum,
    "frieda": load_frieda,
    "visres": load_visres,
    "mmmu": load_mmmu,
    "mmmu_pro": load_mmmu_pro,
    "textvqa": load_textvqa,
    "ocrvqa": load_ocrvqa,
    "vizwiz": load_vizwiz,
}


def load_benchmark(name: str, n: int | None, split: str = "eval",
                   ratio: float = FEW_SHOT_RATIO, seed: int = SPLIT_SEED) -> list[dict]:
    if name not in _LOADERS:
        raise ValueError(f"unknown benchmark {name!r}; choose from {BENCHMARKS}")
    return _LOADERS[name](n, split, ratio, seed)


# ──────────────────────────────────────────────────────────────────────────────
#  Few-shot training pool
# ──────────────────────────────────────────────────────────────────────────────

class FewShotDataset(Dataset):
    """Few-shot split of the reasoning benchmarks, as chat messages."""

    def __init__(self, cap_per_benchmark: int = 200, ratio: float = FEW_SHOT_RATIO, seed: int = SPLIT_SEED):
        self.samples = []
        for name in REASONING_BENCHMARKS:
            s = load_benchmark(name, cap_per_benchmark, split="train", ratio=ratio, seed=seed)
            print(f"[data:few-shot] {name:12s} samples={len(s)}")
            self.samples.extend(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return [
            {"role": "user", "content": [{"type": "image", "image": im} for im in s["images"]]
                                        + [{"type": "text", "text": str(s["question"]).strip()}]},
            {"role": "assistant", "content": str(s["answer"]).strip()},
        ]
