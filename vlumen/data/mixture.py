"""Training data mixtures (paper §3.1).

Mid-training: document/chart, spatial, diagram and natural-scene sources, sampled
with fixed per-source weights. ScienceQA targets include the lecture
("Reasoning: ... Final Answer: ...") in this stage only.
SFT: six reasoning-oriented sources, uniform weights, answer-only targets.

Each source is truncated to its first `cap` training rows. Samples are drawn
from a seeded RNG, independently of the DataLoader index.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = (224, 224)
_PLACEHOLDER_COLORS = [
    (220, 50, 50), (50, 180, 50), (50, 50, 220), (200, 180, 50),
    (50, 200, 200), (200, 50, 200), (130, 130, 130), (220, 140, 50),
]


def _text(x) -> str:
    if x is None:
        return ""
    return x.strip() if isinstance(x, str) else str(x).strip()


def _image(v, idx: int) -> Image.Image:
    if isinstance(v, Image.Image):
        return v.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    # A few rows have no image; use a solid placeholder so the sample keeps its shape.
    return Image.new("RGB", IMAGE_SIZE, color=_PLACEHOLDER_COLORS[idx % len(_PLACEHOLDER_COLORS)])


def chat(image: Image.Image, question: str, answer: str) -> list:
    return [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": _text(question) or "Describe this image."},
        ]},
        {"role": "assistant", "content": _text(answer)},
    ]


def _map_cauldron(row, idx):
    imgs, texts = row.get("images") or [], row.get("texts") or []
    qa = texts[0] if texts else {}
    return chat(_image(imgs[0] if imgs else None, idx), qa.get("user", ""), qa.get("assistant", ""))


def _map_chartqa(row, idx):
    return chat(_image(row.get("image"), idx), row.get("query", ""), row.get("label", ""))


def _map_textcaps(row, idx):
    answer = row.get("answer", row.get("caption_str", row.get("caption", "")))
    return chat(_image(row.get("image"), idx), row.get("query", "Read and answer from the image."), answer)


def _map_gqa(row, idx):
    return chat(_image(row.get("image"), idx), row.get("question", ""), row.get("answer", ""))


def _map_refcoco(row, idx):
    expr = row.get("sentences", row.get("sentence", row.get("expression", "")))
    if isinstance(expr, list):
        expr = expr[0] if expr else ""
    answer = row.get("answer", row.get("bbox", row.get("ref_id", "")))
    return chat(_image(row.get("image"), idx), f"Locate the referred object: {expr}", answer)


def _map_scienceqa(row, idx, with_lecture: bool):
    q, choices, hint = row.get("question", ""), row.get("choices") or [], row.get("hint", "")
    if choices:
        q = f"{q}\nChoices: {' '.join(f'({i}) {c}' for i, c in enumerate(choices))}".strip()
    if hint:
        q = f"{q}\nContext: {hint}".strip()
    answer = row.get("answer", "")
    if isinstance(answer, int) and choices and 0 <= answer < len(choices):
        answer = choices[answer]
    answer = _text(answer)
    lecture = _text(row.get("lecture", ""))
    target = f"Reasoning:\n{lecture}\nFinal Answer: {answer}".strip() if (with_lecture and lecture) else answer
    return chat(_image(row.get("image"), idx), q, target)


@dataclass(frozen=True)
class Source:
    repo_id: str
    config: str | None
    map_fn: Callable


SOURCES: dict[str, Source] = {
    "docvqa":         Source("HuggingFaceM4/the_cauldron", "docvqa", _map_cauldron),
    "chartqa":        Source("HuggingFaceM4/ChartQA", None, _map_chartqa),
    "infographicvqa": Source("HuggingFaceM4/the_cauldron", "infographic_vqa", _map_cauldron),
    "gqa":            Source("lmms-lab/GQA", "train_balanced_instructions", _map_gqa),
    "refcoco":        Source("jxu124/refcoco", None, _map_refcoco),
    "ai2d":           Source("HuggingFaceM4/the_cauldron", "ai2d", _map_cauldron),
    "aokvqa":         Source("HuggingFaceM4/the_cauldron", "aokvqa", _map_cauldron),
    "textcaps":       Source("lmms-lab/TextCaps", None, _map_textcaps),
    "scienceqa":      Source("derek-thomas/ScienceQA", None, None),
}

# (source, sampling weight). Group weights: document 0.30, spatial 0.25,
# diagram 0.20, natural 0.25, split evenly over the sources of each group.
# The spatial group was designed with three sources (Visual Genome, GQA,
# RefCOCO); Visual Genome was not available in our runs, so its third of the
# spatial weight is unused. These are the weights the released checkpoints were
# trained with.
MIDTRAIN_MIXTURE: list[tuple[str, float]] = [
    ("docvqa", 0.30 / 3), ("chartqa", 0.30 / 3), ("infographicvqa", 0.30 / 3),
    ("gqa", 0.25 / 3), ("refcoco", 0.25 / 3),
    ("ai2d", 0.20 / 1),
    ("aokvqa", 0.25 / 3), ("textcaps", 0.25 / 3), ("scienceqa", 0.25 / 3),
]
SFT_MIXTURE: list[tuple[str, float]] = [
    (name, 1.0 / 6) for name in ("scienceqa", "ai2d", "docvqa", "chartqa", "gqa", "aokvqa")
]


class _SourceDataset(Dataset):
    def __init__(self, name: str, cap: int | None, scienceqa_lecture: bool):
        self.name = name
        self.src = SOURCES[name]
        self.scienceqa_lecture = scienceqa_lecture
        split = "train" if cap is None else f"train[:{cap}]"
        self.ds = load_dataset(self.src.repo_id, self.src.config, split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        if self.name == "scienceqa":
            return _map_scienceqa(row, idx, self.scienceqa_lecture)
        return self.src.map_fn(row, idx)


class MixtureDataset(Dataset):
    """Weighted mixture; __getitem__ ignores the index and draws (source, row) from a seeded RNG."""

    def __init__(self, stage: str, cap: int | None, seed: int = 42):
        if stage == "midtrain":
            mixture, lecture = MIDTRAIN_MIXTURE, True
        elif stage == "sft":
            mixture, lecture = SFT_MIXTURE, False
        else:
            raise ValueError(f"unknown stage {stage!r}")
        self.rng = random.Random(seed)
        self.names = [n for n, _ in mixture]
        self.weights = [w for _, w in mixture]
        self.sources = {n: _SourceDataset(n, cap, scienceqa_lecture=lecture) for n in self.names}
        for n, w in mixture:
            print(f"[data:{stage}] {n:15s} weight={w:.4f} rows={len(self.sources[n]):,}")
        self._len = sum(len(d) for d in self.sources.values())

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        name = self.rng.choices(self.names, weights=self.weights, k=1)[0]
        ds = self.sources[name]
        return ds[self.rng.randrange(len(ds))]
