# V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models

**Miso Choi\*, Daekeun Kim\*, Eunji Kim, Jungbeom Lee** · NeurIPS 2026 · [arXiv:2610.37019](https://arxiv.org/abs/2610.37019)

Official PyTorch implementation of **"V-LUMEN: Visual Lookup Memory for Embedding Scaling
in Vision-Language Models"**, accepted at **NeurIPS 2026**.

V-LUMEN adds a large, sparsely read **visual lookup memory** to a frozen vision-language model.
Visual features are discretized with a frozen codebook, turned into spatially aware and
text-conditioned hash keys, and the retrieved memory rows are fused into the backbone
through a gated residual. All addressing is non-parametric, and a lookup costs the same
whatever the table size, so capacity grows in storage rather than in per-token compute.

This repository contains the code to train and evaluate V-LUMEN on Qwen3-VL-2B-Instruct
and to reproduce the main results of the paper (Tab. 1–3), plus the seed-1 checkpoints
of all Tab. 2 settings.

### Motivation

Dense scaling grows capacity and per-token compute together. A lookup memory (e.g.,
Engram for text) reads one row per token, so capacity grows while compute per token
stays constant. V-LUMEN brings this to vision-language models, whose visual features
are continuous, spatially structured and query-dependent rather than discrete tokens.

<p align="center"><img src="assets/v-lumen-motivation-figure.png" width="90%" alt="Dense scaling vs. lookup memory"></p>

### Overview

<p align="center"><img src="assets/v-lumen-main-figure.png" width="100%" alt="V-LUMEN overview"></p>

## Method in one paragraph

At the input of the backbone, each visual token is mapped to a codebook ID by a frozen
random projection and the frozen LaVIT codebook (**§2.2**). On the 8×8 grid of IDs,
**Spatial Aggregation-based Hashing** (SAH, §2.3) blends row/column summaries (axial
pooling) and a coarse position bucket (coordinate bias) into each key, and
**Text-Conditioned Hashing** (TCH, §2.4) XORs a per-patch text-relevance bin computed
from the question (PRH, DRH, or their ensemble). Each of the 2 n-gram orders × 8 heads =
16 **memory cells** hashes the key to one row of its table (81,920+ rows each); the rows
are fused after decoder layers 2 and 8 by **Multi-Head Conditional Gating** (one sigmoid
gate per cell), a depthwise convolution and RMSNorm, as a scaled residual (§2.5).
Only the memory rows and the fusion parameters are trained.

## Setup

Tested with Python 3.12, PyTorch 2.11 and one NVIDIA RTX 6000 Ada (48 GB).

```bash
git clone https://github.com/miso-choi/V-Lumen.git && cd V-Lumen

# with uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

# or with venv + pip
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The backbone (`Qwen/Qwen3-VL-2B-Instruct`), the LaVIT codebook (`rain1011/LaVIT-7B-v2`,
only the visual-tokenizer file) and all datasets are downloaded from the Hugging Face Hub
on first use. Set `HF_HOME` to a disk with enough space (training data + benchmarks ≈ 60 GB).

## Checkpoints

Seed-1 checkpoints of every Tab. 2 setting (SFT stage, ~0.35–0.5 GB each; their scores are
in [notes/seed1_results.md](notes/seed1_results.md)):

| File | Setting | Trainable modules |
|---|---|---|
| `vlumen_only_seed1.pt` | V-LUMEN-only | V-LUMEN (86.1M) on the frozen backbone |
| `lora_only_seed1.pt` | LoRA-only = LoRA(1+k) | LoRA rank 321 (128.9M) |
| `lora_vlumen_seed1.pt` | LoRA + V-LUMEN = LoRA(k) + V-LUMEN(1) | LoRA rank 107 + V-LUMEN |
| `moe_only_seed1.pt` | MoE-only = MoE(1+k) | 31 experts × 344, top-1 (131.2M) |
| `moe_vlumen_seed1.pt` | MoE + V-LUMEN = MoE(k) + V-LUMEN(1) | 21 experts × 168, top-1 + V-LUMEN |

```bash
python scripts/download_checkpoints.py        # -> checkpoints/*.pt (SHA-256 verified)
```

Google Drive folder: _link to be added_.

## Reproducing the paper

| | Command | Needs | Time (1 GPU) |
|---|---|---|---|
| Tab. 1 | `bash scripts/reproduce_tab1.sh` | an idle GPU | ~1 min |
| Tab. 2 (released checkpoints) | `bash scripts/reproduce_tab2.sh` | `checkpoints/` | ~2 h per setting |
| Tab. 2 (from scratch, 3 seeds) | `bash scripts/reproduce_tab2.sh train` | — | ~10 h training per seed + eval |
| Tab. 3 | `bash scripts/reproduce_tab3.sh` | `checkpoints/vlumen_only_seed1.pt` | ~1 h |

Tab. 1 needs no checkpoint; it prints the table and writes `results/tab1_lookup_latency.json`.
Tab. 2 / 3 evaluation scores the trained model and the vanilla backbone in the same pass;
add `--no_vanilla` to `scripts/evaluate.py` to halve the time.

### Training a single setting

```bash
bash scripts/train_and_eval.sh vlumen 1        # V-LUMEN-only, seed 1
bash scripts/train_and_eval.sh lora_vlumen 1   # LoRA(k) + V-LUMEN(1)
# settings: vlumen | lora | lora_vlumen | moe | moe_vlumen
```

This runs mid-training → SFT → evaluation and writes everything to `runs/<setting>_seed<seed>/`.
The two stages can also be run directly:

```bash
python scripts/train.py --stage midtrain --setting vlumen --steps 10000 --seed 1 --out runs/v/midtrain.pt
python scripts/train.py --stage sft --init runs/v/midtrain.pt --steps 2000 --seed 1 --out runs/v/sft.pt
python scripts/evaluate.py --ckpt runs/v/sft.pt --benchmarks reasoning     # or: zero_shot
python scripts/summarize.py runs/*/eval_reasoning.json                      # mean ± std table
```

LoRA / MoE shapes are matched to the V-LUMEN parameter budget automatically (App. G).
Training data, hyperparameters, budget matching and compute are in
[notes/training_recipe.md](notes/training_recipe.md).

## Results

Numbers from the camera-ready paper. Single-seed numbers of the released checkpoints, a
from-scratch retraining check and a Tab. 1 re-measurement are in
[notes/seed1_results.md](notes/seed1_results.md).

**Tab. 1 — Lookup latency vs. memory-table size** (one fusion layer, 64 visual tokens,
bf16, RTX 6000 Ada, mean over 50 runs; relative to the default 1×)

| Rows / cell | Table params | Memory | Lookup total (ms) | Gather only (ms) |
|---|---|---|---|---|
| 16,384 (0.2×) | 8,433,472 | 16.9 MB | 11.22 ± 0.15 (1.005×) | 0.463 (0.983×) |
| 81,920 (1×) | 41,976,448 | 84.0 MB | 11.16 ± 0.11 (1.000×) | 0.471 (1.000×) |
| 327,680 (4×) | 167,825,792 | 335.7 MB | 11.16 ± 0.11 (1.000×) | 0.462 (0.981×) |
| 655,360 (8×) | 335,605,248 | 671.2 MB | 11.23 ± 0.21 (1.006×) | 0.491 (1.042×) |

**Tab. 2 — Accuracy (%) on five reasoning benchmarks** (mean ± std over three seeds;
Avg Δ = average gain over the vanilla backbone, pp)

| Setting | ChartMus. | FRIEDA | VisRes | MMMU | MMMU-Pro | Avg | Avg Δ |
|---|---|---|---|---|---|---|---|
| Vanilla | 1.69 | 2.27 | 5.00 | 1.63 | 0.30 | 2.18 | – |
| V-LUMEN-only | 16.70 ± 0.79 | 4.77 ± 0.23 | 24.50 ± 2.07 | 33.38 ± 0.95 | 20.50 ± 0.82 | 19.97 ± 0.47 | +17.79 ± 0.47 |
| LoRA-only | 17.87 ± 0.28 | 5.53 ± 0.13 | 22.03 ± 0.50 | 41.87 ± 1.47 | 26.50 ± 1.93 | 22.76 ± 0.85 | +20.58 ± 0.85 |
| LoRA + V-LUMEN | 18.24 ± 0.41 | 5.61 ± 0.35 | 21.70 ± 2.39 | 41.91 ± 1.30 | 29.20 ± 0.66 | 23.33 ± 0.02 | +21.15 ± 0.02 |
| MoE-only | 13.48 ± 0.90 | 4.85 ± 0.35 | 23.53 ± 3.87 | 33.21 ± 2.22 | 20.10 ± 0.26 | 19.03 ± 0.95 | +16.85 ± 0.95 |
| MoE + V-LUMEN | 14.19 ± 0.49 | 4.70 ± 1.02 | 25.07 ± 1.43 | 32.54 ± 1.17 | 19.00 ± 2.59 | 19.10 ± 0.91 | +16.92 ± 0.91 |

**Tab. 3 — Zero-shot accuracy (%)** of V-LUMEN-only on benchmarks outside the training
curriculum (frozen backbone, no few-shot samples, images disjoint from training data)

| Benchmark | Vanilla | + V-LUMEN (3 seeds) |
|---|---|---|
| TextVQA | 11.00 | 37.17 ± 3.42 (+26.17) |
| OCR-VQA | 13.80 | 47.67 ± 0.76 (+33.87) |
| VizWiz-VQA | 2.00 | 9.97 ± 1.84 (+7.97) |

Evaluation protocol: held-out 90% of each benchmark under a fixed split seed (42), up to
1,000 examples, greedy decoding with at most 32 new tokens, 224×224 images, relaxed
accuracy (numbers within 5% relative error; otherwise exact match after normalising case,
punctuation and whitespace; MMMU answers are reduced to the option letter).

## Code structure

```
vlumen/
  config.py         VLumenConfig: every setting of the paper; stored in each checkpoint
  codebook.py       frozen codebook: random projection + LaVIT nearest-code lookup (§2.2)
  memory_table.py   memory cells, SAH (axial pooling, coordinate bias) and TCH salting (§2.3–2.5)
  fusion.py         VLumenModule: lookup + Multi-Head Conditional Gating + residual fusion
  model.py          VLumenVLM: hooks on the frozen Qwen3-VL; TCH relevance (PRH / DRH)
  moe.py            routed MoE adapter for the MoE baselines
  budget.py         parameter-budget matching for LoRA / MoE (App. G)
  checkpoint.py     model construction and checkpoint I/O
  evaluation.py     greedy generation and relaxed accuracy
  data/
    mixture.py      mid-training and SFT data mixtures
    benchmarks.py   evaluation benchmarks and the few-shot split
    collate.py      answer-only loss masking
scripts/
  train.py                     one training stage (midtrain | sft)
  evaluate.py                  evaluate a checkpoint (reasoning | zero_shot | list)
  benchmark_lookup_latency.py  Tab. 1
  summarize.py                 eval JSONs -> Markdown table
  download_checkpoints.py      released checkpoints
  train_and_eval.sh            midtrain -> SFT -> eval for one setting / seed
  reproduce_tab{1,2,3}.sh
notes/
  training_recipe.md           data, hyperparameters, budget matching, compute
  seed1_results.md             released seed-1 scores, retraining check, Tab. 1 re-run
```

How the pieces connect at inference (prefill; decode steps reuse the KV cache):

```
input tokens ─► Qwen3-VL decoder
   layer 0  (pre-hook)   visual hidden states ──frozen codebook──► code IDs
   layer 2  (pre-hook)   question & patch states ──PRH/DRH──► relevance bins (TCH)
   layer 2  (post-hook)  VLumenModule: SAH+TCH keys → 16 memory cells → gated residual
   layer 8  (post-hook)  VLumenModule (own memory)                          [+ MoE adapter]
```

## Citation

```bibtex
@inproceedings{choi2026vlumen,
  title     = {V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models},
  author    = {Choi, Miso and Kim, Daekeun and Kim, Eunji and Lee, Jungbeom},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code and the released checkpoints are available for **noncommercial use only**
(research, education, personal and nonprofit use) under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use is not permitted.
For commercial licensing inquiries, please contact the authors.

The backbone (Qwen3-VL-2B-Instruct), the LaVIT codebook and the datasets are not part of
this repository and are subject to their own licenses.
