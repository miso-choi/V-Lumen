# Training recipe

The recipe behind every setting in Tab. 2 and Tab. 3. `scripts/train.py` uses these values
by default, so `bash scripts/train_and_eval.sh <setting> <seed>` needs no extra flags.

## Stages

| | Mid-training | SFT |
|---|---|---|
| Data | DocVQA, ChartQA, InfographicVQA, GQA, RefCOCO, AI2D, A-OKVQA, TextCaps, ScienceQA (with lecture) | ScienceQA, AI2D, DocVQA, ChartQA, GQA, A-OKVQA |
| Rows per source | 6,000 | 4,000 (10% held out for validation) |
| + few-shot split | 10% of each reasoning benchmark (≤ 200 per benchmark) | same |
| Steps (V-LUMEN-only / adapter settings) | 10,000 / 2,000 | 2,000 / 1,000 |
| Batch size | 4 | 8 |
| LR (adapters, fusion) | 4e-4 | 1e-4 |
| LR (memory rows) | 5 × LR | 0.1 × LR |
| Schedule | cosine to 0.1 × LR | cosine to 0.05 × LR |

The few-shot split is drawn with a fixed seed (42), and evaluation uses the remaining 90%.
The SFT validation hold-out also uses seed 42.

## Optimisation

- AdamW, weight decay 0.1. Memory rows, norms and the residual scale use no weight decay.
- bf16 frozen backbone; V-LUMEN and the LoRA / MoE adapters train in fp32.
- Answer-only loss: prompt and image tokens are masked.
- Gradient clipping: per V-LUMEN module at √(memory_dim / 128) = 2, and at 1.0 for LoRA / MoE.
- MoE adapters add a load-balancing loss with weight 1e-2.

## Parameter-budget matching (App. G)

From the V-LUMEN parameter count P_v and `--budget_scale k` (default 0.5),
`vlumen/budget.py` derives the adapter shapes, so that LoRA-only and MoE-only match
LoRA + V-LUMEN and MoE + V-LUMEN in trainable parameters:

| Setting | Adapter |
|---|---|
| V-LUMEN-only | P_v = 86.1M |
| LoRA-only = LoRA(1+k) | rank 321, alpha 97 (128.9M) |
| LoRA + V-LUMEN = LoRA(k) + V-LUMEN(1) | rank 107, alpha 32 |
| MoE-only = MoE(1+k) | 31 experts × 344, top-1 (131.2M) |
| MoE + V-LUMEN = MoE(k) + V-LUMEN(1) | 21 experts × 168, top-1 |

`train.py` prints the derived shapes at start-up.

## Compute

On one RTX 6000 Ada (48 GB):

| Setting | Mid-training | SFT |
|---|---|---|
| V-LUMEN-only | ~3 h | ~1 h |
| Each adapter setting | ~50 min | ~50 min |

Evaluation scores the trained model and the vanilla backbone in the same pass. It takes
about 1 to 2 h for the five reasoning benchmarks, or roughly half that with `--no_vanilla`.

## Data notes

- The paper lists Visual Genome in the mid-training data, but it was not loaded in the
  runs behind the released checkpoints. The mixture in `vlumen/data/mixture.py` matches
  those runs.
- `--seed` fixes both the initialisation and the data order. Runs on different GPUs or
  with different library versions are not bit-identical. See
  [seed1_results.md](seed1_results.md) for a from-scratch retraining check.
