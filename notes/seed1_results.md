# Seed-1 results

The [README](../README.md) reports the paper's numbers (mean ± std over three seeds).
This note gives the single-seed numbers behind the released checkpoints, a from-scratch
retraining check, and a re-measurement of Tab. 1 with this code.

All accuracies are in %. The evaluation protocol is the one described in the README
(n ≤ 1,000 per benchmark, greedy decoding, relaxed accuracy).

## Released checkpoints

The released checkpoints are the seed-1 runs of the paper. The table shows what
`bash scripts/reproduce_tab2.sh` and `bash scripts/reproduce_tab3.sh` give with them.

**Tab. 2, reasoning benchmarks**

| Setting | ChartMus. | FRIEDA | VisRes | MMMU | MMMU-Pro | Avg | Avg Δ |
|---|---|---|---|---|---|---|---|
| Vanilla | 1.69 | 2.27 | 5.00 | 1.63 | 0.30 | 2.18 | – |
| V-LUMEN-only | 15.88 | 4.77 | 26.70 | 34.38 | 20.70 | 20.49 | +18.31 |
| LoRA-only | 17.91 | 5.45 | 22.10 | 42.16 | 26.10 | 22.74 | +20.57 |
| LoRA + V-LUMEN | 18.69 | 5.68 | 19.60 | 42.66 | 29.90 | 23.31 | +21.13 |
| MoE-only | 12.61 | 4.55 | 21.20 | 31.74 | 19.90 | 18.00 | +15.82 |
| MoE + V-LUMEN | 14.41 | 3.64 | 23.50 | 31.24 | 18.20 | 18.20 | +16.02 |

Benchmark sizes: ChartMuseum 888, FRIEDA 440, VisRes 1,000, MMMU 797, MMMU-Pro 1,000.

**Tab. 3, zero-shot (V-LUMEN-only)**

| Benchmark | Vanilla | V-LUMEN-only (seed 1) |
|---|---|---|
| TextVQA | 11.00 | 34.90 |
| OCR-VQA | 13.80 | 46.80 |
| VizWiz-VQA | 2.00 | 8.40 |

These match the per-sample outputs of the original evaluation, with one exception. On
MoE-only, one MMMU vanilla answer and one MMMU-Pro answer differ because of GPU
nondeterminism, so MMMU-Pro is 19.90 here and 19.80 in the original run.

## Retraining from scratch

We retrained all five settings with `bash scripts/train_and_eval.sh <setting> 1` on one
RTX 6000 Ada per setting and evaluated the resulting checkpoints.

**Tab. 2, reasoning benchmarks**

| Setting | ChartMus. | FRIEDA | VisRes | MMMU | MMMU-Pro | Avg | Paper Avg (3 seeds) |
|---|---|---|---|---|---|---|---|
| V-LUMEN-only | 16.33 | 5.00 | 23.60 | 32.12 | 19.40 | 19.29 | 19.97 ± 0.47 |
| LoRA-only | 17.34 | 6.14 | 22.10 | 39.02 | 26.60 | 22.24 | 22.76 ± 0.85 |
| LoRA + V-LUMEN | 18.02 | 5.23 | 21.40 | 41.66 | 28.90 | 23.04 | 23.33 ± 0.02 |
| MoE-only | 14.53 | 4.32 | 28.20 | 32.12 | 19.40 | 19.71 | 19.03 ± 0.95 |
| MoE + V-LUMEN | 14.53 | 5.68 | 22.50 | 34.38 | 19.30 | 19.28 | 19.10 ± 0.91 |

**Tab. 3, zero-shot (V-LUMEN-only)**

| Benchmark | Retrained (seed 1) | Paper (3 seeds) |
|---|---|---|
| TextVQA | 36.80 | 37.17 ± 3.42 |
| OCR-VQA | 46.30 | 47.67 ± 0.76 |
| VizWiz-VQA | 10.50 | 9.97 ± 1.84 |

All averages lie within about 1.5 standard deviations of the paper's 3-seed mean. The
per-benchmark differences are a few examples each, which is below the sampling error of a
1,000-example benchmark (about ±1.3 pp).

The memory-table initialisation was not seeded in the original runs. A retrained model
therefore does not match the released checkpoint bit for bit, even with the same `--seed`.
With a single seed, the order of settings that are close in Tab. 2 can change. For
example, the retrained V-LUMEN-only is below the retrained MoE-only here, while the
3-seed means put it above.

## Tab. 1, re-measured

`bash scripts/reproduce_tab1.sh` on an idle RTX 6000 Ada:

| Rows / cell | Lookup total (ms) | Gather only (ms) |
|---|---|---|
| 16,384 (0.2×) | 11.16 ± 0.31 (1.003×) | 0.488 (1.024×) |
| 81,920 (1×) | 11.12 ± 0.18 (1.000×) | 0.477 (1.000×) |
| 327,680 (4×) | 11.24 ± 0.24 (1.011×) | 0.476 (0.999×) |
| 655,360 (8×) | 11.11 ± 0.19 (0.999×) | 0.480 (1.008×) |

Absolute times depend on the GPU and on other load. Without enough warm-up (the script
defaults to 100 iterations), the first configurations are measured before the GPU clock
settles and come out slower.
