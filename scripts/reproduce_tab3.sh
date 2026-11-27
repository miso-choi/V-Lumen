#!/usr/bin/env bash
# Tab. 3: zero-shot accuracy of V-LUMEN-only (frozen backbone) on TextVQA / OCR-VQA / VizWiz.
#   bash scripts/reproduce_tab3.sh                    # released seed-1 checkpoint
#   bash scripts/reproduce_tab3.sh runs/vlumen_seed*/sft.pt   # your own checkpoints
set -euo pipefail
cd "$(dirname "$0")/.."
CKPTS=("$@"); [[ ${#CKPTS[@]} -eq 0 ]] && CKPTS=(checkpoints/vlumen_only_seed1.pt)
OUTS=()
for ck in "${CKPTS[@]}"; do
  name="$(basename "$(dirname "$ck")")_$(basename "$ck" .pt)"
  python scripts/evaluate.py --ckpt "$ck" --benchmarks zero_shot --out "results/tab3/${name}.json"
  OUTS+=("results/tab3/${name}.json")
done
python scripts/summarize.py "${OUTS[@]}"
