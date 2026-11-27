#!/usr/bin/env bash
# Tab. 2: accuracy on the five reasoning benchmarks.
#   bash scripts/reproduce_tab2.sh            # evaluate the released seed-1 checkpoints in checkpoints/
#   bash scripts/reproduce_tab2.sh train      # train + evaluate all settings for seeds 1 2 3 (SEEDS="1 2 3")
set -euo pipefail
cd "$(dirname "$0")/.."
SETTINGS=(vlumen lora lora_vlumen moe moe_vlumen)
declare -A CKPT=([vlumen]=vlumen_only [lora]=lora_only [lora_vlumen]=lora_vlumen [moe]=moe_only [moe_vlumen]=moe_vlumen)

if [[ "${1:-eval}" == "train" ]]; then
  for seed in ${SEEDS:-1 2 3}; do
    for s in "${SETTINGS[@]}"; do bash scripts/train_and_eval.sh "$s" "$seed"; done
  done
  python scripts/summarize.py runs/*/eval_reasoning.json
else
  for s in "${SETTINGS[@]}"; do
    python scripts/evaluate.py --ckpt "checkpoints/${CKPT[$s]}_seed1.pt" --benchmarks reasoning \
      --out "results/tab2/${CKPT[$s]}_seed1.json"
  done
  python scripts/summarize.py results/tab2/*.json
fi
