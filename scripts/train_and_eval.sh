#!/usr/bin/env bash
# Train one Tab. 2 setting (mid-training -> SFT) and evaluate it.
#   bash scripts/train_and_eval.sh <vlumen|lora|lora_vlumen|moe|moe_vlumen> [seed]
# Outputs: runs/<setting>_seed<seed>/{midtrain.pt, sft.pt, eval_*.json, *.log}
set -euo pipefail
cd "$(dirname "$0")/.."
SETTING="$1"; SEED="${2:-1}"
OUT="${RUN_DIR:-runs}/${SETTING}_seed${SEED}"
mkdir -p "$OUT"

# Training schedule (App. E.2, Tab. 4)
case "$SETTING" in
  vlumen)                         MID_STEPS=10000; SFT_STEPS=2000 ;;
  lora|lora_vlumen|moe|moe_vlumen) MID_STEPS=2000;  SFT_STEPS=1000 ;;
  *) echo "unknown setting: $SETTING" >&2; exit 1 ;;
esac

python scripts/train.py --stage midtrain --setting "$SETTING" --seed "$SEED" --steps "$MID_STEPS" \
  --out "$OUT/midtrain.pt" 2>&1 | tee "$OUT/midtrain.log"
python scripts/train.py --stage sft --init "$OUT/midtrain.pt" --seed "$SEED" --steps "$SFT_STEPS" \
  --out "$OUT/sft.pt" 2>&1 | tee "$OUT/sft.log"
python scripts/evaluate.py --ckpt "$OUT/sft.pt" --benchmarks reasoning \
  --out "$OUT/eval_reasoning.json" 2>&1 | tee "$OUT/eval_reasoning.log"
if [[ "$SETTING" == "vlumen" ]]; then
  python scripts/evaluate.py --ckpt "$OUT/sft.pt" --benchmarks zero_shot \
    --out "$OUT/eval_zero_shot.json" 2>&1 | tee "$OUT/eval_zero_shot.log"
fi
