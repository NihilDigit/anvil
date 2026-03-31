#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-$PWD/data/vimeo_triplet}"
PREALIGNED_DIR="${PREALIGNED_DIR:-$DATA_ROOT/prealigned_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/artifacts/checkpoints}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-4}"
VAL_INTERVAL="${VAL_INTERVAL:-3}"
PATIENCE="${PATIENCE:-7}"
MIN_DELTA="${MIN_DELTA:-0.10}"
SUBSET="${SUBSET:-}"

if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=("D-unet-v3bs-nomv" "D-unet-v3bm-nomv")
fi

mkdir -p "$OUTPUT_DIR"

echo "ANVIL reproduction training"
echo "  data:        $DATA_ROOT"
echo "  prealigned:  $PREALIGNED_DIR"
echo "  output:      $OUTPUT_DIR"
echo "  models:      ${MODELS[*]}"
echo "  epochs:      $EPOCHS"
echo "  batch-size:  $BATCH_SIZE"
echo "  device:      $DEVICE"

for model in "${MODELS[@]}"; do
  cmd=(
    pixi run python -m anvil_exp01.train
    --model "$model"
    --data-dir "$DATA_ROOT"
    --prealigned-dir "$PREALIGNED_DIR"
    --output-dir "$OUTPUT_DIR"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --workers "$WORKERS"
    --val-interval "$VAL_INTERVAL"
    --patience "$PATIENCE"
    --min-delta "$MIN_DELTA"
    --resume auto
    --device "$DEVICE"
  )
  if [[ -n "$SUBSET" ]]; then
    cmd+=(--subset "$SUBSET")
  fi
  "${cmd[@]}"
done
