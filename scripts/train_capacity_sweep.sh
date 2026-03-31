#!/usr/bin/env bash
# =============================================================================
# ANVIL — Capacity sweep training (paper Table ablation_capacity)
# =============================================================================
#
# Trains the 5 D-nomv capacity sweep models on full Vimeo90K (51313 triplets):
#   D-tiny-nomv   (1.8K params)   — 4-layer plain conv, 8ch
#   D-mini-nomv   (10.6K params)  — 6-layer plain conv, 16ch
#   D-mid-nomv    (33K params)    — 8-layer plain conv, 24ch
#   D-unet-s-nomv (129K params)   — 3-level U-Net, base_ch=16
#   D-unet-l-nomv (289K params)   — 3-level U-Net, base_ch=24
#
# IMPORTANT: These models use prealignment v1 (basic block-level MV warp),
# NOT the smoothed v2 used by the final ANVIL-S/M models. Point PREALIGNED_DIR
# to v1 prealigned data (output of anvil_exp01.data.prealign, NOT prealign_v2).
#
# Usage:
#   bash scripts/train_capacity_sweep.sh                  # train all 5
#   bash scripts/train_capacity_sweep.sh D-mid-nomv       # train one
#
# Environment variables:
#   DATA_ROOT       — Vimeo90K root (default: $PWD/data/vimeo_triplet)
#   PREALIGNED_DIR  — v1 prealigned directory (default: $DATA_ROOT/prealigned)
#   OUTPUT_DIR      — checkpoint output (default: $PWD/artifacts/checkpoints_capacity)
#   DEVICE          — training device (default: cuda)
#   WORKERS         — dataloader workers (default: 8)
#   EPOCHS          — max epochs (default: 100)
#   BATCH_SIZE      — training batch size (default: 16)
#   LR              — learning rate (default: 2e-4)
#   SUBSET          — fractional subset for quick runs, e.g. 0.25 (default: empty = full)
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-$PWD/data/vimeo_triplet}"
PREALIGNED_DIR="${PREALIGNED_DIR:-$DATA_ROOT/prealigned}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/artifacts/checkpoints_capacity}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-4}"
VAL_INTERVAL="${VAL_INTERVAL:-3}"
PATIENCE="${PATIENCE:-7}"
MIN_DELTA="${MIN_DELTA:-0.05}"
SUBSET="${SUBSET:-}"

if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(
    "D-tiny-nomv"
    "D-mini-nomv"
    "D-mid-nomv"
    "D-unet-s-nomv"
    "D-unet-l-nomv"
  )
fi

mkdir -p "$OUTPUT_DIR"

echo "ANVIL capacity sweep training (prealign v1)"
echo "  data:        $DATA_ROOT"
echo "  prealigned:  $PREALIGNED_DIR  (v1 — basic block-level MV warp)"
echo "  output:      $OUTPUT_DIR"
echo "  models:      ${MODELS[*]}"
echo "  epochs:      $EPOCHS"
echo "  batch-size:  $BATCH_SIZE"
echo "  lr:          $LR"
echo "  device:      $DEVICE"

for model in "${MODELS[@]}"; do
  train_log="$OUTPUT_DIR/$model/train.log"
  if [[ -f "$train_log" ]] && grep -q "Done\.\|Early stopping" "$train_log"; then
    echo "[$model] Already completed (found Done/Early stopping in train.log), skipping."
    continue
  fi

  echo ""
  echo "--- Training: $model ---"

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
    --lpips-warmup "$EPOCHS"
    --resume auto
    --device "$DEVICE"
  )
  if [[ -n "$SUBSET" ]]; then
    cmd+=(--subset "$SUBSET")
  fi
  "${cmd[@]}"
done
