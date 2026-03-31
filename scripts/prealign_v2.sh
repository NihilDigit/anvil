#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-$PWD/data/vimeo_triplet}"
MV_DIR="${MV_DIR:-$DATA_ROOT/mv_cache}"
PREALIGNED_DIR="${PREALIGNED_DIR:-$DATA_ROOT/prealigned_v2}"
SPLIT="${SPLIT:-both}"
WORKERS="${WORKERS:-$(nproc)}"

pixi run python -m anvil_exp01.data.prealign_v2 \
  --data-dir "$DATA_ROOT" \
  --mv-dir "$MV_DIR" \
  --output-dir "$PREALIGNED_DIR" \
  --method med_gauss \
  --split "$SPLIT" \
  --workers "$WORKERS"
