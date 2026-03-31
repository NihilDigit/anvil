#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PWD/artifacts/checkpoints}"
ONNX_OUT="${ONNX_OUT:-$PWD/artifacts/onnx}"
RESOLUTIONS="${RESOLUTIONS:-vimeo 1080p}"
MODELS=("$@")

cmd=(
  pixi run python -m anvil_exp01.export_onnx_npu
  --out-dir "$ONNX_OUT"
  --checkpoint-dir "$CHECKPOINT_DIR"
)

if [[ "${#MODELS[@]}" -gt 0 ]]; then
  cmd+=(--models "${MODELS[@]}")
fi

cmd+=(--resolutions)
for res in $RESOLUTIONS; do
  cmd+=("$res")
done

"${cmd[@]}"
