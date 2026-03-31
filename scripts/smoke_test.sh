#!/usr/bin/env bash
# Smoke test: verify all release scripts run without errors.
# Uses minimal data (2 samples / 1 epoch) to complete quickly.
#
# Prerequisites: data + checkpoints must be in place.
# Device-dependent scripts (HTP benchmarks) are skipped.
#
# Usage:
#   bash scripts/smoke_test.sh
#   bash scripts/smoke_test.sh --skip-rife   # skip RIFE-dependent tests
set -uo pipefail

cd "$(dirname "$0")/.."

PASS=0
FAIL=0
SKIP=0
SKIP_RIFE=0

for arg in "$@"; do
    case "$arg" in
        --skip-rife) SKIP_RIFE=1 ;;
    esac
done

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

run_test() {
    local name="$1"; shift
    printf "  %-40s " "$name"
    if "$@" > "$TMPDIR/$name.log" 2>&1; then
        echo "PASS"
        ((PASS++))
    else
        echo "FAIL (see $TMPDIR/$name.log)"
        ((FAIL++))
    fi
}

skip_test() {
    printf "  %-40s SKIP (%s)\n" "$1" "$2"
    ((SKIP++))
}

echo "=== ANVIL Open-Source Release — Smoke Tests ==="
echo ""

# --- Data pipeline (check files exist) ---
echo "[Data checks]"
if [ -f "data/vimeo_triplet/tri_testlist.txt" ]; then
    run_test "vimeo_testlist_exists" test -f data/vimeo_triplet/tri_testlist.txt
else
    skip_test "vimeo_testlist_exists" "data not downloaded"
fi

if [ -d "data/xiph_1080p/sequences" ]; then
    run_test "xiph_sequences_exist" test -d data/xiph_1080p/sequences
else
    skip_test "xiph_sequences_exist" "data not downloaded"
fi

# --- ANVIL eval on Vimeo (--limit 2) ---
echo ""
echo "[ANVIL evaluation]"
ANVIL_S_CKPT="artifacts/checkpoints/D-unet-v3bs-nomv/best.pt"
ANVIL_M_CKPT="artifacts/checkpoints/D-unet-v3bm-nomv/best.pt"

if [ -f "$ANVIL_S_CKPT" ] && [ -f "data/vimeo_triplet/tri_testlist.txt" ]; then
    run_test "eval_vimeo_anvil_s" pixi run python scripts/eval_vimeo.py \
        --model D-unet-v3bs-nomv \
        --checkpoint "$ANVIL_S_CKPT" \
        --data-dir data/vimeo_triplet \
        --prealigned-dir data/vimeo_triplet/prealigned_v2 \
        --output-dir "$TMPDIR/eval_vimeo" \
        --limit 2
else
    skip_test "eval_vimeo_anvil_s" "checkpoint or data missing"
fi

if [ -f "$ANVIL_S_CKPT" ] && [ -d "data/xiph_1080p/sequences" ]; then
    run_test "eval_xiph_anvil_s" pixi run python scripts/eval_xiph.py \
        --model D-unet-v3bs-nomv \
        --checkpoint "$ANVIL_S_CKPT" \
        --data-dir data/xiph_1080p \
        --prealigned-dir data/xiph_1080p/prealigned_v2 \
        --output-dir "$TMPDIR/eval_xiph" \
        --limit 2
else
    skip_test "eval_xiph_anvil_s" "checkpoint or data missing"
fi

# --- Baselines (--limit 2) ---
echo ""
echo "[Baselines]"
if [ -f "data/vimeo_triplet/tri_testlist.txt" ]; then
    run_test "eval_baselines_vimeo" pixi run python scripts/eval_baselines.py \
        --dataset vimeo \
        --methods naive_blend mv_blend_v2 \
        --output-dir "$TMPDIR/baselines" \
        --limit 2 --force
else
    skip_test "eval_baselines_vimeo" "data missing"
fi

# --- RIFE native ---
echo ""
echo "[RIFE baselines]"
RIFE_REPO="vendor/ECCV2022-RIFE"
RIFE_WEIGHTS="$RIFE_REPO/train_log/flownet.pkl"
IFRNET_REPO="vendor/AMT"
IFRNET_WEIGHTS="vendor/IFRNet_Vimeo90K.pth"
if [ "$SKIP_RIFE" -eq 1 ]; then
    skip_test "eval_rife_native" "--skip-rife"
elif [ -d "$RIFE_REPO/train_log" ] && [ -f "data/vimeo_triplet/tri_testlist.txt" ]; then
    run_test "eval_rife_native" pixi run python scripts/eval_rife_native.py \
        --rife-repo "$RIFE_REPO" \
        --data-dir data/vimeo_triplet \
        --dataset vimeo \
        --limit 2 \
        --output-dir "$TMPDIR/rife_native"
else
    skip_test "eval_rife_native" "RIFE repo or data missing"
fi

# --- RIFE ONNX export ---
if [ "$SKIP_RIFE" -eq 1 ]; then
    skip_test "export_rife_onnx" "--skip-rife"
elif [ -d "$RIFE_REPO/train_log" ] && [ -f "$RIFE_WEIGHTS" ]; then
    run_test "export_rife_onnx" pixi run python scripts/export_rife_onnx.py \
        --rife-repo "$RIFE_REPO" \
        --output-dir "$TMPDIR/rife_onnx" \
        --weights "$RIFE_WEIGHTS" \
        --resolutions 360p \
        --modes flow
else
    skip_test "export_rife_onnx" "RIFE repo or weights missing"
fi

# --- IFRNet ONNX export ---
if [ -d "$IFRNET_REPO" ] && [ -f "$IFRNET_WEIGHTS" ]; then
    run_test "export_ifrnet_onnx" pixi run python scripts/export_ifrnet_onnx.py \
        --amt-repo "$IFRNET_REPO" \
        --weights "$IFRNET_WEIGHTS" \
        --output-dir "$TMPDIR/ifrnet_onnx"
else
    skip_test "export_ifrnet_onnx" "IFRNet repo or weights missing"
fi

# --- INT8 cross-method (--n-eval 2 --n-calib 2) ---
echo ""
echo "[INT8 quantization]"
INT8_ONNX_DIR="$TMPDIR/int8_onnx"
mkdir -p "$INT8_ONNX_DIR"
if [ -f "$TMPDIR/rife_onnx/rife_flow_360p.onnx" ]; then
    cp "$TMPDIR/rife_onnx/rife_flow_360p.onnx" "$INT8_ONNX_DIR/"
fi
if [ -f "$TMPDIR/ifrnet_onnx/ifrnet_flow.onnx" ]; then
    cp "$TMPDIR/ifrnet_onnx/ifrnet_flow.onnx" "$INT8_ONNX_DIR/"
fi
if [ -f "$TMPDIR/ifrnet_onnx/ifrnet_frame.onnx" ]; then
    cp "$TMPDIR/ifrnet_onnx/ifrnet_frame.onnx" "$INT8_ONNX_DIR/"
fi
if [ -f "$ANVIL_S_CKPT" ] && [ -f "$ANVIL_M_CKPT" ] && [ -d "data/xiph_1080p/sequences" ] && \
   [ -f "$INT8_ONNX_DIR/rife_flow_360p.onnx" ] && [ -f "$INT8_ONNX_DIR/ifrnet_flow.onnx" ] && \
   [ -f "$INT8_ONNX_DIR/ifrnet_frame.onnx" ]; then
    run_test "eval_int8_cross_method" pixi run python scripts/eval_int8_cross_method.py \
        --onnx-dir "$INT8_ONNX_DIR" \
        --xiph-dir data/xiph_1080p \
        --prealigned-dir data/xiph_1080p/prealigned_v2 \
        --anvil-s-ckpt "$ANVIL_S_CKPT" \
        --anvil-m-ckpt "$ANVIL_M_CKPT" \
        --output-dir "$TMPDIR/int8_cross" \
        --n-eval 2 --n-calib 2
else
    skip_test "eval_int8_cross_method" "checkpoints, ONNX baselines, or data missing"
fi

# --- INT8 instrumented (--n-test 2 --n-calib 2) ---
if [ "$SKIP_RIFE" -eq 1 ]; then
    skip_test "eval_int8_instrumented" "--skip-rife"
elif [ -d "data/xiph_1080p/sequences" ]; then
    if [ -f "$INT8_ONNX_DIR/rife_flow_360p.onnx" ] && [ -f "$INT8_ONNX_DIR/ifrnet_flow.onnx" ]; then
        run_test "eval_int8_instrumented" pixi run python scripts/eval_int8_instrumented.py \
            --onnx-dir "$INT8_ONNX_DIR" \
            --xiph-dir data/xiph_1080p \
            --models rife_flow_360p ifrnet_flow \
            --output-dir "$TMPDIR/int8_instrumented" \
            --n-test 2 --n-calib 2
    else
        skip_test "eval_int8_instrumented" "RIFE/IFRNet ONNX not available"
    fi
else
    skip_test "eval_int8_instrumented" "data missing"
fi

# --- Deploy-time fusion verification ---
echo ""
echo "[Deploy-time fusion]"
if [ -f "$ANVIL_S_CKPT" ]; then
    run_test "eval_deploy_fusion" pixi run python scripts/eval_deploy_fusion.py \
        --model D-unet-v3bs-nomv \
        --checkpoint "$ANVIL_S_CKPT" \
        --n-inputs 2 \
        --output-dir "$TMPDIR/deploy_fusion"
else
    skip_test "eval_deploy_fusion" "checkpoint missing"
fi

# --- Temporal consistency (--limit 1 --max-pairs 2) ---
echo ""
echo "[Temporal consistency]"
if [ "$SKIP_RIFE" -eq 1 ]; then
    skip_test "eval_temporal_consistency" "--skip-rife"
elif [ -f "$ANVIL_S_CKPT" ] && [ -f "$ANVIL_M_CKPT" ] && [ -d "$RIFE_REPO/train_log" ] && [ -d "data/xiph_1080p/sequences" ]; then
    run_test "eval_temporal_consistency" pixi run python scripts/eval_temporal_consistency.py \
        --xiph-dir data/xiph_1080p \
        --anvil-s-ckpt "$ANVIL_S_CKPT" \
        --anvil-m-ckpt "$ANVIL_M_CKPT" \
        --rife-repo "$RIFE_REPO" \
        --output-dir "$TMPDIR/temporal" \
        --limit 1 --max-pairs 2
else
    skip_test "eval_temporal_consistency" "checkpoints, RIFE, or data missing"
fi

# --- Visual comparison figure (--limit 1) ---
echo ""
echo "[Figure generation]"
if [ "$SKIP_RIFE" -eq 1 ]; then
    skip_test "gen_fig_visual_comparison" "--skip-rife"
elif [ -f "$ANVIL_M_CKPT" ] && [ -d "$RIFE_REPO/train_log" ] && [ -d "data/xiph_1080p/sequences" ]; then
    run_test "gen_fig_visual_comparison" pixi run python scripts/gen_fig_visual_comparison.py \
        --xiph-dir data/xiph_1080p \
        --anvil-ckpt "$ANVIL_M_CKPT" \
        --rife-repo "$RIFE_REPO" \
        --output-dir "$TMPDIR/figures" \
        --limit 1
else
    skip_test "gen_fig_visual_comparison" "checkpoints, RIFE, or data missing"
fi

# --- Device-dependent scripts (always skip) ---
echo ""
echo "[Device-dependent (skipped)]"
skip_test "bench_htp_latency" "requires QAIRT SDK + Android device"
skip_test "bench_operators" "requires QAIRT SDK + Android device"
skip_test "eval_rife_qnn_int8" "requires QAIRT SDK + Android device"

# --- RIFE reduced-resolution ORT ---
echo ""
echo "[RIFE reduced-resolution]"
if [ "$SKIP_RIFE" -eq 1 ]; then
    skip_test "eval_rife_reduced_res" "--skip-rife"
else
    RIFE_ONNX_DIR="$TMPDIR/rife_onnx"
    if [ -d "$RIFE_ONNX_DIR" ] && [ -d "data/xiph_1080p/sequences" ]; then
        run_test "eval_rife_reduced_res" pixi run python scripts/eval_rife_reduced_res.py \
            --onnx-dir "$RIFE_ONNX_DIR" \
            --xiph-dir data/xiph_1080p \
            --output-dir "$TMPDIR/rife_reduced" \
            --limit 2
    else
        skip_test "eval_rife_reduced_res" "RIFE ONNX or data missing"
    fi
fi

# --- Summary ---
echo ""
echo "========================================"
echo "  Results: $PASS pass, $FAIL fail, $SKIP skip"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    echo "  Log files in: $TMPDIR/"
    # Don't delete tmpdir on failure
    trap - EXIT
    exit 1
fi
exit 0
