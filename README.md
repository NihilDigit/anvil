# ANVIL Open-Source Reproduction Bundle

> **Paper**: [ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors](https://arxiv.org/abs/2603.26835)

This folder is a cleaned release candidate for training, evaluation, and ONNX export of the ANVIL models presented in the paper.

## Included Scripts

- `scripts/train_v3b.sh` — Train final ANVIL-S / ANVIL-M (prealign v2)
- `scripts/train_capacity_sweep.sh` — Train 5 capacity sweep models D-tiny through D-unet-l (prealign v1, Table ablation_capacity)
- `scripts/eval_vimeo.py` — Vimeo90K evaluation (PSNR / SSIM / LPIPS)
- `scripts/eval_xiph.py` — Xiph 1080p evaluation
- `scripts/eval_baselines.py` — Zero-network baseline evaluation (Naive Blend, MV Blend)
- `scripts/eval_rife_native.py` — RIFE HDv3 native-resolution baseline (Table main_quality)
- `scripts/eval_rife_reduced_res.py` — RIFE reduced-resolution ORT INT8 evaluation (Table rife_reduced, ORT column)
- `scripts/eval_rife_qnn_int8.py` — RIFE reduced-resolution QNN HTP INT8 evaluation (Table rife_reduced, QNN column)
- `scripts/eval_int8_cross_method.py` — Cross-method ORT INT8 quality comparison (Table int8_quality)
- `scripts/eval_int8_instrumented.py` — Per-operator instrumented quantization analysis (Table instrumented)
- `scripts/export_rife_onnx.py` — Export RIFE HDv3 to flow/frame ONNX variants
- `scripts/export_ifrnet_onnx.py` — Export IFRNet to frame/flow ONNX variants
- `scripts/export_onnx.sh` — Export ANVIL models to ONNX
- `scripts/prealign_v2.sh` — Generate prealignment v2 data (smoothed MV flow)
- `scripts/bench_htp_latency.py` — HTP INT8/FP16 latency benchmark (Table latency)
- `scripts/bench_operators.py` — Operator-level NPU latency benchmark (Table operator_compat, HTP column)
- `scripts/eval_deploy_fusion.py` — Deploy-time BN fusion mathematical equivalence verification
- `scripts/eval_temporal_consistency.py` — Temporal consistency evaluation (tOF / WE metrics)
- `scripts/gen_fig_visual_comparison.py` — Figure 6: visual quality comparison on Xiph 1080p
- `scripts/smoke_test.sh` — Smoke test for all scripts (`--limit 2`)
- `docs/DATA_LAYOUT.md` — Expected dataset directory layout
- `docs/MEDIATEK_PIPELINE.md` — MediaTek NeuroPilot Public deployment pipeline (Table operator_compat, APU column)
- `pixi.toml` — Complete runtime environment for training and evaluation

## What Is Not Included

- Device-side Android app (see [mpv-android-anvil](https://github.com/NihilDigit/mpv-android-anvil) repo)

## Environment

```bash
pixi install
```

If you are working from the full research monorepo instead of the standalone release repo, run `cd open_source_release` first.

## Data Setup

### Vimeo90K

Download and prepare the Vimeo90K triplet dataset:

```bash
# 1. Download (~30 GB for full dataset, ~1.7 GB for test-only)
pixi run python -m anvil_exp01.data.download_vimeo90k --output-dir data/vimeo_triplet

# 2. Extract H.264 motion vectors
pixi run python -m anvil_exp01.data.extract_mv --data-dir data/vimeo_triplet

# 3. Convert block MVs to dense flow fields
pixi run python -m anvil_exp01.data.mv_to_dense --data-dir data/vimeo_triplet
```

### Prealignment

Two prealignment versions exist. Which one you need depends on the models you want to train or evaluate:

```bash
# Prealignment v1 (basic block-level MV warp)
# Required for: capacity sweep models (D-tiny-nomv through D-unet-l-nomv)
pixi run python -m anvil_exp01.data.prealign --data-dir data/vimeo_triplet

# Prealignment v2 (median + Gaussian smoothed MV flow)
# Required for: final ANVIL-S / ANVIL-M models
pixi run prealign-v2
```

### Xiph 1080p

```bash
pixi run python -m anvil_exp01.data.download_xiph --data-dir data/xiph_1080p
```

After download, run MV extraction, dense flow, and prealignment on Xiph the same way as Vimeo90K (adjusting `--data-dir` to `data/xiph_1080p`).

### Expected Layout

See `docs/DATA_LAYOUT.md` for the full directory structure. Typical layout:

```text
open_source_release/
  data/
    vimeo_triplet/
      sequences/
      tri_trainlist.txt
      tri_testlist.txt
      mv_cache/
      prealigned/        # v1, for capacity sweep
      prealigned_v2/     # v2, for ANVIL-S/M
      motion_labels.csv
    xiph_1080p/
      sequences/
      prealigned_v2/
  artifacts/
    checkpoints/
    checkpoints_capacity/
    eval/
    onnx/
```

## Pretrained Weights

Download pretrained checkpoints and ONNX models from [HuggingFace](https://huggingface.co/NihilDigit/anvil):

```bash
# All checkpoints + ONNX (~258MB)
pixi run download-weights

# Or individual models:
pixi run -- huggingface-cli download NihilDigit/anvil checkpoints/D-unet-v3bs-nomv/best.pt --local-dir artifacts  # ANVIL-S (10MB)
pixi run -- huggingface-cli download NihilDigit/anvil checkpoints/D-unet-v3bm-nomv/best.pt --local-dir artifacts  # ANVIL-M (31MB)
```

## Quick Start

Train final models (ANVIL-S and ANVIL-M):

```bash
pixi run train-anvil-s
pixi run train-anvil-m
```

Evaluate on Vimeo90K:

```bash
pixi run python scripts/eval_vimeo.py \
  --model D-unet-v3bm-nomv \
  --checkpoint artifacts/checkpoints/D-unet-v3bm-nomv/best.pt \
  --data-dir data/vimeo_triplet \
  --prealigned-dir data/vimeo_triplet/prealigned_v2 \
  --motion-csv data/vimeo_triplet/motion_labels.csv \
  --output-dir artifacts/eval/vimeo/D-unet-v3bm-nomv
```

Evaluate on Xiph 1080p:

```bash
pixi run python scripts/eval_xiph.py \
  --model D-unet-v3bm-nomv \
  --checkpoint artifacts/checkpoints/D-unet-v3bm-nomv/best.pt \
  --data-dir data/xiph_1080p \
  --prealigned-dir data/xiph_1080p/prealigned_v2 \
  --output-dir artifacts/eval/xiph/D-unet-v3bm-nomv
```

Export ONNX:

```bash
pixi run export-onnx
```

## Reproduction Scripts

Each paper table has a corresponding `pixi run reproduce-*` shortcut.
Some tables require external vendor repos (RIFE, IFRNet/AMT) or hardware
(QAIRT SDK + Qualcomm device). See the dependency column below.

```bash
# Offline tables (GPU only, no vendor repos needed)
pixi run reproduce-main-quality      # Table main_quality: ANVIL models + baselines
pixi run reproduce-ablation          # Table ablation_capacity: 5-model sweep
pixi run reproduce-deploy-fusion     # Deploy-time fusion verification
pixi run reproduce-prealign-v2       # Prealign v2 improvement (+1.28 dB)
pixi run reproduce-fig-visual        # Figure 6: visual comparison

# Offline tables (require RIFE/IFRNet/AMT vendor repos)
pixi run reproduce-rife-reduced      # Table rife_reduced: RIFE 360p/480p ORT INT8
pixi run reproduce-int8-quality      # Table int8_quality: cross-method INT8 (note: ANVIL
                                     #   INT8 deltas are archived QNN device measurements,
                                     #   not live ORT quantization — see script docstring)
pixi run reproduce-instrumented      # Table instrumented: per-operator CosSim
pixi run reproduce-temporal          # Temporal consistency (tOF / WE)

# ALL offline tables in one go (needs vendor repos for RIFE/IFRNet rows)
pixi run reproduce-all-offline

# Device-dependent (require QAIRT SDK + Android device)
pixi run reproduce-latency           # Table latency: HTP FP16/INT8
pixi run reproduce-operator-compat   # Table operator_compat: HTP column
```

## Reproducing Paper Tables

### Table main_quality — Main Quality Comparison

ANVIL models on Vimeo90K and Xiph 1080p, plus RIFE native-resolution baseline:

```bash
# ANVIL-S on Vimeo90K
pixi run python scripts/eval_vimeo.py \
  --model D-unet-v3bs-nomv \
  --checkpoint artifacts/checkpoints/D-unet-v3bs-nomv/best.pt \
  --data-dir data/vimeo_triplet \
  --prealigned-dir data/vimeo_triplet/prealigned_v2 \
  --output-dir artifacts/eval/vimeo/D-unet-v3bs-nomv

# ANVIL-M on Vimeo90K
pixi run python scripts/eval_vimeo.py \
  --model D-unet-v3bm-nomv \
  --checkpoint artifacts/checkpoints/D-unet-v3bm-nomv/best.pt \
  --data-dir data/vimeo_triplet \
  --prealigned-dir data/vimeo_triplet/prealigned_v2 \
  --output-dir artifacts/eval/vimeo/D-unet-v3bm-nomv

# ANVIL-S on Xiph 1080p
pixi run python scripts/eval_xiph.py \
  --model D-unet-v3bs-nomv \
  --checkpoint artifacts/checkpoints/D-unet-v3bs-nomv/best.pt \
  --data-dir data/xiph_1080p \
  --prealigned-dir data/xiph_1080p/prealigned_v2 \
  --output-dir artifacts/eval/xiph/D-unet-v3bs-nomv

# ANVIL-M on Xiph 1080p
pixi run python scripts/eval_xiph.py \
  --model D-unet-v3bm-nomv \
  --checkpoint artifacts/checkpoints/D-unet-v3bm-nomv/best.pt \
  --data-dir data/xiph_1080p \
  --prealigned-dir data/xiph_1080p/prealigned_v2 \
  --output-dir artifacts/eval/xiph/D-unet-v3bm-nomv

# NAFNet-ceiling (17.1M params, quality upper bound) on Vimeo90K
pixi run python scripts/eval_vimeo.py \
  --model D-nafnet-nomv \
  --checkpoint artifacts/checkpoints/D-nafnet-nomv/best.pt \
  --data-dir data/vimeo_triplet \
  --prealigned-dir data/vimeo_triplet/prealigned_v2 \
  --output-dir artifacts/eval/vimeo/D-nafnet-nomv

# NAFNet-ceiling on Xiph 1080p
pixi run python scripts/eval_xiph.py \
  --model D-nafnet-nomv \
  --checkpoint artifacts/checkpoints/D-nafnet-nomv/best.pt \
  --data-dir data/xiph_1080p \
  --prealigned-dir data/xiph_1080p/prealigned_v2 \
  --output-dir artifacts/eval/xiph/D-nafnet-nomv

# Zero-network baselines (Naive Blend, MV Blend)
pixi run python scripts/eval_baselines.py \
  --output-dir artifacts/eval/baselines

# RIFE HDv3 native-resolution baseline
pixi run python scripts/eval_rife_native.py \
  --rife-repo vendor/ECCV2022-RIFE \
  --data-dir data/vimeo_triplet \
  --xiph-dir data/xiph_1080p \
  --output-dir artifacts/eval/rife_native
```

### Table rife_reduced — RIFE Reduced-Resolution Deployment

```bash
# Export RIFE ONNX models first
pixi run python scripts/export_rife_onnx.py \
  --rife-repo vendor/ECCV2022-RIFE \
  --weights vendor/ECCV2022-RIFE/train_log/flownet.pkl \
  --output-dir artifacts/onnx/rife

# ORT INT8 evaluation (offline, no device needed)
pixi run python scripts/eval_rife_reduced_res.py \
  --onnx-dir artifacts/onnx/rife \
  --xiph-dir data/xiph_1080p \
  --vimeo-dir data/vimeo_triplet \
  --output-dir artifacts/eval/rife_reduced

# QNN HTP INT8 evaluation (requires QAIRT SDK + Snapdragon device)
pixi run python scripts/eval_rife_qnn_int8.py \
  --onnx-dir artifacts/onnx/rife \
  --xiph-dir data/xiph_1080p \
  --qairt-root /opt/qcom/aistack/qairt/2.42.0.251225 \
  --output-dir artifacts/eval/rife_qnn_int8
```

### Table int8_quality — Cross-Method INT8 Quality

```bash
# Export the ONNX baselines needed by this table
pixi run python scripts/export_rife_onnx.py \
  --rife-repo vendor/ECCV2022-RIFE \
  --weights vendor/ECCV2022-RIFE/train_log/flownet.pkl \
  --output-dir artifacts/onnx/int8_cross \
  --resolutions 360p \
  --modes flow

pixi run python scripts/export_ifrnet_onnx.py \
  --amt-repo vendor/AMT \
  --weights vendor/IFRNet_Vimeo90K.pth \
  --output-dir artifacts/onnx/int8_cross

pixi run python scripts/eval_int8_cross_method.py \
  --onnx-dir artifacts/onnx/int8_cross \
  --xiph-dir data/xiph_1080p \
  --vimeo-dir data/vimeo_triplet \
  --prealigned-dir data/xiph_1080p/prealigned_v2 \
  --anvil-s-ckpt artifacts/checkpoints/D-unet-v3bs-nomv/best.pt \
  --anvil-m-ckpt artifacts/checkpoints/D-unet-v3bm-nomv/best.pt \
  --output-dir artifacts/eval/int8_cross_method
```

### Table instrumented — Per-Operator Instrumented Quantization

```bash
# RIFE flow and IFRNet flow are both required
pixi run python scripts/export_rife_onnx.py \
  --rife-repo vendor/ECCV2022-RIFE \
  --weights vendor/ECCV2022-RIFE/train_log/flownet.pkl \
  --output-dir artifacts/onnx/int8_instrumented \
  --resolutions 360p \
  --modes flow

pixi run python scripts/export_ifrnet_onnx.py \
  --amt-repo vendor/AMT \
  --weights vendor/IFRNet_Vimeo90K.pth \
  --output-dir artifacts/onnx/int8_instrumented \
  --modes flow

pixi run python scripts/eval_int8_instrumented.py \
  --onnx-dir artifacts/onnx/int8_instrumented \
  --xiph-dir data/xiph_1080p \
  --vimeo-dir data/vimeo_triplet \
  --models rife_flow_360p ifrnet_flow \
  --output-dir artifacts/eval/int8_instrumented
```

### Table latency — HTP Latency Benchmark

Requires QAIRT SDK and an Android device with Snapdragon SoC.

INT8 benchmarks require calibration data (.raw files + input_list.txt). Generate it first:

```bash
# Generate INT8 calibration data (100 samples, NCHW .raw files)
# Paper protocol: Vimeo90K training set (no test-set leakage)
pixi run python -m anvil_exp01.gen_calibration_data \
  --source vimeo \
  --prealigned-dir data/vimeo_triplet/prealigned_v2 \
  --train-list data/vimeo_triplet/tri_trainlist.txt \
  --n-samples 100 \
  --resolution 1080p \
  --out-dir artifacts/calib/anvil_s_1080p

# Legacy fallback: Xiph 1080p (NOT recommended — uses test set for calibration)
# pixi run python -m anvil_exp01.gen_calibration_data \
#   --source xiph \
#   --xiph-dir data/xiph_1080p \
#   --n-samples 100 \
#   --resolution 1080p \
#   --out-dir artifacts/calib/anvil_s_1080p

# Run latency benchmark (FP16 + INT8)
pixi run python scripts/bench_htp_latency.py \
  --onnx-dir artifacts/onnx \
  --calib-dir artifacts/calib/anvil_s_1080p \
  --qairt-root /opt/qcom/aistack/qairt/2.42.0.251225 \
  --output-dir artifacts/eval/latency
```

Use `--precision fp16` to skip INT8 (no calibration needed for FP16-only).

### Table operator_compat — Operator-Level NPU Compatibility

#### Qualcomm HTP Column

Requires QAIRT SDK and an Android device:

```bash
pixi run python scripts/bench_operators.py \
  --qairt-root /opt/qcom/aistack/qairt/2.42.0.251225 \
  --output-dir artifacts/eval/operator_compat
```

#### MediaTek APU Column

The MediaTek APU column uses a different toolchain: `mtk_converter` (ONNX to TFLite) and an NDK-compiled benchmark binary using the NeuroPilot TFLite Shim API.

See `docs/MEDIATEK_PIPELINE.md` for the full build and evaluation pipeline.

Key findings: PReLU and LayerNorm are NEURON_UNMAPPABLE on MediaTek APU. GridSample converts to `MTKEXT_GRID_SAMPLE_2D` custom op requiring the Premium SDK (not available via Public SDK). RIFE is completely non-deployable on the MediaTek NeuroPilot Public path.

**Disclaimer**: Latency data uses NeuroPilot Public SDK (runtime compilation), not Premium SDK (offline DLA compilation). Absolute latencies are conservative and should not be directly compared with Qualcomm QNN (offline-compiled) numbers.

### Table ablation_capacity — Capacity Sweep Ablation

Train 5 models with prealign v1, then evaluate:

```bash
# Train all 5 capacity sweep models
bash scripts/train_capacity_sweep.sh

# Evaluate each on Vimeo90K
for model in D-tiny-nomv D-mini-nomv D-mid-nomv D-unet-s-nomv D-unet-l-nomv; do
  pixi run python scripts/eval_vimeo.py \
    --model "$model" \
    --checkpoint "artifacts/checkpoints_capacity/$model/best.pt" \
    --data-dir data/vimeo_triplet \
    --prealigned-dir data/vimeo_triplet/prealigned \
    --output-dir "artifacts/eval/capacity_sweep/$model"
done
```

### Prealignment v2 Improvement (+1.28 dB)

The paper claims MV Blend improves from 29.92 to 31.20 dB (+1.28 dB) with prealign v2 smoothing. To reproduce, evaluate MV Blend with both prealignment versions:

```bash
# MV Blend with basic prealignment — requires prealign v1 data
pixi run python scripts/eval_baselines.py \
  --dataset vimeo \
  --methods mv_blend \
  --prealigned-v1-vimeo data/vimeo_triplet/prealigned \
  --output-dir artifacts/eval/baselines_v1

# MV Blend with smoothed prealignment (median + Gaussian)
pixi run python scripts/eval_baselines.py \
  --dataset vimeo \
  --methods mv_blend_v2 \
  --output-dir artifacts/eval/baselines_v2
```

Generate prealign v1 data with `pixi run prealign-v1` before running the v1 comparison.

### Deploy-Time Fusion Verification

Verifies that BN fusion is mathematically exact (max abs diff < 1.5e-5):

```bash
# ANVIL-S
pixi run python scripts/eval_deploy_fusion.py \
  --model D-unet-v3bs-nomv \
  --checkpoint artifacts/checkpoints/D-unet-v3bs-nomv/best.pt \
  --output-dir artifacts/eval/deploy_fusion/D-unet-v3bs-nomv

# ANVIL-M
pixi run python scripts/eval_deploy_fusion.py \
  --model D-unet-v3bm-nomv \
  --checkpoint artifacts/checkpoints/D-unet-v3bm-nomv/best.pt \
  --output-dir artifacts/eval/deploy_fusion/D-unet-v3bm-nomv
```

### Temporal Consistency (tOF / WE)

Evaluates temporal fidelity using optical flow magnitude (tOF) and warping error (WE) on Xiph 1080p. Requires CUDA (~8 GB for RAFT-Small at 540p):

```bash
pixi run python scripts/eval_temporal_consistency.py \
  --xiph-dir data/xiph_1080p \
  --anvil-s-ckpt artifacts/checkpoints/D-unet-v3bs-nomv/best.pt \
  --anvil-m-ckpt artifacts/checkpoints/D-unet-v3bm-nomv/best.pt \
  --rife-repo vendor/ECCV2022-RIFE \
  --output-dir artifacts/eval/temporal_consistency
```

Takes ~2 hours for 12 sequences x 30 pairs. Use `--max-pairs 5` for a quick check (~20 min).

### Figure 6 — Visual Comparison

Generate the 3-scenario visual comparison figure (ANVIL-M vs RIFE HDv3 vs Ground Truth):

```bash
pixi run python scripts/gen_fig_visual_comparison.py \
  --xiph-dir data/xiph_1080p \
  --anvil-ckpt artifacts/checkpoints/D-unet-v3bm-nomv/best.pt \
  --rife-repo vendor/ECCV2022-RIFE \
  --output-dir artifacts/figures
```

Output: `artifacts/figures/fig_visual_comparison.pdf`

### Table e2e_latency — End-to-End System Latency

The end-to-end pipeline runs in the [mpv-android-anvil](https://github.com/NihilDigit/mpv-android-anvil) repository, a research demo fork of mpv-android with ANVIL integrated as a native video filter (`vf_anvil`, ~2800 lines C). A pre-built release APK is available in that repo's [Releases](https://github.com/NihilDigit/mpv-android-anvil/releases) page.

**Paper claim**: median VFI latency = **28.4 ms** on SM8650, measured over 54,623 consecutively logged frame pairs during a 30-minute continuous playback of 1080p H.264 30fps content (94.9% within the 33.3 ms budget). See `bench_paper_e2e/` in the mpv-android-anvil repo for raw data. Note: full-frame logging adds ~4°C thermal overhead; the released APK uses sampled logging (every 30th frame) for normal use.

**Three-accelerator pipeline** (CPU + GPU Vulkan + HTP INT8, pipelined):

| Stage | Hardware | Latency | Description |
|-------|----------|---------|-------------|
| P1a | CPU | 2.9 ms | MV densify + downsample + YUV pack (NEON) |
| P1b+P2 | GPU (Adreno 750) | 3.7 ms | Prealign v2 + quantized warp |
| Copy | CPU | 0.9 ms | 12 MB uint8 NHWC memcpy (NEON prefetch) |
| P3 | HTP V75 (INT8) | 17.0 ms | ANVIL-S inference (pipelined async) |
| P4 | GPU (Adreno 750) | 3.3 ms | Residual + RGB→YUV420 |
| **Total** | | **28.4 ms** | Median over 30-min 30fps playback (n=54,623) |

**Stress test data** (in the mpv-android-anvil repo under `bench_paper_e2e/`):
- `anvil_timing.csv` — parsed per-stage timings (n=54,623)
- `timing_summary.json` — paper-ready statistics (median, p95, min, max)
- `telemetry.csv` — system telemetry (battery, thermal, CPU/GPU freq, 10s interval)
- `meta.json` — device info, battery drop, duration

**Key source files** (in the mpv-android-anvil repo):
- `anvil/filter/vf_anvil.c` — core VFI filter (~2800 lines, three-accelerator pipeline)
- `anvil/shaders/*.comp` — 4 Vulkan compute shaders (median, Gauss, warp+quant, residual+YUV)
- `app/src/main/assets/anvil/context.serialized.bin` — pre-compiled QNN HTP INT8 model (ANVIL-S 1080p)

**Build prerequisites:**
- Linux x86_64, Qualcomm QAIRT SDK 2.42+, Android NDK r29
- ~20 GB disk space for NDK/SDK + dependencies
- See the [mpv-android-anvil README](https://github.com/NihilDigit/mpv-android-anvil) for full build instructions

## External Dependencies

### RIFE HDv3

Required for: Tables main_quality, rife_reduced, int8_quality, instrumented.

```bash
git clone https://github.com/megvii-research/ECCV2022-RIFE vendor/ECCV2022-RIFE
# Download pretrained weights into vendor/ECCV2022-RIFE/train_log/
```

### IFRNet

Required for: Tables int8_quality and instrumented.

IFRNet weights and architecture are loaded from the AMT vendor repository (`vendor/AMT`), which bundles IFRNet as a baseline for comparison. This avoids maintaining a separate IFRNet clone.

```bash
git clone https://github.com/MCG-NKU/AMT vendor/AMT
# Download IFRNet_Vimeo90K.pth and place it at vendor/IFRNet_Vimeo90K.pth
```

### QAIRT SDK

Required for: Tables rife_reduced (QNN column), latency, operator_compat.

Install the Qualcomm AI Engine Direct SDK. Scripts expect the path via `--qairt-root` or `QAIRT_SDK_ROOT` environment variable.

### Android Device

Required for: device-side QNN/HTP benchmarks and the end-to-end demo app. Tested on Snapdragon 8 Gen 2 (SM8550), 8 Gen 3 (SM8650), 7+ Gen 2 (SM7475), and MediaTek Dimensity 9300/9400+.

## Smoke Tests

Run a quick sanity check on all scripts (requires data + checkpoints in place):

```bash
bash scripts/smoke_test.sh
```

This runs each evaluation script with `--limit 2` (or equivalent). Device-dependent scripts (HTP benchmarks) are skipped. Use `--skip-rife` to also skip RIFE-dependent tests.

## Notes

- The release bundle assumes CUDA is available for training and LPIPS evaluation.
- `cv2`, `scipy`, `onnxruntime`, `lpips`, and `motion-vector-extractor` are included in `pixi.toml`.
- Capacity sweep models use prealign v1 (basic); final ANVIL-S/M use prealign v2 (smoothed). Do not mix them.
- Device-specific deployment code lives in the [mpv-android-anvil](https://github.com/NihilDigit/mpv-android-anvil) repository.
