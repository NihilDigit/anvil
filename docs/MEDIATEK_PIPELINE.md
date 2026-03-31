# MediaTek NeuroPilot Public Deployment Pipeline

This document describes the end-to-end pipeline for deploying ANVIL on MediaTek APU devices using the NeuroPilot Public SDK. No NDA or Premium SDK is required.

**Paper reference**: This pipeline reproduces the **MediaTek APU column** in Table `operator_compat` and the cross-vendor validation results reported in the paper.

> **Disclaimer**: Latency data uses NeuroPilot Public SDK (runtime compilation via TFLite Shim API),
> not MediaTek Premium SDK (offline DLA compilation via ncc-tflite). Absolute latencies include
> significant Shim runtime overhead and should NOT be directly compared with Qualcomm QNN
> (offline-compiled) numbers. Cross-platform comparison should use normalized ratios (conv3x3 = 1.0x).
> Operator compatibility conclusions (UNMAPPABLE, MTKEXT failures) are unaffected by the runtime path.

## Cross-Vendor Validation Summary

| Platform | NPU | Device | ANVIL-S INT8 1080p | 33.3ms? | INT8 Quality |
|----------|-----|--------|:------------------:|:-------:|:------------:|
| D9300 | APU 790 | vivo Pad3 Pro (PA2473) | 24.4ms | Yes | +0.02 dB |
| D9400+ | APU 890 | vivo X200s (V2458A) | 25.5ms | Yes | ~0 dB |

RIFE is **completely blocked** on the NeuroPilot Public path: `mtk_converter` converts GridSample to `MTKEXT_GRID_SAMPLE_2D` custom op, which the Public Shim API cannot execute. PReLU is also `NEURON_UNMAPPABLE`, further blocking RIFE. Both D9300 and D9400+ exhibit identical operator support limitations.

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| mtk_converter | 8.13.0 | ONNX to TFLite conversion (requires Python 3.11) |
| NeuroPilotTFLiteShim.h | 8.x | Header-only dlopen wrapper, no linking needed |
| NDK | r29+ | clang++ cross-compilation for the benchmark binary |
| Device | libtflite_mtk.so pre-installed | Tested: vivo Pad3 Pro (D9300), vivo X200s (D9400+) |
| onnxsim | latest | ONNX graph simplification (required by mtk_converter) |

### Environment Setup

```bash
# Create a Python 3.11 virtual environment for mtk_converter
python3.11 -m venv mtk-env
source mtk-env/bin/activate
pip install mtk_converter
```

## Step 1: ONNX Simplify

`mtk_converter` does not accept ONNX models with dynamic shapes or redundant nodes. Use `onnxsim` to clean the graph first.

```bash
python -m onnxsim \
    models/anvil_s_1080p.onnx \
    build/mtk/anvil_s_1080p_sim.onnx
```

## Step 2: ONNX to TFLite (FP32)

```bash
source mtk-env/bin/activate
python - <<'PYEOF'
from mtk_converter import OnnxConverter

converter = OnnxConverter.from_model_proto("build/mtk/anvil_s_1080p_sim.onnx")
converter.convert_to_tflite(output_file="build/mtk/anvil_s_mtk_1080p.tflite")
PYEOF
```

## Step 3: INT8 PTQ Quantization

### 3a. Generate Calibration Data

The calibration script outputs NCHW float32 `.raw` files + `input_list.txt`:

```bash
python -m anvil_exp01.gen_calibration_data \
    --source vimeo \
    --prealigned-dir data/vimeo_triplet/prealigned_v2 \
    --train-list data/vimeo_triplet/tri_trainlist.txt \
    --n-samples 100 \
    --resolution 1080p \
    --out-dir build/mtk/calib_anvil_s_1080p
```

### 3b. Convert .raw to .npy for mtk_converter

`mtk_converter` expects numpy arrays, not raw files. Convert:

```bash
source mtk-env/bin/activate
python - <<'PYEOF'
import numpy as np
from pathlib import Path

calib_dir = Path("build/mtk/calib_anvil_s_1080p")
# Each .raw is NCHW float32 (6, 1080, 1920)
shape = (6, 1080, 1920)
for raw_file in sorted(calib_dir.glob("*.raw")):
    arr = np.fromfile(raw_file, dtype=np.float32).reshape(1, *shape)
    np.save(raw_file.with_suffix(".npy"), arr)
print(f"Converted {len(list(calib_dir.glob('*.npy')))} files")
PYEOF
```

### 3c. Quantized Conversion

```bash
source mtk-env/bin/activate
python - <<'PYEOF'
import numpy as np
from pathlib import Path
from mtk_converter import OnnxConverter

converter = OnnxConverter.from_model_proto("build/mtk/anvil_s_1080p_sim.onnx")

# Load calibration data (list of numpy arrays, NCHW float32)
calib_dir = Path("build/mtk/calib_anvil_s_1080p")
calib_data = [np.load(f) for f in sorted(calib_dir.glob("*.npy"))[:100]]

# input_value_ranges = [0, 1] (normalized RGB range)
converter.quantize(
    calibration_data_gen=iter(calib_data),
    input_value_ranges=[(0.0, 1.0)],
    quantization_method="symmetric",
    per_channel=True,
)

converter.convert_to_tflite(
    output_file="build/mtk/anvil_s_mtk_1080p_int8.tflite"
)
PYEOF
```

**Note:** 1080p INT8 calibration with 100 samples requires approximately 12GB RAM (100 x 6ch x 1080 x 1920 x float32).

## Step 4: Compile Benchmark Binary

Single-file C++ source (`scripts/bench_neuropilot.cc`) compiled directly with NDK clang++. No CMake required.

**Prerequisite**: Download `NeuroPilotTFLiteShim.h` from the [MediaTek NeuroPilot SDK](https://neuropilot.mediatek.com/) and place it in a directory (e.g., `neuropilot/`). This header is MediaTek proprietary and cannot be redistributed.

```bash
NDK=/path/to/android-ndk-r29
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang++ \
    -O2 -std=c++17 \
    -I neuropilot \
    -o build/mtk/bench_neuropilot \
    scripts/bench_neuropilot.cc \
    -ldl -lm -static-libstdc++
```

The `-ldl` flag is needed for Shim API's `dlopen(libtflite_mtk.so)`. No TFLite libraries need to be linked.

## Step 5: Device Deployment

```bash
DEVICE_DIR=/data/local/tmp/anvil_bench

adb push build/mtk/bench_neuropilot $DEVICE_DIR/
adb push build/mtk/anvil_s_mtk_1080p.tflite $DEVICE_DIR/
adb push build/mtk/anvil_s_mtk_1080p_int8.tflite $DEVICE_DIR/

# Push test input (NHWC float32 raw, for quality verification)
adb push test_input.raw $DEVICE_DIR/

adb shell chmod +x $DEVICE_DIR/bench_neuropilot
```

No need to push `libtflite_mtk.so` -- it is pre-installed on the device.

## Step 6: Run Benchmarks

### Latency Benchmark

```bash
# FP32 + NEURON delegate
adb shell "$DEVICE_DIR/bench_neuropilot \
    --graph=$DEVICE_DIR/anvil_s_mtk_1080p.tflite \
    --accel=neuron \
    --num_runs=50 --warmup=10"

# INT8 + NEURON delegate
adb shell "$DEVICE_DIR/bench_neuropilot \
    --graph=$DEVICE_DIR/anvil_s_mtk_1080p_int8.tflite \
    --accel=neuron \
    --num_runs=50 --warmup=10"
```

### Quality Verification (output raw file)

```bash
adb shell "$DEVICE_DIR/bench_neuropilot \
    --graph=$DEVICE_DIR/anvil_s_mtk_1080p_int8.tflite \
    --accel=neuron \
    --input_raw=$DEVICE_DIR/test_input.raw \
    --output_raw=$DEVICE_DIR/test_output.raw \
    --num_runs=1"

adb pull $DEVICE_DIR/test_output.raw ./
```

Output is NHWC float32 raw. Reshape with numpy and compute PSNR against the FP32 reference.

## Known Limitations

1. **NeuroPilot Public Shim runtime overhead** -- The Shim API calls `dlopen` at runtime to access `libtflite_mtk.so`, adding dispatch overhead compared to Premium SDK's offline `ncc-tflite` to DLA compilation path. Expect 2-3x slower than Premium SDK. Actual APU kernel time is shorter than measured end-to-end latency. This is confirmed by D9400+ (APU 890) showing nearly identical latency to D9300 (APU 790) despite higher NPU specs -- runtime overhead dominates, not compute.

2. **MTKEXT_GRID_SAMPLE_2D** -- `mtk_converter` converts ONNX GridSample to this custom op, but the NeuroPilot Public Shim API cannot execute it (requires Premium SDK's custom op registry). **RIFE and all flow-warp VFI methods (7/9 surveyed) are completely blocked on this path.** Verified on both D9300 and D9400+.

3. **PReLU / LayerNorm** -- The NEURON delegate reports `NEURON_UNMAPPABLE` for both operators, falling back to CPU. This additionally blocks RIFE (PReLU) and transformer-based VFI methods like VFIformer and EMA-VFI (LayerNorm). ANVIL's pure Conv+ReLU architecture is unaffected.

4. **INT8 calibration memory** -- 1080p calibration with 100 samples requires approximately 12GB RAM, completed in a single Python process.

5. **Resize is extremely slow** -- Bilinear resize 2x on the APU takes 1200ms at 1080p, an order of magnitude slower than Qualcomm HTP. ANVIL uses ConvTranspose instead and is unaffected. This is a cross-platform bottleneck: pyramid-based VFI methods face deployment barriers on both vendors.

## Key Findings Summary

These findings reproduce the MediaTek APU column in the paper's operator compatibility table (Table `operator_compat`):

| Operator | Freq in VFI | APU Status | Impact |
|----------|:-----------:|:----------:|--------|
| Conv2d 3x3 | 9/9 | 1.0x (baseline) | ANVIL sweet spot |
| Conv2d 1x1 | 8/9 | 1.0x | ANVIL sweet spot |
| Conv + ReLU | 5/9 | 1.0x | ANVIL sweet spot |
| ConvTranspose2d | 7/9 | 2.7x | Usable |
| Residual Add | 9/9 | 1.8x | Usable |
| GridSample | 7/9 | **MTKEXT (blocked)** | 7/9 VFI methods out |
| Resize 2x | 7/9 | **12.6x** | Pyramid methods bottlenecked |
| Conv + Sigmoid | 7/9 | 0.9x | OK |
| Conv + PReLU | 1/9 | **UNMAPPABLE** | RIFE blocked |
| Conv + LeakyReLU | 2/9 | 1.0x | OK |
| Conv + LayerNorm | 2/9 | **UNMAPPABLE** | Transformer VFI blocked |
| Conv + GELU | 1/9 | **5.8x** | Very slow |
| DepthwiseConv 3x3 | 1/9 | 0.9x | OK |

Normalized ratios use conv3x3 as the 1.0x baseline. All measurements at 1080p resolution, 32 channels.

**Key takeaway**: ANVIL uses only Conv + ReLU (1.0x baseline on both Qualcomm HTP and MediaTek APU), making it the only surveyed VFI method deployable at 1080p INT8 across both major mobile NPU vendors.

## Tested Devices

| SoC | NPU | NeuroPilot Version | Device |
|-----|-----|--------------------|--------|
| Dimensity 9300 (MT6989) | APU 790 | 8.2.10 | vivo Pad3 Pro (PA2473) |
| Dimensity 9400+ (MT6991) | APU 890 | 8.2.26 | vivo X200s (V2458A) |
