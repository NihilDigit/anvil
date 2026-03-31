"""Operator-level HTP latency benchmark for 17 VFI operators.

Reproduces paper Table II (operator compatibility, HTP column).  For each of
17 operators drawn from a survey of 9 mainstream VFI methods, this script:

  1. Generates a minimal ONNX model programmatically (onnx.helper)
  2. Compiles to a QNN FP16 context binary via Docker + QAIRT SDK
  3. Pushes to device and runs qnn-net-run for latency measurement
  4. Parses profiling output and records latency
  5. Normalizes all latencies relative to conv3x3 = 1.0x

Operators that fail at any stage (OOM, unsupported ONNX op, HTP compile
failure) are recorded with their failure mode -- these failures are themselves
the key data points showing deployment obstacles.

Operator catalog (from literature survey of RIFE, IFRNet, AMT, VFIformer,
EMA-VFI, M2M, ABME, FLAVR, UPR-Net):

  Group A (universal CNN):  conv3x3, conv1x1, conv_relu, convtranspose, add_residual
  Group B (flow-warp):      grid_sample, resize_2x, conv_sigmoid, conv_prelu, conv_leakyrelu
  Group C (advanced):       conv_layernorm, conv_gelu, dwconv3x3, self_attention, deformable_conv
  Group D (composite):      iter_accum_3, warp_chain

Prerequisites:
  - Docker (for QAIRT conversion)
  - QAIRT SDK on host
  - Android device connected via ADB with Hexagon HTP support
  - Python packages: numpy, onnx (for ONNX model generation)

Usage:
    # Full pipeline: generate ONNX, convert, benchmark, report
    python scripts/bench_operators.py \\
        --qairt-root /opt/qcom/aistack/qairt/2.42.0.251225 \\
        --output-dir artifacts/bench/operators

    # Skip conversion (reuse existing QNN artifacts):
    python scripts/bench_operators.py --skip-convert --skip-generate \\
        --output-dir artifacts/bench/operators

    # Only specific operators:
    python scripts/bench_operators.py --operators conv3x3,grid_sample,iter_accum_3 \\
        --output-dir artifacts/bench/operators
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    print("ERROR: onnx package required. Install with: pip install onnx", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCKER_IMAGE = "qairt-converter"
CONTAINER_QAIRT = "/opt/qairt"

DEFAULT_DSP_ARCH = "v75"
DEFAULT_SOC_ID = 57
DEFAULT_HTP_VERSION = "V75"

RESOLUTIONS = {
    "vimeo": (256, 448),
    "540p": (540, 960),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}

# ---------------------------------------------------------------------------
# Operator catalog
# ---------------------------------------------------------------------------

OPERATORS = OrderedDict([
    # Group A: Universal CNN ops
    ("conv3x3",        {"group": "A", "desc": "Conv2d 3x3",                       "freq": "9/9"}),
    ("conv1x1",        {"group": "A", "desc": "Conv2d 1x1",                       "freq": "8/9"}),
    ("conv_relu",      {"group": "A", "desc": "Conv + ReLU",                      "freq": "5/9"}),
    ("convtranspose",  {"group": "A", "desc": "ConvTranspose2d 4x4 (stride=2)",   "freq": "7/9"}),
    ("add_residual",   {"group": "A", "desc": "Residual Add (Conv+Conv+Add)",      "freq": "9/9"}),
    # Group B: Flow-warp ops
    ("grid_sample",    {"group": "B", "desc": "GridSample bilinear",               "freq": "7/9"}),
    ("resize_2x",      {"group": "B", "desc": "Resize bilinear 2x",               "freq": "7/9"}),
    ("conv_sigmoid",   {"group": "B", "desc": "Conv + Sigmoid",                    "freq": "7/9"}),
    ("conv_prelu",     {"group": "B", "desc": "Conv + PReLU",                      "freq": "1/9"}),
    ("conv_leakyrelu", {"group": "B", "desc": "Conv + LeakyReLU(0.01)",            "freq": "2/9"}),
    # Group C: Advanced / Transformer ops
    ("conv_layernorm", {"group": "C", "desc": "Conv + LayerNorm",                  "freq": "2/9"}),
    ("conv_gelu",      {"group": "C", "desc": "Conv + GELU (x*sigmoid(1.702x))",   "freq": "1/9"}),
    ("dwconv3x3",      {"group": "C", "desc": "DepthwiseConv 3x3 + Pointwise",    "freq": "1/9"}),
    ("self_attention",  {"group": "C", "desc": "Multi-Head Self-Attention (4-head)","freq": "2/9"}),
    ("deformable_conv", {"group": "C", "desc": "Deformable Conv2d",                "freq": "1/9"}),
    # Group D: Composite patterns
    ("iter_accum_3",   {"group": "D", "desc": "3-stage iterative accumulation",    "freq": "4/9"}),
    ("warp_chain",     {"group": "D", "desc": "Conv->GridSample->Conv->GridSample","freq": "7/9"}),
])

# Operators with a secondary grid input (1, H, W, 2)
GRID_INPUT_OPS = {"grid_sample", "warp_chain"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str, *, check: bool = True, capture: bool = False, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, check=check, capture_output=capture, text=True, **kwargs)


def adb(cmd: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return run(f"adb {cmd}", check=check, capture=capture)


def _make_weight(shape: tuple, name: str) -> onnx.TensorProto:
    """Create a random float32 weight initializer."""
    rng = np.random.default_rng(hash(name) % (2**31))
    data = (rng.standard_normal(shape) * 0.01).astype(np.float32)
    return numpy_helper.from_array(data, name=name)


def _make_bias(size: int, name: str) -> onnx.TensorProto:
    """Create a zero bias initializer."""
    data = np.zeros(size, dtype=np.float32)
    return numpy_helper.from_array(data, name=name)


def _conv_node(
    name: str,
    input_name: str,
    output_name: str,
    weight_name: str,
    bias_name: str,
    kernel: int = 3,
    pads: list[int] | None = None,
    group: int = 1,
    strides: list[int] | None = None,
) -> onnx.NodeProto:
    """Create a Conv node."""
    if pads is None:
        p = kernel // 2
        pads = [p, p, p, p]
    if strides is None:
        strides = [1, 1]
    return helper.make_node(
        "Conv", [input_name, weight_name, bias_name], [output_name],
        name=name, kernel_shape=[kernel, kernel], pads=pads,
        strides=strides, group=group,
    )


# ---------------------------------------------------------------------------
# ONNX model generators (one per operator)
# ---------------------------------------------------------------------------

def _gen_conv3x3(ch: int, H: int, W: int) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])
    w = _make_weight((ch, ch, 3, 3), "W")
    b = _make_bias(ch, "B")
    conv = _conv_node("conv", "input", "output", "W", "B", kernel=3)
    graph = helper.make_graph([conv], "conv3x3", [inp], [out], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_conv1x1(ch: int, H: int, W: int) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])
    w = _make_weight((ch, ch, 1, 1), "W")
    b = _make_bias(ch, "B")
    conv = _conv_node("conv", "input", "output", "W", "B", kernel=1, pads=[0, 0, 0, 0])
    graph = helper.make_graph([conv], "conv1x1", [inp], [out], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_conv_relu(ch: int, H: int, W: int) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])
    w = _make_weight((ch, ch, 3, 3), "W")
    b = _make_bias(ch, "B")
    conv = _conv_node("conv", "input", "conv_out", "W", "B", kernel=3)
    relu = helper.make_node("Relu", ["conv_out"], ["output"], name="relu")
    graph = helper.make_graph([conv, relu], "conv_relu", [inp], [out], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_convtranspose(ch: int, H: int, W: int) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H * 2, W * 2])
    w = _make_weight((ch, ch, 4, 4), "W")
    b = _make_bias(ch, "B")
    convt = helper.make_node(
        "ConvTranspose", ["input", "W", "B"], ["output"],
        name="convt", kernel_shape=[4, 4], strides=[2, 2], pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph([convt], "convtranspose", [inp], [out], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_add_residual(ch: int, H: int, W: int) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])
    w_a = _make_weight((ch, ch, 3, 3), "W_a")
    b_a = _make_bias(ch, "B_a")
    w_b = _make_weight((ch, ch, 3, 3), "W_b")
    b_b = _make_bias(ch, "B_b")
    conv_a = _conv_node("conv_a", "input", "feat_a", "W_a", "B_a")
    conv_b = _conv_node("conv_b", "input", "feat_b", "W_b", "B_b")
    add = helper.make_node("Add", ["feat_a", "feat_b"], ["output"], name="add")
    graph = helper.make_graph(
        [conv_a, conv_b, add], "add_residual", [inp], [out],
        initializer=[w_a, b_a, w_b, b_b],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_grid_sample(ch: int, H: int, W: int) -> onnx.ModelProto:
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, ch, H, W])
    grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, H, W, 2])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])
    gs = helper.make_node(
        "GridSample", ["image", "grid"], ["output"],
        name="grid_sample", mode="bilinear", padding_mode="zeros", align_corners=1,
    )
    graph = helper.make_graph([gs], "grid_sample", [image, grid], [out])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_resize_2x(ch: int, H: int, W: int) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H * 2, W * 2])
    w = _make_weight((ch, ch, 3, 3), "W")
    b = _make_bias(ch, "B")
    conv = _conv_node("conv", "input", "conv_out", "W", "B")
    roi = numpy_helper.from_array(np.array([], dtype=np.float32), "roi")
    scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), "scales")
    resize = helper.make_node(
        "Resize", ["conv_out", "roi", "scales"], ["output"],
        name="resize", mode="linear",
    )
    graph = helper.make_graph(
        [conv, resize], "resize_2x", [inp], [out],
        initializer=[w, b, roi, scales],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_conv_sigmoid(ch: int, H: int, W: int) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])
    w = _make_weight((ch, ch, 3, 3), "W")
    b = _make_bias(ch, "B")
    conv = _conv_node("conv", "input", "conv_out", "W", "B")
    sig = helper.make_node("Sigmoid", ["conv_out"], ["output"], name="sigmoid")
    graph = helper.make_graph([conv, sig], "conv_sigmoid", [inp], [out], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_conv_prelu(ch: int, H: int, W: int) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])
    w = _make_weight((ch, ch, 3, 3), "W")
    b = _make_bias(ch, "B")
    slope = numpy_helper.from_array(
        np.full((ch, 1, 1), 0.25, dtype=np.float32), "slope"
    )
    conv = _conv_node("conv", "input", "conv_out", "W", "B")
    prelu = helper.make_node("PReLU", ["conv_out", "slope"], ["output"], name="prelu")
    graph = helper.make_graph(
        [conv, prelu], "conv_prelu", [inp], [out],
        initializer=[w, b, slope],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_conv_leakyrelu(ch: int, H: int, W: int) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])
    w = _make_weight((ch, ch, 3, 3), "W")
    b = _make_bias(ch, "B")
    conv = _conv_node("conv", "input", "conv_out", "W", "B")
    lrelu = helper.make_node(
        "LeakyRelu", ["conv_out"], ["output"], name="leakyrelu", alpha=0.01,
    )
    graph = helper.make_graph(
        [conv, lrelu], "conv_leakyrelu", [inp], [out], initializer=[w, b],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_conv_layernorm(ch: int, H: int, W: int) -> onnx.ModelProto:
    """Conv + LayerNormalization (ONNX opset 17)."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])
    w = _make_weight((ch, ch, 3, 3), "W")
    b = _make_bias(ch, "B")
    conv = _conv_node("conv", "input", "conv_out", "W", "B")
    # Transpose NCHW -> NHWC for LayerNorm
    tr1 = helper.make_node(
        "Transpose", ["conv_out"], ["nhwc"], name="to_nhwc", perm=[0, 2, 3, 1],
    )
    ln_scale = numpy_helper.from_array(np.ones(ch, dtype=np.float32), "ln_scale")
    ln_bias = numpy_helper.from_array(np.zeros(ch, dtype=np.float32), "ln_bias")
    ln = helper.make_node(
        "LayerNormalization", ["nhwc", "ln_scale", "ln_bias"], ["ln_out"],
        name="layernorm", axis=-1, epsilon=1e-5,
    )
    # Transpose back NHWC -> NCHW
    tr2 = helper.make_node(
        "Transpose", ["ln_out"], ["output"], name="to_nchw", perm=[0, 3, 1, 2],
    )
    graph = helper.make_graph(
        [conv, tr1, ln, tr2], "conv_layernorm", [inp], [out],
        initializer=[w, b, ln_scale, ln_bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _gen_conv_gelu(ch: int, H: int, W: int) -> onnx.ModelProto:
    """Conv + GELU approximation: x * sigmoid(1.702 * x)."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])
    w = _make_weight((ch, ch, 3, 3), "W")
    b = _make_bias(ch, "B")
    conv = _conv_node("conv", "input", "conv_out", "W", "B")
    alpha = numpy_helper.from_array(np.array([1.702], dtype=np.float32), "alpha")
    mul1 = helper.make_node("Mul", ["conv_out", "alpha"], ["scaled"], name="mul_alpha")
    sig = helper.make_node("Sigmoid", ["scaled"], ["gate"], name="sigmoid")
    mul2 = helper.make_node("Mul", ["conv_out", "gate"], ["output"], name="mul_gate")
    graph = helper.make_graph(
        [conv, mul1, sig, mul2], "conv_gelu", [inp], [out],
        initializer=[w, b, alpha],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_dwconv3x3(ch: int, H: int, W: int) -> onnx.ModelProto:
    """DepthwiseConv 3x3 + Pointwise 1x1."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])
    dw_w = _make_weight((ch, 1, 3, 3), "dw_W")
    dw_b = _make_bias(ch, "dw_B")
    pw_w = _make_weight((ch, ch, 1, 1), "pw_W")
    pw_b = _make_bias(ch, "pw_B")
    dw = _conv_node("dw_conv", "input", "dw_out", "dw_W", "dw_B", kernel=3, group=ch)
    pw = _conv_node("pw_conv", "dw_out", "output", "pw_W", "pw_B", kernel=1, pads=[0, 0, 0, 0])
    graph = helper.make_graph(
        [dw, pw], "dwconv3x3", [inp], [out],
        initializer=[dw_w, dw_b, pw_w, pw_b],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_self_attention(ch: int, H: int, W: int) -> onnx.ModelProto | None:
    """Multi-head self-attention (4 heads) with 4x downsampling.

    At 1080p this creates ~270x480 feature maps for attention, which may
    still OOM on HTP. This is expected and is itself a key finding.
    """
    n_heads = 4
    head_dim = ch // n_heads
    Hd, Wd = H // 4, W // 4
    N = Hd * Wd

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])

    nodes = []
    initializers = []

    # Downsample: 2 strided convolutions
    dw1 = _make_weight((ch, ch, 3, 3), "down1_W")
    db1 = _make_bias(ch, "down1_B")
    initializers.extend([dw1, db1])
    nodes.append(helper.make_node(
        "Conv", ["input", "down1_W", "down1_B"], ["d1"],
        name="down1", kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1],
    ))
    nodes.append(helper.make_node("Relu", ["d1"], ["d1r"], name="relu1"))

    dw2 = _make_weight((ch, ch, 3, 3), "down2_W")
    db2 = _make_bias(ch, "down2_B")
    initializers.extend([dw2, db2])
    nodes.append(helper.make_node(
        "Conv", ["d1r", "down2_W", "down2_B"], ["d2"],
        name="down2", kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1],
    ))
    nodes.append(helper.make_node("Relu", ["d2"], ["d2r"], name="relu2"))

    # QKV projection (1x1 conv -> 3*ch)
    qkv_w = _make_weight((ch * 3, ch, 1, 1), "qkv_W")
    qkv_b = _make_bias(ch * 3, "qkv_B")
    initializers.extend([qkv_w, qkv_b])
    nodes.append(_conv_node("qkv", "d2r", "qkv_out", "qkv_W", "qkv_B", kernel=1, pads=[0, 0, 0, 0]))

    # Reshape (1, 3*ch, Hd, Wd) -> (1, 3, n_heads, head_dim, N)
    shape1 = numpy_helper.from_array(
        np.array([1, 3, n_heads, head_dim, N], dtype=np.int64), "shape1"
    )
    initializers.append(shape1)
    nodes.append(helper.make_node("Reshape", ["qkv_out", "shape1"], ["qkv_5d"], name="reshape1"))

    # Split Q, K, V along dim=1
    nodes.append(helper.make_node(
        "Split", ["qkv_5d"], ["q_4d", "k_4d", "v_4d"],
        name="split_qkv", axis=1,
    ))

    # Squeeze dim=1: (1, 1, heads, hdim, N) -> (1, heads, hdim, N)
    sq_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "sq_axes")
    initializers.append(sq_axes)
    nodes.append(helper.make_node("Squeeze", ["q_4d", "sq_axes"], ["q"], name="sq_q"))
    nodes.append(helper.make_node("Squeeze", ["k_4d", "sq_axes"], ["k"], name="sq_k"))
    nodes.append(helper.make_node("Squeeze", ["v_4d", "sq_axes"], ["v"], name="sq_v"))

    # Transpose to (1, heads, N, hdim)
    nodes.append(helper.make_node("Transpose", ["q"], ["qt"], name="tr_q", perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("Transpose", ["k"], ["kt"], name="tr_k", perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("Transpose", ["v"], ["vt"], name="tr_v", perm=[0, 1, 3, 2]))

    # Attention: Qt @ Kt^T / sqrt(head_dim)
    nodes.append(helper.make_node("Transpose", ["kt"], ["kt_t"], name="tr_k2", perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("MatMul", ["qt", "kt_t"], ["attn_raw"], name="matmul1"))
    scale_val = numpy_helper.from_array(
        np.array([1.0 / (head_dim ** 0.5)], dtype=np.float32), "scale_val"
    )
    initializers.append(scale_val)
    nodes.append(helper.make_node("Mul", ["attn_raw", "scale_val"], ["attn_scaled"], name="scale"))
    nodes.append(helper.make_node("Softmax", ["attn_scaled"], ["attn_weights"], name="softmax", axis=-1))

    # attn_weights @ Vt -> (1, heads, N, hdim)
    nodes.append(helper.make_node("MatMul", ["attn_weights", "vt"], ["attn_out"], name="matmul2"))

    # Transpose back and reshape to (1, ch, Hd, Wd)
    nodes.append(helper.make_node(
        "Transpose", ["attn_out"], ["attn_perm"], name="tr_out", perm=[0, 1, 3, 2],
    ))
    shape2 = numpy_helper.from_array(
        np.array([1, ch, Hd, Wd], dtype=np.int64), "shape2"
    )
    initializers.append(shape2)
    nodes.append(helper.make_node("Reshape", ["attn_perm", "shape2"], ["attn_feat"], name="reshape2"))

    # Output projection
    proj_w = _make_weight((ch, ch, 1, 1), "proj_W")
    proj_b = _make_bias(ch, "proj_B")
    initializers.extend([proj_w, proj_b])
    nodes.append(_conv_node("proj", "attn_feat", "proj_out", "proj_W", "proj_B", kernel=1, pads=[0, 0, 0, 0]))

    # Upsample back: 2x ConvTranspose
    up1_w = _make_weight((ch, ch, 4, 4), "up1_W")
    up1_b = _make_bias(ch, "up1_B")
    initializers.extend([up1_w, up1_b])
    nodes.append(helper.make_node(
        "ConvTranspose", ["proj_out", "up1_W", "up1_B"], ["u1"],
        name="up1", kernel_shape=[4, 4], strides=[2, 2], pads=[1, 1, 1, 1],
    ))
    nodes.append(helper.make_node("Relu", ["u1"], ["u1r"], name="relu3"))

    up2_w = _make_weight((ch, ch, 4, 4), "up2_W")
    up2_b = _make_bias(ch, "up2_B")
    initializers.extend([up2_w, up2_b])
    nodes.append(helper.make_node(
        "ConvTranspose", ["u1r", "up2_W", "up2_B"], ["output"],
        name="up2", kernel_shape=[4, 4], strides=[2, 2], pads=[1, 1, 1, 1],
    ))

    graph = helper.make_graph(nodes, "self_attention", [inp], [out], initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_iter_accum_3(ch: int, H: int, W: int) -> onnx.ModelProto:
    """3-stage iterative accumulation: accum = accum + conv(cat(input, accum))."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ch, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])

    # Initial accumulator = zeros (as a constant)
    zeros = numpy_helper.from_array(
        np.zeros((1, ch, H, W), dtype=np.float32), "zeros"
    )

    nodes = []
    initializers = [zeros]

    for stage in range(3):
        w = _make_weight((ch, ch * 2, 3, 3), f"W{stage}")
        b = _make_bias(ch, f"B{stage}")
        initializers.extend([w, b])

        accum_in = "zeros" if stage == 0 else f"accum{stage - 1}"
        cat_out = f"cat{stage}"
        conv_out = f"delta{stage}"
        accum_out = f"accum{stage}"

        nodes.append(helper.make_node(
            "Concat", ["input", accum_in], [cat_out],
            name=f"concat{stage}", axis=1,
        ))
        nodes.append(_conv_node(
            f"conv{stage}", cat_out, conv_out, f"W{stage}", f"B{stage}",
        ))
        nodes.append(helper.make_node(
            "Add", [accum_in, conv_out], [accum_out], name=f"add{stage}",
        ))

    # Rename final accumulator to output
    nodes.append(helper.make_node("Identity", ["accum2"], ["output"], name="identity_out"))

    graph = helper.make_graph(nodes, "iter_accum_3", [inp], [out], initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def _gen_warp_chain(ch: int, H: int, W: int) -> onnx.ModelProto:
    """Conv -> GridSample -> Conv -> GridSample."""
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, ch, H, W])
    grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, H, W, 2])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, H, W])

    w1 = _make_weight((ch, ch, 3, 3), "W1")
    b1 = _make_bias(ch, "B1")
    w2 = _make_weight((ch, ch, 3, 3), "W2")
    b2 = _make_bias(ch, "B2")

    conv1 = _conv_node("conv1", "image", "c1", "W1", "B1")
    gs1 = helper.make_node(
        "GridSample", ["c1", "grid"], ["g1"],
        name="gs1", mode="bilinear", padding_mode="zeros", align_corners=1,
    )
    conv2 = _conv_node("conv2", "g1", "c2", "W2", "B2")
    gs2 = helper.make_node(
        "GridSample", ["c2", "grid"], ["output"],
        name="gs2", mode="bilinear", padding_mode="zeros", align_corners=1,
    )

    graph = helper.make_graph(
        [conv1, gs1, conv2, gs2], "warp_chain", [image, grid], [out],
        initializer=[w1, b1, w2, b2],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


# Map operator names to generator functions
GENERATORS: dict[str, callable] = {
    "conv3x3": _gen_conv3x3,
    "conv1x1": _gen_conv1x1,
    "conv_relu": _gen_conv_relu,
    "convtranspose": _gen_convtranspose,
    "add_residual": _gen_add_residual,
    "grid_sample": _gen_grid_sample,
    "resize_2x": _gen_resize_2x,
    "conv_sigmoid": _gen_conv_sigmoid,
    "conv_prelu": _gen_conv_prelu,
    "conv_leakyrelu": _gen_conv_leakyrelu,
    "conv_layernorm": _gen_conv_layernorm,
    "conv_gelu": _gen_conv_gelu,
    "dwconv3x3": _gen_dwconv3x3,
    "self_attention": _gen_self_attention,
    # deformable_conv: ONNX standard does not support this op
    "iter_accum_3": _gen_iter_accum_3,
    "warp_chain": _gen_warp_chain,
}


# ---------------------------------------------------------------------------
# Step 1: Generate ONNX models
# ---------------------------------------------------------------------------

def generate_onnx_models(
    onnx_dir: Path,
    operators: list[str],
    ch: int,
    H: int,
    W: int,
    resolution_tag: str,
) -> dict[str, str]:
    """Generate ONNX models for each operator. Returns status dict."""
    onnx_dir.mkdir(parents=True, exist_ok=True)
    status = {}

    for op_name in operators:
        onnx_path = onnx_dir / f"{op_name}_{resolution_tag}.onnx"

        if op_name == "deformable_conv":
            print(f"  [{op_name}] SKIP: ONNX standard does not support deformable convolution")
            status[op_name] = "onnx_unsupported"
            continue

        if onnx_path.exists():
            print(f"  [{op_name}] ONNX exists, skip: {onnx_path.name}")
            status[op_name] = "ok"
            continue

        if op_name not in GENERATORS:
            print(f"  [{op_name}] SKIP: no ONNX generator defined")
            status[op_name] = "no_generator"
            continue

        try:
            model = GENERATORS[op_name](ch, H, W)
            if model is None:
                status[op_name] = "generator_returned_none"
                continue
            onnx.checker.check_model(model)
            onnx.save(model, str(onnx_path))
            size_kb = onnx_path.stat().st_size / 1024
            print(f"  [{op_name}] OK ({size_kb:.0f} KB): {onnx_path.name}")
            status[op_name] = "ok"
        except Exception as e:
            print(f"  [{op_name}] FAILED: {e}")
            status[op_name] = f"onnx_error: {e}"

    return status


# ---------------------------------------------------------------------------
# Step 2: Generate calibration data (for future INT8 use)
# ---------------------------------------------------------------------------

def generate_calibration_data(
    calib_dir: Path,
    operators: list[str],
    ch: int,
    H: int,
    W: int,
    n_samples: int = 5,
) -> None:
    """Generate random NCHW calibration data for QNN converter."""
    rng = np.random.default_rng(42)

    for op_name in operators:
        if op_name == "deformable_conv":
            continue

        op_calib = calib_dir / op_name
        op_calib.mkdir(parents=True, exist_ok=True)

        has_grid = op_name in GRID_INPUT_OPS
        in_ch = ch

        lines = []
        for i in range(n_samples):
            data = (rng.standard_normal((1, in_ch, H, W)) * 0.5).astype(np.float32)
            raw_path = op_calib / f"input_{i:03d}.raw"
            data.tofile(str(raw_path))

            if has_grid:
                # Identity grid with small perturbation
                xs = np.linspace(-1, 1, W, dtype=np.float32)
                ys = np.linspace(-1, 1, H, dtype=np.float32)
                gx, gy = np.meshgrid(xs, ys)
                grid = np.stack([gx, gy], axis=-1)[np.newaxis]
                grid += rng.normal(0, 0.02, size=grid.shape).astype(np.float32)
                grid = np.clip(grid, -1, 1)
                grid_path = op_calib / f"grid_{i:03d}.raw"
                grid.tofile(str(grid_path))
                lines.append(f"{raw_path.name} {grid_path.name}")
            else:
                lines.append(raw_path.name)

        (op_calib / "input_list.txt").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Step 3: QNN Conversion
# ---------------------------------------------------------------------------

def ensure_docker_image() -> None:
    result = run(f"docker image inspect {DOCKER_IMAGE}", check=False, capture=True)
    if result.returncode == 0:
        return
    print("  Building Docker image ...")
    dockerfile = """\
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3.10 python3.10-dev python3-pip libc++-dev libc++abi-dev \\
    libatomic1 && \\
    pip3 install "numpy<2" pyyaml packaging pandas onnx==1.16.2 && \\
    apt-get clean && rm -rf /var/lib/apt/lists/*
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".Dockerfile", delete=False) as f:
        f.write(dockerfile)
        f.flush()
        run(f"docker build --network=host -t {DOCKER_IMAGE} -f {f.name} .")
        os.unlink(f.name)


def qnn_convert_operator(
    op_name: str,
    resolution_tag: str,
    onnx_dir: Path,
    qnn_dir: Path,
    calib_dir: Path,
    qairt_root: str,
    phone_dir: str,
    precision: str,
    dsp_arch: str,
    soc_id: int,
) -> str:
    """Convert a single operator ONNX to QNN context binary.

    Returns status string: "ok", "skip", or failure description.
    """
    graph_name = f"{op_name}_{resolution_tag}"
    onnx_file = f"{graph_name}.onnx"
    qnn_subdir = f"{graph_name}_{precision}"
    out = qnn_dir / qnn_subdir
    ctx_file = out / f"{graph_name}.serialized.bin"

    if not (onnx_dir / onnx_file).exists():
        return "no_onnx"

    if ctx_file.exists():
        return "ok"

    out.mkdir(parents=True, exist_ok=True)

    # Write HTP configs
    backend_cfg = {
        "graphs": [{"graph_names": [graph_name], "vtcm_mb": 0, "O": 3}],
        "devices": [{"dsp_arch": dsp_arch, "soc_id": soc_id,
                      "pd_session": "unsigned", "device_id": 0}],
    }
    (out / "htp_backend_config.json").write_text(json.dumps(backend_cfg, indent=2))
    convert_cfg = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": "/out/htp_backend_config.json",
        }
    }
    (out / "htp_config_convert.json").write_text(json.dumps(convert_cfg, indent=2))
    device_cfg = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": f"{phone_dir}/qnn/{qnn_subdir}/htp_backend_config.json",
        }
    }
    (out / "htp_config.json").write_text(json.dumps(device_cfg, indent=2))

    onnx_abs = onnx_dir.resolve()
    out_abs = out.resolve()

    docker_base = (
        f"docker run --rm --user $(id -u):$(id -g) -w /out "
        f"-v {qairt_root}:{CONTAINER_QAIRT}:ro "
        f"-v {onnx_abs}:/onnx:ro "
        f"-v {out_abs}:/out"
    )
    docker_root = (
        f"docker run --rm -w /out "
        f"-v {qairt_root}:{CONTAINER_QAIRT}:ro "
        f"-v {onnx_abs}:/onnx:ro "
        f"-v {out_abs}:/out"
    )

    converter_extra = ""
    if precision == "int8":
        op_calib = calib_dir / op_name
        if not (op_calib / "input_list.txt").exists():
            return "no_calibration_data"
        calib_abs = op_calib.resolve()
        docker_base += f" -v {calib_abs}:/calib:ro"
        docker_root += f" -v {calib_abs}:/calib:ro"

        # Build Docker-aware input list
        with open(op_calib / "input_list.txt") as f:
            lines = f.read().strip().splitlines()
        docker_lines = []
        for line in lines:
            tokens = line.strip().split()
            docker_lines.append(" ".join(f"/calib/{t}" for t in tokens))
        (out / "input_list_docker.txt").write_text("\n".join(docker_lines) + "\n")

        input_name = "image" if op_name in GRID_INPUT_OPS else "input"
        converter_extra = (
            f"--input_list /out/input_list_docker.txt "
            f"--act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 32 "
            f"--act_quantizer_calibration percentile --percentile_calibration_value 99.99 "
            f"--act_quantizer_schema asymmetric --param_quantizer_schema symmetric "
            f"--use_per_channel_quantization "
            f"--input_layout {input_name} NCHW"
        )
    else:
        converter_extra = "--float_bitwidth 16"

    docker_base += f" {DOCKER_IMAGE}"
    docker_root += f" {DOCKER_IMAGE}"

    # Step 1: ONNX -> QNN
    result = run(
        f'{docker_base} bash -c "'
        f"export PYTHONPATH={CONTAINER_QAIRT}/lib/python:\\$PYTHONPATH; "
        f"export LD_LIBRARY_PATH={CONTAINER_QAIRT}/lib/x86_64-linux-clang:\\$LD_LIBRARY_PATH; "
        f"python3 {CONTAINER_QAIRT}/bin/x86_64-linux-clang/qnn-onnx-converter "
        f"  --input_network /onnx/{onnx_file} "
        f"  --output_path /out/{graph_name} "
        f'  {converter_extra}"',
        check=False, capture=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "")[-500:]
        return f"converter_failed: {err}"

    if (out / graph_name).exists() and not (out / f"{graph_name}.cpp").exists():
        (out / graph_name).rename(out / f"{graph_name}.cpp")

    # Step 2: QNN -> .so
    result = run(
        f'{docker_root} bash -c "'
        f"export PYTHONPATH={CONTAINER_QAIRT}/lib/python:\\$PYTHONPATH; "
        f"export LD_LIBRARY_PATH={CONTAINER_QAIRT}/lib/x86_64-linux-clang:\\$LD_LIBRARY_PATH; "
        f"export PATH={CONTAINER_QAIRT}/bin/x86_64-linux-clang:\\$PATH; "
        f"export TMPDIR=/out; "
        f"python3 {CONTAINER_QAIRT}/bin/x86_64-linux-clang/qnn-model-lib-generator "
        f"  -c /out/{graph_name}.cpp -b /out/{graph_name}.bin "
        f'  -o /out/ -t x86_64-linux-clang"',
        check=False, capture=True,
    )
    if result.returncode != 0:
        return "lib_generator_failed"
    run(f'{docker_root} chown -R "$(id -u):$(id -g)" /out/', check=False, capture=True)

    # Step 3: .so -> context binary
    result = run(
        f'{docker_base} bash -c "'
        f"export LD_LIBRARY_PATH={CONTAINER_QAIRT}/lib/x86_64-linux-clang:\\$LD_LIBRARY_PATH; "
        f"export PATH={CONTAINER_QAIRT}/bin/x86_64-linux-clang:\\$PATH; "
        f"qnn-context-binary-generator "
        f"  --model /out/x86_64-linux-clang/lib{graph_name}.so "
        f"  --backend {CONTAINER_QAIRT}/lib/x86_64-linux-clang/libQnnHtp.so "
        f"  --binary_file {graph_name}.serialized --output_dir /out/ "
        f'  --config_file /out/htp_config_convert.json"',
        check=False, capture=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "")[-300:]
        return f"context_binary_failed: {err}"

    if ctx_file.exists():
        return "ok"
    return "context_binary_missing"


# ---------------------------------------------------------------------------
# Step 4: Device push + benchmark
# ---------------------------------------------------------------------------

def push_runtime(qairt_root: str, phone_dir: str, htp_version: str, dsp_arch: str) -> None:
    """Push QNN runtime libraries to device."""
    run(f'adb shell "mkdir -p {phone_dir}/qnn"', check=False, capture=True)
    lib_android = f"{qairt_root}/lib/aarch64-android"
    for lib in ["libQnnHtp.so", f"libQnnHtp{htp_version}Stub.so", "libQnnSystem.so",
                "libQnnHtpPrepare.so", "libQnnHtpNetRunExtensions.so"]:
        path = f"{lib_android}/{lib}"
        if os.path.isfile(path):
            run(f'adb push "{path}" "{phone_dir}/"', check=False, capture=True)

    skel = f"{qairt_root}/lib/hexagon-{dsp_arch}/unsigned/libQnnHtp{htp_version}Skel.so"
    if os.path.isfile(skel):
        run(f'adb push "{skel}" "{phone_dir}/"', check=False, capture=True)

    qnn_bin = f"{qairt_root}/bin/aarch64-android/qnn-net-run"
    if os.path.isfile(qnn_bin):
        run(f'adb push "{qnn_bin}" "{phone_dir}/"', check=False, capture=True)
        run(f'adb shell "chmod +x {phone_dir}/qnn-net-run"', check=False)


def benchmark_operator(
    op_name: str,
    resolution_tag: str,
    qnn_dir: Path,
    phone_dir: str,
    precision: str,
    iterations: int,
    perf_profile: str,
    qairt_root: str,
) -> dict:
    """Benchmark a single operator on device.

    Returns dict with keys: latency_ms (float or None), status (str).
    """
    graph_name = f"{op_name}_{resolution_tag}"
    qnn_subdir = f"{graph_name}_{precision}"
    local_dir = qnn_dir / qnn_subdir
    ctx_file = local_dir / f"{graph_name}.serialized.bin"

    if not ctx_file.exists():
        return {"latency_ms": None, "status": "no_context_binary"}

    device_subdir = f"{phone_dir}/qnn/{qnn_subdir}"
    run(f'adb shell "mkdir -p {device_subdir}"', check=False, capture=True)

    # Push context binary and configs
    run(f'adb push "{ctx_file}" "{device_subdir}/"', check=False, capture=True)
    for cfg in local_dir.glob("htp_*.json"):
        run(f'adb push "{cfg}" "{device_subdir}/"', check=False, capture=True)

    # Clean old output
    run(f'adb shell "rm -rf {device_subdir}/output"', check=False, capture=True)

    # Run benchmark (no input_list = qnn-net-run generates random input)
    qnn_cmd = (
        f"cd {device_subdir} && "
        f"LD_LIBRARY_PATH={phone_dir} ADSP_LIBRARY_PATH={phone_dir} "
        f"{phone_dir}/qnn-net-run "
        f"  --retrieve_context {graph_name}.serialized.bin "
        f"  --backend libQnnHtp.so "
        f"  --perf_profile {perf_profile} "
        f"  --num_inferences {iterations} "
        f"  --profiling_level basic "
        f"  --output_dir output "
        f"  --config_file htp_config.json"
    )

    result = run(f'adb shell "{qnn_cmd}"', check=False, capture=True)
    if result.returncode != 0:
        stderr = (result.stderr or "")[-300:]
        if "TCM" in stderr or "memory" in stderr.lower():
            return {"latency_ms": None, "status": "OOM"}
        return {"latency_ms": None, "status": f"run_failed: {stderr}"}

    # Pull and parse profiling log
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
        prof_local = tmp.name

    pull_result = run(
        f'adb pull "{device_subdir}/output/qnn-profiling-data_0.log" "{prof_local}"',
        check=False, capture=True,
    )
    if pull_result.returncode != 0:
        os.unlink(prof_local)
        return {"latency_ms": None, "status": "no_profiling_log"}

    # Parse via qnn-profile-viewer
    viewer_result = run(
        f"docker run --rm "
        f"-v {qairt_root}:{CONTAINER_QAIRT}:ro "
        f"-v \"{prof_local}:/data/profiling.log:ro\" "
        f"{DOCKER_IMAGE} bash -c \""
        f"export LD_LIBRARY_PATH={CONTAINER_QAIRT}/lib/x86_64-linux-clang:\\$LD_LIBRARY_PATH; "
        f"{CONTAINER_QAIRT}/bin/x86_64-linux-clang/qnn-profile-viewer "
        f'  --input_log /data/profiling.log"',
        check=False, capture=True,
    )
    os.unlink(prof_local)

    output = viewer_result.stdout or ""

    # Extract NetRun average (microseconds)
    match = re.search(
        r"Execute Stats \(Average\).*?NetRun[^:]*:\s*(\d+)\s*us",
        output, re.DOTALL,
    )
    if match:
        avg_us = int(match.group(1))
        return {"latency_ms": avg_us / 1000.0, "status": "ok"}

    # Extract min as fallback
    match = re.search(
        r"Execute Stats \(Min\).*?NetRun[^:]*:\s*(\d+)\s*us",
        output, re.DOTALL,
    )
    if match:
        min_us = int(match.group(1))
        return {"latency_ms": min_us / 1000.0, "status": "ok_min_only"}

    return {"latency_ms": None, "status": "parse_failed"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Operator-level HTP latency benchmark for 17 VFI operators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--qairt-root", type=str,
        default="/opt/qcom/aistack/qairt/2.42.0.251225",
        help="Path to QAIRT SDK on host.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/bench/operators"),
        help="Directory for ONNX models, QNN artifacts, and output CSV.",
    )
    parser.add_argument(
        "--phone-root", type=str, default="/data/local/tmp/qnn_op_bench",
        help="Device path for QNN runtime and artifacts.",
    )
    parser.add_argument(
        "--resolution", type=str, default="1080p",
        choices=list(RESOLUTIONS.keys()),
        help="Spatial resolution for operator benchmarks.",
    )
    parser.add_argument(
        "--channels", type=int, default=32,
        help="Number of channels for operator models.",
    )
    parser.add_argument(
        "--precision", type=str, default="fp16",
        choices=["fp16", "int8"],
        help="Precision mode for benchmarking.",
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of inference iterations.",
    )
    parser.add_argument(
        "--perf-profile", type=str, default="burst",
        choices=["burst", "sustained", "balanced", "power_saver"],
        help="HTP performance profile.",
    )
    parser.add_argument(
        "--operators", type=str, default="all",
        help="Comma-separated operator names or 'all'.",
    )
    parser.add_argument(
        "--dsp-arch", type=str, default=DEFAULT_DSP_ARCH,
        help="Hexagon DSP architecture.",
    )
    parser.add_argument(
        "--soc-id", type=int, default=DEFAULT_SOC_ID,
        help="SoC ID for HTP backend config.",
    )
    parser.add_argument(
        "--htp-version", type=str, default=DEFAULT_HTP_VERSION,
        help="HTP version string.",
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip ONNX model generation.",
    )
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="Skip QNN conversion.",
    )
    parser.add_argument(
        "--skip-push", action="store_true",
        help="Skip pushing runtime to device.",
    )
    args = parser.parse_args()

    H, W = RESOLUTIONS[args.resolution]
    ch = args.channels
    resolution_tag = args.resolution

    # Determine operator list
    if args.operators == "all":
        ops = list(OPERATORS.keys())
    else:
        ops = [o.strip() for o in args.operators.split(",")]
        unknown = [o for o in ops if o not in OPERATORS]
        if unknown:
            print(f"ERROR: Unknown operators: {unknown}", file=sys.stderr)
            print(f"  Valid: {list(OPERATORS.keys())}", file=sys.stderr)
            sys.exit(1)

    print("=" * 60)
    print("ANVIL Operator-Level HTP Benchmark")
    print("=" * 60)
    print(f"Resolution: {resolution_tag} ({H}x{W})")
    print(f"Channels: {ch}")
    print(f"Precision: {args.precision}")
    print(f"Operators: {len(ops)}")
    print(f"Iterations: {args.iterations}")
    print(f"Perf profile: {args.perf_profile}")
    print()

    onnx_dir = args.output_dir / "onnx"
    qnn_dir = args.output_dir / "qnn"
    calib_dir = args.output_dir / "calibration"

    # Step 1: Generate ONNX models
    if not args.skip_generate:
        print("=== Step 1: Generate ONNX Models ===")
        gen_status = generate_onnx_models(onnx_dir, ops, ch, H, W, resolution_tag)
        # Also generate calibration data for INT8
        if args.precision == "int8":
            print("\n  Generating calibration data ...")
            generate_calibration_data(calib_dir, ops, ch, H, W)
        print()
    else:
        print("=== Step 1: SKIP (--skip-generate) ===")
        gen_status = {}

    # Step 2: QNN Conversion
    convert_status: dict[str, str] = {}
    if not args.skip_convert:
        print("=== Step 2: QNN Conversion ===")
        ensure_docker_image()
        for op_name in ops:
            print(f"  [{op_name}/{args.precision}] Converting ...")
            status = qnn_convert_operator(
                op_name, resolution_tag, onnx_dir, qnn_dir, calib_dir,
                args.qairt_root, args.phone_root, args.precision,
                args.dsp_arch, args.soc_id,
            )
            convert_status[op_name] = status
            marker = "OK" if status == "ok" else status.upper()
            print(f"    -> {marker}")
        print()
    else:
        print("=== Step 2: SKIP (--skip-convert) ===")

    # Step 3: Push runtime
    if not args.skip_push:
        print("=== Step 3: Push QNN Runtime ===")
        push_runtime(args.qairt_root, args.phone_root, args.htp_version, args.dsp_arch)
        print()
    else:
        print("=== Step 3: SKIP (--skip-push) ===")

    # Step 4: Benchmark
    print("=== Step 4: Benchmark ===")
    results: list[dict] = []

    header = f"  {'Grp':<4} {'Operator':<20} {'Freq':<6} {'Latency(ms)':>12} {'Status':<20}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for op_name in ops:
        op_info = OPERATORS[op_name]

        # Check if this operator was skipped at earlier stages
        if gen_status.get(op_name, "ok") not in ("ok",):
            status = gen_status[op_name]
            results.append({
                "operator": op_name,
                "group": op_info["group"],
                "freq": op_info["freq"],
                "latency_ms": "",
                "normalized": "",
                "status": status,
            })
            print(f"  {op_info['group']:<4} {op_name:<20} {op_info['freq']:<6} {'N/A':>12} {status}")
            continue

        if convert_status.get(op_name, "ok") not in ("ok",):
            status = convert_status.get(op_name, "not_converted")
            results.append({
                "operator": op_name,
                "group": op_info["group"],
                "freq": op_info["freq"],
                "latency_ms": "",
                "normalized": "",
                "status": status,
            })
            print(f"  {op_info['group']:<4} {op_name:<20} {op_info['freq']:<6} {'N/A':>12} {status}")
            continue

        bench = benchmark_operator(
            op_name, resolution_tag, qnn_dir, args.phone_root,
            args.precision, args.iterations, args.perf_profile,
            args.qairt_root,
        )

        lat_str = f"{bench['latency_ms']:.1f}" if bench["latency_ms"] is not None else "N/A"
        results.append({
            "operator": op_name,
            "group": op_info["group"],
            "freq": op_info["freq"],
            "latency_ms": f"{bench['latency_ms']:.2f}" if bench["latency_ms"] is not None else "",
            "normalized": "",  # filled in below
            "status": bench["status"],
        })
        print(f"  {op_info['group']:<4} {op_name:<20} {op_info['freq']:<6} {lat_str:>12} {bench['status']}")

    # Normalize relative to conv3x3
    conv3x3_lat = None
    for r in results:
        if r["operator"] == "conv3x3" and r["latency_ms"]:
            conv3x3_lat = float(r["latency_ms"])
            break

    if conv3x3_lat and conv3x3_lat > 0:
        for r in results:
            if r["latency_ms"]:
                norm = float(r["latency_ms"]) / conv3x3_lat
                r["normalized"] = f"{norm:.1f}x"

    # Step 5: Write CSV
    csv_path = args.output_dir / "operator_latency.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["operator", "group", "freq", "latency_ms", "normalized", "status"],
        )
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Saved to: {csv_path}")
    if conv3x3_lat:
        print(f"Baseline: conv3x3 = {conv3x3_lat:.1f}ms = 1.0x")
    print()

    print(f"  {'Grp':<4} {'Operator':<20} {'Freq':<6} {'ms':>8} {'Norm':>8} {'Status':<20}")
    print("  " + "-" * 70)
    for r in results:
        lat_str = r["latency_ms"] if r["latency_ms"] else "N/A"
        norm_str = r["normalized"] if r["normalized"] else "N/A"
        print(f"  {r['group']:<4} {r['operator']:<20} {r['freq']:<6} {lat_str:>8} {norm_str:>8} {r['status']}")

    # Highlight deployment-blocking operators
    print()
    print("Deployment blockers (failed or > 3x conv3x3 latency):")
    for r in results:
        blocked = False
        reason = ""
        if r["status"] not in ("ok", "ok_min_only"):
            blocked = True
            reason = r["status"]
        elif r["normalized"] and conv3x3_lat:
            norm_val = float(r["latency_ms"]) / conv3x3_lat
            if norm_val > 3.0:
                blocked = True
                reason = f"{norm_val:.1f}x baseline"
        if blocked:
            print(f"  {r['operator']:<20} [{r['group']}] {r['freq']}  -> {reason}")


if __name__ == "__main__":
    main()
