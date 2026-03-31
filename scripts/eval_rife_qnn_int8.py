#!/usr/bin/env python3
"""Reproduce RIFE QNN HTP INT8 quality collapse (paper Table rife_reduced, QNN Δ column).

End-to-end pipeline: ONNX → QNN INT8 context binary → on-device inference → quality eval.

Prerequisites:
  - Qualcomm AI Engine Direct SDK (QAIRT) installed, with qnn-onnx-converter and
    qnn-context-binary-generator on PATH (or set --qairt-root).
  - Android device connected via ADB with qnn-net-run and HTP libs pushed to device
    (see --phone-root).
  - RIFE ONNX models in --onnx-dir (same as eval_rife_reduced_res.py).
  - Xiph 1080p dataset in --xiph-dir.

The script performs:
  1. QNN offline compilation: ONNX → QNN model → INT8 context binary (host-side)
  2. Input preparation: Xiph frames → downsampled NHWC raw files
  3. Device inference: adb push inputs → qnn-net-run → adb pull outputs
  4. Output parsing: raw → float32 (dequantize if needed) → postprocess → quality metrics

Usage:
    pixi run python scripts/eval_rife_qnn_int8.py \\
        --onnx-dir artifacts/onnx/rife \\
        --xiph-dir data/xiph_1080p \\
        --qairt-root /opt/qcom/aistack/qairt/2.42.0.251225 \\
        --output-dir artifacts/eval/rife_qnn_int8

    # Quick subset:
    pixi run python scripts/eval_rife_qnn_int8.py \\
        --onnx-dir artifacts/onnx/rife \\
        --xiph-dir data/xiph_1080p \\
        --qairt-root /opt/qcom/aistack/qairt/2.42.0.251225 \\
        --limit 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESOLUTIONS = {
    "360p": (384, 640),
    "480p": (512, 864),
}

# Sequence-level calibration/evaluation split for Xiph 1080p.
# Calibration sequences chosen to cover diverse motion (small + dense)
# while keeping paper-featured sequences (tractor, old_town_cross, etc.) in eval.
CALIB_SEQUENCES = {"sunflower", "pedestrian_area"}

SEED = 42
N_CALIB = 100  # QNN calibration: 100 stratified samples (from Vimeo train when --vimeo-dir provided)
CHUNK_SIZE = 20  # Samples per qnn-net-run invocation


# ---------------------------------------------------------------------------
# Shell / ADB helpers
# ---------------------------------------------------------------------------

def run_cmd(
    cmd: list[str], *, check: bool = True, capture_output: bool = True,
    env: dict | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True, env=env)


def adb_shell(command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_cmd(["adb", "shell", command], check=check)


def adb_push(src: Path, dst: str) -> None:
    run_cmd(["adb", "push", str(src), dst])


def adb_pull(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["adb", "pull", src, str(dst)])


def ensure_adb_device() -> None:
    proc = run_cmd(["adb", "devices"], check=True)
    lines = [l for l in proc.stdout.strip().split("\n") if l.endswith("\tdevice")]
    if not lines:
        print("ERROR: No ADB device connected. Connect a Snapdragon device and retry.")
        sys.exit(1)
    print(f"  ADB device: {lines[0].split()[0]}")


# ---------------------------------------------------------------------------
# QNN compilation (host-side)
# ---------------------------------------------------------------------------

def find_qairt_tool(qairt_root: Path, tool_name: str) -> Path:
    """Find a QAIRT tool binary."""
    candidates = [
        qairt_root / "bin" / "x86_64-linux-clang" / tool_name,
        qairt_root / "bin" / tool_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: assume on PATH
    return Path(tool_name)


def compile_qnn_int8_context(
    onnx_path: Path,
    output_dir: Path,
    qairt_root: Path,
    calib_dir: Path,
    input_layout: str = "NCHW",
    htp_soc: str = "sm8650",
    use_docker: bool = False,
    docker_image: str = "qairt-converter",
) -> Path:
    """ONNX → QNN model.cpp → INT8 context binary.

    Returns path to the compiled context binary directory (containing .serialized.bin,
    *_net.json, and config files).
    """
    ctx_dir = output_dir / f"{onnx_path.stem}_int8"
    ctx_bin = ctx_dir / f"{onnx_path.stem}.serialized.bin"
    if ctx_bin.exists():
        print(f"    Context binary cached: {ctx_bin.name}")
        return ctx_dir

    ctx_dir.mkdir(parents=True, exist_ok=True)
    converter = find_qairt_tool(qairt_root, "qnn-onnx-converter")
    ctx_gen = find_qairt_tool(qairt_root, "qnn-context-binary-generator")

    # Build calibration input list
    raws = sorted(calib_dir.glob("*.raw"))
    if not raws:
        raise FileNotFoundError(f"No calibration .raw files in {calib_dir}")

    if use_docker:
        # QAIRT converter requires Python 3.10 + libPyIrGraph native libs.
        # Use Docker container (same approach as prepare_int8_analysis.sh).
        _compile_via_docker(
            onnx_path, ctx_dir, ctx_bin, qairt_root, calib_dir, raws,
            input_layout, htp_soc, docker_image,
        )
    else:
        _compile_native(
            onnx_path, ctx_dir, ctx_bin, qairt_root, calib_dir, raws,
            input_layout, htp_soc, converter, ctx_gen,
        )

    # Write backend config for device-side execution
    _write_device_configs(ctx_dir)

    print(f"    Context binary ready: {ctx_bin.name}")
    return ctx_dir


def _write_htp_compile_config(ctx_dir: Path, soc: str) -> Path:
    """Write HTP compilation config JSON."""
    config = {
        "graphs": {
            "0": {
                "graph_names": ["*"],
                "vtcm_mb": 4,
                "O": 3,
            }
        },
        "devices": [{"soc_model": soc}],
    }
    p = ctx_dir / "compile_config.json"
    p.write_text(json.dumps(config, indent=2))
    return p


def _compile_via_docker(
    onnx_path: Path, ctx_dir: Path, ctx_bin: Path,
    qairt_root: Path, calib_dir: Path, raws: list[Path],
    input_layout: str, htp_soc: str, docker_image: str,
) -> None:
    """Compile QNN INT8 context binary inside a Docker container.

    Three-step pipeline (matches prepare_int8_analysis.sh):
      1. qnn-onnx-converter  → model.cpp + model.bin
      2. qnn-model-lib-generator  → libmodel.so
      3. qnn-context-binary-generator  → *.serialized.bin
    """
    container_qairt = "/opt/qcom/aistack/qairt/2.42.0.251225"
    graph_name = "model"
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    soc_map = {"sm8650": ("v75", 57), "sm8550": ("v73", 43), "sm7475": ("v69", 41)}
    dsp_arch, soc_id = soc_map.get(htp_soc, ("v75", 57))

    # Copy ONNX to ctx_dir for Docker mount
    onnx_in_ctx = ctx_dir / onnx_path.name
    if not onnx_in_ctx.exists():
        shutil.copy2(onnx_path, onnx_in_ctx)

    # Calibration input list (container paths)
    (ctx_dir / "calib_input_list_docker.txt").write_text(
        "\n".join(f"/calib/{r.name}" for r in raws) + "\n"
    )

    # HTP config files
    (ctx_dir / "htp_backend_config.json").write_text(json.dumps({
        "graphs": [{"graph_names": [graph_name], "vtcm_mb": 0, "O": 3}],
        "devices": [{"dsp_arch": dsp_arch, "soc_id": soc_id,
                      "pd_session": "unsigned", "device_id": 0}],
    }, indent=2))
    (ctx_dir / "htp_config_convert.json").write_text(json.dumps({
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": "/out/htp_backend_config.json",
        }
    }, indent=2))

    base_docker = [
        "docker", "run", "--rm", "-w", "/out",
        "-v", f"{ctx_dir.resolve()}:/out",
        "-v", f"{calib_dir.resolve()}:/calib:ro",
        "-v", f"{qairt_root.resolve()}:{container_qairt}:ro",
    ]
    docker_user = base_docker + ["--user", uid_gid, docker_image, "bash", "-c"]
    docker_root = base_docker + [docker_image, "bash", "-c"]

    env = (
        f"export PYTHONPATH={container_qairt}/lib/python:$PYTHONPATH && "
        f"export LD_LIBRARY_PATH={container_qairt}/lib/x86_64-linux-clang:${{LD_LIBRARY_PATH:-}} && "
        f"export PATH={container_qairt}/bin/x86_64-linux-clang:$PATH && "
    )

    # Step 1: ONNX → QNN model
    print(f"    [1/3] ONNX → QNN model ({len(raws)} calib samples)...")
    proc = run_cmd(docker_user + [env +
        f"python3 {container_qairt}/bin/x86_64-linux-clang/qnn-onnx-converter "
        f"--input_network /out/{onnx_path.name} "
        f"--output_path /out/{graph_name} "
        f"--input_list /out/calib_input_list_docker.txt "
        f"--input_layout input {input_layout} "
        f"--act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 32 "
        f"--act_quantizer_calibration percentile "
        f"--percentile_calibration_value 99.99 "
        f"--act_quantizer_schema asymmetric "
        f"--param_quantizer_schema symmetric "
        f"--use_per_channel_quantization"
    ], check=False)
    if proc.returncode != 0:
        print(f"    FAILED:\n{(proc.stderr or proc.stdout)[-2000:]}")
        raise RuntimeError("qnn-onnx-converter (Docker) failed")

    # Fix missing .cpp extension (converter sometimes omits it)
    cpp_path = ctx_dir / f"{graph_name}.cpp"
    bare_path = ctx_dir / graph_name
    if bare_path.exists() and not cpp_path.exists():
        bare_path.rename(cpp_path)

    # Step 2: QNN model → .so
    print("    [2/3] Building model .so...")
    proc = run_cmd(docker_root + [env +
        f"export TMPDIR=/out && "
        f"python3 {container_qairt}/bin/x86_64-linux-clang/qnn-model-lib-generator "
        f"-c /out/{graph_name}.cpp -b /out/{graph_name}.bin "
        f"-o /out/ -t x86_64-linux-clang && "
        f"chown -R {uid_gid} /out/"
    ], check=False)
    if proc.returncode != 0:
        print(f"    FAILED:\n{(proc.stderr or proc.stdout)[-2000:]}")
        raise RuntimeError("qnn-model-lib-generator (Docker) failed")

    # Step 3: .so → HTP context binary
    print(f"    [3/3] Generating HTP context binary ({htp_soc})...")
    proc = run_cmd(docker_user + [env +
        f"qnn-context-binary-generator "
        f"--model /out/x86_64-linux-clang/lib{graph_name}.so "
        f"--backend {container_qairt}/lib/x86_64-linux-clang/libQnnHtp.so "
        f"--binary_file {onnx_path.stem}.serialized "
        f"--output_dir /out "
        f"--config_file /out/htp_config_convert.json"
    ], check=False)
    if proc.returncode != 0:
        print(f"    FAILED:\n{(proc.stderr or proc.stdout)[-2000:]}")
        raise RuntimeError("qnn-context-binary-generator (Docker) failed")


def _compile_native(
    onnx_path: Path, ctx_dir: Path, ctx_bin: Path,
    qairt_root: Path, calib_dir: Path, raws: list[Path],
    input_layout: str, htp_soc: str,
    converter: Path, ctx_gen: Path,
) -> None:
    """Compile QNN INT8 context binary natively (requires matching Python)."""
    qairt_env = dict(os.environ)
    qairt_python = str(qairt_root / "lib" / "python")
    existing = qairt_env.get("PYTHONPATH", "")
    qairt_env["PYTHONPATH"] = f"{qairt_python}:{existing}" if existing else qairt_python
    qairt_lib = str(qairt_root / "lib" / "x86_64-linux-clang")
    existing_ld = qairt_env.get("LD_LIBRARY_PATH", "")
    qairt_env["LD_LIBRARY_PATH"] = f"{qairt_lib}:{existing_ld}" if existing_ld else qairt_lib

    model_cpp = ctx_dir / "model.cpp"
    calib_input_list = ctx_dir / "calib_input_list.txt"
    calib_input_list.write_text("\n".join(str(r) for r in raws) + "\n")

    print(f"    Converting ONNX → QNN model ({len(raws)} calibration samples)...")
    convert_cmd = [
        str(converter),
        "--input_network", str(onnx_path),
        "--output_path", str(model_cpp),
        "--input_list", str(calib_input_list),
        "--input_layout", "input", input_layout,
        "--act_bitwidth", "8", "--weights_bitwidth", "8", "--bias_bitwidth", "32",
        "--algorithms", "cle",
        "--percentile_calibration_value", "99.99",
        "--use_per_channel_quantization",
    ]
    proc = run_cmd(convert_cmd, check=False, env=qairt_env)
    if proc.returncode != 0:
        print(f"    Converter failed:\n{proc.stderr[-2000:]}")
        raise RuntimeError("qnn-onnx-converter failed")

    print(f"    Generating HTP context binary for {htp_soc}...")
    lib_dir = qairt_root / "lib" / "x86_64-linux-clang"
    ctx_cmd = [
        str(ctx_gen),
        "--model", str(model_cpp.with_suffix(".bin")),
        "--backend", str(lib_dir / "libQnnHtp.so"),
        "--binary_file", onnx_path.stem,
        "--output_dir", str(ctx_dir),
        "--config_file", str(_write_htp_compile_config(ctx_dir, htp_soc)),
    ]
    proc = run_cmd(ctx_cmd, check=False, env=qairt_env)
    if proc.returncode != 0:
        print(f"    Context generation failed:\n{proc.stderr[-2000:]}")
        raise RuntimeError("qnn-context-binary-generator failed")


def _write_device_configs(ctx_dir: Path) -> None:
    """Write HTP backend + runtime configs for qnn-net-run."""
    backend_cfg = {
        "graphs": {
            "0": {
                "graph_names": ["*"],
                "vtcm_mb": 4,
                "O": 3,
            }
        },
    }
    (ctx_dir / "htp_backend_config.json").write_text(json.dumps(backend_cfg, indent=2))

    # Runtime config pointing to backend config (path will be rewritten for device)
    runtime_cfg = {
        "backend_extensions": {
            "config_file_path": "htp_backend_config.json",
            "shared_library_path": "",
        }
    }
    (ctx_dir / "htp_config.json").write_text(json.dumps(runtime_cfg, indent=2))


# ---------------------------------------------------------------------------
# Calibration data generation
# ---------------------------------------------------------------------------

def generate_calibration_data(
    xiph_dir: Path,
    target_hw: tuple[int, int],
    output_dir: Path,
    n_samples: int,
    triplets: list[str],
    vimeo_dir: Path | None = None,
) -> Path:
    """Generate NCHW float32 calibration .raw files for QNN converter.

    Note: calibration data uses NCHW layout (matching the ONNX model's native
    layout and ``--input_layout input NCHW``).  The converter handles any
    internal layout transformations.  Inference inputs (for qnn-net-run) use
    NHWC — see ``prepare_rife_nhwc_raw()``.

    When *vimeo_dir* is provided, calibration samples are drawn from Vimeo90K
    training set.  Otherwise uses Xiph triplets from a disjoint split
    (recommended for RIFE — matches 1080p deployment distribution).
    """
    use_vimeo = vimeo_dir is not None
    source_tag = "vimeo" if use_vimeo else "xiph"
    calib_dir = output_dir / f"calib_{target_hw[0]}x{target_hw[1]}_{source_tag}"
    if calib_dir.exists() and len(list(calib_dir.glob("*.raw"))) >= n_samples:
        print(f"    Calibration data cached ({n_samples} samples, source={source_tag})")
        return calib_dir

    calib_dir.mkdir(parents=True, exist_ok=True)
    h, w = target_hw
    rng = random.Random(SEED)

    if use_vimeo:
        # Load Vimeo90K training triplet IDs
        train_list = vimeo_dir / "tri_trainlist.txt"
        if not train_list.exists():
            raise FileNotFoundError(
                f"Vimeo train list not found: {train_list}. "
                "Expected vimeo_dir to contain sequences/ and tri_trainlist.txt."
            )
        with open(train_list) as f:
            vimeo_tids = [line.strip() for line in f if line.strip()]
        selected = rng.sample(vimeo_tids, min(n_samples, len(vimeo_tids)))
        seq_root = vimeo_dir / "sequences"

        for i, tid in enumerate(selected):
            td = seq_root / tid
            img0 = cv2.cvtColor(cv2.imread(str(td / "im1.png")), cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(cv2.imread(str(td / "im3.png")), cv2.COLOR_BGR2RGB)
            lo0 = cv2.resize(img0, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            lo1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            nchw = np.concatenate([lo0, lo1], axis=2).transpose(2, 0, 1)[np.newaxis]
            nchw.tofile(calib_dir / f"calib_{i:04d}.raw")
    else:
        # Use disjoint Xiph triplets for calibration
        seq_root = xiph_dir / "sequences"
        selected = rng.sample(triplets, min(n_samples, len(triplets)))

        for i, tid in enumerate(selected):
            seq, fid = tid.split("/")
            td = seq_root / seq / fid
            img0 = cv2.cvtColor(cv2.imread(str(td / "im1.png")), cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(cv2.imread(str(td / "im3.png")), cv2.COLOR_BGR2RGB)
            lo0 = cv2.resize(img0, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            lo1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            # NCHW layout: (1, 6, H, W) — matches ONNX model input
            nchw = np.concatenate([lo0, lo1], axis=2).transpose(2, 0, 1)[np.newaxis]
            nchw.tofile(calib_dir / f"calib_{i:04d}.raw")

    print(f"    Generated {len(selected)} calibration samples at {h}x{w} NCHW (source={source_tag})")
    return calib_dir


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def discover_triplets(xiph_dir: Path) -> list[str]:
    seq_root = xiph_dir / "sequences"
    triplets = []
    for sd in sorted(seq_root.iterdir()):
        if sd.is_dir():
            for td in sorted(sd.iterdir()):
                if td.is_dir() and (td / "im2.png").exists():
                    triplets.append(f"{sd.name}/{td.name}")
    return triplets


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    return float(10.0 * np.log10(255.0**2 / mse)) if mse > 0 else 100.0


# ---------------------------------------------------------------------------
# Input preparation (NHWC for qnn-net-run)
# ---------------------------------------------------------------------------

def prepare_rife_nhwc_raw(
    img0: np.ndarray, img1: np.ndarray, target_hw: tuple[int, int],
) -> np.ndarray:
    """Downsample and produce NHWC float32 for qnn-net-run."""
    h, w = target_hw
    lo0 = cv2.resize(img0, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    lo1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return np.concatenate([lo0, lo1], axis=2)  # (H, W, 6) NHWC


# ---------------------------------------------------------------------------
# QNN output parsing
# ---------------------------------------------------------------------------

def load_qnn_tensor_specs(net_json_path: Path) -> dict:
    """Parse *_net.json to get all tensor metadata.

    Returns the full tensors dict keyed by tensor name.  The ``type`` field
    is an integer: 0 = app_write (graph input), 3 = native (internal),
    4 = static/app_read.  Actual graph outputs are identified by matching
    the ONNX output tensor names in the dict keys.
    """
    data = json.loads(net_json_path.read_text())
    return data["graph"]["tensors"]


def parse_qnn_raw_output(
    raw_path: Path, tensor_info: dict,
) -> np.ndarray:
    """Read qnn-net-run output .raw file, return NCHW float32.

    qnn-net-run writes dequantized float32 by default (unless
    ``--use_native_output_files``), so we can simply read as float32 and
    apply ``permute_order_to_src`` to convert from the graph's native
    layout (typically NHWC) back to NCHW.
    """
    dims = tuple(int(x) for x in tensor_info["dims"])
    expected = int(np.prod(dims))
    size_bytes = raw_path.stat().st_size

    if size_bytes == expected * 4:
        raw = np.fromfile(str(raw_path), dtype=np.float32)
    elif size_bytes == expected * 2:
        raw = np.fromfile(str(raw_path), dtype=np.float16).astype(np.float32)
    else:
        raise ValueError(
            f"{raw_path}: unexpected size {size_bytes} "
            f"(expected {expected * 4} for float32)"
        )

    raw = raw.reshape(dims)

    # Permute from native layout (NHWC) back to source layout (NCHW)
    perm = tensor_info.get("permute_order_to_src")
    if perm:
        raw = np.transpose(raw, axes=[int(p) for p in perm])

    return raw


# ---------------------------------------------------------------------------
# RIFE postprocessing (same as eval_rife_reduced_res.py)
# ---------------------------------------------------------------------------

def postprocess_flow_up(
    flow_nchw: np.ndarray, mask_nchw: np.ndarray,
    img0_u8: np.ndarray, img1_u8: np.ndarray,
) -> np.ndarray:
    src_h, src_w = flow_nchw.shape[2], flow_nchw.shape[3]
    tgt_h, tgt_w = img0_u8.shape[0], img0_u8.shape[1]
    sy, sx = tgt_h / src_h, tgt_w / src_w
    fl = flow_nchw[0]
    up = lambda a: cv2.resize(a, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
    f01x, f01y = up(fl[0]) * sx, up(fl[1]) * sy
    f10x, f10y = up(fl[2]) * sx, up(fl[3]) * sy
    mask_full = cv2.resize(mask_nchw[0, 0], (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
    gy, gx = np.mgrid[0:tgt_h, 0:tgt_w].astype(np.float32)
    w0 = cv2.remap(img0_u8, gx + f01x.astype(np.float32), gy + f01y.astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    w1 = cv2.remap(img1_u8, gx + f10x.astype(np.float32), gy + f10y.astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = mask_full[:, :, np.newaxis].astype(np.float32)
    return np.clip(m * w0.astype(np.float32) + (1 - m) * w1.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)


def postprocess_frame_up(output_nchw: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    hwc = np.clip(output_nchw[0].transpose(1, 2, 0), 0, 1)
    up = cv2.resize(hwc, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(up * 255 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Device inference for a chunk
# ---------------------------------------------------------------------------

def run_device_chunk(
    *,
    ctx_dir: Path,
    input_raws: list[Path],
    work_dir: Path,
    phone_root: str,
) -> list[Path]:
    """Push inputs, run qnn-net-run on device, pull outputs. Returns output .raw paths."""
    ctx_bin = next(ctx_dir.glob("*.serialized.bin"))

    # Ensure device dir
    device_dir = f"{phone_root}/qnn_eval"
    adb_shell(f"mkdir -p {shlex.quote(device_dir)}", check=True)

    # Push context + configs (only once per ctx_dir)
    marker = work_dir / f".pushed_{ctx_dir.name}"
    if not marker.exists():
        adb_push(ctx_bin, f"{device_dir}/")
        for cfg in ctx_dir.glob("*.json"):
            adb_push(cfg, f"{device_dir}/")
        # Rewrite htp_config.json device path
        local_cfg = work_dir / "htp_config_device.json"
        local_cfg.write_text(json.dumps({
            "backend_extensions": {
                "config_file_path": f"{device_dir}/htp_backend_config.json",
                "shared_library_path": "",
            }
        }))
        adb_push(local_cfg, f"{device_dir}/htp_config.json")
        marker.touch()

    # Push inputs
    chunk_name = f"chunk_{id(input_raws) % 10000:04d}"
    device_inputs = f"{device_dir}/{chunk_name}"
    device_outputs = f"{device_dir}/{chunk_name}_out"
    adb_shell(f"rm -rf {shlex.quote(device_inputs)} {shlex.quote(device_outputs)}")
    adb_shell(f"mkdir -p {shlex.quote(device_inputs)}")

    input_list_lines = []
    for raw_path in input_raws:
        adb_push(raw_path, f"{device_inputs}/")
        input_list_lines.append(f"{chunk_name}/{raw_path.name}")

    input_list_local = work_dir / f"{chunk_name}_input_list.txt"
    input_list_local.write_text("\n".join(input_list_lines) + "\n")
    adb_push(input_list_local, f"{device_dir}/")

    # Run qnn-net-run
    cmd = (
        f"cd {shlex.quote(device_dir)} && "
        f"LD_LIBRARY_PATH={shlex.quote(phone_root)} "
        f"ADSP_LIBRARY_PATH={shlex.quote(phone_root)} "
        f"{shlex.quote(phone_root)}/qnn-net-run "
        f"--retrieve_context {ctx_bin.name} "
        f"--backend libQnnHtp.so "
        f"--input_list {chunk_name}_input_list.txt "
        f"--output_dir {chunk_name}_out "
        f"--config_file htp_config.json"
    )
    proc = adb_shell(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"qnn-net-run failed: {proc.stderr[:1000]}")

    # Pull outputs
    local_outputs = work_dir / f"{chunk_name}_out"
    if local_outputs.exists():
        shutil.rmtree(local_outputs)
    adb_pull(device_outputs, local_outputs)

    # Collect output raws sorted by Result_N directory.
    # qnn-net-run creates Result_0/, Result_1/, ... with tensor_name.raw files.
    result_dirs = sorted(
        [d for d in local_outputs.iterdir() if d.is_dir() and d.name.startswith("Result_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    output_raws = []
    for rd in result_dirs:
        output_raws.append(sorted(rd.glob("*.raw")))

    # Cleanup device
    adb_shell(f"rm -rf {shlex.quote(device_inputs)} {shlex.quote(device_outputs)}")

    return output_raws


# ---------------------------------------------------------------------------
# Per-variant evaluation
# ---------------------------------------------------------------------------

def evaluate_variant_qnn(
    *,
    name: str,
    onnx_path: Path,
    mode: str,
    res_name: str,
    target_hw: tuple[int, int],
    xiph_dir: Path,
    eval_tids: list[str],
    triplets_all: list[str],
    qairt_root: Path,
    output_dir: Path,
    phone_root: str,
    use_docker: bool = False,
    docker_image: str = "qairt-converter",
    vimeo_dir: Path | None = None,
) -> dict:
    """Full QNN INT8 evaluation for one RIFE variant."""
    print(f"\n  [{name}] QNN INT8 — {mode}↑ @ {res_name}")
    seq_root = xiph_dir / "sequences"
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    work_dir = variant_dir / "_work"
    work_dir.mkdir(exist_ok=True)

    if not onnx_path.exists():
        print(f"    ONNX not found: {onnx_path}")
        return {"name": name, "error": "ONNX not found"}

    # Step 1: Generate calibration data
    calib_dir = generate_calibration_data(
        xiph_dir, target_hw, variant_dir, N_CALIB, triplets_all,
        vimeo_dir=vimeo_dir,
    )

    # Step 2: Compile QNN INT8 context binary
    ctx_dir = compile_qnn_int8_context(
        onnx_path, variant_dir, qairt_root, calib_dir,
        use_docker=use_docker, docker_image=docker_image,
    )

    # Load tensor specs for output parsing
    net_json = next(ctx_dir.glob("*_net.json"))
    tensor_specs = load_qnn_tensor_specs(net_json)

    # Identify output tensor names.  For flow mode the ONNX has two outputs
    # ("flow" and "mask"); for frame mode it has one ("output").
    if mode == "flow":
        flow_spec = tensor_specs.get("flow")
        mask_spec = tensor_specs.get("mask")
        if not flow_spec or not mask_spec:
            print(f"    ERROR: Cannot find 'flow'/'mask' tensors in {net_json.name}")
            return {"name": name, "error": "missing output tensor specs"}
    else:
        out_spec = tensor_specs.get("output")
        if not out_spec:
            print(f"    ERROR: Cannot find 'output' tensor in {net_json.name}")
            return {"name": name, "error": "missing output tensor spec"}

    # Step 3: Run inference in chunks
    psnr_values = []

    for chunk_start in range(0, len(eval_tids), CHUNK_SIZE):
        chunk_tids = eval_tids[chunk_start : chunk_start + CHUNK_SIZE]
        input_raws = []
        sample_meta = []

        for i, tid in enumerate(chunk_tids):
            seq, fid = tid.split("/")
            td = seq_root / seq / fid
            try:
                img0 = load_image_rgb(td / "im1.png")
                img1 = load_image_rgb(td / "im3.png")
                gt = load_image_rgb(td / "im2.png")
            except FileNotFoundError:
                continue

            nhwc = prepare_rife_nhwc_raw(img0, img1, target_hw)
            raw_path = work_dir / f"input_{chunk_start + i:04d}.raw"
            nhwc.tofile(raw_path)
            input_raws.append(raw_path)
            sample_meta.append({"tid": tid, "img0": img0, "img1": img1, "gt": gt})

        if not input_raws:
            continue

        # Device inference
        chunk_outputs = run_device_chunk(
            ctx_dir=ctx_dir,
            input_raws=input_raws,
            work_dir=work_dir,
            phone_root=phone_root,
        )

        # Parse outputs — each element of chunk_outputs is a list of .raw
        # files in one Result_N/ directory.  File stems match tensor names.
        for j, (meta, out_raws) in enumerate(zip(sample_meta, chunk_outputs)):
            raw_by_name = {r.stem: r for r in out_raws}

            if mode == "flow":
                flow_raw = raw_by_name.get("flow")
                mask_raw = raw_by_name.get("mask")
                if not flow_raw or not mask_raw:
                    print(f"    WARN: missing flow/mask in {out_raws}")
                    continue
                flow = parse_qnn_raw_output(flow_raw, flow_spec)
                mask = parse_qnn_raw_output(mask_raw, mask_spec)
                pred = postprocess_flow_up(flow, mask, meta["img0"], meta["img1"])
            else:
                out_raw = raw_by_name.get("output")
                if not out_raw:
                    print(f"    WARN: missing output in {out_raws}")
                    continue
                frame = parse_qnn_raw_output(out_raw, out_spec)
                pred = postprocess_frame_up(frame, meta["gt"].shape[:2])

            psnr_values.append(psnr(pred, meta["gt"]))

        print(f"    {min(chunk_start + CHUNK_SIZE, len(eval_tids))}/{len(eval_tids)}", flush=True)

    if not psnr_values:
        return {"name": name, "error": "No valid samples"}

    arr = np.array(psnr_values)
    result = {
        "name": name,
        "resolution": res_name,
        "mode": mode,
        "n_eval": len(arr),
        "qnn_int8_psnr": round(float(arr.mean()), 2),
        "qnn_int8_std": round(float(arr.std()), 2),
    }
    print(f"    QNN INT8: {result['qnn_int8_psnr']:.2f} ± {result['qnn_int8_std']:.2f} dB (n={len(arr)})")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RIFE QNN HTP INT8 quality evaluation (paper Table rife_reduced, QNN column).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--onnx-dir", type=Path, required=True,
                        help="Directory with RIFE ONNX models.")
    parser.add_argument("--xiph-dir", type=Path, required=True,
                        help="Xiph 1080p root (containing sequences/).")
    parser.add_argument("--vimeo-dir", type=Path, default=None,
                        help="Vimeo90K root (contains sequences/ + tri_trainlist.txt). "
                             "When provided, calibration uses Vimeo training set. "
                             "When omitted, calibration uses disjoint Xiph split (recommended for RIFE).")
    parser.add_argument("--qairt-root", type=Path, required=True,
                        help="QAIRT SDK root (e.g. /opt/qcom/aistack/qairt/2.42.0.251225).")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/rife_qnn_int8"))
    parser.add_argument("--phone-root", default="/data/local/tmp/qnn",
                        help="Device-side directory with QNN libs + qnn-net-run.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N eval triplets (0 = use paper protocol: 20 stratified).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--variants", default="all",
                        help="Comma-separated: 360p-flow,360p-frame,480p-flow,480p-frame or 'all'.")
    parser.add_argument("--use-docker", action="store_true",
                        help="Run QNN converter inside Docker (needed if host Python != 3.10).")
    parser.add_argument("--docker-image", default="qairt-converter",
                        help="Docker image name for QNN compilation (default: qairt-converter).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Verify prerequisites
    print("Checking prerequisites...")
    ensure_adb_device()

    qairt_bin = args.qairt_root / "bin"
    if not qairt_bin.exists():
        print(f"ERROR: QAIRT bin not found at {qairt_bin}")
        sys.exit(1)
    print(f"  QAIRT: {args.qairt_root}")

    # Calibration source
    if args.vimeo_dir is not None:
        print(f"  Calibration source: Vimeo90K train ({args.vimeo_dir})")
    else:
        print("  Calibration source: sequence-level disjoint Xiph split.")

    # Discover and sample triplets
    triplets = discover_triplets(args.xiph_dir)
    print(f"  Xiph: {len(triplets)} triplets")

    random.seed(args.seed)
    n_eval = args.limit if args.limit > 0 else 20  # Paper protocol: 20-sample stratified

    if args.vimeo_dir is not None:
        # Vimeo for calibration — all Xiph triplets available for eval
        calib_pool = triplets  # passed to evaluate_variant_qnn but unused when vimeo_dir set
        eval_tids = random.sample(triplets, min(n_eval, len(triplets)))
        print(f"  Eval: {len(eval_tids)} triplets (all sequences), Seed: {args.seed}")
    else:
        # Sequence-level disjoint split: calib and eval share no sequences
        calib_pool = [t for t in triplets if t.split("/")[0] in CALIB_SEQUENCES]
        eval_pool = [t for t in triplets if t.split("/")[0] not in CALIB_SEQUENCES]
        eval_tids = random.sample(eval_pool, min(n_eval, len(eval_pool)))
        calib_seqs_str = ", ".join(sorted(CALIB_SEQUENCES))
        print(
            f"  Sequence-level disjoint split: calib sequences = {{{calib_seqs_str}}} "
            f"({len(calib_pool)} available), eval = {len(eval_tids)} from remaining sequences, "
            f"Seed: {args.seed}"
        )

    # Build variant list
    all_variants = []
    for res_name, hw in RESOLUTIONS.items():
        for mode, suffix in [("flow", f"rife_flow_{res_name}.onnx"),
                              ("frame", f"rife_frame_{res_name}.onnx")]:
            key = f"{res_name}-{mode}"
            all_variants.append({
                "key": key,
                "name": f"RIFE-{res_name}-{mode}-up",
                "onnx_path": args.onnx_dir / suffix,
                "mode": mode,
                "res_name": res_name,
                "target_hw": hw,
            })

    if args.variants != "all":
        selected = set(args.variants.split(","))
        all_variants = [v for v in all_variants if v["key"] in selected]

    # Run evaluations
    print(f"\n{'=' * 70}")
    print("RIFE QNN HTP INT8 Quality Evaluation")
    print(f"{'=' * 70}")

    results = []
    for v in all_variants:
        result = evaluate_variant_qnn(
            name=v["name"],
            onnx_path=v["onnx_path"],
            mode=v["mode"],
            res_name=v["res_name"],
            target_hw=v["target_hw"],
            xiph_dir=args.xiph_dir,
            eval_tids=eval_tids,
            triplets_all=calib_pool,
            qairt_root=args.qairt_root,
            output_dir=args.output_dir,
            phone_root=args.phone_root,
            use_docker=args.use_docker,
            docker_image=args.docker_image,
            vimeo_dir=args.vimeo_dir,
        )
        results.append(result)

    # Save results
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 70}")
    print("QNN HTP INT8 Results (paper Table rife_reduced, QNN Δ column)")
    print(f"{'=' * 70}")
    print(f"{'Res.':<6} {'Mode':<10} {'QNN INT8':>10} {'N':>5}")
    print("-" * 35)
    for r in results:
        if "error" in r:
            print(f"{r.get('resolution', '?'):<6} {r.get('mode', '?'):<10} ERROR")
        else:
            print(f"{r['resolution']:<6} {r['mode'] + '↑':<10} {r['qnn_int8_psnr']:>10.2f} {r['n_eval']:>5}")

    print(f"\nTo compute Δ, compare against FP32 values from eval_rife_reduced_res.py.")
    print(f"Results: {summary_path}")


if __name__ == "__main__":
    main()
