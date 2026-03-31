"""HTP INT8/FP16 latency benchmark for ANVIL and RIFE models.

Reproduces paper Table III (latency) and Table IV (RIFE reduced-resolution INT8
latency column).  End-to-end pipeline:

  1. ONNX -> QNN context binary (host-side, via Docker + QAIRT SDK)
  2. Push context binary + QNN runtime libs to device via ADB
  3. Run qnn-net-run with --num_inferences for latency measurement
  4. Parse qnn-profile-viewer output for avg/min/max latency
  5. Output: latency.csv

Prerequisites:
  - Docker (for QAIRT conversion)
  - QAIRT SDK on host (default /opt/qcom/aistack/qairt/2.42.0.251225)
  - Android device connected via ADB with Hexagon HTP support
  - ONNX models in --onnx-dir (export via export_onnx.sh)
  - For INT8: calibration data in --calib-dir (NCHW .raw files + input_list.txt)

Usage:
    python scripts/bench_htp_latency.py \\
        --onnx-dir artifacts/onnx \\
        --qairt-root /opt/qcom/aistack/qairt/2.42.0.251225 \\
        --output-dir artifacts/bench/latency \\
        --precision both \\
        --models anvil_s,anvil_m

    # INT8 with calibration data:
    python scripts/bench_htp_latency.py \\
        --onnx-dir artifacts/onnx \\
        --qairt-root /opt/qcom/aistack/qairt/2.42.0.251225 \\
        --output-dir artifacts/bench/latency \\
        --precision int8 \\
        --calib-dir artifacts/calibration \\
        --models all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCKER_IMAGE = "qairt-converter"
CONTAINER_QAIRT = "/opt/qairt"

# Default HTP device parameters (SM8650 / V75)
DEFAULT_DSP_ARCH = "v75"
DEFAULT_SOC_ID = 57
DEFAULT_HTP_VERSION = "V75"

# Known models and their expected ONNX filenames
# Keys are user-facing names, values are ONNX stem patterns
KNOWN_MODELS = {
    "anvil_s": "D_unet_v3bs_nomv",
    "anvil_m": "D_unet_v3bm_nomv",
    "rife_360p_flow": "rife_flow_360p",
    "rife_360p_frame": "rife_frame_360p",
    "rife_480p_flow": "rife_flow_480p",
    "rife_480p_frame": "rife_frame_480p",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str, *, check: bool = True, capture: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Run a shell command with consistent settings."""
    return subprocess.run(
        cmd, shell=True, check=check, capture_output=capture,
        text=True, **kwargs,
    )


def adb(cmd: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run an ADB command."""
    return run(f"adb {cmd}", check=check, capture=capture)


def check_adb_device() -> None:
    """Verify an ADB device is connected."""
    result = adb("devices", capture=True)
    lines = [l for l in result.stdout.strip().splitlines()[1:] if l.strip()]
    devices = [l for l in lines if "device" in l.split()[-1:]]
    if not devices:
        print("ERROR: No ADB device connected.", file=sys.stderr)
        sys.exit(1)
    print(f"  ADB device: {devices[0].split()[0]}")


def ensure_docker_image() -> None:
    """Build the Docker image for QAIRT conversion if it doesn't exist."""
    result = run(f"docker image inspect {DOCKER_IMAGE}", check=False, capture=True)
    if result.returncode == 0:
        return
    print(f"  Building Docker image {DOCKER_IMAGE} ...")
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


def discover_onnx_models(onnx_dir: Path, models_filter: list[str] | None) -> list[dict]:
    """Discover ONNX models in the directory.

    Returns a list of dicts with keys: name, onnx_path, stem, resolution.
    """
    found = []
    onnx_files = sorted(onnx_dir.glob("*.onnx"))

    if models_filter and "all" not in models_filter:
        # Match requested models against known aliases and file stems
        requested_stems = set()
        for m in models_filter:
            m_clean = m.strip().lower()
            if m_clean in KNOWN_MODELS:
                requested_stems.add(KNOWN_MODELS[m_clean])
            else:
                # Try direct stem match
                requested_stems.add(m.replace("-", "_"))

        for p in onnx_files:
            stem = p.stem
            # Check if any requested stem is a prefix/match
            matched = any(stem.startswith(rs) or rs == stem for rs in requested_stems)
            if not matched:
                # Also try checking if any requested token appears in the stem
                matched = any(rs in stem for rs in requested_stems)
            if matched:
                found.append(_parse_onnx_info(p))
    else:
        for p in onnx_files:
            found.append(_parse_onnx_info(p))

    return found


def _parse_onnx_info(onnx_path: Path) -> dict:
    """Extract model name and resolution from ONNX filename."""
    stem = onnx_path.stem
    # Try to detect resolution from suffix
    resolution = "unknown"
    for res_tag in ("1080p", "720p", "540p", "480p", "360p", "vimeo"):
        if stem.endswith(f"_{res_tag}"):
            resolution = res_tag
            break
    # Derive a human-readable name
    name = stem
    for res_tag in ("1080p", "720p", "540p", "480p", "360p", "vimeo"):
        name = name.removesuffix(f"_{res_tag}")
    return {
        "name": name,
        "onnx_path": onnx_path,
        "stem": stem,
        "resolution": resolution,
    }


# ---------------------------------------------------------------------------
# QNN Conversion
# ---------------------------------------------------------------------------

def write_htp_configs(
    out_dir: Path,
    graph_name: str,
    phone_dir: str,
    qnn_subdir: str,
    dsp_arch: str,
    soc_id: int,
) -> None:
    """Write HTP backend config JSON files for conversion and device execution."""
    backend_cfg = {
        "graphs": [{"graph_names": [graph_name], "vtcm_mb": 0, "O": 3}],
        "devices": [{"dsp_arch": dsp_arch, "soc_id": soc_id,
                      "pd_session": "unsigned", "device_id": 0}],
    }
    (out_dir / "htp_backend_config.json").write_text(json.dumps(backend_cfg, indent=2))

    convert_cfg = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": "/out/htp_backend_config.json",
        }
    }
    (out_dir / "htp_config_convert.json").write_text(json.dumps(convert_cfg, indent=2))

    device_cfg = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": f"{phone_dir}/qnn/{qnn_subdir}/htp_backend_config.json",
        }
    }
    (out_dir / "htp_config.json").write_text(json.dumps(device_cfg, indent=2))


def qnn_convert_fp16(
    model_info: dict,
    qnn_dir: Path,
    onnx_dir: Path,
    qairt_root: str,
    phone_dir: str,
    dsp_arch: str,
    soc_id: int,
) -> Path | None:
    """Convert ONNX to QNN FP16 context binary. Returns context binary path or None."""
    stem = model_info["stem"]
    qnn_subdir = f"{stem}_fp16"
    out = qnn_dir / qnn_subdir
    ctx_file = out / f"{stem}.serialized.bin"

    if ctx_file.exists():
        print(f"    [{stem}/fp16] Context binary exists, skip.")
        return ctx_file

    out.mkdir(parents=True, exist_ok=True)
    write_htp_configs(out, stem, phone_dir, qnn_subdir, dsp_arch, soc_id)

    onnx_abs = model_info["onnx_path"].parent.resolve()
    onnx_file = model_info["onnx_path"].name
    out_abs = out.resolve()

    docker_base = (
        f"docker run --rm --user $(id -u):$(id -g) "
        f"-v {qairt_root}:{CONTAINER_QAIRT}:ro "
        f"-v {onnx_abs}:/onnx:ro "
        f"-v {out_abs}:/out "
        f"{DOCKER_IMAGE}"
    )
    docker_root = (
        f"docker run --rm "
        f"-v {qairt_root}:{CONTAINER_QAIRT}:ro "
        f"-v {onnx_abs}:/onnx:ro "
        f"-v {out_abs}:/out "
        f"{DOCKER_IMAGE}"
    )

    # Step 1: ONNX -> QNN model
    print(f"    [{stem}/fp16] ONNX -> QNN FP16 ...")
    result = run(
        f'{docker_base} bash -c "'
        f"export PYTHONPATH={CONTAINER_QAIRT}/lib/python:\\$PYTHONPATH; "
        f"export LD_LIBRARY_PATH={CONTAINER_QAIRT}/lib/x86_64-linux-clang:\\$LD_LIBRARY_PATH; "
        f"python3 {CONTAINER_QAIRT}/bin/x86_64-linux-clang/qnn-onnx-converter "
        f"  --input_network /onnx/{onnx_file} "
        f"  --output_path /out/{stem} "
        f'  --float_bitwidth 16"',
        check=False, capture=True,
    )
    if result.returncode != 0:
        print(f"    [{stem}/fp16] FAILED at onnx-converter: {result.stderr[-500:]}")
        return None

    # Fix missing .cpp extension
    if (out / stem).exists() and not (out / f"{stem}.cpp").exists():
        (out / stem).rename(out / f"{stem}.cpp")

    # Step 2: QNN -> .so
    print(f"    [{stem}/fp16] QNN -> .so ...")
    result = run(
        f'{docker_root} bash -c "'
        f"export PYTHONPATH={CONTAINER_QAIRT}/lib/python:\\$PYTHONPATH; "
        f"export LD_LIBRARY_PATH={CONTAINER_QAIRT}/lib/x86_64-linux-clang:\\$LD_LIBRARY_PATH; "
        f"export PATH={CONTAINER_QAIRT}/bin/x86_64-linux-clang:\\$PATH; "
        f"export TMPDIR=/out; "
        f"python3 {CONTAINER_QAIRT}/bin/x86_64-linux-clang/qnn-model-lib-generator "
        f"  -c /out/{stem}.cpp -b /out/{stem}.bin "
        f'  -o /out/ -t x86_64-linux-clang"',
        check=False, capture=True,
    )
    if result.returncode != 0:
        print(f"    [{stem}/fp16] FAILED at lib-generator: {result.stderr[-500:]}")
        return None
    run(f'{docker_root} chown -R "$(id -u):$(id -g)" /out/', check=False, capture=True)

    # Step 3: .so -> context binary
    print(f"    [{stem}/fp16] .so -> context binary ...")
    result = run(
        f'{docker_base} bash -c "'
        f"export LD_LIBRARY_PATH={CONTAINER_QAIRT}/lib/x86_64-linux-clang:\\$LD_LIBRARY_PATH; "
        f"export PATH={CONTAINER_QAIRT}/bin/x86_64-linux-clang:\\$PATH; "
        f"qnn-context-binary-generator "
        f"  --model /out/x86_64-linux-clang/lib{stem}.so "
        f"  --backend {CONTAINER_QAIRT}/lib/x86_64-linux-clang/libQnnHtp.so "
        f"  --binary_file {stem}.serialized --output_dir /out/ "
        f'  --config_file /out/htp_config_convert.json"',
        check=False, capture=True,
    )
    if result.returncode != 0:
        print(f"    [{stem}/fp16] FAILED at context-binary-generator: {result.stderr[-500:]}")
        return None

    if ctx_file.exists():
        size_kb = ctx_file.stat().st_size / 1024
        print(f"    [{stem}/fp16] OK ({size_kb:.0f} KB)")
        return ctx_file
    else:
        print(f"    [{stem}/fp16] FAILED: context binary not created")
        return None


def qnn_convert_int8(
    model_info: dict,
    qnn_dir: Path,
    onnx_dir: Path,
    calib_dir: Path,
    qairt_root: str,
    phone_dir: str,
    dsp_arch: str,
    soc_id: int,
) -> Path | None:
    """Convert ONNX to QNN INT8 context binary. Returns context binary path or None."""
    stem = model_info["stem"]
    qnn_subdir = f"{stem}_int8"
    out = qnn_dir / qnn_subdir
    ctx_file = out / f"{stem}.serialized.bin"

    if ctx_file.exists():
        print(f"    [{stem}/int8] Context binary exists, skip.")
        return ctx_file

    if not (calib_dir / "input_list.txt").exists():
        print(f"    [{stem}/int8] SKIP: no calibration data at {calib_dir}/input_list.txt")
        return None

    out.mkdir(parents=True, exist_ok=True)
    write_htp_configs(out, stem, phone_dir, qnn_subdir, dsp_arch, soc_id)

    onnx_abs = model_info["onnx_path"].parent.resolve()
    onnx_file = model_info["onnx_path"].name
    out_abs = out.resolve()
    calib_abs = calib_dir.resolve()

    # Generate Docker-aware input list
    with open(calib_dir / "input_list.txt") as f:
        lines = f.read().strip().splitlines()
    docker_lines = ["/calib/" + l.strip() for l in lines if l.strip()]
    (out / "input_list_docker.txt").write_text("\n".join(docker_lines) + "\n")

    docker_base = (
        f"docker run --rm --user $(id -u):$(id -g) -w /out "
        f"-v {qairt_root}:{CONTAINER_QAIRT}:ro "
        f"-v {onnx_abs}:/onnx:ro "
        f"-v {calib_abs}:/calib:ro "
        f"-v {out_abs}:/out "
        f"{DOCKER_IMAGE}"
    )
    docker_root = (
        f"docker run --rm -w /out "
        f"-v {qairt_root}:{CONTAINER_QAIRT}:ro "
        f"-v {onnx_abs}:/onnx:ro "
        f"-v {calib_abs}:/calib:ro "
        f"-v {out_abs}:/out "
        f"{DOCKER_IMAGE}"
    )

    # Step 1: ONNX -> QNN INT8
    print(f"    [{stem}/int8] ONNX -> QNN INT8 (W8A8) ...")
    result = run(
        f'{docker_base} bash -c "'
        f"export PYTHONPATH={CONTAINER_QAIRT}/lib/python:\\$PYTHONPATH; "
        f"export LD_LIBRARY_PATH={CONTAINER_QAIRT}/lib/x86_64-linux-clang:\\$LD_LIBRARY_PATH; "
        f"python3 {CONTAINER_QAIRT}/bin/x86_64-linux-clang/qnn-onnx-converter "
        f"  --input_network /onnx/{onnx_file} "
        f"  --output_path /out/{stem} "
        f"  --input_list /out/input_list_docker.txt "
        f"  --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 32 "
        f"  --act_quantizer_calibration percentile --percentile_calibration_value 99.99 "
        f"  --act_quantizer_schema asymmetric --param_quantizer_schema symmetric "
        f"  --use_per_channel_quantization "
        f'  --input_layout input NCHW"',
        check=False, capture=True,
    )
    if result.returncode != 0:
        print(f"    [{stem}/int8] FAILED at onnx-converter: {result.stderr[-500:]}")
        return None

    if (out / stem).exists() and not (out / f"{stem}.cpp").exists():
        (out / stem).rename(out / f"{stem}.cpp")

    # Step 2: QNN -> .so
    print(f"    [{stem}/int8] QNN -> .so ...")
    result = run(
        f'{docker_root} bash -c "'
        f"export PYTHONPATH={CONTAINER_QAIRT}/lib/python:\\$PYTHONPATH; "
        f"export LD_LIBRARY_PATH={CONTAINER_QAIRT}/lib/x86_64-linux-clang:\\$LD_LIBRARY_PATH; "
        f"export PATH={CONTAINER_QAIRT}/bin/x86_64-linux-clang:\\$PATH; "
        f"export TMPDIR=/out; "
        f"python3 {CONTAINER_QAIRT}/bin/x86_64-linux-clang/qnn-model-lib-generator "
        f"  -c /out/{stem}.cpp -b /out/{stem}.bin "
        f'  -o /out/ -t x86_64-linux-clang"',
        check=False, capture=True,
    )
    if result.returncode != 0:
        print(f"    [{stem}/int8] FAILED at lib-generator: {result.stderr[-500:]}")
        return None
    run(f'{docker_root} chown -R "$(id -u):$(id -g)" /out/', check=False, capture=True)

    # Step 3: .so -> context binary
    print(f"    [{stem}/int8] .so -> context binary ...")
    result = run(
        f'{docker_base} bash -c "'
        f"export LD_LIBRARY_PATH={CONTAINER_QAIRT}/lib/x86_64-linux-clang:\\$LD_LIBRARY_PATH; "
        f"export PATH={CONTAINER_QAIRT}/bin/x86_64-linux-clang:\\$PATH; "
        f"qnn-context-binary-generator "
        f"  --model /out/x86_64-linux-clang/lib{stem}.so "
        f"  --backend {CONTAINER_QAIRT}/lib/x86_64-linux-clang/libQnnHtp.so "
        f"  --binary_file {stem}.serialized --output_dir /out/ "
        f'  --config_file /out/htp_config_convert.json"',
        check=False, capture=True,
    )
    if result.returncode != 0:
        print(f"    [{stem}/int8] FAILED at context-binary-generator: {result.stderr[-500:]}")
        return None

    if ctx_file.exists():
        size_kb = ctx_file.stat().st_size / 1024
        print(f"    [{stem}/int8] OK ({size_kb:.0f} KB)")
        return ctx_file
    else:
        print(f"    [{stem}/int8] FAILED: context binary not created")
        return None


# ---------------------------------------------------------------------------
# Device Push
# ---------------------------------------------------------------------------

def push_qnn_runtime(qairt_root: str, phone_dir: str, htp_version: str, dsp_arch: str) -> None:
    """Push QNN runtime libraries to the device."""
    print("  Pushing QNN runtime libs ...")
    adb(f'shell "mkdir -p {phone_dir}/qnn"', check=False)

    lib_android = f"{qairt_root}/lib/aarch64-android"
    libs = [
        "libQnnHtp.so",
        f"libQnnHtp{htp_version}Stub.so",
        "libQnnSystem.so",
        "libQnnHtpPrepare.so",
        "libQnnHtpNetRunExtensions.so",
    ]
    for lib in libs:
        lib_path = f"{lib_android}/{lib}"
        if os.path.isfile(lib_path):
            adb(f'push "{lib_path}" "{phone_dir}/"', check=False, capture=True)
        else:
            print(f"    [WARN] {lib} not found at {lib_path}")

    skel = f"{qairt_root}/lib/hexagon-{dsp_arch}/unsigned/libQnnHtp{htp_version}Skel.so"
    if os.path.isfile(skel):
        adb(f'push "{skel}" "{phone_dir}/"', check=False, capture=True)

    qnn_bin = f"{qairt_root}/bin/aarch64-android/qnn-net-run"
    if os.path.isfile(qnn_bin):
        adb(f'push "{qnn_bin}" "{phone_dir}/"', check=False, capture=True)
        adb(f'shell "chmod +x {phone_dir}/qnn-net-run"', check=False)

    print("  Runtime push complete.")


def push_model_artifacts(
    qnn_dir: Path,
    stem: str,
    precision: str,
    phone_dir: str,
    onnx_dir: Path,
) -> bool:
    """Push context binary, configs, and dummy input to device. Returns True on success."""
    qnn_subdir = f"{stem}_{precision}"
    local_dir = qnn_dir / qnn_subdir
    ctx_file = local_dir / f"{stem}.serialized.bin"

    if not ctx_file.exists():
        return False

    device_subdir = f"{phone_dir}/qnn/{qnn_subdir}"
    adb(f'shell "mkdir -p {device_subdir}"', check=False)

    # Push context binary
    adb(f'push "{ctx_file}" "{device_subdir}/"', check=False, capture=True)

    # Push HTP configs
    for cfg in local_dir.glob("htp_*.json"):
        adb(f'push "{cfg}" "{device_subdir}/"', check=False, capture=True)

    # Create and push a dummy input raw file for latency benchmarking
    # (content doesn't matter for latency, just needs correct size)
    dummy_raw = local_dir / f"{stem}_dummy.raw"
    if not dummy_raw.exists():
        # Try to infer input size from ONNX; fall back to generating a small file
        # For latency benchmarking, qnn-net-run generates random input if no --input_list
        pass

    # Push dummy input list (qnn-net-run needs at least one input)
    raw_file = onnx_dir / f"{stem}.raw"
    if raw_file.exists():
        adb(f'push "{raw_file}" "{device_subdir}/"', check=False, capture=True)
        # Write input list referencing the raw file
        input_list = local_dir / "input_list.txt"
        input_list.write_text(f"{stem}.raw\n")
        adb(f'push "{input_list}" "{device_subdir}/"', check=False, capture=True)

    return True


# ---------------------------------------------------------------------------
# Benchmark Execution & Parsing
# ---------------------------------------------------------------------------

def run_benchmark(
    stem: str,
    precision: str,
    phone_dir: str,
    iterations: int,
    warmup: int,
    perf_profile: str,
) -> dict | None:
    """Run qnn-net-run on device and return parsed latency dict, or None on failure."""
    qnn_subdir = f"{stem}_{precision}"
    device_dir = f"{phone_dir}/qnn/{qnn_subdir}"
    ctx_file = f"{stem}.serialized.bin"

    # Clean old output
    adb(f'shell "rm -rf {device_dir}/output"', check=False, capture=True)

    # Build qnn-net-run command
    total_iters = warmup + iterations
    qnn_cmd = (
        f"cd {device_dir} && "
        f"LD_LIBRARY_PATH={phone_dir} ADSP_LIBRARY_PATH={phone_dir} "
        f"{phone_dir}/qnn-net-run "
        f"  --retrieve_context {ctx_file} "
        f"  --backend libQnnHtp.so "
        f"  --perf_profile {perf_profile} "
        f"  --num_inferences {total_iters} "
        f"  --profiling_level basic "
        f"  --output_dir output "
        f"  --config_file htp_config.json"
    )

    # Check if input_list exists on device; if so, add it
    check_result = adb(
        f'shell "test -f {device_dir}/input_list.txt && echo yes || echo no"',
        capture=True, check=False,
    )
    if check_result.stdout.strip() == "yes":
        qnn_cmd += f" --input_list input_list.txt"

    result = adb(f'shell "{qnn_cmd}"', check=False, capture=True)
    if result.returncode != 0:
        print(f"    [{stem}/{precision}] qnn-net-run FAILED")
        if result.stderr:
            print(f"    stderr: {result.stderr[-300:]}")
        return None

    return parse_profiling(stem, precision, phone_dir)


def parse_profiling(
    stem: str,
    precision: str,
    phone_dir: str,
) -> dict | None:
    """Pull and parse profiling log from device. Returns latency dict or None."""
    qnn_subdir = f"{stem}_{precision}"
    device_dir = f"{phone_dir}/qnn/{qnn_subdir}"

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
        prof_local = tmp.name

    result = adb(
        f'pull "{device_dir}/output/qnn-profiling-data_0.log" "{prof_local}"',
        check=False, capture=True,
    )
    if result.returncode != 0:
        print(f"    [{stem}/{precision}] Profiling log not found on device")
        os.unlink(prof_local)
        return None

    # Parse via qnn-profile-viewer in Docker
    viewer_result = run(
        f"docker run --rm "
        f"-v \"{prof_local}:/data/profiling.log:ro\" "
        f"{DOCKER_IMAGE} bash -c \""
        f"if [ -x /opt/qairt/bin/x86_64-linux-clang/qnn-profile-viewer ]; then "
        f"  export LD_LIBRARY_PATH=/opt/qairt/lib/x86_64-linux-clang:\\$LD_LIBRARY_PATH; "
        f"  /opt/qairt/bin/x86_64-linux-clang/qnn-profile-viewer --input_log /data/profiling.log; "
        f"else echo NO_VIEWER; fi\"",
        check=False, capture=True,
    )

    # Also try with QAIRT mount if viewer not in image
    if "NO_VIEWER" in (viewer_result.stdout or ""):
        # Viewer might need QAIRT mount; we don't have qairt_root here,
        # so fall back to parsing raw qnn-net-run stdout
        os.unlink(prof_local)
        return _parse_netrun_stdout(stem, precision, phone_dir)

    os.unlink(prof_local)
    output = viewer_result.stdout or ""

    # Extract timings: "Execute Stats (Average):" section -> "NetRun: NNN us"
    netrun_avg = _extract_stat(output, "Average", "NetRun")
    netrun_min = _extract_stat(output, "Min", "NetRun")
    netrun_max = _extract_stat(output, "Max", "NetRun")
    accel_avg = _extract_stat(output, "Average", "Accelerator")

    if netrun_avg is None:
        # Fall back to parsing qnn-net-run log output directly
        return _parse_netrun_stdout(stem, precision, phone_dir)

    return {
        "avg_us": netrun_avg,
        "min_us": netrun_min,
        "max_us": netrun_max,
        "accel_avg_us": accel_avg,
    }


def _extract_stat(text: str, stat_type: str, key: str) -> int | None:
    """Extract a timing value from qnn-profile-viewer output.

    Looks for patterns like:
        Execute Stats (Average):
            NetRun: 612 us
    """
    pattern = rf"Execute Stats \({stat_type}\).*?{key}[^:]*:\s*(\d+)\s*us"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return int(match.group(1))
    return None


def _parse_netrun_stdout(stem: str, precision: str, phone_dir: str) -> dict | None:
    """Fallback: parse qnn-net-run timing from its stdout/log file on device."""
    qnn_subdir = f"{stem}_{precision}"
    device_dir = f"{phone_dir}/qnn/{qnn_subdir}"

    # Try to read the log file from device output
    result = adb(
        f'shell "cat {device_dir}/output/qnn-net-run.log 2>/dev/null || echo NOLOG"',
        capture=True, check=False,
    )
    text = result.stdout or ""
    if "NOLOG" in text:
        return None

    # Look for "Total Inference Time" or individual inference times
    # Common pattern: "Inference N: NNN us"
    times = re.findall(r"Inference \d+:\s*(\d+)\s*us", text)
    if not times:
        # Try "Total Inference Time: NNN us"
        match = re.search(r"Total Inference Time:\s*(\d+)\s*us", text)
        if match:
            total = int(match.group(1))
            # Count inferences
            n_match = re.search(r"Num Inferences:\s*(\d+)", text)
            n = int(n_match.group(1)) if n_match else 1
            return {"avg_us": total // n, "min_us": None, "max_us": None, "accel_avg_us": None}
        return None

    times_int = [int(t) for t in times]
    return {
        "avg_us": int(sum(times_int) / len(times_int)),
        "min_us": min(times_int),
        "max_us": max(times_int),
        "accel_avg_us": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTP INT8/FP16 latency benchmark for ANVIL and RIFE models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--onnx-dir", type=Path, required=True,
        help="Directory containing ONNX model files.",
    )
    parser.add_argument(
        "--qairt-root", type=str,
        default="/opt/qcom/aistack/qairt/2.42.0.251225",
        help="Path to QAIRT SDK on host.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/bench/latency"),
        help="Directory for output CSV and QNN artifacts.",
    )
    parser.add_argument(
        "--phone-root", type=str, default="/data/local/tmp/qnn",
        help="Device path for QNN libs and artifacts.",
    )
    parser.add_argument(
        "--precision", choices=["fp16", "int8", "both"], default="both",
        help="Quantization mode(s) to benchmark.",
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of inference iterations for benchmarking.",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup iterations before measurement.",
    )
    parser.add_argument(
        "--models", type=str, default="all",
        help="Comma-separated model names or 'all'. Known aliases: "
             + ", ".join(KNOWN_MODELS.keys()),
    )
    parser.add_argument(
        "--calib-dir", type=Path, default=None,
        help="Directory with INT8 calibration data (NCHW .raw + input_list.txt). "
             "Required for INT8.",
    )
    parser.add_argument(
        "--perf-profile", type=str, default="burst",
        choices=["burst", "sustained", "balanced", "power_saver"],
        help="HTP performance profile.",
    )
    parser.add_argument(
        "--dsp-arch", type=str, default=DEFAULT_DSP_ARCH,
        help="Hexagon DSP architecture (e.g. v73, v75).",
    )
    parser.add_argument(
        "--soc-id", type=int, default=DEFAULT_SOC_ID,
        help="SoC ID for HTP backend config.",
    )
    parser.add_argument(
        "--htp-version", type=str, default=DEFAULT_HTP_VERSION,
        help="HTP version string (e.g. V73, V75).",
    )
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="Skip QNN conversion (reuse existing context binaries).",
    )
    parser.add_argument(
        "--skip-push", action="store_true",
        help="Skip pushing artifacts to device.",
    )
    args = parser.parse_args()

    # Validate
    precisions = []
    if args.precision in ("fp16", "both"):
        precisions.append("fp16")
    if args.precision in ("int8", "both"):
        precisions.append("int8")
        if args.calib_dir is None and not args.skip_convert:
            print("WARNING: INT8 selected but no --calib-dir provided. "
                  "Conversion will fail without calibration data.")

    models_filter = None if args.models == "all" else [m.strip() for m in args.models.split(",")]

    # Discover models
    print("=" * 60)
    print("ANVIL HTP Latency Benchmark")
    print("=" * 60)

    models = discover_onnx_models(args.onnx_dir, models_filter)
    if not models:
        print(f"ERROR: No ONNX models found in {args.onnx_dir}", file=sys.stderr)
        if models_filter:
            print(f"  Filter: {models_filter}", file=sys.stderr)
            print(f"  Available: {[p.stem for p in args.onnx_dir.glob('*.onnx')]}", file=sys.stderr)
        sys.exit(1)

    print(f"Models ({len(models)}):")
    for m in models:
        print(f"  {m['stem']} ({m['resolution']})")
    print(f"Precisions: {precisions}")
    print(f"Iterations: {args.warmup} warmup + {args.iterations} measured")
    print(f"Perf profile: {args.perf_profile}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qnn_dir = args.output_dir / "qnn"
    qnn_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: QNN Conversion
    if not args.skip_convert:
        print("=== Step 1: QNN Conversion ===")
        ensure_docker_image()
        for model_info in models:
            for prec in precisions:
                if prec == "fp16":
                    qnn_convert_fp16(
                        model_info, qnn_dir, args.onnx_dir, args.qairt_root,
                        args.phone_root, args.dsp_arch, args.soc_id,
                    )
                else:
                    calib = args.calib_dir or (args.onnx_dir / "calibration")
                    qnn_convert_int8(
                        model_info, qnn_dir, args.onnx_dir, calib,
                        args.qairt_root, args.phone_root, args.dsp_arch, args.soc_id,
                    )
        print()
    else:
        print("=== Step 1: SKIP (--skip-convert) ===")

    # Step 2: Push to device
    if not args.skip_push:
        print("=== Step 2: Push to Device ===")
        check_adb_device()
        push_qnn_runtime(args.qairt_root, args.phone_root, args.htp_version, args.dsp_arch)
        for model_info in models:
            for prec in precisions:
                push_model_artifacts(
                    qnn_dir, model_info["stem"], prec,
                    args.phone_root, args.onnx_dir,
                )
        print()
    else:
        print("=== Step 2: SKIP (--skip-push) ===")

    # Step 3: Benchmark
    print("=== Step 3: Benchmark ===")
    check_adb_device()

    results: list[dict] = []
    header = f"  {'Model':<35} {'Res':<8} {'Prec':<6} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model_info in models:
        for prec in precisions:
            stem = model_info["stem"]
            latency = run_benchmark(
                stem, prec, args.phone_root,
                args.iterations, args.warmup, args.perf_profile,
            )
            row = {
                "model": model_info["name"],
                "resolution": model_info["resolution"],
                "precision": prec,
                "min_ms": "",
                "avg_ms": "",
                "max_ms": "",
            }
            if latency:
                avg_ms = latency["avg_us"] / 1000 if latency["avg_us"] else None
                min_ms = latency["min_us"] / 1000 if latency.get("min_us") else None
                max_ms = latency["max_us"] / 1000 if latency.get("max_us") else None

                row["avg_ms"] = f"{avg_ms:.2f}" if avg_ms else "N/A"
                row["min_ms"] = f"{min_ms:.2f}" if min_ms else "N/A"
                row["max_ms"] = f"{max_ms:.2f}" if max_ms else "N/A"

                print(
                    f"  {model_info['name']:<35} {model_info['resolution']:<8} {prec:<6} "
                    f"{row['avg_ms']:>10} {row['min_ms']:>10} {row['max_ms']:>10}"
                )
            else:
                row["avg_ms"] = "ERROR"
                row["min_ms"] = "ERROR"
                row["max_ms"] = "ERROR"
                print(
                    f"  {model_info['name']:<35} {model_info['resolution']:<8} {prec:<6} "
                    f"{'ERROR':>10} {'ERROR':>10} {'ERROR':>10}"
                )

            results.append(row)

    # Step 4: Write CSV
    csv_path = args.output_dir / "latency.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "resolution", "precision", "min_ms", "avg_ms", "max_ms"])
        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Saved to: {csv_path}")
    print()

    # Pretty-print summary
    for row in results:
        status = "OK" if row["avg_ms"] not in ("", "ERROR") else "FAILED"
        avg_str = row["avg_ms"] if row["avg_ms"] else "N/A"
        deadline = ""
        try:
            avg_val = float(row["avg_ms"])
            deadline = " < 33.3ms" if avg_val < 33.3 else " > 33.3ms OVER BUDGET"
        except (ValueError, TypeError):
            pass
        print(f"  {row['model']:<30} {row['resolution']:<8} {row['precision']:<6} "
              f"avg={avg_str:>8}ms  [{status}]{deadline}")

    print()
    print("Target: < 33.3ms per frame for 30->60fps interpolation")


if __name__ == "__main__":
    main()
