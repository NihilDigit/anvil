"""Ablation benchmark: architecture optimization decisions.

Reproduces paper latency ablation by benchmarking three architecture
variants in a single device session, isolating each design decision:

  Step 1: Symmetric channels → Asymmetric channels   (-28 to -40%)
  Step 2: Concat skip → Additive skip + BN fusion    (-17 to -26%)

Tested variants (S tier example):
  v3-s  symmetric  [20,40,80,80]  concat     baseline
  v3s   asymmetric [16,32,64,64]  concat     channel optimization only
  v3bs  asymmetric [16,32,64,64]  additive   + skip optimization (= ANVIL-S)

Model weights are random (untrained) since latency measurement does
not depend on weight values — only graph structure matters.

Prerequisites:
  - Docker (for QAIRT conversion)
  - QAIRT SDK on host (default /opt/qcom/aistack/qairt/2.42.0.251225)
  - Android device connected via ADB with Hexagon HTP support
  - Calibration data in --calib-dir (NCHW .raw files + input_list.txt)

Usage:
    pixi run python scripts/bench_deploy_optimization.py \\
        --qairt-root /opt/qcom/aistack/qairt/2.42.0.251225 \\
        --calib-dir artifacts/calibration_1080p \\
        --output-dir artifacts/bench/deploy_optimization \\
        --num-inferences 100
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
from pathlib import Path

import torch

from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from anvil_exp01.models.conv_vfi import build_model, count_parameters


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCKER_IMAGE = "qairt-converter"
CONTAINER_QAIRT = "/opt/qairt"
DEFAULT_DSP_ARCH = "v75"
DEFAULT_SOC_ID = 57

# Ablation variants: (display_name, model_id, description)
# Each consecutive pair isolates one design decision.
ABLATION_MODELS = [
    # --- S tier ---
    ("v3_s_symmetric", "D-unet-v3-s-nomv", "symmetric [20,40,80,80] concat"),
    ("v3s_asymmetric", "D-unet-v3s-nomv", "asymmetric [16,32,64,64] concat"),
    ("v3bs_final", "D-unet-v3bs-nomv", "asymmetric [16,32,64,64] additive+BN (=ANVIL-S)"),
    # --- M tier ---
    ("v3_m_symmetric", "D-unet-v3-m-nomv", "symmetric [24,48,96,96] concat"),
    ("v3m_asymmetric", "D-unet-v3m-nomv", "asymmetric [16,32,96,96] concat"),
    ("v3bm_final", "D-unet-v3bm-nomv", "asymmetric [16,32,96,96] additive+BN (=ANVIL-M)"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str, *, check: bool = True, capture: bool = False, **kwargs):
    return subprocess.run(
        cmd, shell=True, check=check, capture_output=capture, text=True, **kwargs,
    )


def adb(cmd: str, *, check: bool = True, capture: bool = False):
    return run(f"adb {cmd}", check=check, capture=capture)


def check_adb_device() -> str:
    result = adb("devices", capture=True)
    lines = [l for l in result.stdout.strip().splitlines()[1:] if l.strip()]
    devices = [l for l in lines if "device" in l.split()[-1:]]
    if not devices:
        print("ERROR: No ADB device connected.", file=sys.stderr)
        sys.exit(1)
    serial = devices[0].split()[0]
    # Get SoC info
    soc = adb("shell getprop ro.soc.model", capture=True).stdout.strip()
    print(f"  Device: {serial} ({soc})")
    return soc


def ensure_docker_image():
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


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(model_id: str, output_path: Path) -> int:
    """Export model to ONNX at 1080p. Returns param count."""
    model = build_model(model_id)
    model.eval()
    n_params = count_parameters(model)
    x = torch.randn(1, 6, 1080, 1920)
    torch.onnx.export(
        model, x, str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )
    return n_params


# ---------------------------------------------------------------------------
# QNN INT8 Compilation (Docker)
# ---------------------------------------------------------------------------

def compile_int8(
    name: str,
    onnx_path: Path,
    calib_dir: Path,
    output_dir: Path,
    qairt_root: Path,
    dsp_arch: str,
    soc_id: int,
) -> Path:
    """Compile ONNX → QNN INT8 context binary. Returns path to .serialized.bin."""
    out = output_dir / f"{name}_int8"
    graph_name = f"{name}_1080p"
    ctx_bin = out / f"{graph_name}.serialized.bin"

    if ctx_bin.exists():
        print(f"  [{name}] Context binary exists, skipping compilation")
        return ctx_bin

    out.mkdir(parents=True, exist_ok=True)

    # HTP configs
    (out / "htp_backend_config.json").write_text(json.dumps({
        "graphs": [{"graph_names": [graph_name], "vtcm_mb": 0, "O": 3}],
        "devices": [{"dsp_arch": dsp_arch, "soc_id": soc_id,
                      "pd_session": "unsigned", "device_id": 0}],
    }, indent=2))

    (out / "htp_config_convert.json").write_text(json.dumps({
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": "/out/htp_backend_config.json",
        }
    }, indent=2))

    # Docker input list
    input_list = calib_dir / "input_list.txt"
    if not input_list.exists():
        # Auto-generate from .raw files
        raws = sorted(calib_dir.glob("calib_*.raw"))
        input_list.write_text("\n".join(r.name for r in raws) + "\n")

    docker_input_list = out / "input_list_docker.txt"
    docker_input_list.write_text(
        "\n".join(f"/calib/{l.strip()}" for l in input_list.read_text().splitlines() if l.strip())
    )

    onnx_dir = onnx_path.parent
    onnx_file = onnx_path.name

    docker_base = (
        f"docker run --rm --user $(id -u):$(id -g) -w /out "
        f"-v {qairt_root}:{CONTAINER_QAIRT}:ro "
        f"-v {onnx_dir}:/onnx:ro "
        f"-v {calib_dir}:/calib:ro "
        f"-v {out}:/out "
        f"{DOCKER_IMAGE}"
    )
    docker_root = docker_base.replace(f"--user $(id -u):$(id -g)", "")

    env_setup = (
        f"export PYTHONPATH={CONTAINER_QAIRT}/lib/python:$PYTHONPATH && "
        f"export LD_LIBRARY_PATH={CONTAINER_QAIRT}/lib/x86_64-linux-clang:$LD_LIBRARY_PATH && "
        f"export PATH={CONTAINER_QAIRT}/bin/x86_64-linux-clang:$PATH"
    )

    # ONNX → QNN model (W8A8)
    print(f"  [{name}] ONNX → QNN INT8 ...")
    run(f'{docker_base} bash -c "{env_setup} && '
        f"python3 {CONTAINER_QAIRT}/bin/x86_64-linux-clang/qnn-onnx-converter "
        f"--input_network /onnx/{onnx_file} "
        f"--output_path /out/{graph_name} "
        f"--input_list /out/input_list_docker.txt "
        f"--act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 32 "
        f"--act_quantizer_calibration percentile "
        f"--percentile_calibration_value 99.99 "
        f"--act_quantizer_schema asymmetric "
        f"--param_quantizer_schema symmetric "
        f"--use_per_channel_quantization "
        f'--input_layout input NCHW"')

    # Fix extension
    cpp = out / f"{graph_name}.cpp"
    no_ext = out / graph_name
    if no_ext.exists() and not cpp.exists():
        no_ext.rename(cpp)

    # QNN model → .so
    print(f"  [{name}] QNN → .so ...")
    run(f'{docker_root} bash -c "{env_setup} && '
        f"export TMPDIR=/out && "
        f"python3 {CONTAINER_QAIRT}/bin/x86_64-linux-clang/qnn-model-lib-generator "
        f"-c /out/{graph_name}.cpp "
        f"-b /out/{graph_name}.bin "
        f"-o /out/ "
        f'-t x86_64-linux-clang"')
    run(f'{docker_root} chown -R "$(id -u):$(id -g)" /out/')

    # .so → context binary
    print(f"  [{name}] .so → context binary ...")
    run(f'{docker_base} bash -c "{env_setup} && '
        f"qnn-context-binary-generator "
        f"--model /out/x86_64-linux-clang/lib{graph_name}.so "
        f"--backend {CONTAINER_QAIRT}/lib/x86_64-linux-clang/libQnnHtp.so "
        f"--binary_file {graph_name}.serialized "
        f"--output_dir /out/ "
        f'--config_file /out/htp_config_convert.json"')

    if ctx_bin.exists():
        print(f"  [{name}] OK: {ctx_bin.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"  [{name}] FAILED", file=sys.stderr)
        sys.exit(1)

    return ctx_bin


# ---------------------------------------------------------------------------
# Device Benchmark
# ---------------------------------------------------------------------------

def push_qnn_libs(qairt_root: Path, device_lib: str):
    """Push QNN runtime libraries to device (once)."""
    adb(f'shell "mkdir -p {device_lib}"', check=False)
    arm_lib = qairt_root / "lib" / "aarch64-android"
    htp_lib = qairt_root / "lib" / "hexagon-v75" / "unsigned"
    for lib in ["libQnnHtpV75Stub.so", "libQnnHtp.so",
                "libQnnHtpNetRunExtensions.so", "libQnnSystem.so"]:
        adb(f"push {arm_lib / lib} {device_lib}/", check=False)
    adb(f"push {htp_lib / 'libQnnHtpV75Skel.so'} {device_lib}/", check=False)
    adb(f"push {qairt_root / 'bin' / 'aarch64-android' / 'qnn-net-run'} {device_lib}/",
        check=False)


def benchmark_one(
    name: str,
    ctx_bin: Path,
    device_dir: str,
    device_lib: str,
    num_inferences: int,
    output_dir: Path,
    qairt_root: Path,
) -> dict:
    """Run one model on device, return parsed timing dict."""
    graph_name = f"{name}_1080p"
    remote = f"{device_dir}/{name}"

    adb(f'shell "mkdir -p {remote}/output"', check=False)

    # Push context binary + config
    adb(f"push {ctx_bin} {remote}/")

    htp_backend = {
        "graphs": [{"graph_names": [graph_name], "vtcm_mb": 0, "O": 3}],
        "devices": [{"dsp_arch": DEFAULT_DSP_ARCH, "soc_id": DEFAULT_SOC_ID,
                      "pd_session": "unsigned", "device_id": 0}],
    }
    htp_config = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": f"{remote}/htp_backend_config.json",
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(htp_backend, f)
        f.flush()
        adb(f"push {f.name} {remote}/htp_backend_config.json")
        os.unlink(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(htp_config, f)
        f.flush()
        adb(f"push {f.name} {remote}/htp_config.json")
        os.unlink(f.name)

    # Run inference
    result = adb(
        f'shell "cd {device_dir} && '
        f"export LD_LIBRARY_PATH={device_lib} && "
        f"export ADSP_LIBRARY_PATH={device_lib} && "
        f"{device_lib}/qnn-net-run "
        f"--backend libQnnHtp.so "
        f"--retrieve_context {remote}/{graph_name}.serialized.bin "
        f"--input_list {device_dir}/input/input_list.txt "
        f"--num_inferences {num_inferences} "
        f"--perf_profile burst "
        f"--profiling_level basic "
        f"--config_file {remote}/htp_config.json "
        f'--output_dir {remote}/output"',
        capture=True,
    )

    # Pull and parse profiling log
    prof_log = output_dir / f"{name}_profile.log"
    adb(f"pull {remote}/output/qnn-profiling-data.log {prof_log}", check=False)

    if not prof_log.exists():
        # Try alternate path
        pull_result = adb(f'shell "ls {remote}/output/*.log"', capture=True, check=False)
        if pull_result.stdout.strip():
            remote_log = pull_result.stdout.strip().split("\n")[0].strip()
            adb(f"pull {remote_log} {prof_log}", check=False)

    timing = {"name": name, "avg_us": 0, "min_us": 0, "max_us": 0}

    if prof_log.exists():
        # Use qnn-profile-viewer via Docker
        viewer_out = output_dir / f"{name}_viewer.txt"
        run(
            f"docker run --rm "
            f"-v {qairt_root}:{qairt_root}:ro "
            f"-v {output_dir}:/data "
            f"{DOCKER_IMAGE} bash -c \""
            f"export LD_LIBRARY_PATH={qairt_root}/lib/x86_64-linux-clang:$LD_LIBRARY_PATH && "
            f"{qairt_root}/bin/x86_64-linux-clang/qnn-profile-viewer "
            f'--input_log /data/{prof_log.name} --output /data/{viewer_out.name}"',
            check=False,
        )

        if viewer_out.exists():
            text = viewer_out.read_text()

            # Try CSV format first (QAIRT 2.42+):
            # columns: timestamp,phase,time_us,unit,source,level,detail
            # EXECUTE rows with NETRUN source = per-inference latency
            netrun_times = []
            for line in text.splitlines():
                parts = line.split(",")
                if (len(parts) >= 5
                        and parts[1].strip() == "EXECUTE"
                        and parts[4].strip() == "NETRUN"):
                    try:
                        netrun_times.append(int(parts[2].strip()))
                    except ValueError:
                        pass

            if netrun_times:
                timing["avg_us"] = int(sum(netrun_times) / len(netrun_times))
                timing["min_us"] = min(netrun_times)
                timing["max_us"] = max(netrun_times)
            else:
                # Fallback: text format (older QAIRT)
                for section, key in [("Average", "avg_us"), ("Min", "min_us"),
                                     ("Max", "max_us")]:
                    pattern = rf"Execute Stats \({section}\):.*?NetRun:\s*(\d+)\s*us"
                    m = re.search(pattern, text, re.DOTALL)
                    if m:
                        timing[key] = int(m.group(1))

    return timing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A/B benchmark: deploy-time optimization (additive vs concat skip).",
    )
    parser.add_argument("--qairt-root", type=Path,
                        default=Path("/opt/qcom/aistack/qairt/2.42.0.251225"))
    parser.add_argument("--calib-dir", type=Path, required=True,
                        help="Calibration data directory (NCHW .raw files).")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/bench/deploy_optimization"))
    parser.add_argument("--num-inferences", type=int, default=100)
    parser.add_argument("--dsp-arch", default=DEFAULT_DSP_ARCH)
    parser.add_argument("--soc-id", type=int, default=DEFAULT_SOC_ID)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = args.output_dir / "onnx"
    qnn_dir = args.output_dir / "qnn"
    onnx_dir.mkdir(exist_ok=True)
    qnn_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  A/B Benchmark: Deploy-Time Optimization")
    print("  Additive skip + BN fusion vs. Concat skip")
    print("=" * 60)

    # Check prerequisites
    print("\n[1/5] Checking prerequisites ...")
    soc = check_adb_device()
    ensure_docker_image()

    # Export ONNX models
    print("\n[2/5] Exporting ONNX models ...")
    model_info = {}
    for display_name, model_id, description in ABLATION_MODELS:
        onnx_path = onnx_dir / f"{display_name}_1080p.onnx"
        if onnx_path.exists():
            n = count_parameters(build_model(model_id))
            print(f"  [{display_name}] ONNX exists ({n:,} params)")
        else:
            n = export_onnx(model_id, onnx_path)
            print(f"  [{display_name}] Exported ({n:,} params)")
        model_info[display_name] = {
            "model_id": model_id, "description": description,
            "params": n, "onnx_path": onnx_path,
        }

    # Compile INT8 context binaries
    print("\n[3/5] Compiling INT8 context binaries ...")
    for display_name, info in model_info.items():
        ctx = compile_int8(
            name=display_name,
            onnx_path=info["onnx_path"],
            calib_dir=args.calib_dir.resolve(),
            output_dir=qnn_dir,
            qairt_root=args.qairt_root,
            dsp_arch=args.dsp_arch,
            soc_id=args.soc_id,
        )
        info["ctx_bin"] = ctx

    # Push QNN libs + dummy input
    print("\n[4/5] Pushing to device ...")
    device_dir = "/data/local/tmp/anvil/bench_deploy_opt"
    device_lib = "/data/local/tmp/anvil/qnn_lib"
    push_qnn_libs(args.qairt_root, device_lib)

    # Create dummy input on device
    adb(f'shell "mkdir -p {device_dir}/input"', check=False)
    # Generate a dummy raw file (content irrelevant for latency)
    dummy_raw = args.output_dir / "dummy_1080p.raw"
    if not dummy_raw.exists():
        import numpy as np
        np.random.randn(1, 6, 1080, 1920).astype(np.float32).tofile(dummy_raw)
    adb(f"push {dummy_raw} {device_dir}/input/dummy.raw")
    adb(f'shell "echo \'{device_dir}/input/dummy.raw\' > {device_dir}/input/input_list.txt"')

    # Benchmark all models in same session
    print(f"\n[5/5] Running benchmark ({args.num_inferences} inferences each) ...")
    print()

    results = []
    for display_name, info in model_info.items():
        print(f"  --- {display_name} ({info['description']}) ---")
        timing = benchmark_one(
            name=display_name,
            ctx_bin=info["ctx_bin"],
            device_dir=device_dir,
            device_lib=device_lib,
            num_inferences=args.num_inferences,
            output_dir=args.output_dir,
            qairt_root=args.qairt_root,
        )
        timing["description"] = info["description"]
        timing["params"] = info["params"]
        results.append(timing)

        avg_ms = timing["avg_us"] / 1000
        min_ms = timing["min_us"] / 1000
        print(f"    avg={avg_ms:.1f}ms  min={min_ms:.1f}ms")

    # Print summary
    print("\n" + "=" * 70)
    print("  Results Summary")
    print("=" * 70)
    print(f"  {'Model':<20} {'Params':>8} {'Avg(ms)':>8} {'Min(ms)':>8}  Description")
    print("  " + "-" * 66)

    for r in results:
        print(f"  {r['name']:<20} {r['params']:>8,} "
              f"{r['avg_us']/1000:>8.1f} {r['min_us']/1000:>8.1f}  {r['description']}")

    # Compute step-by-step reductions
    print()
    print("  Step-by-step latency reduction (avg):")
    steps = [
        # (before, after, label, design_decision)
        ("v3_s_symmetric", "v3s_asymmetric", "S", "asymmetric channels"),
        ("v3s_asymmetric", "v3bs_final", "S", "additive skip + BN fusion"),
        ("v3_s_symmetric", "v3bs_final", "S", "total (symmetric → ANVIL-S)"),
        ("v3_m_symmetric", "v3m_asymmetric", "M", "asymmetric channels"),
        ("v3m_asymmetric", "v3bm_final", "M", "additive skip + BN fusion"),
        ("v3_m_symmetric", "v3bm_final", "M", "total (symmetric → ANVIL-M)"),
    ]
    for before_name, after_name, tier, decision in steps:
        before = next((r for r in results if r["name"] == before_name), None)
        after = next((r for r in results if r["name"] == after_name), None)
        if before and after and before["avg_us"] > 0:
            reduction = (before["avg_us"] - after["avg_us"]) / before["avg_us"] * 100
            print(f"    [{tier}] {decision}: "
                  f"{before['avg_us']/1000:.1f} → {after['avg_us']/1000:.1f}ms "
                  f"({reduction:+.1f}%)")

    # Save CSV
    csv_path = args.output_dir / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "description", "params",
                                          "avg_us", "min_us", "max_us"])
        w.writeheader()
        w.writerows(results)
    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    main()
