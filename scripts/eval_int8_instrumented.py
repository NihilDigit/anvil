#!/usr/bin/env python3
"""Per-operator instrumented INT8 quantization analysis.

Reproduces paper Table instrumented: CosSim causal chain showing how progressive
operator quantization degrades RIFE and IFRNet output quality.

Protocol:
  - Progressively add operator types to ORT W8A8 static quantization:
      Step 1: Conv + ConvTranspose only
      Step 2: + PReLU
      Step 3: + Add
      Step 4: Full W8A8 (all ops)
      Step 5: Resize only (control)
  - 3 calibration inputs from Vimeo90K train (--vimeo-dir) or disjoint Xiph split
  - 5 test inputs from Xiph 1080p
  - Metric: CosSim (cosine similarity) between FP32 and quantized raw outputs

Key finding: Add quantization is the collapse trigger for iterative-accumulation
VFI methods. Conv introduces initial error (~0.92-0.99), PReLU adds zero,
Add amplifies to ~0.77-0.99, and Full W8A8 equals Conv+Add.

Usage:
    # Paper protocol: Vimeo train for calibration, Xiph for test
    pixi run python scripts/eval_int8_instrumented.py \\
        --onnx-dir artifacts/onnx \\
        --xiph-dir data/xiph_1080p \\
        --vimeo-dir data/vimeo_triplet \\
        --output-dir artifacts/eval/int8_instrumented

    # Without Vimeo: uses disjoint Xiph split for calibration
    pixi run python scripts/eval_int8_instrumented.py \\
        --onnx-dir artifacts/onnx \\
        --xiph-dir data/xiph_1080p \\
        --models rife_flow_360p ifrnet_flow
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CALIB = 3
N_TEST = 5
SEED = 42

# Sequence-level calibration/evaluation split for Xiph 1080p.
# Calibration sequences chosen to cover diverse motion (small + dense)
# while keeping paper-featured sequences (tractor, old_town_cross, etc.) in eval.
CALIB_SEQUENCES = {"sunflower", "pedestrian_area"}

# RIFE / IFRNet input resolutions
RIFE_H, RIFE_W = 384, 640
IFR_H, IFR_W = 256, 448

# Progressive quantization steps: (label, op_types_to_quantize or None for full)
QUANT_STEPS = [
    ("Conv_ConvTr", ["Conv", "ConvTranspose"]),
    ("+PReLU", ["Conv", "ConvTranspose", "PReLU"]),
    ("+Add", ["Conv", "ConvTranspose", "PReLU", "Add"]),
    ("Full_W8A8", None),  # None = quantize all ops
    ("Resize_only", ["Resize"]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def discover_triplets(seq_dir: Path) -> list[str]:
    """Return sorted list of 'sequence/triplet' IDs."""
    triplets = []
    for sd in sorted(seq_dir.iterdir()):
        if not sd.is_dir():
            continue
        for td in sorted(sd.iterdir()):
            if td.is_dir():
                triplets.append(f"{sd.name}/{td.name}")
    return triplets


def load_frames(seq_dir: Path, tid: str) -> tuple[np.ndarray, np.ndarray]:
    """Load im1 and im3 from a Xiph triplet."""
    s, t = tid.split("/")
    i1 = cv2.cvtColor(cv2.imread(str(seq_dir / s / t / "im1.png")), cv2.COLOR_BGR2RGB)
    i3 = cv2.cvtColor(cv2.imread(str(seq_dir / s / t / "im3.png")), cv2.COLOR_BGR2RGB)
    return i1, i3


def prep_rife(i1: np.ndarray, i3: np.ndarray) -> dict[str, np.ndarray]:
    a = cv2.resize(i1, (RIFE_W, RIFE_H)).astype(np.float32) / 255.0
    b = cv2.resize(i3, (RIFE_W, RIFE_H)).astype(np.float32) / 255.0
    return {"input": np.concatenate([a, b], axis=2).transpose(2, 0, 1)[np.newaxis]}


def prep_ifrnet(i1: np.ndarray, i3: np.ndarray) -> dict[str, np.ndarray]:
    a = cv2.resize(i1, (IFR_W, IFR_H)).astype(np.float32) / 255.0
    b = cv2.resize(i3, (IFR_W, IFR_H)).astype(np.float32) / 255.0
    return {
        "img0": a.transpose(2, 0, 1)[np.newaxis],
        "img1": b.transpose(2, 0, 1)[np.newaxis],
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two arrays (flattened to 1D)."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def concat_outputs(outputs: list[np.ndarray]) -> np.ndarray:
    """Concatenate all output arrays into a single flat vector."""
    return np.concatenate([o.flatten() for o in outputs])


# ---------------------------------------------------------------------------
# Calibration reader
# ---------------------------------------------------------------------------
class XiphCalibrationReader(CalibrationDataReader):
    def __init__(self, tids: list[str], seq_dir: Path, prep_fn):
        self._tids = list(tids)
        self._seq_dir = seq_dir
        self._prep_fn = prep_fn
        self._idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        while self._idx < len(self._tids):
            tid = self._tids[self._idx]
            self._idx += 1
            try:
                i1, i3 = load_frames(self._seq_dir, tid)
                return self._prep_fn(i1, i3)
            except Exception:
                continue
        return None


class VimeoCalibrationReader(CalibrationDataReader):
    """Feed calibration samples from Vimeo90K training set (paper protocol)."""

    def __init__(self, vimeo_dir: Path, n_samples: int, prep_fn, seed: int = 42):
        train_list = vimeo_dir / "tri_trainlist.txt"
        triplets = [l.strip() for l in train_list.read_text().splitlines() if l.strip()]
        rng = random.Random(seed)
        self._tids = rng.sample(triplets, min(n_samples, len(triplets)))
        self._seq_dir = vimeo_dir / "sequences"
        self._prep_fn = prep_fn
        self._idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        while self._idx < len(self._tids):
            tid = self._tids[self._idx]
            self._idx += 1
            try:
                td = self._seq_dir / tid
                i1 = cv2.cvtColor(cv2.imread(str(td / "im1.png")), cv2.COLOR_BGR2RGB)
                i3 = cv2.cvtColor(cv2.imread(str(td / "im3.png")), cv2.COLOR_BGR2RGB)
                return self._prep_fn(i1, i3)
            except Exception:
                continue
        return None


# ---------------------------------------------------------------------------
# Instrumented analysis for a single model
# ---------------------------------------------------------------------------
def analyze_model(
    model_name: str,
    onnx_path: Path,
    prep_fn,
    test_inputs: list[dict[str, np.ndarray]],
    calib_tids: list[str],
    seq_dir: Path,
    cache_dir: Path,
    vimeo_dir: Path | None = None,
    n_calib: int = N_CALIB,
    seed: int = SEED,
) -> list[dict]:
    """Run progressive quantization and measure CosSim at each step."""

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4

    # FP32 baseline outputs
    print(f"  Running FP32 baseline ...", flush=True)
    sess_fp32 = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    fp32_outputs = []
    for feed in test_inputs:
        outs = sess_fp32.run(None, feed)
        fp32_outputs.append(concat_outputs(outs))
    del sess_fp32

    rows = []
    prev_cossim = 1.0

    for step_label, op_types in QUANT_STEPS:
        int8_path = cache_dir / f"{model_name}_{step_label}.onnx"

        if not int8_path.exists():
            print(f"  Quantizing [{step_label}] ...", flush=True)
            t0 = time.time()
            if vimeo_dir is not None:
                reader = VimeoCalibrationReader(vimeo_dir, n_calib, prep_fn, seed=seed)
            else:
                reader = XiphCalibrationReader(calib_tids, seq_dir, prep_fn)
            kwargs = {
                "model_input": str(onnx_path),
                "model_output": str(int8_path),
                "calibration_data_reader": reader,
                "quant_format": QuantFormat.QOperator,
                "weight_type": QuantType.QInt8,
                "activation_type": QuantType.QUInt8,
                "calibrate_method": CalibrationMethod.Percentile,
                "extra_options": {"ActivationSymmetric": False, "CalibPercentile": 99.99},
            }
            if op_types is not None:
                kwargs["op_types_to_quantize"] = op_types
            quantize_static(**kwargs)
            print(f"    Done in {time.time() - t0:.1f}s", flush=True)
        else:
            print(f"  Using cached [{step_label}]: {int8_path.name}")

        # Run quantized model
        sess_q = ort.InferenceSession(str(int8_path), opts, providers=["CPUExecutionProvider"])
        cossims = []
        for i, feed in enumerate(test_inputs):
            q_outs = sess_q.run(None, feed)
            q_flat = concat_outputs(q_outs)
            cs = cosine_similarity(fp32_outputs[i], q_flat)
            cossims.append(cs)
        del sess_q

        mean_cs = float(np.mean(cossims))
        delta = mean_cs - prev_cossim
        rows.append({
            "model": model_name,
            "quantized_ops": step_label,
            "cossim": round(mean_cs, 3),
            "delta_vs_prev": round(delta, 3),
        })
        print(f"    [{step_label}] CosSim={mean_cs:.3f}  delta={delta:+.3f}")
        prev_cossim = mean_cs

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-operator instrumented INT8 analysis (CosSim causal chain)."
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        required=True,
        help="Directory with ONNX models (rife_flow_360p.onnx, ifrnet_flow_vimeo.onnx).",
    )
    parser.add_argument(
        "--xiph-dir",
        type=Path,
        required=True,
        help="Xiph 1080p root (contains sequences/<seq>/<triplet>/im{1,3}.png).",
    )
    parser.add_argument(
        "--vimeo-dir",
        type=Path,
        default=None,
        help="Vimeo90K root for calibration. "
        "Contains tri_trainlist.txt + sequences/. "
        "When omitted, calibration uses disjoint Xiph split (recommended for RIFE/IFRNet).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rife_flow_360p", "ifrnet_flow"],
        help="Model names to analyze (default: rife_flow_360p ifrnet_flow).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eval/int8_instrumented"),
    )
    parser.add_argument("--n-calib", type=int, default=N_CALIB, help="Calibration samples.")
    parser.add_argument("--n-test", type=int, default=N_TEST, help="Test samples.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "int8_models"
    cache_dir.mkdir(exist_ok=True)

    seq_dir = args.xiph_dir / "sequences"
    if not seq_dir.is_dir():
        seq_dir = args.xiph_dir

    # Discover triplets and sample
    triplets = discover_triplets(seq_dir)
    print(f"Xiph: {len(triplets)} total triplets")

    random.seed(args.seed)
    shuffled = list(triplets)
    random.shuffle(shuffled)

    if args.vimeo_dir is not None:
        # Paper protocol: Vimeo train for calibration, all Xiph for test
        calib_tids = []  # unused — VimeoCalibrationReader handles its own sampling
        test_tids = shuffled[: args.n_test]
        print(
            f"Protocol: calib={args.n_calib} from Vimeo train, "
            f"test={len(test_tids)} from Xiph (all sequences), seed={args.seed}"
        )
    else:
        # Sequence-level disjoint split: calib and eval share no sequences
        calib_pool = [t for t in triplets if t.split("/")[0] in CALIB_SEQUENCES]
        test_pool = [t for t in triplets if t.split("/")[0] not in CALIB_SEQUENCES]
        random.shuffle(test_pool)
        calib_tids = random.sample(calib_pool, min(args.n_calib, len(calib_pool)))
        test_tids = test_pool[: args.n_test]
        calib_seqs_str = ", ".join(sorted(CALIB_SEQUENCES))
        print(
            f"Protocol: sequence-level disjoint split — "
            f"calib sequences = {{{calib_seqs_str}}} ({len(calib_tids)} samples from {len(calib_pool)} available), "
            f"test = {len(test_tids)} from remaining sequences, seed={args.seed}"
        )

    # Model registry: name -> (onnx filename, prep function)
    model_registry = {
        "rife_flow_360p": ("rife_flow_360p.onnx", prep_rife),
        "ifrnet_flow": ("ifrnet_flow_vimeo.onnx", prep_ifrnet),
    }

    all_rows: list[dict] = []

    for model_name in args.models:
        if model_name not in model_registry:
            print(f"\n[{model_name}] Unknown model, skipping.")
            continue
        onnx_filename, prep_fn = model_registry[model_name]
        onnx_path = args.onnx_dir / onnx_filename
        if not onnx_path.exists():
            print(f"\n[{model_name}] SKIP (not found: {onnx_path})")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{model_name}] Instrumented INT8 Analysis")
        print(f"{'=' * 60}")

        # Prepare test inputs
        test_inputs = []
        for tid in test_tids:
            try:
                i1, i3 = load_frames(seq_dir, tid)
                test_inputs.append(prep_fn(i1, i3))
            except Exception:
                pass
        if len(test_inputs) < 1:
            print(f"  No valid test inputs, skipping.")
            continue
        print(f"  Loaded {len(test_inputs)} test inputs")

        rows = analyze_model(
            model_name=model_name,
            onnx_path=onnx_path,
            prep_fn=prep_fn,
            test_inputs=test_inputs,
            calib_tids=calib_tids,
            seq_dir=seq_dir,
            cache_dir=cache_dir,
            vimeo_dir=args.vimeo_dir,
            n_calib=args.n_calib,
            seed=args.seed,
        )
        all_rows.extend(rows)

    # ── Save results ──
    csv_path = args.output_dir / "instrumented_cossim.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "quantized_ops", "cossim", "delta_vs_prev"])
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = args.output_dir / "instrumented_cossim.json"
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    # ── Print summary table ──
    print(f"\n{'=' * 70}")
    print(f"Instrumented INT8 CosSim (calib={args.n_calib}, test={args.n_test}, seed={args.seed})")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'Quantized Ops':<16} {'CosSim':>8} {'Delta':>8}")
    print("-" * 56)

    # Add FP32 baseline row for display
    seen_models = set()
    for row in all_rows:
        if row["model"] not in seen_models:
            seen_models.add(row["model"])
            print(f"{row['model']:<20} {'None (FP32)':<16} {'1.000':>8} {'':>8}")
        print(
            f"{'':<20} {row['quantized_ops']:<16} "
            f"{row['cossim']:8.3f} {row['delta_vs_prev']:+8.3f}"
        )

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
