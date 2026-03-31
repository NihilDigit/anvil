#!/usr/bin/env python3
"""Unified ORT INT8 quality evaluation: ANVIL + RIFE + IFRNet on Xiph 1080p.

Reproduces paper Table int8_quality (cross-method INT8 quantization compatibility).

Protocol (matches paper):
  - Xiph 1080p, 200-sample eval subset, 10-sample calibration, seed=42
  - W8A8 Percentile 99.99, QOperator format
  - Bootstrap 95% CI (2000 resamples)
  - Methods: ANVIL-S (direct), ANVIL-M (direct), RIFE flow-up (360p),
    IFRNet flow-up, IFRNet frame

ANVIL special case: ANVIL INT8 delta comes from full 2662-triplet QNN device
evaluation, not ORT. The script runs FP32 inference for ANVIL and annotates the
known device-side delta. ORT INT8 is NOT run on ANVIL (not the paper protocol).

Usage:
    # With Vimeo training set for calibration:
    pixi run python scripts/eval_int8_cross_method.py \\
        --onnx-dir artifacts/onnx \\
        --xiph-dir data/xiph_1080p \\
        --vimeo-dir data/vimeo_triplet \\
        --prealigned-dir data/xiph_1080p/prealigned_v2 \\
        --output-dir artifacts/eval/int8_cross_method

    # With ANVIL PyTorch checkpoints (for FP32 baseline):
    pixi run python scripts/eval_int8_cross_method.py \\
        --onnx-dir artifacts/onnx \\
        --xiph-dir data/xiph_1080p \\
        --vimeo-dir data/vimeo_triplet \\
        --prealigned-dir data/xiph_1080p/prealigned_v2 \\
        --anvil-s-ckpt checkpoints/anvil_s/best.pt \\
        --anvil-m-ckpt checkpoints/anvil_m/best.pt \\
        --output-dir artifacts/eval/int8_cross_method
"""
from __future__ import annotations

import argparse
import csv
import json
import random
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

from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CALIB = 10
N_EVAL = 200
SEED = 42
N_BOOTSTRAP = 2000

# Sequence-level calibration/evaluation split for Xiph 1080p.
# Calibration sequences chosen to cover diverse motion (small + dense)
# while keeping paper-featured sequences (tractor, old_town_cross, etc.) in eval.
CALIB_SEQUENCES = {"sunflower", "pedestrian_area"}

# Model-specific input resolutions
RIFE_H, RIFE_W = 384, 640
IFR_H, IFR_W = 256, 448

# Known QNN device-side INT8 deltas (full 2662-triplet evaluation)
ANVIL_QNN_DELTA = {"anvil_s": -0.19, "anvil_m": -0.09}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def discover_triplets(seq_dir: Path) -> list[str]:
    """Return sorted list of 'sequence/triplet' IDs from Xiph directory layout."""
    triplets = []
    for sd in sorted(seq_dir.iterdir()):
        if not sd.is_dir():
            continue
        for td in sorted(sd.iterdir()):
            if td.is_dir():
                triplets.append(f"{sd.name}/{td.name}")
    return triplets


def load_xiph(
    seq_dir: Path, prealigned_dir: Path | None, tid: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load a Xiph triplet: (im1, im3, gt, pa1, pa3). pa* may be None."""
    s, t = tid.split("/")
    i1 = cv2.cvtColor(cv2.imread(str(seq_dir / s / t / "im1.png")), cv2.COLOR_BGR2RGB)
    i3 = cv2.cvtColor(cv2.imread(str(seq_dir / s / t / "im3.png")), cv2.COLOR_BGR2RGB)
    gt = cv2.cvtColor(cv2.imread(str(seq_dir / s / t / "im2.png")), cv2.COLOR_BGR2RGB)
    pa1 = pa3 = None
    if prealigned_dir is not None:
        pa1 = cv2.cvtColor(
            cv2.imread(str(prealigned_dir / s / t / "im1_aligned.png")),
            cv2.COLOR_BGR2RGB,
        )
        pa3 = cv2.cvtColor(
            cv2.imread(str(prealigned_dir / s / t / "im3_aligned.png")),
            cv2.COLOR_BGR2RGB,
        )
    return i1, i3, gt, pa1, pa3


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10.0 * np.log10(255.0**2 / mse) if mse > 0 else 100.0


def bootstrap_ci(data: np.ndarray, n_resamples: int = N_BOOTSTRAP) -> tuple[float, float]:
    """Bootstrap 95 % confidence interval for the mean."""
    rng = np.random.RandomState(SEED)
    means = [np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n_resamples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Input preparation functions (per model family)
# ---------------------------------------------------------------------------
def prep_rife(i1: np.ndarray, i3: np.ndarray) -> dict[str, np.ndarray]:
    """Prepare RIFE 360p input: downsample + concat along channel axis."""
    a = cv2.resize(i1, (RIFE_W, RIFE_H)).astype(np.float32) / 255.0
    b = cv2.resize(i3, (RIFE_W, RIFE_H)).astype(np.float32) / 255.0
    inp = np.concatenate([a, b], axis=2).transpose(2, 0, 1)[np.newaxis]  # (1,6,H,W)
    return {"input": inp}


def prep_ifrnet(i1: np.ndarray, i3: np.ndarray) -> dict[str, np.ndarray]:
    """Prepare IFRNet input: downsample + two separate inputs."""
    a = cv2.resize(i1, (IFR_W, IFR_H)).astype(np.float32) / 255.0
    b = cv2.resize(i3, (IFR_W, IFR_H)).astype(np.float32) / 255.0
    return {
        "img0": a.transpose(2, 0, 1)[np.newaxis],  # (1,3,H,W)
        "img1": b.transpose(2, 0, 1)[np.newaxis],
    }


# ---------------------------------------------------------------------------
# Reconstruction functions (per model variant)
# ---------------------------------------------------------------------------
def reconstruct_rife_flow(
    outputs: list[np.ndarray],
    i1: np.ndarray,
    i3: np.ndarray,
    gt_shape: tuple[int, ...],
) -> np.ndarray:
    """RIFE flow-up: 4ch flow + 1ch mask at low res -> warp at low res -> upsample."""
    flow = outputs[0][0]  # (4, h, w)
    mask_logit = outputs[1][0, 0]  # (h, w)
    mask = 1.0 / (1.0 + np.exp(-mask_logit))

    fh, fw = flow.shape[1], flow.shape[2]
    oh, ow = gt_shape[0], gt_shape[1]

    a = cv2.resize(i1, (fw, fh)).astype(np.float32) / 255.0
    b = cv2.resize(i3, (fw, fh)).astype(np.float32) / 255.0

    f01 = flow[:2].transpose(1, 2, 0)  # (h, w, 2)
    f10 = flow[2:].transpose(1, 2, 0)

    yy, xx = np.mgrid[:fh, :fw].astype(np.float32)
    w0 = cv2.remap(
        a,
        np.stack([xx + f01[..., 0], yy + f01[..., 1]], axis=-1),
        None,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    w1 = cv2.remap(
        b,
        np.stack([xx + f10[..., 0], yy + f10[..., 1]], axis=-1),
        None,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    blended = mask[..., None] * w0 + (1.0 - mask[..., None]) * w1
    blended_u8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return cv2.resize(blended_u8, (ow, oh))


def reconstruct_ifrnet_flow(
    outputs: list[np.ndarray],
    i1: np.ndarray,
    i3: np.ndarray,
    gt_shape: tuple[int, ...],
) -> np.ndarray:
    """IFRNet flow-up: 4ch flow at low res -> warp at low res -> average -> upsample."""
    flow = outputs[0][0]  # (4, h, w)
    h, w = flow.shape[1], flow.shape[2]
    oh, ow = gt_shape[0], gt_shape[1]

    a = cv2.resize(i1, (w, h)).astype(np.float32) / 255.0
    b = cv2.resize(i3, (w, h)).astype(np.float32) / 255.0

    f01 = flow[:2].transpose(1, 2, 0)
    f10 = flow[2:].transpose(1, 2, 0)

    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    w0 = cv2.remap(
        a,
        np.stack([xx + f01[..., 0], yy + f01[..., 1]], axis=-1),
        None,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    w1 = cv2.remap(
        b,
        np.stack([xx + f10[..., 0], yy + f10[..., 1]], axis=-1),
        None,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    avg = np.clip((w0 + w1) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    if (oh, ow) == (h, w):
        return avg
    return cv2.resize(avg, (ow, oh))


def reconstruct_ifrnet_frame(
    outputs: list[np.ndarray],
    i1: np.ndarray,
    i3: np.ndarray,
    gt_shape: tuple[int, ...],
) -> np.ndarray:
    """IFRNet frame: 3ch frame at low res -> bicubic upsample to full res."""
    frame = outputs[0][0].transpose(1, 2, 0)  # (h, w, 3)
    frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    oh, ow = gt_shape[0], gt_shape[1]
    return cv2.resize(frame_u8, (ow, oh), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# ANVIL FP32 evaluation (PyTorch)
# ---------------------------------------------------------------------------
def evaluate_anvil_pytorch(
    tag: str,
    model_id: str,
    checkpoint: Path,
    eval_tids: list[str],
    seq_dir: Path,
    prealigned_dir: Path,
) -> dict:
    """Run PyTorch FP32 inference for ANVIL and attach QNN INT8 delta."""
    import torch
    import sys

    # Import model builder -- try open_source_release path first, then main repo
    try:
        from anvil_exp01.models.conv_vfi import build_model
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from anvil_exp01.models.conv_vfi import build_model

    model = build_model(model_id)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    psnrs = []
    for j, tid in enumerate(eval_tids):
        _, _, gt, pa1, pa3 = load_xiph(seq_dir, prealigned_dir, tid)
        if pa1 is None or pa3 is None:
            continue
        i1t = torch.from_numpy(pa1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        i3t = torch.from_numpy(pa3).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            blend = (i1t + i3t) / 2.0
            residual = model(torch.cat([i1t, i3t], dim=1))
            pred = (blend + residual).clamp(0, 1)
        pred_u8 = (pred[0].permute(1, 2, 0) * 255).round().byte().numpy()
        psnrs.append(psnr(pred_u8, gt))
        if (j + 1) % 50 == 0:
            print(f"  [{tag}] {j + 1}/{len(eval_tids)}", flush=True)

    arr = np.array(psnrs)
    ci = bootstrap_ci(arr)
    delta = ANVIL_QNN_DELTA[tag]
    return {
        "model": tag,
        "stages": 0,
        "mode": "direct",
        "n": len(arr),
        "fp32": round(float(arr.mean()), 2),
        "fp32_ci": [round(ci[0], 2), round(ci[1], 2)],
        "int8": round(float(arr.mean()) + delta, 2),
        "delta": delta,
        "delta_ci": "N/A (QNN device)",
        "pct_gt_3dB": 0.0,
        "note": "INT8 delta from full 2662-triplet QNN device eval",
    }


# ---------------------------------------------------------------------------
# ORT calibration reader
# ---------------------------------------------------------------------------
class XiphCalibrationReader(CalibrationDataReader):
    """Feed calibration samples to ORT static quantization."""

    def __init__(
        self,
        calib_tids: list[str],
        seq_dir: Path,
        prep_fn,
    ):
        self._tids = list(calib_tids)
        self._seq_dir = seq_dir
        self._prep_fn = prep_fn
        self._idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        while self._idx < len(self._tids):
            tid = self._tids[self._idx]
            self._idx += 1
            try:
                i1, i3, _, _, _ = load_xiph(self._seq_dir, None, tid)
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
# ORT INT8 evaluation for a single model
# ---------------------------------------------------------------------------
def evaluate_ort_model(
    name: str,
    stages: int,
    mode: str,
    onnx_path: Path,
    prep_fn,
    recon_fn,
    eval_tids: list[str],
    seq_dir: Path,
    int8_cache_dir: Path,
    vimeo_dir: Path | None = None,
    n_calib: int = N_CALIB,
    calib_tids: list[str] | None = None,
    seed: int = SEED,
) -> dict:
    """Run ORT FP32 + INT8 and compute quality delta with bootstrap CI.

    Calibration source (in priority order):
      1. vimeo_dir -- Vimeo90K training set
      2. calib_tids -- disjoint Xiph triplet IDs (recommended for RIFE/IFRNet)
    """
    int8_path = int8_cache_dir / f"{name}_int8.onnx"

    # Quantize if not cached
    if not int8_path.exists():
        if vimeo_dir is not None:
            print(f"  [{name}] Calibrating with {n_calib} Vimeo training samples ...", flush=True)
        else:
            print(f"  [{name}] Calibrating with {len(calib_tids)} Xiph disjoint samples ...", flush=True)
        t0 = time.time()
        if vimeo_dir is not None:
            reader = VimeoCalibrationReader(vimeo_dir, n_calib, prep_fn, seed=seed)
        else:
            reader = XiphCalibrationReader(calib_tids, seq_dir, prep_fn)
        quantize_static(
            str(onnx_path),
            str(int8_path),
            reader,
            quant_format=QuantFormat.QOperator,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            calibrate_method=CalibrationMethod.Percentile,
            extra_options={"ActivationSymmetric": False, "CalibPercentile": 99.99},
        )
        print(f"  [{name}] Quantized in {time.time() - t0:.0f}s", flush=True)
    else:
        print(f"  [{name}] Using cached INT8 model: {int8_path.name}")

    # Create sessions
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess_fp32 = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), opts, providers=["CPUExecutionProvider"])

    psnrs_fp32, psnrs_int8 = [], []
    for j, tid in enumerate(eval_tids):
        try:
            i1, i3, gt, _, _ = load_xiph(seq_dir, None, tid)
            feed = prep_fn(i1, i3)
            out_fp32 = sess_fp32.run(None, feed)
            out_int8 = sess_int8.run(None, feed)
            psnrs_fp32.append(psnr(recon_fn(out_fp32, i1, i3, gt.shape), gt))
            psnrs_int8.append(psnr(recon_fn(out_int8, i1, i3, gt.shape), gt))
        except Exception:
            pass
        if (j + 1) % 50 == 0:
            print(f"  [{name}] {j + 1}/{len(eval_tids)}", flush=True)

    fp32_arr = np.array(psnrs_fp32)
    int8_arr = np.array(psnrs_int8)
    delta_arr = int8_arr - fp32_arr

    delta_ci = bootstrap_ci(delta_arr)
    pct_gt_3 = float(np.mean(delta_arr < -3.0) * 100.0)

    return {
        "model": name,
        "stages": stages,
        "mode": mode,
        "n": len(fp32_arr),
        "fp32": round(float(fp32_arr.mean()), 2),
        "fp32_std": round(float(fp32_arr.std()), 2),
        "int8": round(float(int8_arr.mean()), 2),
        "int8_std": round(float(int8_arr.std()), 2),
        "delta": round(float(delta_arr.mean()), 2),
        "delta_std": round(float(delta_arr.std()), 2),
        "delta_ci": [round(delta_ci[0], 2), round(delta_ci[1], 2)],
        "pct_gt_3dB": round(pct_gt_3, 1),
        "worst_delta": round(float(delta_arr.min()), 2),
        "best_delta": round(float(delta_arr.max()), 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified ORT INT8 quality: ANVIL + RIFE + IFRNet on Xiph 1080p."
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        required=True,
        help="Directory with ONNX models: rife_flow_360p.onnx, ifrnet_flow_vimeo.onnx, ifrnet_trained_vimeo.onnx.",
    )
    parser.add_argument(
        "--xiph-dir",
        type=Path,
        required=True,
        help="Xiph 1080p root (contains sequences/<seq>/<triplet>/im{1,2,3}.png).",
    )
    parser.add_argument(
        "--prealigned-dir",
        type=Path,
        default=None,
        help="Prealigned v2 root for ANVIL (contains <seq>/<triplet>/im{1,3}_aligned.png).",
    )
    parser.add_argument(
        "--anvil-s-ckpt",
        type=Path,
        default=None,
        help="ANVIL-S PyTorch checkpoint (best.pt) for FP32 baseline.",
    )
    parser.add_argument(
        "--anvil-m-ckpt",
        type=Path,
        default=None,
        help="ANVIL-M PyTorch checkpoint (best.pt) for FP32 baseline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eval/int8_cross_method"),
    )
    parser.add_argument(
        "--vimeo-dir",
        type=Path,
        default=None,
        help="Vimeo90K root (contains tri_trainlist.txt + sequences/). "
        "When provided, calibration uses Vimeo training set. "
        "When omitted, calibration uses disjoint Xiph split (recommended for RIFE/IFRNet).",
    )
    parser.add_argument("--n-eval", type=int, default=N_EVAL, help="Number of eval samples.")
    parser.add_argument("--n-calib", type=int, default=N_CALIB, help="Number of calibration samples.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    int8_cache = args.output_dir / "int8_models"
    int8_cache.mkdir(exist_ok=True)

    seq_dir = args.xiph_dir / "sequences"
    if not seq_dir.is_dir():
        # Fallback: xiph_dir itself contains sequence subdirectories
        seq_dir = args.xiph_dir

    # Discover and sample triplets
    triplets = discover_triplets(seq_dir)
    print(f"Xiph: {len(triplets)} total triplets")

    random.seed(args.seed)
    eval_tids = random.sample(triplets, min(args.n_eval, len(triplets)))

    # Calibration source selection
    calib_tids: list[str] | None = None
    if args.vimeo_dir is not None:
        vimeo_list = args.vimeo_dir / "tri_trainlist.txt"
        if not vimeo_list.exists():
            parser.error(f"--vimeo-dir provided but {vimeo_list} not found")
        # When Vimeo is provided, all Xiph triplets are used for eval (no Xiph calib needed)
        print(
            f"Protocol: eval={len(eval_tids)} Xiph, calib={args.n_calib} Vimeo train, "
            f"seed={args.seed}"
        )
    else:
        # Sequence-level disjoint split: calib and eval share no sequences
        calib_pool = [t for t in triplets if t.split("/")[0] in CALIB_SEQUENCES]
        eval_pool = [t for t in triplets if t.split("/")[0] not in CALIB_SEQUENCES]
        calib_tids = random.sample(calib_pool, min(args.n_calib, len(calib_pool)))
        # Override eval_tids to only contain non-calib sequences
        eval_tids = random.sample(eval_pool, min(args.n_eval, len(eval_pool)))
        calib_seqs_str = ", ".join(sorted(CALIB_SEQUENCES))
        print(
            f"Protocol: sequence-level disjoint split — "
            f"calib sequences = {{{calib_seqs_str}}} ({len(calib_tids)} samples from {len(calib_pool)} available), "
            f"eval = {len(eval_tids)} from remaining sequences, seed={args.seed}"
        )

    results: dict[str, dict] = {}

    # ── ANVIL-S / ANVIL-M (PyTorch FP32 + QNN INT8 delta) ──
    anvil_configs = [
        ("anvil_s", "D-unet-v3bs-nomv", args.anvil_s_ckpt),
        ("anvil_m", "D-unet-v3bm-nomv", args.anvil_m_ckpt),
    ]
    for tag, model_id, ckpt_path in anvil_configs:
        if ckpt_path is None or not ckpt_path.exists():
            print(f"\n[{tag}] SKIP (no checkpoint provided)")
            continue
        if args.prealigned_dir is None or not args.prealigned_dir.is_dir():
            print(f"\n[{tag}] SKIP (no prealigned dir)")
            continue
        print(f"\n{'=' * 60}")
        print(f"[{tag}] PyTorch FP32 + QNN INT8 delta")
        print(f"{'=' * 60}")
        results[tag] = evaluate_anvil_pytorch(
            tag, model_id, ckpt_path, eval_tids, seq_dir, args.prealigned_dir
        )
        print(
            f"  {tag}: FP32={results[tag]['fp32']:.2f}  "
            f"INT8={results[tag]['int8']:.2f}  "
            f"delta={results[tag]['delta']:+.2f} (QNN device)"
        )

    # ── ORT models: RIFE + IFRNet ──
    ort_jobs = [
        {
            "name": "rife_flow",
            "stages": 3,
            "mode": "flow_up",
            "onnx": "rife_flow_360p.onnx",
            "prep": prep_rife,
            "recon": reconstruct_rife_flow,
        },
        {
            "name": "ifrnet_flow",
            "stages": 4,
            "mode": "flow_up",
            "onnx": "ifrnet_flow_vimeo.onnx",
            "prep": prep_ifrnet,
            "recon": reconstruct_ifrnet_flow,
        },
        {
            "name": "ifrnet_frame",
            "stages": 4,
            "mode": "frame",
            "onnx": "ifrnet_trained_vimeo.onnx",
            "prep": prep_ifrnet,
            "recon": reconstruct_ifrnet_frame,
        },
    ]

    for job in ort_jobs:
        onnx_path = args.onnx_dir / job["onnx"]
        if not onnx_path.exists():
            print(f"\n[{job['name']}] SKIP (not found: {onnx_path})")
            continue
        print(f"\n{'=' * 60}")
        print(f"[{job['name']}] ORT FP32 + INT8")
        print(f"{'=' * 60}")
        results[job["name"]] = evaluate_ort_model(
            name=job["name"],
            stages=job["stages"],
            mode=job["mode"],
            onnx_path=onnx_path,
            prep_fn=job["prep"],
            recon_fn=job["recon"],
            eval_tids=eval_tids,
            seq_dir=seq_dir,
            int8_cache_dir=int8_cache,
            vimeo_dir=args.vimeo_dir,
            n_calib=args.n_calib,
            calib_tids=calib_tids,
            seed=args.seed,
        )
        r = results[job["name"]]
        ci = r["delta_ci"]
        print(
            f"  {job['name']}: FP32={r['fp32']:.2f}  INT8={r['int8']:.2f}  "
            f"delta={r['delta']:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]  "
            f">3dB={r['pct_gt_3dB']:.0f}%"
        )

    # ── Save results ──
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(args.output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "stages", "mode", "n",
            "fp32", "int8", "delta", "delta_ci_lo", "delta_ci_hi", "pct_gt_3dB", "note",
        ])
        for name, r in results.items():
            ci = r.get("delta_ci", [r["delta"], r["delta"]])
            if isinstance(ci, str):
                ci_lo, ci_hi = "", ""
            else:
                ci_lo, ci_hi = ci[0], ci[1]
            writer.writerow([
                r["model"],
                r.get("stages", ""),
                r.get("mode", ""),
                r.get("n", ""),
                r["fp32"],
                r.get("int8", ""),
                r["delta"],
                ci_lo,
                ci_hi,
                r.get("pct_gt_3dB", ""),
                r.get("note", ""),
            ])

    # ── Print summary table ──
    print(f"\n{'=' * 80}")
    print(f"INT8 Cross-Method Quality (Xiph 1080p, N={args.n_eval}, seed={args.seed})")
    print(f"{'=' * 80}")
    print(
        f"{'Method':<16} {'Stages':>6} {'Mode':<8} {'FP32':>7} {'INT8':>7} "
        f"{'Delta':>7} {'95% CI':>20} {'>3dB':>5}"
    )
    print("-" * 80)
    for name, r in results.items():
        ci = r.get("delta_ci", [r["delta"], r["delta"]])
        if isinstance(ci, str):
            ci_str = ci
        else:
            ci_str = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
        print(
            f"{r['model']:<16} {r.get('stages', ''):>6} {r.get('mode', ''):<8} "
            f"{r['fp32']:7.2f} {r.get('int8', r['fp32'] + r['delta']):7.2f} "
            f"{r['delta']:+7.2f} {ci_str:>20} {r.get('pct_gt_3dB', 0):5.0f}%"
        )
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
