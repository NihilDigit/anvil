"""Reproduce Table: RIFE resolution reduction strategies (paper Table rife_reduced).

Evaluates RIFE HDv3 at 360p and 480p with flow-upsample and frame-upsample modes,
measuring FP32 quality and ORT INT8 quality degradation on Xiph 1080p.

This script requires pre-exported RIFE ONNX models.  See --help for paths.
To export RIFE to ONNX, clone the official RIFE repo and use the provided export
instructions, or use the flow/frame split ONNX files from the ANVIL artifact bundle.

Expected ONNX models:
  - rife_flow_360p.onnx   : 6ch input (384x640),  outputs flow(4ch) + mask(1ch)
  - rife_frame_360p.onnx  : 6ch input (384x640),  outputs frame(3ch)
  - rife_flow_480p.onnx   : 6ch input (512x864),  outputs flow(4ch) + mask(1ch)
  - rife_frame_480p.onnx  : 6ch input (512x864),  outputs frame(3ch)

Usage:
    # Paper protocol: calibrate from Vimeo training set (recommended)
    pixi run python scripts/eval_rife_reduced_res.py \\
        --onnx-dir artifacts/onnx/rife \\
        --xiph-dir data/xiph_1080p \\
        --vimeo-dir data/vimeo_triplet \\
        --output-dir artifacts/eval/rife_reduced

    # Quick subset for development:
    pixi run python scripts/eval_rife_reduced_res.py \\
        --onnx-dir artifacts/onnx/rife \\
        --xiph-dir data/xiph_1080p \\
        --vimeo-dir data/vimeo_triplet \\
        --limit 50
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

RESOLUTIONS = {
    "360p": (384, 640),
    "480p": (512, 864),
}

# Sequence-level calibration/evaluation split for Xiph 1080p.
# Calibration sequences chosen to cover diverse motion (small + dense)
# while keeping paper-featured sequences (tractor, old_town_cross, etc.) in eval.
CALIB_SEQUENCES = {"sunflower", "pedestrian_area"}

SEED = 42
N_CALIB = 10


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def discover_triplets(xiph_dir: Path) -> list[str]:
    """Return sorted list of 'sequence/triplet_id' strings."""
    seq_root = xiph_dir / "sequences"
    triplets = []
    for seq_dir in sorted(seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        for tid_dir in sorted(seq_dir.iterdir()):
            if tid_dir.is_dir() and (tid_dir / "im2.png").exists():
                triplets.append(f"{seq_dir.name}/{tid_dir.name}")
    return triplets


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(10.0 * np.log10(255.0**2 / mse))


def ssim_channel(a: np.ndarray, b: np.ndarray) -> float:
    """Compute SSIM for a single-channel pair (uint8)."""
    C1, C2 = 6.5025, 58.5225  # (0.01*255)^2, (0.03*255)^2
    af, bf = a.astype(np.float64), b.astype(np.float64)
    mu_a = cv2.GaussianBlur(af, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(bf, (11, 11), 1.5)
    sigma_a2 = cv2.GaussianBlur(af * af, (11, 11), 1.5) - mu_a * mu_a
    sigma_b2 = cv2.GaussianBlur(bf * bf, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(af * bf, (11, 11), 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sigma_a2 + sigma_b2 + C2)
    ssim_map = num / den
    return float(ssim_map.mean())


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean SSIM over RGB channels."""
    return float(np.mean([ssim_channel(pred[:, :, c], gt[:, :, c]) for c in range(3)]))


def bootstrap_ci(data: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.RandomState(SEED)
    means = [np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# RIFE input preparation
# ---------------------------------------------------------------------------

def prepare_rife_input(
    img0: np.ndarray,
    img1: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Downsample two frames, concatenate to 6ch NCHW float32 in [0,1]."""
    h, w = target_hw
    lo0 = cv2.resize(img0, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    lo1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    # (H, W, 3) + (H, W, 3) -> (H, W, 6) -> (1, 6, H, W)
    return np.concatenate([lo0, lo1], axis=2).transpose(2, 0, 1)[np.newaxis]


# ---------------------------------------------------------------------------
# RIFE postprocessing
# ---------------------------------------------------------------------------

def postprocess_flow_up(
    flow_nchw: np.ndarray,
    mask_nchw: np.ndarray,
    img0_u8: np.ndarray,
    img1_u8: np.ndarray,
) -> np.ndarray:
    """Flow+mask at low res -> bilinear upsample flow -> CPU warp at full res -> blend."""
    src_h, src_w = flow_nchw.shape[2], flow_nchw.shape[3]
    tgt_h, tgt_w = img0_u8.shape[0], img0_u8.shape[1]
    scale_y, scale_x = tgt_h / src_h, tgt_w / src_w

    flow = flow_nchw[0]  # (4, src_h, src_w)

    def _up(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(arr, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)

    flow_01_x = _up(flow[0]) * scale_x
    flow_01_y = _up(flow[1]) * scale_y
    flow_10_x = _up(flow[2]) * scale_x
    flow_10_y = _up(flow[3]) * scale_y

    mask = mask_nchw[0, 0]
    mask_full = cv2.resize(mask, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)

    gy, gx = np.mgrid[0:tgt_h, 0:tgt_w].astype(np.float32)

    warped0 = cv2.remap(
        img0_u8,
        gx + flow_01_x.astype(np.float32),
        gy + flow_01_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped1 = cv2.remap(
        img1_u8,
        gx + flow_10_x.astype(np.float32),
        gy + flow_10_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    mask_3d = mask_full[:, :, np.newaxis].astype(np.float32)
    blended = mask_3d * warped0.astype(np.float32) + (1.0 - mask_3d) * warped1.astype(np.float32)
    return np.clip(blended + 0.5, 0.0, 255.0).astype(np.uint8)


def postprocess_frame_up(
    output_nchw: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Frame output at low res -> bicubic upsample to full res."""
    frame_hwc = output_nchw[0].transpose(1, 2, 0)  # CHW -> HWC
    frame_hwc = np.clip(frame_hwc, 0.0, 1.0)
    tgt_h, tgt_w = target_hw
    frame_up = cv2.resize(frame_hwc, (tgt_w, tgt_h), interpolation=cv2.INTER_CUBIC)
    return np.clip(frame_up * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# ORT INT8 quantization
# ---------------------------------------------------------------------------

class VimeoCalibReader(CalibrationDataReader):
    """Feed calibration samples from Vimeo90K training set."""

    def __init__(
        self,
        vimeo_dir: Path,
        n_samples: int,
        target_hw: tuple[int, int],
        seed: int = 42,
    ) -> None:
        train_list = vimeo_dir / "tri_trainlist.txt"
        triplets = [l.strip() for l in train_list.read_text().splitlines() if l.strip()]
        rng = random.Random(seed)
        self._tids = rng.sample(triplets, min(n_samples, len(triplets)))
        self._seq_dir = vimeo_dir / "sequences"
        self._hw = target_hw
        self._idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        while self._idx < len(self._tids):
            tid = self._tids[self._idx]
            self._idx += 1
            td = self._seq_dir / tid
            try:
                img0 = load_image_rgb(td / "im1.png")
                img1 = load_image_rgb(td / "im3.png")
                inp = prepare_rife_input(img0, img1, self._hw)
                return {"input": inp}
            except Exception:
                continue
        return None


def quantize_model(
    onnx_path: Path,
    int8_path: Path,
    xiph_dir: Path,
    calib_tids: list[str],
    target_hw: tuple[int, int],
    vimeo_dir: Path | None = None,
    n_calib: int = N_CALIB,
    seed: int = SEED,
) -> Path:
    """Static W8A8 quantization with Percentile 99.99 calibration.

    When *vimeo_dir* is provided, calibration samples come from the Vimeo90K
    training set.  Otherwise uses *calib_tids* drawn from a disjoint Xiph
    split (preferred for RIFE/IFRNet -- matches deployment at 1080p).
    """
    if int8_path.exists():
        print(f"    INT8 model cached: {int8_path.name}")
        return int8_path

    seq_root = xiph_dir / "sequences"

    class XiphCalibReader(CalibrationDataReader):
        def __init__(self) -> None:
            self._tids = list(calib_tids)
            self._idx = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            while self._idx < len(self._tids):
                tid = self._tids[self._idx]
                self._idx += 1
                seq, frame_id = tid.split("/")
                td = seq_root / seq / frame_id
                try:
                    img0 = load_image_rgb(td / "im1.png")
                    img1 = load_image_rgb(td / "im3.png")
                    inp = prepare_rife_input(img0, img1, target_hw)
                    return {"input": inp}
                except Exception:
                    continue
            return None

    if vimeo_dir is not None:
        calib_reader = VimeoCalibReader(vimeo_dir, n_calib, target_hw, seed=seed)
        calib_source = f"Vimeo train ({n_calib} samples)"
    else:
        calib_reader = XiphCalibReader()
        calib_source = f"Xiph disjoint ({len(calib_tids)} samples)"

    print(f"    Calibrating from {calib_source}...", flush=True)
    t0 = time.time()
    quantize_static(
        str(onnx_path),
        str(int8_path),
        calib_reader,
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"ActivationSymmetric": False, "CalibPercentile": 99.99},
    )
    print(f"    Calibrated in {time.time() - t0:.0f}s", flush=True)
    return int8_path


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_variant(
    *,
    name: str,
    onnx_path: Path,
    mode: str,
    res_name: str,
    target_hw: tuple[int, int],
    xiph_dir: Path,
    eval_tids: list[str],
    calib_tids: list[str],
    int8_cache_dir: Path,
    vimeo_dir: Path | None = None,
    n_calib: int = N_CALIB,
    seed: int = SEED,
) -> dict:
    """Run FP32 + INT8 evaluation for one RIFE variant. Returns summary dict."""
    print(f"\n  [{name}] {mode} @ {res_name} ({target_hw[0]}x{target_hw[1]})")
    seq_root = xiph_dir / "sequences"

    if not onnx_path.exists():
        print(f"    ONNX not found: {onnx_path}")
        return {"name": name, "error": "ONNX not found"}

    # INT8 model
    int8_path = int8_cache_dir / f"{onnx_path.stem}_int8.onnx"
    quantize_model(onnx_path, int8_path, xiph_dir, calib_tids, target_hw,
                   vimeo_dir=vimeo_dir, n_calib=n_calib, seed=seed)

    # Sessions
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess_fp32 = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), opts, providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name

    psnr_fp32, psnr_int8, ssim_fp32 = [], [], []

    for i, tid in enumerate(eval_tids):
        seq, frame_id = tid.split("/")
        td = seq_root / seq / frame_id
        try:
            img0 = load_image_rgb(td / "im1.png")
            img1 = load_image_rgb(td / "im3.png")
            gt = load_image_rgb(td / "im2.png")
        except FileNotFoundError:
            continue

        inp = prepare_rife_input(img0, img1, target_hw)
        feed = {input_name: inp}

        # FP32
        out_fp32 = sess_fp32.run(None, feed)
        if mode == "flow":
            pred_fp32 = postprocess_flow_up(out_fp32[0], out_fp32[1], img0, img1)
        else:
            pred_fp32 = postprocess_frame_up(out_fp32[0], gt.shape[:2])

        # INT8
        out_int8 = sess_int8.run(None, feed)
        if mode == "flow":
            pred_int8 = postprocess_flow_up(out_int8[0], out_int8[1], img0, img1)
        else:
            pred_int8 = postprocess_frame_up(out_int8[0], gt.shape[:2])

        psnr_fp32.append(psnr(pred_fp32, gt))
        psnr_int8.append(psnr(pred_int8, gt))
        ssim_fp32.append(compute_ssim(pred_fp32, gt))

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(eval_tids)}", flush=True)

    fp32_arr = np.array(psnr_fp32)
    int8_arr = np.array(psnr_int8)
    delta_arr = int8_arr - fp32_arr
    ssim_arr = np.array(ssim_fp32)

    delta_ci = bootstrap_ci(delta_arr)

    result = {
        "name": name,
        "resolution": res_name,
        "mode": mode,
        "n_eval": len(fp32_arr),
        "fp32_psnr": round(float(fp32_arr.mean()), 2),
        "fp32_ssim": round(float(ssim_arr.mean()), 4),
        "int8_psnr": round(float(int8_arr.mean()), 2),
        "int8_delta": round(float(delta_arr.mean()), 2),
        "int8_delta_ci": [round(delta_ci[0], 2), round(delta_ci[1], 2)],
        "pct_gt_3dB": round(float(np.mean(delta_arr < -3) * 100), 1),
    }

    print(
        f"    FP32: {result['fp32_psnr']:.2f} dB  "
        f"INT8: {result['int8_psnr']:.2f} dB  "
        f"Δ: {result['int8_delta']:+.2f} dB  "
        f"CI: [{delta_ci[0]:+.2f}, {delta_ci[1]:+.2f}]"
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RIFE resolution reduction strategies (Table rife_reduced).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--onnx-dir", type=Path, required=True,
        help="Directory containing RIFE ONNX models (rife_flow_360p.onnx, etc.).",
    )
    parser.add_argument(
        "--xiph-dir", type=Path, required=True,
        help="Xiph 1080p root directory (containing sequences/).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/eval/rife_reduced"),
        help="Output directory for results.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit evaluation to N triplets (0 = all, default: all).",
    )
    parser.add_argument(
        "--n-calib", type=int, default=N_CALIB,
        help=f"Number of calibration samples for INT8 (default: {N_CALIB}).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed for sampling (default: {SEED}).",
    )
    parser.add_argument(
        "--vimeo-dir", type=Path, default=None,
        help="Vimeo90K root (contains sequences/ and tri_trainlist.txt). "
             "When provided, calibration uses Vimeo training set. "
             "When omitted, calibration uses disjoint Xiph split (recommended for RIFE).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    int8_cache = args.output_dir / "int8_models"
    int8_cache.mkdir(exist_ok=True)

    # Discover triplets
    triplets = discover_triplets(args.xiph_dir)
    print(f"Xiph 1080p: {len(triplets)} triplets found")

    random.seed(args.seed)
    if args.limit > 0:
        eval_tids = random.sample(triplets, min(args.limit, len(triplets)))
    else:
        eval_tids = triplets

    # Calibration source: Vimeo train or sequence-level disjoint Xiph split
    if args.vimeo_dir is not None:
        vimeo_seq = args.vimeo_dir / "sequences"
        train_list = args.vimeo_dir / "tri_trainlist.txt"
        if not vimeo_seq.is_dir() or not train_list.exists():
            parser.error(f"--vimeo-dir must contain sequences/ and tri_trainlist.txt: {args.vimeo_dir}")
        calib_tids = []  # placeholder; VimeoCalibReader handles its own sampling
        calib_label = f"Vimeo train ({args.n_calib} samples)"
        # When Vimeo is provided, all Xiph triplets are used for eval (no Xiph calib needed)
    else:
        # Sequence-level disjoint split: calib and eval share no sequences
        calib_pool = [t for t in triplets if t.split("/")[0] in CALIB_SEQUENCES]
        eval_tids_filtered = [t for t in triplets if t.split("/")[0] not in CALIB_SEQUENCES]
        calib_tids = random.sample(calib_pool, min(args.n_calib, len(calib_pool)))
        # Override eval_tids to only contain non-calib sequences
        if args.limit > 0:
            eval_tids = random.sample(eval_tids_filtered, min(args.limit, len(eval_tids_filtered)))
        else:
            eval_tids = eval_tids_filtered
        calib_seqs_str = ", ".join(sorted(CALIB_SEQUENCES))
        print(
            f"Sequence-level disjoint split: calib sequences = {{{calib_seqs_str}}} "
            f"({len(calib_tids)} calib samples from {len(calib_pool)} available), "
            f"eval = {len(eval_tids)} triplets from remaining sequences.",
            flush=True,
        )
        calib_label = f"Xiph sequence-level disjoint ({len(calib_tids)} samples from {{{calib_seqs_str}}})"

    print(f"Eval: {len(eval_tids)} triplets, Calib: {calib_label}, Seed: {args.seed}")

    # Define ONNX model variants
    variants = []
    for res_name, hw in RESOLUTIONS.items():
        for mode, suffix in [("flow", f"rife_flow_{res_name}.onnx"), ("frame", f"rife_frame_{res_name}.onnx")]:
            onnx_path = args.onnx_dir / suffix
            variants.append({
                "name": f"RIFE-{res_name}-{mode}-up",
                "onnx_path": onnx_path,
                "mode": mode,
                "res_name": res_name,
                "target_hw": hw,
            })

    # Run evaluation
    print(f"\n{'=' * 70}")
    print("RIFE Resolution Reduction Strategies — Xiph 1080p")
    print(f"{'=' * 70}")

    results = []
    for v in variants:
        result = evaluate_variant(
            name=v["name"],
            onnx_path=v["onnx_path"],
            mode=v["mode"],
            res_name=v["res_name"],
            target_hw=v["target_hw"],
            xiph_dir=args.xiph_dir,
            eval_tids=eval_tids,
            calib_tids=calib_tids,
            int8_cache_dir=int8_cache,
            vimeo_dir=args.vimeo_dir,
            n_calib=args.n_calib,
            seed=args.seed,
        )
        results.append(result)

    # Write summary
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = args.output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "resolution", "mode", "n_eval",
            "fp32_psnr", "fp32_ssim", "int8_psnr", "int8_delta",
            "int8_delta_ci", "pct_gt_3dB",
        ])
        writer.writeheader()
        for r in results:
            if "error" not in r:
                row = dict(r)
                row["int8_delta_ci"] = str(row["int8_delta_ci"])
                writer.writerow(row)

    # Print summary table (paper Table rife_reduced format)
    print(f"\n{'=' * 70}")
    print("Table: RIFE Resolution Reduction Strategies on Xiph 1080p")
    print(f"{'=' * 70}")
    print(f"{'Res.':<6} {'Mode':<10} {'FP32':>6} {'INT8 Δ':>8} {'CI 95%':>20} {'>3dB%':>6}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r.get('resolution', '?'):<6} {r.get('mode', '?'):<10} {'ERROR':>6}")
            continue
        ci = r["int8_delta_ci"]
        print(
            f"{r['resolution']:<6} {r['mode'] + '↑':<10} "
            f"{r['fp32_psnr']:>6.2f} "
            f"{r['int8_delta']:>+8.2f} "
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}]      "
            f"{r['pct_gt_3dB']:>5.1f}%"
        )

    print(f"\nResults saved to: {args.output_dir}")
    print(
        "\nNote: INT8 latency on HTP V75 (from device benchmarks, not reproduced here):\n"
        "  360p flow↑: 14.7ms   360p frame↑: 17.8ms\n"
        "  480p flow↑: 28.2ms   480p frame↑: 36.3ms (exceeds 33.3ms deadline)\n"
        "  All FP16 latencies exceed 33.3ms deadline (47.4–99.8ms)."
    )


if __name__ == "__main__":
    main()
