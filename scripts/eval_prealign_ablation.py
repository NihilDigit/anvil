"""Prealignment method ablation: generate + evaluate all 6 methods.

Runs prealign_v2.py for each method, then evaluates MV Blend PSNR/SSIM
against GT on the Vimeo90K test set. Produces a summary CSV for the paper's
prealignment ablation table.

Prerequisites:
    - Vimeo90K downloaded (pixi run download-vimeo)
    - MV extracted (pixi run extract-mv)
    - Dense flow generated (pixi run mv-to-dense)

Usage:
    pixi run python scripts/eval_prealign_ablation.py \
        --data-dir data/vimeo_triplet \
        --mv-dir data/vimeo_triplet/mv_cache \
        --dense-flow-dir data/vimeo_triplet/dense_flow \
        --output-dir artifacts/eval/prealign_ablation
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# All methods implemented in prealign_v2.py
METHODS = [
    "med_gauss",
    "block_avg",
    "block_subpel",
    "obmc",
    "aobmc",
    "daala",
]


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 60.0
    return float(10.0 * np.log10(255.0 * 255.0 / mse))


def ssim_simple(pred: np.ndarray, gt: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    pred_f = pred.astype(np.float64)
    gt_f = gt.astype(np.float64)
    mu_p = cv2.GaussianBlur(pred_f, (11, 11), 1.5)
    mu_g = cv2.GaussianBlur(gt_f, (11, 11), 1.5)
    sigma_p_sq = cv2.GaussianBlur(pred_f * pred_f, (11, 11), 1.5) - mu_p * mu_p
    sigma_g_sq = cv2.GaussianBlur(gt_f * gt_f, (11, 11), 1.5) - mu_g * mu_g
    sigma_pg = cv2.GaussianBlur(pred_f * gt_f, (11, 11), 1.5) - mu_p * mu_g
    ssim_map = ((2 * mu_p * mu_g + C1) * (2 * sigma_pg + C2)) / (
        (mu_p * mu_p + mu_g * mu_g + C1) * (sigma_p_sq + sigma_g_sq + C2)
    )
    return float(ssim_map.mean())


def read_split_list(data_dir: Path, split: str) -> list[str]:
    filename = {"train": "tri_trainlist.txt", "test": "tri_testlist.txt"}[split]
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Split list not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def run_prealign(
    data_dir: Path,
    mv_dir: Path,
    output_dir: Path,
    method: str,
    split: str = "test",
) -> Path:
    """Run prealign_v2.py for a single method. Returns the output directory."""
    method_dir = output_dir / f"prealigned_{method}"
    cmd = [
        sys.executable, "-m", "anvil_exp01.data.prealign_v2",
        "--data-dir", str(data_dir),
        "--mv-dir", str(mv_dir),
        "--output-dir", str(method_dir),
        "--method", method,
        "--split", split,
    ]
    print(f"\n{'='*60}")
    print(f"Prealigning: {method} -> {method_dir}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)
    return method_dir


def evaluate_methods(
    data_dir: Path,
    method_dirs: dict[str, Path],
    split: str = "test",
) -> list[dict]:
    """Evaluate MV Blend PSNR/SSIM for each prealigned method."""
    triplet_ids = read_split_list(data_dir, split)
    print(f"\nEvaluating {len(method_dirs)} methods on {len(triplet_ids)} triplets")

    # Also evaluate naive blend and v1 if available
    results: dict[str, list[tuple[float, float]]] = {}
    for name in ["naive_blend"] + list(method_dirs.keys()):
        results[name] = []

    skipped: dict[str, int] = {name: 0 for name in method_dirs}

    for tid in tqdm(triplet_ids, desc="Evaluating", unit="triplet"):
        seq_id, trip_id = tid.split("/")
        gt_path = data_dir / "sequences" / seq_id / trip_id / "im2.png"
        if not gt_path.exists():
            continue
        gt = cv2.imread(str(gt_path))

        # Naive blend
        i0 = cv2.imread(str(data_dir / "sequences" / seq_id / trip_id / "im1.png"))
        i1 = cv2.imread(str(data_dir / "sequences" / seq_id / trip_id / "im3.png"))
        if i0 is not None and i1 is not None:
            blend = np.round(
                (i0.astype(np.float64) + i1.astype(np.float64)) / 2.0
            ).clip(0, 255).astype(np.uint8)
            results["naive_blend"].append((psnr(blend, gt), ssim_simple(blend, gt)))

        # Each prealign method
        for name, pre_dir in method_dirs.items():
            a0_path = pre_dir / seq_id / trip_id / "im1_aligned.png"
            a1_path = pre_dir / seq_id / trip_id / "im3_aligned.png"
            if not a0_path.exists() or not a1_path.exists():
                skipped[name] += 1
                continue
            a0 = cv2.imread(str(a0_path))
            a1 = cv2.imread(str(a1_path))
            blend = np.round(
                (a0.astype(np.float64) + a1.astype(np.float64)) / 2.0
            ).clip(0, 255).astype(np.uint8)
            results[name].append((psnr(blend, gt), ssim_simple(blend, gt)))

    # Compute summaries
    rows = []
    baseline_psnr = None
    print(f"\n{'='*70}")
    print(f"{'Method':<20} {'N':>6} {'PSNR':>8} {'SSIM':>8} {'Δ PSNR':>8}")
    print(f"{'='*70}")

    for name in ["naive_blend"] + list(method_dirs.keys()):
        vals = results[name]
        if not vals:
            continue
        mean_psnr = np.mean([v[0] for v in vals])
        mean_ssim = np.mean([v[1] for v in vals])
        if baseline_psnr is None:
            baseline_psnr = mean_psnr
        delta = mean_psnr - baseline_psnr
        delta_str = f"{delta:+.2f}" if name != "naive_blend" else "base"
        print(f"{name:<20} {len(vals):>6} {mean_psnr:>8.2f} {mean_ssim:>8.4f} {delta_str:>8}")
        rows.append({
            "method": name,
            "n": len(vals),
            "psnr": f"{mean_psnr:.4f}",
            "ssim": f"{mean_ssim:.5f}",
            "delta_psnr": f"{delta:.4f}",
        })
        if skipped.get(name, 0) > 0:
            print(f"  (skipped {skipped[name]} triplets with missing files)")

    print(f"{'='*70}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Vimeo90K root (containing sequences/ and tri_testlist.txt)")
    parser.add_argument("--mv-dir", type=Path, required=True,
                        help="MV cache directory (output of extract_mv)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/prealign_ablation"))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS,
                        help="Methods to evaluate (default: all 6)")
    parser.add_argument("--skip-prealign", action="store_true",
                        help="Skip prealign step, only run evaluation (dirs must exist)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate prealigned frames for each method
    # prealign_v2.py creates a prealigned_{method}/ subdirectory inside --output-dir,
    # so we pass the parent and point evaluation at the nested path.
    method_dirs: dict[str, Path] = {}
    for method in args.methods:
        parent_dir = args.output_dir / f"prealigned_{method}"
        if not args.skip_prealign:
            run_prealign(args.data_dir, args.mv_dir, parent_dir, method, args.split)
        # prealign_v2.py nests output as <output_dir>/prealigned_<method>/
        nested = parent_dir / f"prealigned_{method}"
        method_dirs[method] = nested if nested.is_dir() else parent_dir

    # Step 2: Evaluate all methods
    rows = evaluate_methods(args.data_dir, method_dirs, args.split)

    # Step 3: Save CSV
    csv_path = args.output_dir / "prealign_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "n", "psnr", "ssim", "delta_psnr"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
