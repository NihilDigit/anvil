"""Evaluate zero-network baselines on Vimeo90K and/or Xiph 1080p datasets.

Methods: Naive Blend, MV Blend (basic prealignment), MV Blend (smoothed prealignment).
Reports per-sequence and overall PSNR/SSIM/LPIPS.

Usage:
    pixi run python scripts/eval_baselines.py
    pixi run python scripts/eval_baselines.py --dataset both --methods naive_blend mv_blend_v2
    pixi run python scripts/eval_baselines.py --dataset vimeo --methods naive_blend mv_blend_v2
    pixi run python scripts/eval_baselines.py --dataset xiph --methods naive_blend mv_blend
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from anvil_exp01.eval.metrics import compute_lpips_batch, compute_psnr, compute_ssim


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    name: str
    data_dir: Path
    prealigned_v1: Path | None
    prealigned_v2: Path | None
    lpips_pair_batch: int


DEFAULT_DATASETS: dict[str, DatasetConfig] = {
    "vimeo": DatasetConfig(
        name="vimeo",
        data_dir=Path("data/vimeo_triplet"),
        prealigned_v1=None,  # v1 prealigned not available for Vimeo
        prealigned_v2=Path("data/vimeo_triplet/prealigned_v2"),
        lpips_pair_batch=64,
    ),
    "xiph": DatasetConfig(
        name="xiph",
        data_dir=Path("data/xiph_1080p"),
        prealigned_v1=Path("data/xiph_1080p/prealigned"),
        prealigned_v2=Path("data/xiph_1080p/prealigned_v2"),
        lpips_pair_batch=2,
    ),
}

ALL_METHODS = ["naive_blend", "mv_blend", "mv_blend_v2"]

METHOD_DISPLAY = {
    "naive_blend": "Naive Blend",
    "mv_blend": "MV Blend (basic prealign)",
    "mv_blend_v2": "MV Blend",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_image(path: Path) -> np.ndarray:
    """Load image as RGB uint8 HWC."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_triplet_ids(data_dir: Path) -> list[str]:
    triplet_list = data_dir / "tri_testlist.txt"
    ids: list[str] = []
    with open(triplet_list) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


# ---------------------------------------------------------------------------
# Prediction methods
# ---------------------------------------------------------------------------


def predict_naive_blend(
    i0: np.ndarray, i1: np.ndarray, triplet_id: str, dataset: DatasetConfig
) -> np.ndarray:
    return np.round((i0.astype(np.float32) + i1.astype(np.float32)) / 2).astype(np.uint8)


def predict_mv_blend(
    i0: np.ndarray, i1: np.ndarray, triplet_id: str, dataset: DatasetConfig
) -> np.ndarray:
    """Average of MV-prealigned v1 frames."""
    if dataset.prealigned_v1 is None:
        raise RuntimeError(f"mv_blend requires prealigned v1 dir for {dataset.name}")
    seq, tid = triplet_id.split("/")
    p0 = _load_image(dataset.prealigned_v1 / seq / tid / "im1_aligned.png")
    p1 = _load_image(dataset.prealigned_v1 / seq / tid / "im3_aligned.png")
    return ((p0.astype(np.uint16) + p1.astype(np.uint16)) // 2).astype(np.uint8)


def predict_mv_blend_v2(
    i0: np.ndarray, i1: np.ndarray, triplet_id: str, dataset: DatasetConfig
) -> np.ndarray:
    """Average of MV-prealigned v2 frames (production pipeline)."""
    if dataset.prealigned_v2 is None:
        raise RuntimeError(f"mv_blend_v2 requires prealigned v2 dir for {dataset.name}")
    seq, tid = triplet_id.split("/")
    p0 = _load_image(dataset.prealigned_v2 / seq / tid / "im1_aligned.png")
    p1 = _load_image(dataset.prealigned_v2 / seq / tid / "im3_aligned.png")
    return ((p0.astype(np.uint16) + p1.astype(np.uint16)) // 2).astype(np.uint8)


METHOD_FN = {
    "naive_blend": predict_naive_blend,
    "mv_blend": predict_mv_blend,
    "mv_blend_v2": predict_mv_blend_v2,
}


# ---------------------------------------------------------------------------
# Evaluation loop (per dataset)
# ---------------------------------------------------------------------------


def _try_load_existing(
    output_dir: Path, expected_n: int, methods: list[str],
) -> dict[str, dict[str, float]] | None:
    """Return cached summaries if summary.csv exists and matches expected count."""
    summary_path = output_dir / "summary.csv"
    if not summary_path.is_file():
        return None
    rows: list[dict[str, str]] = []
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None
    # Verify row count matches expected methods
    if len(rows) != len(methods):
        return None
    # Verify triplet count
    for row in rows:
        if int(row["n_triplets"]) != expected_n:
            return None
    result: dict[str, dict[str, float]] = {}
    display_to_key = {v: k for k, v in METHOD_DISPLAY.items()}
    for row in rows:
        key = display_to_key.get(row["method"])
        if key is None or key not in methods:
            return None
        result[key] = {
            "psnr": float(row["psnr"]),
            "ssim": float(row["ssim"]),
            "lpips": float(row["lpips"]),
        }
    return result


def evaluate_dataset(
    dataset: DatasetConfig,
    methods: list[str],
    output_dir: Path,
    device: str,
    force: bool = False,
    limit: int = 0,
) -> dict[str, dict[str, float]]:
    """Run evaluation for one dataset. Returns {method: {psnr, ssim, lpips}}."""

    output_dir.mkdir(parents=True, exist_ok=True)
    triplet_ids = _load_triplet_ids(dataset.data_dir)
    if limit > 0:
        triplet_ids = triplet_ids[:limit]

    if not force:
        cached = _try_load_existing(output_dir, len(triplet_ids), methods)
        if cached is not None:
            print(f"\n  SKIP {dataset.name}: {len(triplet_ids)} results exist at {output_dir}")
            for method, vals in cached.items():
                print(f"    {METHOD_DISPLAY[method]}: PSNR {vals['psnr']:.4f}  "
                      f"SSIM {vals['ssim']:.6f}  LPIPS {vals['lpips']:.6f}")
            return cached

    sequences_dir = dataset.data_dir / "sequences"
    flush_every = dataset.lpips_pair_batch

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.name} ({len(triplet_ids)} triplets)")
    print(f"Methods: {', '.join(METHOD_DISPLAY[m] for m in methods)}")
    print(f"{'='*60}")

    results: dict[str, dict[str, list[float]]] = {
        m: {"psnr": [], "ssim": [], "lpips": []} for m in methods
    }
    tid_list: list[str] = []

    t_start = time.time()

    chunk_preds: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    chunk_gts: list[np.ndarray] = []

    for triplet_id in tqdm(triplet_ids, desc=f"  {dataset.name}", unit="triplet"):
        seq, tid = triplet_id.split("/")
        triplet_dir = sequences_dir / seq / tid

        i0 = _load_image(triplet_dir / "im1.png")
        gt = _load_image(triplet_dir / "im2.png")
        i1 = _load_image(triplet_dir / "im3.png")

        tid_list.append(triplet_id)

        for method in methods:
            pred = METHOD_FN[method](i0, i1, triplet_id, dataset)
            results[method]["psnr"].append(compute_psnr(pred, gt))
            results[method]["ssim"].append(compute_ssim(pred, gt))
            chunk_preds[method].append(pred)

        chunk_gts.append(gt)

        if len(chunk_gts) >= flush_every:
            torch.cuda.empty_cache()
            for method in methods:
                lpips_vals = compute_lpips_batch(
                    chunk_preds[method], chunk_gts,
                    device=device, pair_batch_size=flush_every,
                )
                results[method]["lpips"].extend(lpips_vals)
                chunk_preds[method].clear()
            chunk_gts.clear()
            torch.cuda.empty_cache()

    if chunk_gts:
        for method in methods:
            lpips_vals = compute_lpips_batch(
                chunk_preds[method], chunk_gts,
                device=device, pair_batch_size=max(1, len(chunk_gts)),
            )
            results[method]["lpips"].extend(lpips_vals)

    elapsed = time.time() - t_start
    print(f"  Completed in {elapsed:.1f}s")

    # Per-sequence breakdown
    seq_metrics: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {m: {"psnr": [], "ssim": [], "lpips": []} for m in methods}
    )
    for i, tid in enumerate(tid_list):
        seq = tid.split("/")[0]
        for method in methods:
            seq_metrics[seq][method]["psnr"].append(results[method]["psnr"][i])
            seq_metrics[seq][method]["ssim"].append(results[method]["ssim"][i])
            seq_metrics[seq][method]["lpips"].append(results[method]["lpips"][i])

    # Print results
    summaries: dict[str, dict[str, float]] = {}
    for method in methods:
        display = METHOD_DISPLAY[method]
        mean_psnr = float(np.mean(results[method]["psnr"]))
        mean_ssim = float(np.mean(results[method]["ssim"]))
        mean_lpips = float(np.mean(results[method]["lpips"]))
        summaries[method] = {"psnr": mean_psnr, "ssim": mean_ssim, "lpips": mean_lpips}
        print(f"\n  {display} — OVERALL ({len(tid_list)} triplets)")
        print(f"    PSNR:  {mean_psnr:.4f} dB")
        print(f"    SSIM:  {mean_ssim:.6f}")
        print(f"    LPIPS: {mean_lpips:.6f}")
        print(f"\n    {'Sequence':<20} {'N':>4} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
        print(f"    {'-'*52}")
        for seq in sorted(seq_metrics):
            m = seq_metrics[seq][method]
            n = len(m["psnr"])
            print(f"    {seq:<20} {n:>4} {np.mean(m['psnr']):>8.4f} "
                  f"{np.mean(m['ssim']):>8.6f} {np.mean(m['lpips']):>8.6f}")

    # Save summary CSV
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_triplets", "psnr", "ssim", "lpips"])
        for method in methods:
            w.writerow([
                METHOD_DISPLAY[method],
                len(tid_list),
                f"{np.mean(results[method]['psnr']):.4f}",
                f"{np.mean(results[method]['ssim']):.6f}",
                f"{np.mean(results[method]['lpips']):.6f}",
            ])
    print(f"\n  Summary: {summary_path}")

    # Save per-sequence CSV
    seq_path = output_dir / "per_sequence.csv"
    with open(seq_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["sequence", "n_triplets"]
        for method in methods:
            d = METHOD_DISPLAY[method]
            header.extend([f"{d}_psnr", f"{d}_ssim", f"{d}_lpips"])
        w.writerow(header)
        for seq in sorted(seq_metrics):
            row: list[str | int] = [seq, len(seq_metrics[seq][methods[0]]["psnr"])]
            for method in methods:
                m = seq_metrics[seq][method]
                row.extend([
                    f"{np.mean(m['psnr']):.4f}",
                    f"{np.mean(m['ssim']):.6f}",
                    f"{np.mean(m['lpips']):.6f}",
                ])
            w.writerow(row)
    print(f"  Per-sequence: {seq_path}")

    # Save per-triplet CSV
    triplet_path = output_dir / "per_triplet.csv"
    with open(triplet_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["triplet_id"]
        for method in methods:
            d = METHOD_DISPLAY[method]
            header.extend([f"{d}_psnr", f"{d}_ssim", f"{d}_lpips"])
        w.writerow(header)
        for i, tid in enumerate(tid_list):
            row_t: list[str] = [tid]
            for method in methods:
                row_t.extend([
                    f"{results[method]['psnr'][i]:.4f}",
                    f"{results[method]['ssim'][i]:.6f}",
                    f"{results[method]['lpips'][i]:.6f}",
                ])
            w.writerow(row_t)
    print(f"  Per-triplet: {triplet_path}")

    return summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-network baselines on Vimeo90K / Xiph 1080p"
    )
    parser.add_argument(
        "--dataset", choices=["vimeo", "xiph", "both"], default="both",
        help="Dataset selection (default: both)",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=ALL_METHODS, default=None,
        help="Methods to evaluate (default: naive_blend + mv_blend_v2; "
             "mv_blend requires prealigned v1 dir)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/eval/baselines"),
        help="Output root dir (per-dataset subdirs created automatically)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vimeo-dir", type=Path, default=None,
        help="Override Vimeo data dir",
    )
    parser.add_argument(
        "--xiph-dir", type=Path, default=None,
        help="Override Xiph data dir",
    )
    parser.add_argument(
        "--prealigned-v2-vimeo", type=Path, default=None,
        help="Override Vimeo prealign v2 dir",
    )
    parser.add_argument(
        "--prealigned-v2-xiph", type=Path, default=None,
        help="Override Xiph prealign v2 dir",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if results already exist",
    )
    parser.add_argument(
        "--prealigned-v1-vimeo", type=Path, default=None,
        help="Override Vimeo prealign v1 dir (needed for mv_blend method on Vimeo)",
    )
    parser.add_argument(
        "--prealigned-v1-xiph", type=Path, default=None,
        help="Override Xiph prealign v1 dir",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit to N triplets per dataset (0=all, for smoke testing).",
    )
    args = parser.parse_args()

    # Resolve datasets
    datasets: list[DatasetConfig] = []
    run_vimeo = args.dataset in ("vimeo", "both")
    run_xiph = args.dataset in ("xiph", "both")

    if run_vimeo:
        cfg = DEFAULT_DATASETS["vimeo"]
        v1_override = args.prealigned_v1_vimeo or cfg.prealigned_v1
        v2_override = args.prealigned_v2_vimeo or cfg.prealigned_v2
        if args.vimeo_dir or args.prealigned_v2_vimeo or args.prealigned_v1_vimeo:
            cfg = DatasetConfig(
                name=cfg.name,
                data_dir=args.vimeo_dir or cfg.data_dir,
                prealigned_v1=v1_override,
                prealigned_v2=v2_override,
                lpips_pair_batch=cfg.lpips_pair_batch,
            )
        datasets.append(cfg)

    if run_xiph:
        cfg = DEFAULT_DATASETS["xiph"]
        v1_override = args.prealigned_v1_xiph or cfg.prealigned_v1
        v2_override = args.prealigned_v2_xiph or cfg.prealigned_v2
        if args.xiph_dir or args.prealigned_v2_xiph or args.prealigned_v1_xiph:
            cfg = DatasetConfig(
                name=cfg.name,
                data_dir=args.xiph_dir or cfg.data_dir,
                prealigned_v1=v1_override,
                prealigned_v2=v2_override,
                lpips_pair_batch=cfg.lpips_pair_batch,
            )
        datasets.append(cfg)

    # Default methods: naive_blend + mv_blend_v2 (skip mv_blend v1 unless explicit)
    methods = args.methods or ["naive_blend", "mv_blend_v2"]

    # Validate prealigned dirs exist for requested methods
    for ds in datasets:
        if "mv_blend" in methods and (ds.prealigned_v1 is None or not ds.prealigned_v1.is_dir()):
            print(f"WARNING: skipping mv_blend for {ds.name} — "
                  f"prealigned v1 dir not found: {ds.prealigned_v1}")
            methods = [m for m in methods if m != "mv_blend"]
        if "mv_blend_v2" in methods and (ds.prealigned_v2 is None or not ds.prealigned_v2.is_dir()):
            print(f"ERROR: mv_blend_v2 requires prealigned v2 dir for {ds.name}: "
                  f"{ds.prealigned_v2}")
            raise SystemExit(1)

    if not methods:
        print("ERROR: no methods to evaluate")
        raise SystemExit(1)

    print("=" * 60)
    print("ANVIL Zero-Network Baselines")
    print("=" * 60)
    print(f"Datasets: {', '.join(ds.name for ds in datasets)}")
    print(f"Methods:  {', '.join(METHOD_DISPLAY[m] for m in methods)}")
    print(f"Output:   {args.output_dir}")

    all_summaries: dict[str, dict[str, dict[str, float]]] = {}
    for ds in datasets:
        ds_output = args.output_dir / ds.name
        all_summaries[ds.name] = evaluate_dataset(ds, methods, ds_output, args.device, force=args.force, limit=args.limit)

    # Print combined summary
    print(f"\n{'='*60}")
    print("COMBINED SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<8} {'Method':<16} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("-" * 52)
    for ds_name, summaries in all_summaries.items():
        for method, vals in summaries.items():
            print(f"{ds_name:<8} {METHOD_DISPLAY[method]:<16} "
                  f"{vals['psnr']:>8.4f} {vals['ssim']:>8.6f} {vals['lpips']:>8.6f}")


if __name__ == "__main__":
    main()
