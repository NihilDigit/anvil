"""Experiment 1: Compare 5 blend methods on Vimeo90K test set (zero-network baselines).

Evaluates naive blend, phase correlation blend, MV block-shift blend,
MV blend + 3x3 smoothing, and oracle (RAFT) flow warp blend against
ground-truth middle frames.  Reports PSNR / SSIM / LPIPS overall and
per motion-magnitude bin (small / medium / large).

Usage:
    pixi run python -m anvil_exp01.experiments.exp1_blend_baselines \
        --data-dir data/vimeo_triplet \
        --mv-flow-dir data/mv_dense_flow \
        --raft-flow-dir data/raft_flow \
        --motion-csv data/motion_labels.csv \
        --output-dir results/exp1
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from anvil_exp01.data.prealign import prealign_frames
from anvil_exp01.eval.metrics import compute_lpips_batch, compute_psnr, compute_ssim

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_METHODS = [
    "naive_blend",
    "phase_corr",
    "mv_blend",
    "mv_blend_smooth",
    "oracle_flow",
]

METHOD_DISPLAY = {
    "naive_blend": "Naive Blend",
    "phase_corr": "Phase Corr",
    "mv_blend": "MV Blend",
    "mv_blend_smooth": "MV Blend+Smooth",
    "oracle_flow": "Oracle Flow",
}

MOTION_BINS = ["small", "medium", "large"]

# ---------------------------------------------------------------------------
# Helper: load triplet IDs from tri_testlist.txt
# ---------------------------------------------------------------------------


def _load_test_ids(data_dir: Path) -> list[str]:
    list_path = data_dir / "tri_testlist.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Test list not found: {list_path}")
    ids: list[str] = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


# ---------------------------------------------------------------------------
# Helper: load motion labels CSV -> dict[triplet_id, motion_bin]
# ---------------------------------------------------------------------------


def _load_motion_labels(csv_path: Path) -> dict[str, str]:
    """Return mapping triplet_id -> motion bin ('small'|'medium'|'large')."""
    labels: dict[str, str] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["triplet_id"].strip()
            bin_label = row["motion_bin"].strip()
            labels[tid] = bin_label
    return labels


# ---------------------------------------------------------------------------
# Helper: load image as uint8 HWC numpy array
# ---------------------------------------------------------------------------


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Phase correlation
# ---------------------------------------------------------------------------


def phase_correlate(img0_gray: np.ndarray, img1_gray: np.ndarray) -> tuple[int, int]:
    """Estimate global translation (dx, dy) between two grayscale images
    using phase correlation.  Returns integer pixel offsets.
    """
    f0 = np.fft.fft2(img0_gray.astype(np.float64))
    f1 = np.fft.fft2(img1_gray.astype(np.float64))
    cross_power_raw = f0 * np.conj(f1)
    cross_power = cross_power_raw / (np.abs(cross_power_raw) + 1e-8)
    correlation = np.fft.ifft2(cross_power).real

    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy, dx = int(peak[0]), int(peak[1])

    h, w = img0_gray.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return dx, dy


def _rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Convert HWC uint8 RGB image to HW float64 grayscale."""
    return (
        0.2989 * img[:, :, 0].astype(np.float64)
        + 0.5870 * img[:, :, 1].astype(np.float64)
        + 0.1140 * img[:, :, 2].astype(np.float64)
    )


# ---------------------------------------------------------------------------
# Oracle flow warping (sub-pixel, backward warping)
# ---------------------------------------------------------------------------


def warp_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp *img* (HWC uint8) using *flow* (HW2 float, [dx, dy]) via
    backward warping with bilinear interpolation.

    Sampling convention:
      out(y, x) = img(y + flow_y(y, x), x + flow_x(y, x))
    """
    from scipy.ndimage import map_coordinates

    h, w = img.shape[:2]
    coords_y, coords_x = np.mgrid[0:h, 0:w].astype(np.float64)

    map_x = coords_x + flow[:, :, 0].astype(np.float64)
    map_y = coords_y + flow[:, :, 1].astype(np.float64)

    warped = np.zeros_like(img)
    for c in range(img.shape[2]):
        warped[:, :, c] = (
            map_coordinates(
                img[:, :, c].astype(np.float64),
                [map_y, map_x],
                order=1,
                mode="reflect",
            )
            .clip(0, 255)
            .astype(np.uint8)
        )
    return warped


# ---------------------------------------------------------------------------
# 3x3 mean filter via numpy convolution (no scipy/cv2 dependency)
# ---------------------------------------------------------------------------


def _mean_filter_3x3(img: np.ndarray) -> np.ndarray:
    """Apply a 3x3 mean filter to an HWC uint8 image.

    Uses zero-padded 2-D convolution per channel implemented with numpy
    so there is no dependency on OpenCV or scipy for this simple kernel.
    """
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float64)
        h, w = ch.shape
        padded = np.pad(ch, 1, mode="reflect")
        acc = np.zeros_like(ch)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                acc += padded[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
        out[:, :, c] = np.clip(acc / 9.0, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Per-method prediction functions
# ---------------------------------------------------------------------------


def predict_naive_blend(
    i0: np.ndarray,
    i1: np.ndarray,
    *,
    triplet_id: str,
    mv_flow_dir: Path | None,
    raft_flow_dir: Path | None,
) -> np.ndarray:
    """pred = (I_0 + I_1) / 2, no motion compensation."""
    return ((i0.astype(np.uint16) + i1.astype(np.uint16)) // 2).astype(np.uint8)


def predict_phase_corr(
    i0: np.ndarray,
    i1: np.ndarray,
    *,
    triplet_id: str,
    mv_flow_dir: Path | None,
    raft_flow_dir: Path | None,
) -> np.ndarray:
    """Phase correlation: estimate global translation, apply integer half-shift, blend."""
    g0 = _rgb_to_gray(i0)
    g1 = _rgb_to_gray(i1)
    dx, dy = phase_correlate(g0, g1)

    # Half-shift via numpy roll then zero-fill wrapped edges (avoid wrap-around artifacts)
    half_dx = int(round(dx / 2))
    half_dy = int(round(dy / 2))

    i0_shifted = np.roll(np.roll(i0, half_dy, axis=0), half_dx, axis=1)
    i1_shifted = np.roll(np.roll(i1, -half_dy, axis=0), -half_dx, axis=1)

    # Zero-fill the edges that were incorrectly wrapped by np.roll
    h, w = i0.shape[:2]
    if half_dy > 0:
        i0_shifted[:half_dy, :] = i0[:half_dy, :]
        i1_shifted[-half_dy:, :] = i1[-half_dy:, :]
    elif half_dy < 0:
        i0_shifted[half_dy:, :] = i0[half_dy:, :]
        i1_shifted[:-half_dy, :] = i1[:-half_dy, :]
    if half_dx > 0:
        i0_shifted[:, :half_dx] = i0[:, :half_dx]
        i1_shifted[:, -half_dx:] = i1[:, -half_dx:]
    elif half_dx < 0:
        i0_shifted[:, half_dx:] = i0[:, half_dx:]
        i1_shifted[:, :-half_dx] = i1[:, :-half_dx:]

    return ((i0_shifted.astype(np.uint16) + i1_shifted.astype(np.uint16)) // 2).astype(
        np.uint8
    )


def _triplet_id_to_flow_path(base_dir: Path, triplet_id: str) -> Path:
    """Convert triplet_id like '00001/0001' to '<base_dir>/00001/0001.npy'."""
    parts = triplet_id.split("/")
    return base_dir / parts[0] / f"{parts[1]}.npy"


def predict_mv_blend(
    i0: np.ndarray,
    i1: np.ndarray,
    *,
    triplet_id: str,
    mv_flow_dir: Path | None,
    raft_flow_dir: Path | None,
) -> np.ndarray:
    """MV block-shift pre-alignment then blend."""
    if mv_flow_dir is None:
        raise ValueError("--mv-flow-dir is required for mv_blend method")
    flow_path = _triplet_id_to_flow_path(mv_flow_dir, triplet_id)
    mv_flow = np.load(str(flow_path))  # (H, W, 2) dense flow
    i0_shifted, i1_shifted = prealign_frames(i0, i1, mv_flow)
    return (
        (i0_shifted.astype(np.uint16) + i1_shifted.astype(np.uint16)) // 2
    ).astype(np.uint8)


def predict_mv_blend_smooth(
    i0: np.ndarray,
    i1: np.ndarray,
    *,
    triplet_id: str,
    mv_flow_dir: Path | None,
    raft_flow_dir: Path | None,
) -> np.ndarray:
    """MV block-shift blend followed by 3x3 mean filter."""
    blended = predict_mv_blend(
        i0,
        i1,
        triplet_id=triplet_id,
        mv_flow_dir=mv_flow_dir,
        raft_flow_dir=raft_flow_dir,
    )
    return _mean_filter_3x3(blended)


def predict_oracle_flow(
    i0: np.ndarray,
    i1: np.ndarray,
    *,
    triplet_id: str,
    mv_flow_dir: Path | None,
    raft_flow_dir: Path | None,
) -> np.ndarray:
    """Oracle (RAFT) sub-pixel flow warp then blend."""
    if raft_flow_dir is None:
        raise ValueError("--raft-flow-dir is required for oracle_flow method")
    flow_path = _triplet_id_to_flow_path(raft_flow_dir, triplet_id)
    raft_flow = np.load(str(flow_path))  # (2, H, W) from RAFT
    if raft_flow.ndim == 3 and raft_flow.shape[0] == 2:
        raft_flow = raft_flow.transpose(1, 2, 0)  # -> (H, W, 2)

    half_flow = raft_flow.astype(np.float64) * 0.5
    # RAFT flow here is f_{0->1}. With the backward sampling convention above,
    # midpoint warping should sample I0 at (x - f/2) and I1 at (x + f/2).
    i0_warped = warp_flow(i0, -half_flow)
    i1_warped = warp_flow(i1, half_flow)

    return ((i0_warped.astype(np.uint16) + i1_warped.astype(np.uint16)) // 2).astype(
        np.uint8
    )


# Registry: method_key -> prediction function
METHOD_FN = {
    "naive_blend": predict_naive_blend,
    "phase_corr": predict_phase_corr,
    "mv_blend": predict_mv_blend,
    "mv_blend_smooth": predict_mv_blend_smooth,
    "oracle_flow": predict_oracle_flow,
}


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

_COL_HEADERS = [
    "Method",
    "Overall PSNR",
    "Overall SSIM",
    "Overall LPIPS",
    "Small PSNR",
    "Medium PSNR",
    "Large PSNR",
    "Small LPIPS",
    "Medium LPIPS",
    "Large LPIPS",
]


def _build_summary_rows(
    methods: list[str],
    all_results: dict[str, dict[str, list[float]]],
    motion_labels: dict[str, str],
    triplet_ids: list[str],
) -> list[dict[str, str]]:
    """Build summary rows: one per method, with overall + per-bin aggregates."""

    # Pre-compute bin membership
    bin_indices: dict[str, list[int]] = {b: [] for b in MOTION_BINS}
    for idx, tid in enumerate(triplet_ids):
        b = motion_labels.get(tid, "")
        if b in bin_indices:
            bin_indices[b].append(idx)

    rows: list[dict[str, str]] = []
    for method in methods:
        psnr_all = all_results[method]["psnr"]
        ssim_all = all_results[method]["ssim"]
        lpips_all = all_results[method]["lpips"]

        row: dict[str, str] = {
            "Method": METHOD_DISPLAY.get(method, method),
            "Overall PSNR": f"{np.mean(psnr_all):.4f}",
            "Overall SSIM": f"{np.mean(ssim_all):.6f}",
            "Overall LPIPS": f"{np.mean(lpips_all):.6f}",
        }

        for bin_name in MOTION_BINS:
            idxs = bin_indices[bin_name]
            if idxs:
                bin_psnr = [psnr_all[i] for i in idxs]
                bin_lpips = [lpips_all[i] for i in idxs]
                row[f"{bin_name.capitalize()} PSNR"] = f"{np.mean(bin_psnr):.4f}"
                row[f"{bin_name.capitalize()} LPIPS"] = f"{np.mean(bin_lpips):.6f}"
            else:
                row[f"{bin_name.capitalize()} PSNR"] = "N/A"
                row[f"{bin_name.capitalize()} LPIPS"] = "N/A"

        rows.append(row)
    return rows


def _print_summary_table(rows: list[dict[str, str]]) -> None:
    """Pretty-print the summary table to stdout."""
    # Determine column widths
    widths: dict[str, int] = {}
    for col in _COL_HEADERS:
        widths[col] = len(col)
    for row in rows:
        for col in _COL_HEADERS:
            widths[col] = max(widths[col], len(row.get(col, "")))

    # Header
    header = " | ".join(col.ljust(widths[col]) for col in _COL_HEADERS)
    separator = "-+-".join("-" * widths[col] for col in _COL_HEADERS)
    print(header)
    print(separator)

    # Rows
    for row in rows:
        line = " | ".join(row.get(col, "").ljust(widths[col]) for col in _COL_HEADERS)
        print(line)


def _save_summary_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COL_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary table saved to {path}")


def _save_per_triplet_csv(
    methods: list[str],
    triplet_ids: list[str],
    all_results: dict[str, dict[str, list[float]]],
    motion_labels: dict[str, str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["triplet_id", "motion_bin"]
    for method in methods:
        display = METHOD_DISPLAY.get(method, method)
        fieldnames.extend([
            f"{display}_PSNR",
            f"{display}_SSIM",
            f"{display}_LPIPS",
        ])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, tid in enumerate(triplet_ids):
            row: dict[str, str] = {
                "triplet_id": tid,
                "motion_bin": motion_labels.get(tid, ""),
            }
            for method in methods:
                display = METHOD_DISPLAY.get(method, method)
                row[f"{display}_PSNR"] = f"{all_results[method]['psnr'][idx]:.4f}"
                row[f"{display}_SSIM"] = f"{all_results[method]['ssim'][idx]:.6f}"
                row[f"{display}_LPIPS"] = f"{all_results[method]['lpips'][idx]:.6f}"
            writer.writerow(row)

    print(f"Per-triplet results saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 1: Compare 5 blend methods on Vimeo90K test set "
            "(3782 triplets) under zero-network conditions."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Vimeo90K root (containing sequences/, tri_testlist.txt, etc.)",
    )
    parser.add_argument(
        "--mv-flow-dir",
        type=Path,
        default=None,
        help="Directory with pre-computed MV dense flow .npy files.",
    )
    parser.add_argument(
        "--raft-flow-dir",
        type=Path,
        default=None,
        help="Directory with pre-computed RAFT optical flow .npy files.",
    )
    parser.add_argument(
        "--motion-csv",
        type=Path,
        required=True,
        help="Motion labels CSV (columns: triplet_id, motion_bin).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exp1"),
        help="Output directory for result CSVs (default: results/exp1/).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for LPIPS computation (default: cuda).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=ALL_METHODS,
        help="Which blend methods to evaluate (default: all).",
    )
    parser.add_argument(
        "--batch-triplets",
        type=int,
        default=12,
        help=(
            "How many triplets to process per outer batch. Higher values improve "
            "LPIPS GPU utilization but use more RAM/VRAM (default: 12)."
        ),
    )
    parser.add_argument(
        "--lpips-pair-batch",
        type=int,
        default=64,
        help=(
            "How many image pairs per LPIPS forward pass. Tune this based on VRAM "
            "to balance throughput and memory (default: 64)."
        ),
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods: list[str] = args.methods

    # Validate that required flow dirs are given for selected methods
    if "mv_blend" in methods or "mv_blend_smooth" in methods:
        if args.mv_flow_dir is None:
            parser.error("--mv-flow-dir is required when running mv_blend or mv_blend_smooth")
    if "oracle_flow" in methods:
        if args.raft_flow_dir is None:
            parser.error("--raft-flow-dir is required when running oracle_flow")

    mv_flow_dir = args.mv_flow_dir.resolve() if args.mv_flow_dir else None
    raft_flow_dir = args.raft_flow_dir.resolve() if args.raft_flow_dir else None

    # ------------------------------------------------------------------
    # 1. Load test split
    # ------------------------------------------------------------------
    triplet_ids = _load_test_ids(data_dir)
    print(f"Loaded {len(triplet_ids)} test triplets from {data_dir}")

    # ------------------------------------------------------------------
    # 2. Load motion labels
    # ------------------------------------------------------------------
    motion_labels = _load_motion_labels(args.motion_csv.resolve())
    n_matched = sum(1 for tid in triplet_ids if tid in motion_labels)
    print(f"Motion labels matched for {n_matched}/{len(triplet_ids)} triplets")
    for b in MOTION_BINS:
        count = sum(1 for tid in triplet_ids if motion_labels.get(tid) == b)
        print(f"  {b}: {count}")

    # ------------------------------------------------------------------
    # 3. Initialize results storage
    # ------------------------------------------------------------------
    all_results: dict[str, dict[str, list[float]]] = {}
    for method in methods:
        all_results[method] = {"psnr": [], "ssim": [], "lpips": []}

    # ------------------------------------------------------------------
    # 4. Process each triplet
    # ------------------------------------------------------------------
    seq_dir = data_dir / "sequences"
    t_start = time.time()

    BATCH_TRIPLETS = max(1, int(args.batch_triplets))
    n_methods = len(methods)
    lpips_pairs_per_outer = BATCH_TRIPLETS * n_methods
    print(
        "Batch config: "
        f"triplets={BATCH_TRIPLETS}, "
        f"pairs/outer_batch={lpips_pairs_per_outer}, "
        f"lpips_pair_batch={args.lpips_pair_batch}"
    )

    for batch_start in tqdm(range(0, len(triplet_ids), BATCH_TRIPLETS),
                            desc="Evaluating", unit="batch"):
        batch_ids = triplet_ids[batch_start : batch_start + BATCH_TRIPLETS]

        # Load images and generate predictions for the whole mini-batch
        # preds_flat / gts_flat: len = len(batch_ids) * n_methods
        preds_flat: list[np.ndarray] = []
        gts_flat: list[np.ndarray] = []
        batch_psnr: list[list[float]] = []  # [triplet_idx][method_idx]
        batch_ssim: list[list[float]] = []

        for triplet_id in batch_ids:
            triplet_dir = seq_dir / triplet_id
            i0 = _load_image(triplet_dir / "im1.png")
            i_gt = _load_image(triplet_dir / "im2.png")
            i1 = _load_image(triplet_dir / "im3.png")

            psnr_row: list[float] = []
            ssim_row: list[float] = []

            for method in methods:
                fn = METHOD_FN[method]
                pred = fn(
                    i0, i1,
                    triplet_id=triplet_id,
                    mv_flow_dir=mv_flow_dir,
                    raft_flow_dir=raft_flow_dir,
                )
                preds_flat.append(pred)
                gts_flat.append(i_gt)
                psnr_row.append(compute_psnr(pred, i_gt))
                ssim_row.append(compute_ssim(pred, i_gt))

            batch_psnr.append(psnr_row)
            batch_ssim.append(ssim_row)

        # Single batched LPIPS call: batch_size * n_methods pairs
        lpips_vals = compute_lpips_batch(
            preds_flat,
            gts_flat,
            device=args.device,
            pair_batch_size=args.lpips_pair_batch,
        )

        # Distribute results back
        for t_idx, triplet_id in enumerate(batch_ids):
            for m_idx, method in enumerate(methods):
                flat_idx = t_idx * n_methods + m_idx
                all_results[method]["psnr"].append(batch_psnr[t_idx][m_idx])
                all_results[method]["ssim"].append(batch_ssim[t_idx][m_idx])
                all_results[method]["lpips"].append(lpips_vals[flat_idx])

    elapsed = time.time() - t_start
    print(f"\nEvaluation completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 5. Aggregate & report
    # ------------------------------------------------------------------
    summary_rows = _build_summary_rows(methods, all_results, motion_labels, triplet_ids)

    print()
    _print_summary_table(summary_rows)
    print()

    # Save CSVs
    _save_summary_csv(summary_rows, output_dir / "summary.csv")
    _save_per_triplet_csv(
        methods, triplet_ids, all_results, motion_labels, output_dir / "per_triplet.csv"
    )


if __name__ == "__main__":
    main()
