"""Generate MV pre-aligned frames using block-level integer-pixel shifts.

Applies half-program shift: for each block, compute average flow, halve it,
round to integer, and shift I_0 forward / I_1 backward by that amount.
This produces pre-aligned frames without any sub-pixel interpolation.

Usage:
    pixi run python anvil_exp01/data/prealign.py \
        --data-dir /path/to/vimeo_triplet \
        --flow-dir data/dense_flow \
        --output-dir data/prealigned \
        --split both \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def prealign_frames(
    I_0: np.ndarray,
    I_1: np.ndarray,
    flow_01: np.ndarray,
    block_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-align two frames using MV-derived block-level integer shifts.

    For each block_size x block_size block:
      1. Compute the block-average of flow_01 * 0.5
      2. Round to the nearest integer (block-level integer pixel shift)
      3. Copy I_0 block to the destination shifted by +half_flow
      4. Copy I_1 block to the destination shifted by -half_flow

    Args:
        I_0: Source frame 0, shape (H, W, C), dtype uint8.
        I_1: Source frame 1, shape (H, W, C), dtype uint8.
        flow_01: Dense optical flow from frame 0 to frame 1,
                 shape (H, W, 2), dtype float32.  Channel 0 = dx, channel 1 = dy.
        block_size: Block size for integer shifts (default 16).

    Returns:
        (I_0_shifted, I_1_shifted), both uint8 arrays with same shape as inputs.
    """
    h, w = I_0.shape[:2]
    # Initialize with original frames so that regions not covered by any
    # shifted block retain valid pixel data instead of black (zero) holes.
    I_0_shifted = I_0.copy()
    I_1_shifted = I_1.copy()

    flow_half = flow_01 * 0.5

    for by in range(0, h, block_size):
        bh = min(block_size, h - by)
        for bx in range(0, w, block_size):
            bw = min(block_size, w - bx)

            # Block-average half-flow, rounded to integer
            block_flow = flow_half[by : by + bh, bx : bx + bw]
            dx = int(round(float(block_flow[:, :, 0].mean())))
            dy = int(round(float(block_flow[:, :, 1].mean())))

            # -- Shift I_0 forward by (+dx, +dy) --
            # Source region in I_0: the current block [by:by+bh, bx:bx+bw]
            # Destination region in I_0_shifted: shifted by (dy, dx)
            dst_y0 = by + dy
            dst_x0 = bx + dx

            # Compute the overlap between source block and destination in output
            # Source block coords
            src_y_start = 0
            src_x_start = 0
            src_y_end = bh
            src_x_end = bw

            # Destination coords in output frame
            out_y_start = dst_y0
            out_x_start = dst_x0
            out_y_end = dst_y0 + bh
            out_x_end = dst_x0 + bw

            # Clip destination to frame boundaries, adjust source accordingly
            if out_y_start < 0:
                src_y_start -= out_y_start
                out_y_start = 0
            if out_x_start < 0:
                src_x_start -= out_x_start
                out_x_start = 0
            if out_y_end > h:
                src_y_end -= out_y_end - h
                out_y_end = h
            if out_x_end > w:
                src_x_end -= out_x_end - w
                out_x_end = w

            if out_y_start < out_y_end and out_x_start < out_x_end:
                I_0_shifted[out_y_start:out_y_end, out_x_start:out_x_end] = (
                    I_0[
                        by + src_y_start : by + src_y_end,
                        bx + src_x_start : bx + src_x_end,
                    ]
                )

            # -- Shift I_1 backward by (-dx, -dy) --
            dst_y1 = by - dy
            dst_x1 = bx - dx

            src_y_start = 0
            src_x_start = 0
            src_y_end = bh
            src_x_end = bw

            out_y_start = dst_y1
            out_x_start = dst_x1
            out_y_end = dst_y1 + bh
            out_x_end = dst_x1 + bw

            if out_y_start < 0:
                src_y_start -= out_y_start
                out_y_start = 0
            if out_x_start < 0:
                src_x_start -= out_x_start
                out_x_start = 0
            if out_y_end > h:
                src_y_end -= out_y_end - h
                out_y_end = h
            if out_x_end > w:
                src_x_end -= out_x_end - w
                out_x_end = w

            if out_y_start < out_y_end and out_x_start < out_x_end:
                I_1_shifted[out_y_start:out_y_end, out_x_start:out_x_end] = (
                    I_1[
                        by + src_y_start : by + src_y_end,
                        bx + src_x_start : bx + src_x_end,
                    ]
                )

    return I_0_shifted, I_1_shifted


def _read_split_list(data_dir: Path, split: str) -> list[str]:
    """Read triplet IDs from tri_trainlist.txt / tri_testlist.txt."""
    filename = {
        "train": "tri_trainlist.txt",
        "test": "tri_testlist.txt",
    }[split]
    list_path = data_dir / filename
    if not list_path.exists():
        raise FileNotFoundError(f"Split list not found: {list_path}")
    triplet_ids: list[str] = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                triplet_ids.append(line)
    return triplet_ids


def _process_triplet(
    args: tuple[str, Path, Path, Path, int, bool],
) -> str | None:
    """Process a single triplet: load frames + flow, prealign, save.

    Returns triplet_id on failure, None on success.
    """
    triplet_id, data_dir, flow_dir, output_dir, block_size, force = args

    parts = triplet_id.split("/")
    seq_id, trip_id = parts[0], parts[1]

    # Input paths
    seq_dir = data_dir / "sequences" / seq_id / trip_id
    im1_path = seq_dir / "im1.png"
    im3_path = seq_dir / "im3.png"
    flow_path = flow_dir / seq_id / f"{trip_id}.npy"

    # Output paths
    out_dir = output_dir / seq_id / trip_id
    out_im1_path = out_dir / "im1_aligned.png"
    out_im3_path = out_dir / "im3_aligned.png"
    meta_path = out_dir / "meta.json"

    # Skip only when outputs match the current flow timestamp and block_size.
    if not force and out_im1_path.exists() and out_im3_path.exists():
        if flow_path.exists():
            flow_stat = flow_path.stat()
            flow_mtime = flow_stat.st_mtime
            out_mtime = min(out_im1_path.stat().st_mtime, out_im3_path.stat().st_mtime)
            if out_mtime >= flow_mtime and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    meta = None
                if (
                    isinstance(meta, dict)
                    and meta.get("block_size") == block_size
                    and meta.get("flow_mtime_ns") == flow_stat.st_mtime_ns
                ):
                    return None
        else:
            return None

    # Validate inputs
    if not im1_path.exists():
        return triplet_id
    if not im3_path.exists():
        return triplet_id
    if not flow_path.exists():
        return triplet_id

    # Load data
    I_0 = np.array(Image.open(im1_path))
    I_1 = np.array(Image.open(im3_path))
    flow_01 = np.load(flow_path)

    # Validate shapes
    if flow_01.shape[:2] != I_0.shape[:2]:
        print(
            f"Shape mismatch for {triplet_id}: "
            f"frame {I_0.shape[:2]} vs flow {flow_01.shape[:2]}",
            file=sys.stderr,
        )
        return triplet_id

    # Pre-align
    I_0_shifted, I_1_shifted = prealign_frames(I_0, I_1, flow_01, block_size)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(I_0_shifted).save(out_im1_path)
    Image.fromarray(I_1_shifted).save(out_im3_path)
    meta_path.write_text(
        json.dumps(
            {
                "block_size": block_size,
                "flow_mtime_ns": flow_path.stat().st_mtime_ns,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch generate MV pre-aligned frames for Vimeo90K triplets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Vimeo90K root directory (containing sequences/).",
    )
    parser.add_argument(
        "--flow-dir",
        type=Path,
        required=True,
        help="Directory with dense flow .npy files (output of mv_to_dense.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for pre-aligned frames.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split to process (default: both).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size for integer-pixel shifts (default: 16).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help=f"Number of parallel workers (default: {cpu_count()}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all outputs even if they exist and are up-to-date.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    flow_dir: Path = args.flow_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)
    if not flow_dir.exists():
        print(f"Flow directory does not exist: {flow_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect triplet IDs
    triplet_ids: list[str] = []
    splits_to_process = (
        ["train", "test"] if args.split == "both" else [args.split]
    )
    for split in splits_to_process:
        ids = _read_split_list(data_dir, split)
        print(f"[{split}] Found {len(ids)} triplets")
        triplet_ids.extend(ids)

    if not triplet_ids:
        print("No triplets found. Check --data-dir and --split.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Processing {len(triplet_ids)} triplets with {args.workers} workers, "
        f"block_size={args.block_size}..."
    )

    tasks = [
        (tid, data_dir, flow_dir, output_dir, args.block_size, args.force)
        for tid in triplet_ids
    ]

    failed: list[str] = []
    with Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_triplet, tasks),
            total=len(tasks),
            desc="Pre-aligning frames",
            unit="triplet",
        ):
            if result is not None:
                failed.append(result)

    n_success = len(tasks) - len(failed)
    print(f"\nDone: {n_success}/{len(tasks)} succeeded.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for tid in failed[:20]:
            print(f"  {tid}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
