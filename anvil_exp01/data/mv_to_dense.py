"""Convert block-level H.264 motion vectors to pixel-level dense flow fields.

Provides importable functions (mv_to_dense, smooth_flow) and a CLI mode
for batch conversion of .npz MV files to .npy dense flow fields.

Usage:
    pixi run python anvil_exp01/data/mv_to_dense.py \
        --mv-dir data/mv_cache \
        --output-dir data/dense_flow \
        --smooth \
        --frame-height 256 \
        --frame-width 448
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def mv_to_dense(
    mv_data: dict[str, np.ndarray],
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Convert block-level MVs from .npz to a pixel-level dense flow field.

    Args:
        mv_data: dict from np.load() on an MV .npz file, containing keys:
            blockw, blockh, dstx, dsty, motion_x, motion_y, motion_scale.
        frame_h: Frame height in pixels.
        frame_w: Frame width in pixels.

    Returns:
        Dense flow field of shape (frame_h, frame_w, 2), dtype float32.
        Channel 0 is horizontal (x) motion, channel 1 is vertical (y) motion,
        both in pixel units.
    """
    flow = np.zeros((frame_h, frame_w, 2), dtype=np.float32)

    blockw = mv_data["blockw"].astype(np.int32)
    blockh = mv_data["blockh"].astype(np.int32)
    dstx = mv_data["dstx"].astype(np.int32)
    dsty = mv_data["dsty"].astype(np.int32)
    motion_x = mv_data["motion_x"].astype(np.float32)
    motion_y = mv_data["motion_y"].astype(np.float32)
    motion_scale = float(mv_data["motion_scale"])

    # Convert to pixel units and flip direction.
    # H.264 P-frame MVs are current->reference (I1 -> I0) in our extraction
    # setup; downstream prealignment expects forward half-shift direction
    # (I0 -> I1), so use the negated displacement.
    mx = -(motion_x / motion_scale)
    my = -(motion_y / motion_scale)

    n_blocks = len(blockw)
    for i in range(n_blocks):
        bw = int(blockw[i])
        bh = int(blockh[i])
        dx = int(dstx[i])
        dy = int(dsty[i])

        # Clamp destination block region to frame boundaries
        y_start = max(dy, 0)
        y_end = min(dy + bh, frame_h)
        x_start = max(dx, 0)
        x_end = min(dx + bw, frame_w)

        if y_start >= y_end or x_start >= x_end:
            continue

        flow[y_start:y_end, x_start:x_end, 0] = mx[i]
        flow[y_start:y_end, x_start:x_end, 1] = my[i]

    return flow


def smooth_flow(flow: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a mean filter to smooth block boundaries in a dense flow field.

    Args:
        flow: Dense flow field of shape (H, W, 2), dtype float32.
        kernel_size: Size of the square averaging kernel (must be odd).

    Returns:
        Smoothed flow field, same shape and dtype.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if kernel_size == 1:
        return flow.copy()

    h, w, c = flow.shape
    pad = kernel_size // 2
    smoothed = np.empty_like(flow)

    for ch in range(c):
        channel = flow[:, :, ch]

        # Pad with edge replication
        padded = np.pad(channel, pad, mode="edge")

        # Compute mean filter via cumulative sum for efficiency
        # Vertical cumsum
        cum = np.cumsum(padded, axis=0)
        cum = np.vstack([np.zeros((1, cum.shape[1]), dtype=cum.dtype), cum])
        vert = cum[kernel_size:, :] - cum[:-kernel_size, :]

        # Horizontal cumsum
        cum2 = np.cumsum(vert, axis=1)
        cum2 = np.hstack([np.zeros((cum2.shape[0], 1), dtype=cum2.dtype), cum2])
        box = cum2[:, kernel_size:] - cum2[:, :-kernel_size]

        smoothed[:, :, ch] = box / (kernel_size * kernel_size)

    return smoothed


def _process_single_npz(
    npz_path: Path,
    output_dir: Path,
    frame_h: int,
    frame_w: int,
    do_smooth: bool,
    kernel_size: int,
) -> bool:
    """Convert a single .npz MV file to a .npy dense flow.

    Returns True on success, False on failure.
    """
    try:
        mv_data = dict(np.load(npz_path))
    except Exception as e:
        print(f"Failed to load {npz_path}: {e}", file=sys.stderr)
        return False

    flow = mv_to_dense(mv_data, frame_h, frame_w)

    if do_smooth:
        flow = smooth_flow(flow, kernel_size=kernel_size)

    # Mirror the directory structure: parent_subdir/stem.npy
    # e.g. 00001/0001.npz -> 00001/0001.npy
    rel = npz_path.relative_to(npz_path.parent.parent)
    out_path = output_dir / rel.with_suffix(".npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_path, flow)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch convert block-level .npz MVs to dense .npy flow fields."
    )
    parser.add_argument(
        "--mv-dir",
        type=Path,
        required=True,
        help="Input directory containing .npz MV files (with subdirectory structure).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for .npy dense flow fields.",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply mean-filter smoothing to reduce block boundary artifacts.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Smoothing kernel size (odd integer, default: 3).",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=256,
        help="Frame height in pixels (default: 256 for Vimeo90K).",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=448,
        help="Frame width in pixels (default: 448 for Vimeo90K).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion if .npy already exists.",
    )
    args = parser.parse_args()

    mv_dir: Path = args.mv_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mv_dir.exists():
        print(f"MV directory does not exist: {mv_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all .npz files
    npz_files = sorted(mv_dir.rglob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {mv_dir}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Found {len(npz_files)} .npz files. "
        f"Frame size: {args.frame_height}x{args.frame_width}. "
        f"Smooth: {args.smooth} (kernel={args.kernel_size})."
    )

    n_success = 0
    n_skip = 0
    n_fail = 0

    for npz_path in tqdm(npz_files, desc="Converting to dense flow", unit="file"):
        # Check skip-existing
        if args.skip_existing:
            rel = npz_path.relative_to(mv_dir)
            out_path = output_dir / rel.with_suffix(".npy")
            if out_path.exists():
                n_skip += 1
                continue

        ok = _process_single_npz(
            npz_path,
            output_dir,
            args.frame_height,
            args.frame_width,
            args.smooth,
            args.kernel_size,
        )
        if ok:
            n_success += 1
        else:
            n_fail += 1

    print(
        f"\nDone: {n_success} converted, {n_skip} skipped, {n_fail} failed "
        f"(total {len(npz_files)})."
    )


if __name__ == "__main__":
    main()
