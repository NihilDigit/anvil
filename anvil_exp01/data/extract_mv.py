"""Batch extract H.264 motion vectors from Vimeo90K triplets.

For each triplet, encodes I_0 (im1.png) and I_1 (im3.png) into a temporary
2-frame H.264 video, then reads back the P-frame MVs using mv-extractor.
Saves per-triplet .npz containing block-level MV data.

Usage:
    pixi run python anvil_exp01/data/extract_mv.py \
        --data-dir /path/to/vimeo_triplet \
        --output-dir data/mv_cache \
        --workers 8 \
        --skip-existing
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _read_split_list(data_dir: Path, split: str) -> list[str]:
    """Read triplet IDs from tri_trainlist.txt / tri_testlist.txt.

    Each line is like '00001/0001' (sequence/triplet).
    """
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


def _encode_two_frames(im1_path: str, im3_path: str, output_mp4: str) -> bool:
    """Encode two PNG frames into a 2-frame H.264 video via FFmpeg.

    Uses a concat demuxer approach: write a file list, feed to FFmpeg with
    -framerate 1, libx264 with no B-frames and very large keyint so frame 2
    is a P-frame.

    Returns True on success.
    """
    tmpdir = os.path.dirname(output_mp4)

    # Write a concat file listing the two frames in order
    concat_path = os.path.join(tmpdir, "concat.txt")
    im1_abs = os.path.abspath(im1_path)
    im3_abs = os.path.abspath(im3_path)

    with open(concat_path, "w") as f:
        # Use file protocol for absolute paths
        f.write(f"file '{im1_abs}'\n")
        f.write(f"duration 1\n")
        f.write(f"file '{im3_abs}'\n")
        f.write(f"duration 1\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_path,
        "-c:v", "libx264",
        "-preset", "medium",
        "-x264-params", "keyint=999:bframes=0",
        "-pix_fmt", "yuv420p",
        "-an",
        output_mp4,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(
            f"FFmpeg failed for {im1_path}: {result.stderr.decode(errors='replace')[-500:]}",
            file=sys.stderr,
        )
        return False
    return True


def _extract_mv_from_video(video_path: str) -> tuple[np.ndarray, int] | None:
    """Open a 2-frame H.264 video and extract MV from the P-frame.

    Returns (mv_array, motion_scale) or None on failure.

    Note: mvextractor's ndarray layout can vary by build. In our environment
    it is 10 columns:
      [source, blockw, blockh, srcx, srcy, dstx, dsty, motion_x, motion_y, motion_scale]
    (no separate flags column).
    """
    # Import here so multiprocessing workers each get their own import
    from mvextractor.videocap import VideoCap

    cap = VideoCap()
    if not cap.open(video_path):
        print(f"Failed to open video: {video_path}", file=sys.stderr)
        return None

    motion_scale: int = 4  # H.264 default quarter-pixel precision
    mv_data: np.ndarray | None = None
    saw_p_frame = False

    while True:
        ret, frame, mv, frame_type = cap.read()
        if not ret:
            break
        if frame_type == "P":
            saw_p_frame = True
            if mv is not None and len(mv) > 0:
                mv_data = np.array(mv)
                # mv-extractor exposes motion_scale as attribute on the array
                # or we can read it from the library; H.264 is always 4
                if hasattr(mv_data, "motion_scale"):
                    motion_scale = int(mv_data.motion_scale)
                break

    cap.release()

    if mv_data is None:
        if saw_p_frame:
            # Some clips legitimately encode the P-frame with no inter-block
            # motion vectors (e.g. fully intra-coded content). Treat this as
            # zero motion instead of a hard failure so downstream Route D
            # pipelines can still produce a valid zero flow field.
            mv_data = np.empty((0, 10), dtype=np.float32)
        else:
            return None
    return mv_data, motion_scale


def process_triplet(args: tuple[str, Path, Path, bool]) -> str | None:
    """Process a single triplet: encode -> extract MV -> save .npz.

    Args is a tuple of (triplet_id, data_dir, output_dir, skip_existing).
    Returns the triplet_id on failure, None on success.
    """
    triplet_id, data_dir, output_dir, skip_existing = args

    # Determine paths
    # triplet_id is like "00001/0001"
    seq_dir = data_dir / "sequences" / triplet_id
    im1_path = seq_dir / "im1.png"
    im3_path = seq_dir / "im3.png"

    # Output .npz path mirrors the triplet structure
    # e.g. output_dir/00001/0001.npz
    parts = triplet_id.split("/")
    out_subdir = output_dir / parts[0]
    out_path = out_subdir / f"{parts[1]}.npz"

    if skip_existing and out_path.exists():
        return None

    # Validate inputs exist
    if not im1_path.exists() or not im3_path.exists():
        return triplet_id  # signal failure

    # Create output directory
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Work in a temporary directory
    with tempfile.TemporaryDirectory(prefix="anvil_mv_") as tmpdir:
        tmp_mp4 = os.path.join(tmpdir, "temp.mp4")

        # Step 1: Encode two frames into H.264 video
        if not _encode_two_frames(str(im1_path), str(im3_path), tmp_mp4):
            return triplet_id

        # Step 2: Extract P-frame MVs
        result = _extract_mv_from_video(tmp_mp4)
        if result is None:
            return triplet_id

        mv_array, motion_scale = result

        # Parse column layout robustly across mvextractor variants.
        n_cols = mv_array.shape[1]
        if n_cols == 10:
            # [source, blockw, blockh, srcx, srcy, dstx, dsty, motion_x, motion_y, motion_scale]
            col_motion_x = 7
            col_motion_y = 8
            scale_vals = mv_array[:, 9].astype(np.int32)
        elif n_cols >= 11:
            # [source, blockw, blockh, srcx, srcy, dstx, dsty, flags, motion_x, motion_y, motion_scale]
            col_motion_x = 8
            col_motion_y = 9
            scale_vals = mv_array[:, 10].astype(np.int32)
        else:
            print(f"Unexpected MV column count ({n_cols}) for {triplet_id}", file=sys.stderr)
            return triplet_id

        # Prefer per-row motion_scale from the array when available; fallback
        # to the extractor-level motion_scale value.
        if len(scale_vals) > 0:
            motion_scale = int(np.median(scale_vals))

        np.savez_compressed(
            out_path,
            blockw=mv_array[:, 1].astype(np.int16),
            blockh=mv_array[:, 2].astype(np.int16),
            srcx=mv_array[:, 3].astype(np.int16),
            srcy=mv_array[:, 4].astype(np.int16),
            dstx=mv_array[:, 5].astype(np.int16),
            dsty=mv_array[:, 6].astype(np.int16),
            motion_x=mv_array[:, col_motion_x].astype(np.int16),
            motion_y=mv_array[:, col_motion_y].astype(np.int16),
            motion_scale=np.int16(motion_scale),
        )

    return None  # success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch extract H.264 motion vectors from Vimeo90K triplets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Vimeo90K root directory (containing sequences/, tri_trainlist.txt, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mv_cache"),
        help="Directory to save .npz files (default: data/mv_cache/)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split to process (default: both)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help=f"Number of parallel workers (default: {cpu_count()})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip triplets whose .npz already exists",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
        f"Processing {len(triplet_ids)} triplets with {args.workers} workers..."
    )

    # Build task list
    tasks = [
        (tid, data_dir, output_dir, args.skip_existing) for tid in triplet_ids
    ]

    # Process with multiprocessing pool
    failed: list[str] = []
    with Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_triplet, tasks),
            total=len(tasks),
            desc="Extracting MVs",
            unit="triplet",
        ):
            if result is not None:
                failed.append(result)

    # Report
    n_success = len(tasks) - len(failed)
    print(f"\nDone: {n_success}/{len(tasks)} succeeded.")
    fail_path = output_dir / "failed_triplets.txt"
    if failed:
        print(f"Failed ({len(failed)}):")
        for tid in failed[:20]:
            print(f"  {tid}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

        # Write failed list for re-processing
        with open(fail_path, "w") as f:
            for tid in failed:
                f.write(f"{tid}\n")
        print(f"Full failed list written to {fail_path}")
    elif fail_path.exists():
        fail_path.unlink()


if __name__ == "__main__":
    main()
