"""Download Xiph.org 1080p test sequences for VFI evaluation.

Downloads Y4M files from media.xiph.org/video/derf/, extracts frames with
ffmpeg, and creates VFI triplets (frame0, ground-truth middle, frame2).

These are the standard Xiph 1080p sequences used in VFI benchmarks (RIFE,
IFRNet, EMA-VFI, etc.).  The RIFE HD benchmark uses BlueSky, Kimono1,
ParkScene, sunflower from the same derf collection (as raw YUV); here we
download the full Y4M and extract all non-overlapping triplets.

Usage:
    pixi run python anvil_exp01/data/download_xiph.py
    pixi run python anvil_exp01/data/download_xiph.py --sequences crowd_run park_joy
    pixi run python anvil_exp01/data/download_xiph.py --data-dir /tmp/xiph_test
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

# ── Sequence catalogue ──────────────────────────────────────────────────────
#
# Maps a short name to the exact Y4M filename on media.xiph.org/video/derf/y4m/.
# These are the 1080p sequences most commonly cited in VFI literature.

SEQUENCES: dict[str, str] = {
    "crowd_run": "crowd_run_1080p50.y4m",
    "park_joy": "park_joy_1080p50.y4m",
    "old_town_cross": "old_town_cross_1080p50.y4m",
    "in_to_tree": "in_to_tree_1080p50.y4m",
    "ducks_take_off": "ducks_take_off_1080p50.y4m",
    "sunflower": "sunflower_1080p25.y4m",
    "pedestrian_area": "pedestrian_area_1080p25.y4m",
    "rush_hour": "rush_hour_1080p25.y4m",
    "station2": "station2_1080p25.y4m",
    "tractor": "tractor_1080p25.y4m",
    "blue_sky": "blue_sky_1080p25.y4m",
    "riverbed": "riverbed_1080p25.y4m",
}

BASE_URL = "https://media.xiph.org/video/derf/y4m"

DEFAULT_DATA_DIR = Path("data/xiph_1080p")


# ── Download ─────────────────────────────────────────────────────────────────

class _DownloadProgress(tqdm):
    """tqdm wrapper for urllib reporthook."""

    def update_to(self, blocks: int = 1, bsize: int = 1, total: int = -1) -> None:
        if total > 0:
            self.total = total
        self.update(blocks * bsize - self.n)


def download_y4m(name: str, filename: str, dest: Path) -> bool:
    """Download a single Y4M file.  Returns True on success."""
    url = f"{BASE_URL}/{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[{name}] Already downloaded: {dest.name} ({dest.stat().st_size / (1024**3):.2f} GB)")
        return True

    tmp = dest.with_suffix(".y4m.part")
    print(f"[{name}] Downloading {url}")

    try:
        with _DownloadProgress(
            unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=dest.name,
        ) as pbar:
            urllib.request.urlretrieve(url, tmp, reporthook=pbar.update_to)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"[{name}] WARNING: download failed ({exc}), skipping")
        return False

    tmp.rename(dest)
    print(f"[{name}] Downloaded {dest.name} ({dest.stat().st_size / (1024**3):.2f} GB)")
    return True


# ── Frame extraction ─────────────────────────────────────────────────────────

def extract_frames(y4m_path: Path, frames_dir: Path) -> int:
    """Extract all frames from a Y4M file to PNGs.  Returns frame count."""
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted (look for at least frame 00001.png)
    existing = sorted(frames_dir.glob("*.png"))
    if len(existing) >= 3:
        return len(existing)

    cmd = [
        "ffmpeg", "-y", "-i", str(y4m_path),
        "-vsync", "0",
        str(frames_dir / "%05d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr[:500]}")
        raise RuntimeError(f"ffmpeg failed for {y4m_path.name}")

    extracted = sorted(frames_dir.glob("*.png"))
    return len(extracted)


# ── Triplet creation ─────────────────────────────────────────────────────────

def create_triplets(
    name: str,
    frames_dir: Path,
    seq_out_dir: Path,
    num_frames: int,
) -> list[str]:
    """Create non-overlapping VFI triplets from extracted frames.

    For frames [1, 2, 3, 4, 5, 6, ...] we create:
      triplet 00001: frame 1 (im1), frame 2 (im2=GT), frame 3 (im3)
      triplet 00002: frame 3 (im1), frame 4 (im2=GT), frame 5 (im3)
      triplet 00003: frame 5 (im1), frame 6 (im2=GT), frame 7 (im3)
      ...

    Non-overlapping stride of 2 so each frame appears in at most one triplet
    as an endpoint.
    """
    seq_out_dir.mkdir(parents=True, exist_ok=True)

    triplet_ids: list[str] = []
    triplet_idx = 0

    for start in range(1, num_frames - 1, 2):
        f0 = start
        f1 = start + 1  # ground-truth middle frame
        f2 = start + 2
        if f2 > num_frames:
            break

        triplet_idx += 1
        tid = f"{triplet_idx:05d}"
        triplet_dir = seq_out_dir / tid
        triplet_dir.mkdir(parents=True, exist_ok=True)

        for dst_name, frame_num in [("im1.png", f0), ("im2.png", f1), ("im3.png", f2)]:
            src = frames_dir / f"{frame_num:05d}.png"
            dst = triplet_dir / dst_name
            if not dst.exists():
                # Hard-link to save disk space; fall back to copy
                try:
                    dst.hardlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)

        triplet_ids.append(f"{name}/{tid}")

    return triplet_ids


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Xiph.org 1080p sequences for VFI evaluation.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root output directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=list(SEQUENCES.keys()),
        default=None,
        help="Download only these sequences (default: all)",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep the raw extracted frame directory (default: delete after triplets are created)",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    y4m_dir = data_dir / "y4m"
    seq_dir = data_dir / "sequences"
    tmp_frames_dir = data_dir / "_frames"  # temporary, deleted unless --keep-frames

    selected = args.sequences or list(SEQUENCES.keys())
    print(f"Xiph 1080p VFI dataset builder")
    print(f"  Output:     {data_dir}")
    print(f"  Sequences:  {', '.join(selected)}")
    print()

    # Verify ffmpeg is available
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found in PATH. Install ffmpeg first.")
        sys.exit(1)

    all_triplet_ids: list[str] = []
    summary: list[tuple[str, int, str]] = []  # (name, triplets, status)

    for name in selected:
        filename = SEQUENCES[name]
        y4m_path = y4m_dir / filename

        # 1. Download
        ok = download_y4m(name, filename, y4m_path)
        if not ok:
            summary.append((name, 0, "DOWNLOAD FAILED"))
            continue

        # 2. Extract frames
        frames_dir = tmp_frames_dir / name
        print(f"[{name}] Extracting frames ...")
        try:
            num_frames = extract_frames(y4m_path, frames_dir)
        except RuntimeError:
            summary.append((name, 0, "EXTRACT FAILED"))
            continue
        print(f"[{name}] {num_frames} frames extracted")

        # 3. Create triplets
        seq_out = seq_dir / name
        triplet_ids = create_triplets(name, frames_dir, seq_out, num_frames)
        all_triplet_ids.extend(triplet_ids)
        print(f"[{name}] {len(triplet_ids)} triplets created")

        # 4. Clean up raw frames
        if not args.keep_frames and frames_dir.exists():
            shutil.rmtree(frames_dir)

        summary.append((name, len(triplet_ids), "OK"))

    # Clean up empty tmp dir
    if tmp_frames_dir.exists() and not any(tmp_frames_dir.iterdir()):
        tmp_frames_dir.rmdir()

    # 5. Write tri_testlist.txt
    if all_triplet_ids:
        testlist_path = data_dir / "tri_testlist.txt"
        testlist_path.write_text("\n".join(all_triplet_ids) + "\n", encoding="utf-8")
        print(f"\nWrote {testlist_path} ({len(all_triplet_ids)} triplets)")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Sequence':<20} {'Triplets':>8}   Status")
    print("-" * 60)
    total_triplets = 0
    for name, n, status in summary:
        print(f"{name:<20} {n:>8}   {status}")
        total_triplets += n
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_triplets:>8}")
    print(f"\nDataset root: {data_dir}")


if __name__ == "__main__":
    main()
