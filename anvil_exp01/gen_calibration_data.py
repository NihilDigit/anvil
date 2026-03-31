"""Generate INT8 calibration data for QNN quantization.

Samples representative triplets from the Vimeo90K training set across
motion bins (small/medium/large) and saves prealigned 6ch inputs as
float32 .raw files in CHW layout — the format qnn-onnx-converter expects.

Supports two modes:
  - ANVIL (default): prealigned v2 6ch from Vimeo train set
  - RIFE (--rife-mode): raw frames downsampled to RIFE input resolution
    --source vimeo (default): Vimeo train frames (no test-set leakage)
    --source xiph: Xiph test frames (legacy fallback)

Usage:
    pixi run python -m anvil_exp01.gen_calibration_data
    pixi run python -m anvil_exp01.gen_calibration_data --n-samples 100 --resolution 1080p
    pixi run python -m anvil_exp01.gen_calibration_data --rife-mode 360p --n-samples 100
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np


RIFE_RESOLUTIONS = {
    "360p": (384, 640),
    "480p": (512, 864),
}


def load_image(path: Path) -> np.ndarray:
    """Load image as float32 [0, 1] HWC."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _generate_rife_calibration(args: argparse.Namespace) -> None:
    """Generate RIFE calibration data from Xiph 1080p raw frames."""
    h, w = RIFE_RESOLUTIONS[args.rife_mode]

    list_path = args.xiph_dir / "tri_testlist.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Xiph test list not found: {list_path}")

    triplets = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    samples = random.sample(triplets, min(args.n_samples, len(triplets)))
    print(f"RIFE {args.rife_mode} ({h}x{w}): sampling {len(samples)} from {len(triplets)} Xiph triplets")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_list_lines: list[str] = []
    skipped = 0

    for i, tid in enumerate(samples):
        seq, frame_id = tid.split("/")
        seq_dir = args.xiph_dir / "sequences" / seq / frame_id

        im1_path = seq_dir / "im1.png"
        im3_path = seq_dir / "im3.png"
        if not im1_path.exists() or not im3_path.exists():
            skipped += 1
            continue

        im1 = load_image(im1_path)
        im3 = load_image(im3_path)

        # Downsample to RIFE input resolution
        im1 = cv2.resize(im1, (w, h), interpolation=cv2.INTER_LINEAR)
        im3 = cv2.resize(im3, (w, h), interpolation=cv2.INTER_LINEAR)

        # 6ch CHW tensor (same format as RIFE model input)
        inp = np.concatenate([im1, im3], axis=2).transpose(2, 0, 1)
        inp = np.ascontiguousarray(inp, dtype=np.float32)

        raw_name = f"calib_{i:04d}.raw"
        inp.tofile(args.out_dir / raw_name)
        input_list_lines.append(raw_name)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] {tid} shape={inp.shape}")

    out_list = args.out_dir / "input_list.txt"
    out_list.write_text("\n".join(input_list_lines) + "\n")

    print(f"\nDone: {len(input_list_lines)} RIFE {args.rife_mode} calibration files in {args.out_dir}/")
    if skipped:
        print(f"  ({skipped} skipped — frames not found)")
    print(f"  Input list: {out_list}")
    if input_list_lines:
        print(f"  Shape: (6, {h}, {w}) float32 per sample")


def _generate_rife_vimeo_calibration(args: argparse.Namespace) -> None:
    """Generate RIFE calibration data from Vimeo90K train raw frames (no test-set leakage)."""
    h, w = RIFE_RESOLUTIONS[args.rife_mode]

    # Load train triplet IDs
    train_set: set[str] = set()
    with open(args.train_list) as f:
        for line in f:
            tid = line.strip()
            if tid:
                train_set.add(tid)

    # Stratified sampling if motion labels available, else random
    motion_bins: dict[str, list[str]] = {"small": [], "medium": [], "large": []}
    if args.motion_csv.exists():
        with open(args.motion_csv) as f:
            _header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    tid, _, mbin = parts[0], parts[1], parts[2]
                    if tid in train_set and mbin in motion_bins:
                        motion_bins[mbin].append(tid)

    total_labeled = sum(len(v) for v in motion_bins.values())
    if total_labeled > 0:
        n = args.n_samples
        n_large = max(n // 5, min(len(motion_bins["large"]), n))
        n_remaining = n - n_large
        n_small_pool = len(motion_bins["small"])
        n_medium_pool = len(motion_bins["medium"])
        pool_total = n_small_pool + n_medium_pool
        if pool_total > 0:
            n_small = int(n_remaining * n_small_pool / pool_total)
            n_medium = n_remaining - n_small
        else:
            n_small = n_medium = 0

        samples: list[str] = []
        for mbin, count in [("small", n_small), ("medium", n_medium), ("large", n_large)]:
            pool = motion_bins[mbin]
            count = min(count, len(pool))
            samples.extend(random.sample(pool, count))
        random.shuffle(samples)
        print(
            f"RIFE {args.rife_mode} Vimeo stratified: "
            f"{n_small} small + {n_medium} medium + {min(n_large, len(motion_bins['large']))} large "
            f"= {len(samples)} total"
        )
    else:
        all_train = sorted(train_set)
        samples = random.sample(all_train, min(args.n_samples, len(all_train)))
        print(f"RIFE {args.rife_mode} Vimeo random (no motion labels): {len(samples)} samples")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_list_lines: list[str] = []
    skipped = 0

    for i, tid in enumerate(samples):
        im1_path = args.vimeo_seq_dir / tid / "im1.png"
        im3_path = args.vimeo_seq_dir / tid / "im3.png"

        if not im1_path.exists() or not im3_path.exists():
            skipped += 1
            continue

        im1 = load_image(im1_path)
        im3 = load_image(im3_path)

        # Resize to RIFE input resolution
        im1 = cv2.resize(im1, (w, h), interpolation=cv2.INTER_LINEAR)
        im3 = cv2.resize(im3, (w, h), interpolation=cv2.INTER_LINEAR)

        # 6ch CHW tensor (same format as RIFE model input)
        inp = np.concatenate([im1, im3], axis=2).transpose(2, 0, 1)
        inp = np.ascontiguousarray(inp, dtype=np.float32)

        raw_name = f"calib_{i:04d}.raw"
        inp.tofile(args.out_dir / raw_name)
        input_list_lines.append(raw_name)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] {tid} shape={inp.shape}")

    out_list = args.out_dir / "input_list.txt"
    out_list.write_text("\n".join(input_list_lines) + "\n")

    print(f"\nDone: {len(input_list_lines)} RIFE {args.rife_mode} Vimeo calibration files in {args.out_dir}/")
    if skipped:
        print(f"  ({skipped} skipped — frames not found)")
    print(f"  Input list: {out_list}")
    if input_list_lines:
        print(f"  Shape: (6, {h}, {w}) float32 per sample")


def _generate_anvil_xiph_calibration(args: argparse.Namespace) -> None:
    """Generate ANVIL calibration data from Xiph 1080p prealigned_v2 frames (native 1080p)."""
    xiph_prealigned = args.xiph_dir / "prealigned_v2"
    list_path = args.xiph_dir / "tri_testlist.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Xiph test list not found: {list_path}")
    if not xiph_prealigned.is_dir():
        raise FileNotFoundError(f"Xiph prealigned_v2 not found: {xiph_prealigned}")

    triplets = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    samples = random.sample(triplets, min(args.n_samples, len(triplets)))
    print(f"ANVIL Xiph 1080p (native): sampling {len(samples)} from {len(triplets)} Xiph triplets")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_list_lines: list[str] = []
    skipped = 0

    for i, tid in enumerate(samples):
        seq, frame_id = tid.split("/")
        pre_dir = xiph_prealigned / seq / frame_id
        im1_path = pre_dir / "im1_aligned.png"
        im3_path = pre_dir / "im3_aligned.png"

        if not im1_path.exists() or not im3_path.exists():
            skipped += 1
            continue

        i0 = load_image(im1_path)
        i1 = load_image(im3_path)

        inp = np.concatenate([i0, i1], axis=2).transpose(2, 0, 1)
        inp = np.ascontiguousarray(inp, dtype=np.float32)

        raw_name = f"calib_{i:04d}.raw"
        inp.tofile(args.out_dir / raw_name)
        input_list_lines.append(raw_name)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] {tid} shape={inp.shape}")

    out_list = args.out_dir / "input_list.txt"
    out_list.write_text("\n".join(input_list_lines) + "\n")

    print(f"\nDone: {len(input_list_lines)} ANVIL Xiph 1080p calibration files in {args.out_dir}/")
    if skipped:
        print(f"  ({skipped} skipped — prealigned files not found)")
    print(f"  Input list: {out_list}")
    if input_list_lines:
        print(f"  Shape: {inp.shape} float32 per sample")


def main():
    parser = argparse.ArgumentParser(
        description="Generate INT8 calibration data for QNN"
    )
    parser.add_argument(
        "--prealigned-dir",
        type=Path,
        default=Path("data/vimeo_triplet/prealigned_v2"),
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=Path("data/vimeo_triplet/tri_trainlist.txt"),
    )
    parser.add_argument(
        "--motion-csv",
        type=Path,
        default=Path("data/motion_labels.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("npu_bench/calibration"),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Total calibration samples (stratified across motion bins)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="native",
        choices=["native", "1080p"],
        help="Output resolution: native (256x448) or 1080p (resize for 1080p models)",
    )
    parser.add_argument(
        "--rife-mode",
        choices=["360p", "480p"],
        default=None,
        help="Generate RIFE calibration data (Xiph or Vimeo frames depending on --source)",
    )
    parser.add_argument(
        "--source",
        choices=["vimeo", "xiph"],
        default="vimeo",
        help="Data source: vimeo train set (default) or xiph test set. Works for both ANVIL and RIFE modes.",
    )
    parser.add_argument(
        "--allow-test-leakage",
        action="store_true",
        help="Required when --source xiph to acknowledge test-set leakage. Prefer --source vimeo.",
    )
    parser.add_argument(
        "--xiph-dir",
        type=Path,
        default=Path("data/xiph_1080p"),
        help="Xiph 1080p dataset directory (for --rife-mode or --source xiph)",
    )
    parser.add_argument(
        "--vimeo-seq-dir",
        type=Path,
        default=Path("data/vimeo_triplet/sequences"),
        help="Vimeo90K sequences directory (for --rife-mode --source vimeo)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.source == "xiph" and not args.allow_test_leakage:
        print(
            "ERROR: --source xiph samples calibration data from the test set. "
            "This causes test-set leakage and is NOT recommended.\n"
            "  Use --source vimeo (default) for proper calibration.\n"
            "  If you understand the risk, pass --allow-test-leakage.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.rife_mode:
        if args.source == "vimeo":
            _generate_rife_vimeo_calibration(args)
        else:
            _generate_rife_calibration(args)
        return

    if args.source == "xiph":
        _generate_anvil_xiph_calibration(args)
        return

    # Load motion labels for stratified sampling (Vimeo source)
    motion_bins: dict[str, list[str]] = {"small": [], "medium": [], "large": []}
    train_set: set[str] = set()

    with open(args.train_list) as f:
        for line in f:
            tid = line.strip()
            if tid:
                train_set.add(tid)

    if args.motion_csv.exists():
        with open(args.motion_csv) as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    tid, _, mbin = parts[0], parts[1], parts[2]
                    if tid in train_set and mbin in motion_bins:
                        motion_bins[mbin].append(tid)

    # Stratified sampling: proportional to bin sizes, but ensure large motion
    # gets at least 20% representation (it's the bottleneck)
    total_labeled = sum(len(v) for v in motion_bins.values())
    if total_labeled > 0:
        n = args.n_samples
        # Ensure minimum 20% large motion, rest proportional
        n_large = max(n // 5, min(len(motion_bins["large"]), n))
        n_remaining = n - n_large
        n_small_pool = len(motion_bins["small"])
        n_medium_pool = len(motion_bins["medium"])
        pool_total = n_small_pool + n_medium_pool
        if pool_total > 0:
            n_small = int(n_remaining * n_small_pool / pool_total)
            n_medium = n_remaining - n_small
        else:
            n_small = n_medium = 0

        samples = []
        for mbin, count in [("small", n_small), ("medium", n_medium), ("large", n_large)]:
            pool = motion_bins[mbin]
            count = min(count, len(pool))
            samples.extend(random.sample(pool, count))
        random.shuffle(samples)
        print(f"Stratified sampling: {n_small} small + {n_medium} medium + {min(n_large, len(motion_bins['large']))} large = {len(samples)} total")
    else:
        # Fallback: random from train set
        all_train = sorted(train_set)
        samples = random.sample(all_train, min(args.n_samples, len(all_train)))
        print(f"Random sampling (no motion labels): {len(samples)} samples")

    # Generate .raw files
    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_list_lines = []
    skipped = 0

    for i, tid in enumerate(samples):
        pre_dir = args.prealigned_dir / tid
        im1_path = pre_dir / "im1_aligned.png"
        im3_path = pre_dir / "im3_aligned.png"

        if not im1_path.exists() or not im3_path.exists():
            skipped += 1
            continue

        i0 = load_image(im1_path)  # (H, W, 3) float32 [0, 1]
        i1 = load_image(im3_path)

        # Resize if needed (for 1080p calibration)
        if args.resolution == "1080p":
            i0 = cv2.resize(i0, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            i1 = cv2.resize(i1, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        # Concatenate to 6ch CHW tensor — matches D-nomv input format
        inp = np.concatenate([i0, i1], axis=2)  # (H, W, 6)
        inp = inp.transpose(2, 0, 1)  # (6, H, W)
        inp = np.ascontiguousarray(inp, dtype=np.float32)

        raw_name = f"calib_{i:04d}.raw"
        raw_path = args.out_dir / raw_name
        inp.tofile(raw_path)
        input_list_lines.append(raw_name)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] {tid} shape={inp.shape}")

    # Write input list file (required by qnn-onnx-converter)
    # Bare filenames — bench_npu.sh generates a Docker-aware version with container paths
    list_path = args.out_dir / "input_list.txt"
    list_path.write_text("\n".join(input_list_lines) + "\n")

    print(f"\nDone: {len(input_list_lines)} calibration files in {args.out_dir}/")
    if skipped:
        print(f"  ({skipped} skipped — prealigned files not found)")
    print(f"  Input list: {list_path}")
    print(f"  Shape: {inp.shape} float32 per sample")


if __name__ == "__main__":
    main()
