"""RAFT-based motion magnitude labeling for Vimeo90K triplets.

Computes optical flow from I_0 (im1.png) to I_1 (im3.png) using RAFT-Small,
derives per-triplet motion magnitude (median EPE), and classifies into bins.
Also saves the RAFT flow fields as .npy for later use as Oracle Flow.

Usage:
    pixi run python anvil_exp01/data/motion_label.py --data-dir data/vimeo_triplet
    pixi run python anvil_exp01/data/motion_label.py --data-dir data/vimeo_triplet --split both
    pixi run python anvil_exp01/data/motion_label.py --data-dir data/vimeo_triplet --resume
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from tqdm import tqdm

# Motion magnitude bin thresholds (in pixels).
BIN_SMALL_UPPER = 5.0
BIN_MEDIUM_UPPER = 20.0


def classify_motion(epe_median: float) -> str:
    """Classify a median EPE value into a motion bin."""
    if epe_median < BIN_SMALL_UPPER:
        return "small"
    elif epe_median < BIN_MEDIUM_UPPER:
        return "medium"
    else:
        return "large"


def load_image_as_tensor(path: Path) -> torch.Tensor:
    """Load a PNG image and return a uint8 tensor of shape (1, 3, H, W).

    Keep uint8 here and run the official torchvision RAFT transforms later.
    The preset handles dtype conversion / normalization correctly.
    """
    img = Image.open(path).convert("RGB")
    # HWC uint8 -> CHW uint8
    arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def compute_epe(flow: np.ndarray) -> np.ndarray:
    """Compute per-pixel End-Point Error magnitude from a (2, H, W) flow field."""
    return np.sqrt(flow[0] ** 2 + flow[1] ** 2)


def parse_triplet_list(list_file: Path) -> list[str]:
    """Parse a tri_trainlist.txt / tri_testlist.txt file.

    Each non-empty line is a triplet identifier like "00001/0001".
    """
    if not list_file.exists():
        raise FileNotFoundError(f"List file not found: {list_file}")
    triplets: list[str] = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            triplets.append(stripped)
    return triplets


def get_triplet_ids(data_dir: Path, split: str) -> list[str]:
    """Return triplet IDs for the requested split(s)."""
    triplets: list[str] = []
    if split in ("test", "both"):
        triplets.extend(parse_triplet_list(data_dir / "tri_testlist.txt"))
    if split in ("train", "both"):
        triplets.extend(parse_triplet_list(data_dir / "tri_trainlist.txt"))
    return triplets


def flow_output_path(output_flow_dir: Path, triplet_id: str) -> Path:
    """Return the .npy path for a given triplet's flow."""
    # triplet_id is like "00001/0001" -> save as "00001/0001.npy" (nested)
    parts = triplet_id.split("/")
    return output_flow_dir / parts[0] / f"{parts[1]}.npy"


def load_existing_csv(csv_path: Path) -> dict[str, tuple[float, str]]:
    """Load an existing motion label CSV into a dict keyed by triplet_id."""
    results: dict[str, tuple[float, str]] = {}
    if not csv_path.exists():
        return results
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["triplet_id"]] = (
                float(row["motion_magnitude"]),
                row["motion_bin"],
            )
    return results


def write_csv(
    csv_path: Path,
    results: dict[str, tuple[float, str]],
) -> None:
    """Write motion labels to CSV, sorted by triplet_id for reproducibility."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_ids = sorted(results.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["triplet_id", "motion_magnitude", "motion_bin"])
        for tid in sorted_ids:
            mag, bin_label = results[tid]
            writer.writerow([tid, f"{mag:.4f}", bin_label])


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    default_data_dir = project_root / "data" / "vimeo_triplet"
    default_flow_dir = project_root / "data" / "raft_flow"
    default_csv = project_root / "data" / "motion_labels.csv"

    parser = argparse.ArgumentParser(
        description="Compute RAFT-based motion magnitude labels for Vimeo90K triplets.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help=f"Vimeo90K dataset root (default: {default_data_dir})",
    )
    parser.add_argument(
        "--output-flow-dir",
        type=Path,
        default=default_flow_dir,
        help=f"Directory to save RAFT flow .npy files (default: {default_flow_dir})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_csv,
        help=f"Motion label CSV output path (default: {default_csv})",
    )
    parser.add_argument(
        "--split",
        choices=["test", "train", "both"],
        default="test",
        help="Which split to process (default: test)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for RAFT inference (default: cuda)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last processed triplet (skip existing flow files)",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    output_flow_dir: Path = args.output_flow_dir.resolve()
    output_csv: Path = args.output_csv.resolve()
    device = torch.device(args.device)

    # Validate dataset directory.
    sequences_dir = data_dir / "sequences"
    if not sequences_dir.is_dir():
        print(f"Error: sequences directory not found at {sequences_dir}", file=sys.stderr)
        sys.exit(1)

    # Gather triplet IDs.
    triplet_ids = get_triplet_ids(data_dir, args.split)
    if not triplet_ids:
        print("No triplets found for the requested split.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(triplet_ids)} triplets for split '{args.split}'.")

    # Create output directory.
    output_flow_dir.mkdir(parents=True, exist_ok=True)

    # Always load existing results to preserve entries from other splits.
    existing_results = load_existing_csv(output_csv)
    if existing_results:
        print(f"Loaded {len(existing_results)} existing entries from CSV.")

    # Determine which triplets need processing.
    if args.resume:
        to_process = [
            tid for tid in triplet_ids
            if tid not in existing_results
            or not flow_output_path(output_flow_dir, tid).exists()
        ]
        print(f"Remaining: {len(to_process)} triplets to process.")
    else:
        to_process = triplet_ids

    if not to_process:
        print("All triplets already processed. Nothing to do.")
        # Still write CSV in case it needs updating with sorted order.
        write_csv(output_csv, existing_results)
        return

    # Load RAFT-Small model.
    print(f"Loading RAFT-Small on {device} ...")
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights).eval().to(device)

    # Process triplets one at a time (8GB VRAM constraint).
    results = dict(existing_results)
    skipped = 0

    with torch.no_grad():
        for triplet_id in tqdm(to_process, desc="Computing flow", unit="triplet"):
            triplet_dir = sequences_dir / triplet_id
            img1_path = triplet_dir / "im1.png"
            img3_path = triplet_dir / "im3.png"

            # Validate that images exist.
            if not img1_path.is_file() or not img3_path.is_file():
                tqdm.write(f"Warning: missing images for {triplet_id}, skipping.")
                skipped += 1
                continue

            # Load uint8 images, then apply official RAFT preprocessing.
            img1 = load_image_as_tensor(img1_path)
            img3 = load_image_as_tensor(img3_path)
            img1, img3 = transforms(img1, img3)

            # Compute optical flow: I_0 -> I_1 (im1 -> im3).
            # RAFT returns a list of flow predictions, one per refinement iteration.
            # Use the last (most refined) prediction.
            flow_predictions = model(img1.to(device), img3.to(device))
            flow = flow_predictions[-1]  # (1, 2, H, W)

            # Move to CPU and convert to numpy.
            flow_np = flow.squeeze(0).cpu().numpy()  # (2, H, W)

            # Save flow as .npy for Oracle Flow usage.
            npy_path = flow_output_path(output_flow_dir, triplet_id)
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, flow_np)

            # Compute EPE and derive motion magnitude.
            epe = compute_epe(flow_np)  # (H, W)
            epe_median = float(np.median(epe))
            motion_bin = classify_motion(epe_median)

            results[triplet_id] = (epe_median, motion_bin)

    # Write final CSV.
    write_csv(output_csv, results)

    # Summary statistics.
    total_processed = len(to_process) - skipped
    print(f"\nDone. Processed {total_processed} triplets ({skipped} skipped).")
    print(f"Flow files saved to: {output_flow_dir}")
    print(f"Motion labels saved to: {output_csv}")

    # Print bin distribution.
    bin_counts = {"small": 0, "medium": 0, "large": 0}
    for _, (_, bin_label) in results.items():
        bin_counts[bin_label] += 1
    total = sum(bin_counts.values())
    print(f"\nMotion bin distribution ({total} total):")
    for bin_name in ("small", "medium", "large"):
        count = bin_counts[bin_name]
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"  {bin_name:>8s}: {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
