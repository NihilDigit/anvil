"""Evaluate RIFE HDv3 at native resolution on Vimeo90K and Xiph 1080p.

Reproduces the "RIFE HDv3" row in the ANVIL paper's main quality table:
  - Vimeo90K: 34.26 dB PSNR (3782 test triplets, 256x448)
  - Xiph 1080p: 30.04 dB PSNR (2662 triplets, 1080x1920)

Metrics: RGB uint8 PSNR, SSIM (per-channel mean), AlexNet LPIPS.
This is the same evaluation pipeline used for all ANVIL models.

Prerequisites:
  1. Clone the official RIFE repository:
       git clone https://github.com/megvii-research/ECCV2022-RIFE /path/to/ECCV2022-RIFE

  2. Download the HDv3 pretrained weights.  The weights file (flownet.pkl)
     should be placed at: /path/to/ECCV2022-RIFE/train_log/flownet.pkl
     Download link: https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view

  3. The RIFE repository must contain train_log/RIFE_HDv3.py and
     train_log/IFNet_HDv3.py (shipped with the HDv3 weights download).

  4. Datasets:
     - Vimeo90K: download via scripts or from http://toflow.csail.mit.edu/
       Expected layout: data_dir/sequences/<seq>/<triplet>/im{1,2,3}.png
                        data_dir/tri_testlist.txt
     - Xiph 1080p: download via scripts or from https://media.xiph.org/video/
       Expected layout: data_dir/sequences/<seq>/<triplet>/im{1,2,3}.png
                        data_dir/tri_testlist.txt

Usage:
    # Evaluate on both datasets:
    python scripts/eval_rife_native.py \\
        --rife-repo /path/to/ECCV2022-RIFE \\
        --weights /path/to/ECCV2022-RIFE/train_log/flownet.pkl \\
        --data-dir data/vimeo_triplet \\
        --xiph-dir data/xiph_1080p \\
        --dataset both

    # Vimeo only:
    python scripts/eval_rife_native.py \\
        --rife-repo /path/to/ECCV2022-RIFE \\
        --weights /path/to/ECCV2022-RIFE/train_log/flownet.pkl \\
        --data-dir data/vimeo_triplet \\
        --dataset vimeo

    # Xiph only, smaller LPIPS batch for VRAM-constrained GPUs:
    python scripts/eval_rife_native.py \\
        --rife-repo /path/to/ECCV2022-RIFE \\
        --weights /path/to/ECCV2022-RIFE/train_log/flownet.pkl \\
        --xiph-dir data/xiph_1080p \\
        --dataset xiph \\
        --lpips-batch 1
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from anvil_exp01.eval.metrics import compute_lpips_batch, compute_psnr, compute_ssim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# RIFE model loading
# ---------------------------------------------------------------------------

def load_rife_model(rife_repo: Path, weights_dir: Path) -> object:
    """Load RIFE HDv3 model from the cloned repository.

    The HDv3 pretrained model ships with RIFE_HDv3.py + IFNet_HDv3.py inside
    train_log/.  We import from there.  Falls back to model/RIFE.py if HDv3
    files are not found.

    Args:
        rife_repo: Path to the cloned ECCV2022-RIFE repository root.
        weights_dir: Directory containing flownet.pkl (typically train_log/).

    Returns:
        Loaded RIFE Model object with .inference() method.
    """
    repo_str = str(rife_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    hdv3_py = weights_dir / "RIFE_HDv3.py"
    if hdv3_py.exists():
        from train_log.RIFE_HDv3 import Model
    else:
        from model.RIFE import Model

    model = Model()
    model.load_model(str(weights_dir), -1)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# RIFE inference
# ---------------------------------------------------------------------------

def rife_inference(model: object, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    """Run RIFE inference on two RGB uint8 HWC images.

    Handles padding to multiple of 32 (RIFE requirement) and crops back.

    Args:
        model: Loaded RIFE model with .inference() method.
        img0: First frame, RGB uint8 (H, W, 3).
        img1: Third frame, RGB uint8 (H, W, 3).

    Returns:
        Predicted middle frame as RGB uint8 (H, W, 3).
    """
    h, w, _ = img0.shape

    t0 = torch.from_numpy(img0.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t1 = torch.from_numpy(img1.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t0, t1 = t0.to(DEVICE), t1.to(DEVICE)

    # Pad to multiple of 32
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    pad = (0, pw - w, 0, ph - h)
    t0 = F.pad(t0, pad)
    t1 = F.pad(t1, pad)

    with torch.no_grad():
        mid = model.inference(t0, t1)

    # Crop padding and convert back
    mid = mid[:, :, :h, :w]
    result = (mid[0].clamp(0, 1) * 255).round().byte().cpu().permute(1, 2, 0).numpy()
    return result


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_triplet_ids(data_dir: Path) -> list[str]:
    """Load triplet IDs from tri_testlist.txt."""
    list_path = data_dir / "tri_testlist.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Test list not found: {list_path}")
    ids = []
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB uint8 HWC."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def evaluate_dataset(
    model: object,
    data_dir: Path,
    dataset_name: str,
    output_dir: Path,
    lpips_pair_batch: int = 4,
) -> dict[str, float]:
    """Evaluate RIFE on a dataset.

    Args:
        model: Loaded RIFE model.
        data_dir: Dataset root containing sequences/ and tri_testlist.txt.
        dataset_name: Display name ("vimeo90k" or "xiph_1080p").
        output_dir: Where to write summary.csv, per_sequence.csv, per_triplet.csv.
        lpips_pair_batch: Number of image pairs per LPIPS batch (lower for 1080p).

    Returns:
        Dict with keys "psnr", "ssim", "lpips".
    """
    triplet_ids = load_triplet_ids(data_dir)
    seq_dir = data_dir / "sequences"

    print(f"\n{'=' * 60}")
    print(f"Evaluating RIFE HDv3 on {dataset_name}: {len(triplet_ids)} triplets")
    print(f"{'=' * 60}")

    psnr_list: list[float] = []
    ssim_list: list[float] = []
    lpips_list: list[float] = []
    tid_list: list[str] = []

    chunk_preds: list[np.ndarray] = []
    chunk_gts: list[np.ndarray] = []

    t_start = time.time()

    for triplet_id in tqdm(triplet_ids, desc=dataset_name, unit="triplet"):
        seq, tid = triplet_id.split("/")
        triplet_dir = seq_dir / seq / tid

        img0 = load_image(triplet_dir / "im1.png")
        gt = load_image(triplet_dir / "im2.png")
        img1 = load_image(triplet_dir / "im3.png")

        pred = rife_inference(model, img0, img1)

        psnr_list.append(compute_psnr(pred, gt))
        ssim_list.append(compute_ssim(pred, gt))
        chunk_preds.append(pred)
        chunk_gts.append(gt)
        tid_list.append(triplet_id)

        # Flush LPIPS periodically to manage VRAM
        if len(chunk_preds) >= lpips_pair_batch:
            torch.cuda.empty_cache()
            lpips_list.extend(
                compute_lpips_batch(
                    chunk_preds, chunk_gts,
                    device=str(DEVICE), pair_batch_size=lpips_pair_batch,
                )
            )
            chunk_preds.clear()
            chunk_gts.clear()
            torch.cuda.empty_cache()

    # Flush remaining
    if chunk_preds:
        lpips_list.extend(
            compute_lpips_batch(
                chunk_preds, chunk_gts,
                device=str(DEVICE), pair_batch_size=max(1, len(chunk_preds)),
            )
        )

    elapsed = time.time() - t_start

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_lpips = float(np.mean(lpips_list))

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"\nOVERALL ({len(tid_list)} triplets)")
    print(f"  PSNR:  {mean_psnr:.4f} dB")
    print(f"  SSIM:  {mean_ssim:.6f}")
    print(f"  LPIPS: {mean_lpips:.6f}")

    # Per-sequence breakdown
    seq_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"psnr": [], "ssim": [], "lpips": []}
    )
    for i, tid in enumerate(tid_list):
        seq = tid.split("/")[0]
        seq_metrics[seq]["psnr"].append(psnr_list[i])
        seq_metrics[seq]["ssim"].append(ssim_list[i])
        seq_metrics[seq]["lpips"].append(lpips_list[i])

    print(f"\n{'Sequence':<20} {'N':>4} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("-" * 52)
    for seq in sorted(seq_metrics):
        m = seq_metrics[seq]
        n = len(m["psnr"])
        print(
            f"{seq:<20} {n:>4} {np.mean(m['psnr']):>8.4f} "
            f"{np.mean(m['ssim']):>8.6f} {np.mean(m['lpips']):>8.6f}"
        )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "n_triplets", "psnr", "ssim", "lpips"])
        w.writerow([
            "RIFE_HDv3", dataset_name, len(tid_list),
            f"{mean_psnr:.4f}", f"{mean_ssim:.6f}", f"{mean_lpips:.6f}",
        ])
    print(f"\nSummary: {summary_path}")

    per_sequence_path = output_dir / "per_sequence.csv"
    with open(per_sequence_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "n_triplets", "psnr", "ssim", "lpips"])
        for seq in sorted(seq_metrics):
            m = seq_metrics[seq]
            w.writerow([
                seq, len(m["psnr"]),
                f"{np.mean(m['psnr']):.4f}",
                f"{np.mean(m['ssim']):.6f}",
                f"{np.mean(m['lpips']):.6f}",
            ])
    print(f"Per-sequence: {per_sequence_path}")

    per_triplet_path = output_dir / "per_triplet.csv"
    with open(per_triplet_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["triplet_id", "psnr", "ssim", "lpips"])
        for i, tid in enumerate(tid_list):
            w.writerow([
                tid,
                f"{psnr_list[i]:.4f}",
                f"{ssim_list[i]:.6f}",
                f"{lpips_list[i]:.6f}",
            ])
    print(f"Per-triplet: {per_triplet_path}")

    return {"psnr": mean_psnr, "ssim": mean_ssim, "lpips": mean_lpips}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RIFE HDv3 at native resolution (paper Table main_quality baseline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Expected results (paper Table main_quality):\n"
            "  Vimeo90K: 34.26 dB PSNR  (3782 test triplets)\n"
            "  Xiph 1080p: 30.04 dB PSNR (2662 triplets)\n"
        ),
    )
    parser.add_argument(
        "--rife-repo", type=Path, required=True,
        help="Path to cloned ECCV2022-RIFE repository.",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help=(
            "Path to flownet.pkl.  If not specified, defaults to "
            "<rife-repo>/train_log/flownet.pkl."
        ),
    )
    parser.add_argument(
        "--dataset", choices=["vimeo", "xiph", "both"], default="both",
        help="Which dataset(s) to evaluate on (default: both).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Vimeo90K root directory (containing sequences/ and tri_testlist.txt).",
    )
    parser.add_argument(
        "--xiph-dir", type=Path, default=None,
        help="Xiph 1080p root directory (containing sequences/ and tri_testlist.txt).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/eval/rife_native"),
        help="Base output directory (default: artifacts/eval/rife_native).",
    )
    parser.add_argument(
        "--lpips-batch", type=int, default=4,
        help="LPIPS pair batch size.  Use 1-2 for 1080p on 8GB VRAM (default: 4).",
    )
    args = parser.parse_args()

    # Resolve weights path
    if args.weights is None:
        args.weights = args.rife_repo / "train_log" / "flownet.pkl"

    # Pre-flight checks
    if not args.rife_repo.is_dir():
        print(f"ERROR: RIFE repo not found at {args.rife_repo}")
        print("Clone it: git clone https://github.com/megvii-research/ECCV2022-RIFE")
        sys.exit(1)

    if not args.weights.is_file():
        print(f"ERROR: Weights not found at {args.weights}")
        print("Download HDv3 weights from:")
        print("  https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view")
        sys.exit(1)

    # Resolve dataset directories
    want_vimeo = args.dataset in ("vimeo", "both")
    want_xiph = args.dataset in ("xiph", "both")

    if want_vimeo and args.data_dir is None:
        print("ERROR: --data-dir is required for Vimeo evaluation.")
        sys.exit(1)
    if want_xiph and args.xiph_dir is None:
        print("ERROR: --xiph-dir is required for Xiph evaluation.")
        sys.exit(1)

    # Load model
    weights_dir = args.weights.parent
    print(f"Loading RIFE HDv3 from {args.rife_repo}")
    print(f"Weights: {args.weights}")
    print(f"Device: {DEVICE}")
    model = load_rife_model(args.rife_repo, weights_dir)

    total_params = sum(p.numel() for p in model.flownet.parameters())
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    results: dict[str, dict[str, float]] = {}

    # Vimeo evaluation
    if want_vimeo:
        seq_dir = args.data_dir / "sequences"
        if not seq_dir.exists():
            print(f"WARNING: Vimeo sequences not found at {seq_dir}, skipping.")
        else:
            results["vimeo"] = evaluate_dataset(
                model, args.data_dir, "vimeo90k",
                args.output_dir / "vimeo90k",
                lpips_pair_batch=args.lpips_batch,
            )

    # Xiph evaluation
    if want_xiph:
        seq_dir = args.xiph_dir / "sequences"
        if not seq_dir.exists():
            print(f"WARNING: Xiph sequences not found at {seq_dir}, skipping.")
        else:
            results["xiph"] = evaluate_dataset(
                model, args.xiph_dir, "xiph_1080p",
                args.output_dir / "xiph_1080p",
                lpips_pair_batch=min(args.lpips_batch, 2),
            )

    # Final summary
    if results:
        print(f"\n{'=' * 60}")
        print("RIFE HDv3 NATIVE RESOLUTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Dataset':<12} {'PSNR':>8} {'SSIM':>10} {'LPIPS':>10}")
        print("-" * 42)
        for ds, m in results.items():
            print(f"{ds:<12} {m['psnr']:>8.4f} {m['ssim']:>10.6f} {m['lpips']:>10.6f}")
        print()

        # Reference comparison
        print(f"{'=' * 70}")
        print("COMPARISON (same eval pipeline, RGB uint8)")
        print(f"{'=' * 70}")
        print(f"{'Model':<25} {'Params':>8} {'Vimeo PSNR':>12} {'Xiph PSNR':>12}")
        print("-" * 60)
        vimeo_str = f"{results['vimeo']['psnr']:>11.4f}" if "vimeo" in results else f"{'--':>12}"
        xiph_str = f"{results['xiph']['psnr']:>11.4f}" if "xiph" in results else f"{'--':>12}"
        print(f"{'RIFE HDv3 (this eval)':<25} {f'{total_params/1e6:.2f}M':>8} {vimeo_str} {xiph_str}")
        print(f"{'Anvil-S (855K)':<25} {'855K':>8} {'33.45':>12} {'29.65':>12}")
        print(f"{'Anvil-M (2.66M)':<25} {'2.66M':>8} {'33.66':>12} {'29.74':>12}")
        print(f"{'NAFNet-ceiling (17.1M)':<25} {'17.1M':>8} {'34.58':>12} {'30.30':>12}")

    print("\nDone.")


if __name__ == "__main__":
    main()
