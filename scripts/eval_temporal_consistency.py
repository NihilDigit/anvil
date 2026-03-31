"""Evaluate temporal consistency on Xiph 1080p (tOF + WE metrics).

Reproduces the paper's temporal consistency analysis:
  tOF: mean optical flow magnitude between consecutive interpolated frames
  WE: warping error (backward sampling, lower = better)
  tOF_dev: |tOF_method - tOF_GT| (lower = closer to GT temporal fidelity)

Methods: Naive Blend, MV Blend, ANVIL-S, ANVIL-M, RIFE HDv3, Ground Truth.
All flow computed online with RAFT-Small at 540p.

Usage:
    pixi run python scripts/eval_temporal_consistency.py \
        --xiph-dir data/xiph_1080p \
        --anvil-s-ckpt artifacts/checkpoints/D-unet-v3bs-nomv/best.pt \
        --anvil-m-ckpt artifacts/checkpoints/D-unet-v3bm-nomv/best.pt \
        --rife-repo vendor/ECCV2022-RIFE \
        --output-dir artifacts/eval/temporal_consistency
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from anvil_exp01.models.conv_vfi import build_model


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_img(p: Path) -> np.ndarray:
    """Load an image as RGB uint8 HWC numpy array."""
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# RAFT flow (streaming, no storage)
# ---------------------------------------------------------------------------
def setup_raft(device: torch.device):
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights).to(device).eval()
    transforms = weights.transforms()
    return model, transforms


FLOW_SCALE = 2  # Downsample factor for RAFT (1080p -> 540p to fit 8GB GPU)


@torch.no_grad()
def compute_flow(raft_model, raft_transforms,
                 im1: np.ndarray, im2: np.ndarray,
                 device: torch.device) -> np.ndarray:
    """Returns (H_ds, W_ds, 2) float32 flow at downscaled resolution.

    Downscales inputs by FLOW_SCALE to fit RAFT correlation volume in 8GB GPU.
    tOF is a relative metric (comparing methods, not absolute flow magnitude),
    so consistent downscaling does not affect method ranking.
    """
    h, w = im1.shape[:2]
    dh = (h // FLOW_SCALE) // 8 * 8  # round down to multiple of 8
    dw = (w // FLOW_SCALE) // 8 * 8
    im1_ds = cv2.resize(im1, (dw, dh), interpolation=cv2.INTER_AREA)
    im2_ds = cv2.resize(im2, (dw, dh), interpolation=cv2.INTER_AREA)
    t1 = torch.from_numpy(im1_ds).permute(2, 0, 1).unsqueeze(0).to(device)
    t2 = torch.from_numpy(im2_ds).permute(2, 0, 1).unsqueeze(0).to(device)
    t1, t2 = raft_transforms(t1, t2)
    flow = raft_model(t1, t2)[-1]
    result = flow[0].cpu().permute(1, 2, 0).numpy()
    del t1, t2, flow
    torch.cuda.empty_cache()
    return result


def tof(flow: np.ndarray) -> float:
    """Mean flow magnitude."""
    return float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())


def warping_error(im_prev: np.ndarray, im_curr: np.ndarray,
                  flow: np.ndarray) -> float:
    """MAE after warping prev toward curr using forward flow (prev->curr).

    Backward sampling: sample prev at (x - flow_x, y - flow_y) to reconstruct curr.
    """
    h, w = im_prev.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    warped = cv2.remap(
        im_prev.astype(np.float32) / 255.0,
        xx - flow[..., 0], yy - flow[..., 1],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    target = im_curr.astype(np.float32) / 255.0
    return float(np.mean(np.abs(warped - target)))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_anvil_model(model_id: str, checkpoint: Path, device: torch.device):
    """Load an ANVIL model from checkpoint."""
    model = build_model(model_id)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model


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
# VFI Methods
# ---------------------------------------------------------------------------
def run_naive(im1: np.ndarray, im3: np.ndarray) -> np.ndarray:
    return ((im1.astype(np.float32) + im3.astype(np.float32)) / 2
            ).round().clip(0, 255).astype(np.uint8)


def run_mv_blend(pa1: np.ndarray, pa3: np.ndarray) -> np.ndarray:
    return ((pa1.astype(np.float32) + pa3.astype(np.float32)) / 2
            ).round().clip(0, 255).astype(np.uint8)


def run_anvil(model, pa1: np.ndarray, pa3: np.ndarray,
              device: torch.device) -> np.ndarray:
    i1 = torch.from_numpy(pa1.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    i3 = torch.from_numpy(pa3.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    with torch.no_grad():
        out = ((i1 + i3) / 2 + model(torch.cat([i1, i3], 1))).clamp(0, 1)
    return (out[0].cpu().permute(1, 2, 0) * 255).round().byte().numpy()


def run_rife(model, im1: np.ndarray, im3: np.ndarray,
             device: torch.device) -> np.ndarray:
    h, w = im1.shape[:2]
    t0 = torch.from_numpy(im1.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    t1 = torch.from_numpy(im3.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    t0 = F.pad(t0, (0, pw - w, 0, ph - h))
    t1 = F.pad(t1, (0, pw - w, 0, ph - h))
    with torch.no_grad():
        out = model.inference(t0, t1)
    result = (out[0, :, :h, :w].clamp(0, 1) * 255).round().byte().cpu().permute(1, 2, 0).numpy()
    del t0, t1, out
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate temporal consistency on Xiph 1080p (tOF + WE metrics).",
    )
    parser.add_argument("--xiph-dir", type=Path, required=True,
                        help="Xiph 1080p dataset directory")
    parser.add_argument("--prealigned-dir", type=Path, default=None,
                        help="Prealigned v2 directory (default: xiph-dir/prealigned_v2)")
    parser.add_argument("--anvil-s-ckpt", type=Path, required=True,
                        help="ANVIL-S checkpoint (best.pt)")
    parser.add_argument("--anvil-m-ckpt", type=Path, required=True,
                        help="ANVIL-M checkpoint (best.pt)")
    parser.add_argument("--rife-repo", type=Path, required=True,
                        help="Path to cloned ECCV2022-RIFE repository")
    parser.add_argument("--rife-weights", type=Path, default=None,
                        help="RIFE weights directory (default: rife-repo/train_log)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/eval/temporal_consistency"),
                        help="Output directory for results")
    parser.add_argument("--max-pairs", type=int, default=30,
                        help="Max consecutive pairs per sequence (default: 30)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N sequences (0=all, for smoke testing)")
    args = parser.parse_args()

    # Resolve defaults
    seq_dir = args.xiph_dir / "sequences"
    pa_dir = args.prealigned_dir if args.prealigned_dir else args.xiph_dir / "prealigned_v2"
    rife_weights = args.rife_weights if args.rife_weights else args.rife_repo / "train_log"
    output_dir = args.output_dir

    # Validate paths
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Sequences directory not found: {seq_dir}")
    if not pa_dir.is_dir():
        raise FileNotFoundError(f"Prealigned directory not found: {pa_dir}")
    if not args.anvil_s_ckpt.is_file():
        raise FileNotFoundError(f"ANVIL-S checkpoint not found: {args.anvil_s_ckpt}")
    if not args.anvil_m_ckpt.is_file():
        raise FileNotFoundError(f"ANVIL-M checkpoint not found: {args.anvil_m_ckpt}")
    if not args.rife_repo.is_dir():
        raise FileNotFoundError(f"RIFE repository not found: {args.rife_repo}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------
    # Setup models
    # -------------------------------------------------------------------
    print("Setting up models...")
    raft_model, raft_transforms = setup_raft(device)
    anvil_s_model = load_anvil_model("D-unet-v3bs-nomv", args.anvil_s_ckpt, device)
    anvil_m_model = load_anvil_model("D-unet-v3bm-nomv", args.anvil_m_ckpt, device)
    rife_model = load_rife_model(args.rife_repo, rife_weights)
    print(f"Device: {device}")

    # -------------------------------------------------------------------
    # Discover sequences
    # -------------------------------------------------------------------
    sequences = sorted(d.name for d in seq_dir.iterdir() if d.is_dir())
    if args.limit > 0:
        sequences = sequences[:args.limit]
        print(f"Limiting to {len(sequences)} sequences (--limit {args.limit})")

    methods = ["gt", "naive", "mv_blend", "anvil_s", "anvil_m", "rife"]

    all_rows = []
    t0_total = time.time()

    for seq in sequences:
        this_seq_dir = seq_dir / seq
        triplets = sorted(d.name for d in this_seq_dir.iterdir() if d.is_dir())
        if len(triplets) < 2:
            continue

        n_pairs = min(len(triplets) - 1, args.max_pairs)
        print(f"\n{seq}: {n_pairs} pairs")

        seq_metrics = {m: {"tof": [], "we": [], "tof_dev": []} for m in methods}
        prev_frames = {}

        for i in tqdm(range(n_pairs + 1), desc=f"  {seq}", leave=False):
            trip = triplets[i]
            im1 = load_img(this_seq_dir / trip / "im1.png")
            im2 = load_img(this_seq_dir / trip / "im2.png")
            im3 = load_img(this_seq_dir / trip / "im3.png")

            this_pa_dir = pa_dir / seq / trip
            has_pa = (this_pa_dir / "im1_aligned.png").exists()
            pa1 = load_img(this_pa_dir / "im1_aligned.png") if has_pa else im1
            pa3 = load_img(this_pa_dir / "im3_aligned.png") if has_pa else im3

            # Generate interpolated frames
            current = {}
            current["gt"] = im2
            current["naive"] = run_naive(im1, im3)
            current["mv_blend"] = run_mv_blend(pa1, pa3)
            current["anvil_s"] = run_anvil(anvil_s_model, pa1, pa3, device)
            current["anvil_m"] = run_anvil(anvil_m_model, pa1, pa3, device)
            current["rife"] = run_rife(rife_model, im1, im3, device)

            if i > 0:
                # RAFT 1080p correlation volume needs ~4GB.
                # Offload RIFE flownet to CPU to free GPU memory.
                rife_model.flownet.cpu()
                torch.cuda.empty_cache()

                pair_tof_values = {}
                for m in methods:
                    flow = compute_flow(raft_model, raft_transforms,
                                        prev_frames[m], current[m], device)
                    t = tof(flow)
                    seq_metrics[m]["tof"].append(t)
                    pair_tof_values[m] = t
                    # Warping error at downscaled resolution (matching flow)
                    fh, fw = flow.shape[:2]
                    prev_ds = cv2.resize(prev_frames[m], (fw, fh))
                    curr_ds = cv2.resize(current[m], (fw, fh))
                    seq_metrics[m]["we"].append(warping_error(prev_ds, curr_ds, flow))
                    del flow, prev_ds, curr_ds

                # Pairwise tOF deviation from GT
                gt_tof = pair_tof_values.get("gt", 0)
                for m in methods:
                    if m != "gt":
                        seq_metrics[m]["tof_dev"].append(abs(pair_tof_values[m] - gt_tof))

                # Restore RIFE to GPU for next iteration
                rife_model.flownet.to(device)

            prev_frames = current

        # Aggregate per-sequence
        for m in methods:
            if seq_metrics[m]["tof"]:
                row = {
                    "sequence": seq,
                    "method": m,
                    "n_pairs": len(seq_metrics[m]["tof"]),
                    "tof_mean": float(np.mean(seq_metrics[m]["tof"])),
                    "tof_std": float(np.std(seq_metrics[m]["tof"])),
                    "we_mean": float(np.mean(seq_metrics[m]["we"])),
                    "we_std": float(np.std(seq_metrics[m]["we"])),
                    "tof_dev_mean": float(np.mean(seq_metrics[m]["tof_dev"])) if seq_metrics[m]["tof_dev"] else 0.0,
                }
                all_rows.append(row)

        # Print per-sequence
        for m in methods:
            e = [r for r in all_rows if r["sequence"] == seq and r["method"] == m]
            if e:
                e = e[0]
                print(f"  {m:10s}: tOF={e['tof_mean']:.3f}  WE={e['we_mean']:.4f}")

    elapsed = time.time() - t0_total

    if not all_rows:
        print("No data collected. Check that --xiph-dir contains sequences.")
        return

    # Save detailed results
    csv_path = output_dir / "per_sequence.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)

    # Overall summary
    summary = {}
    for m in methods:
        entries = [r for r in all_rows if r["method"] == m]
        if entries:
            summary[m] = {
                "tof_mean": float(np.mean([r["tof_mean"] for r in entries])),
                "tof_std": float(np.std([r["tof_mean"] for r in entries])),
                "we_mean": float(np.mean([r["we_mean"] for r in entries])),
                "we_std": float(np.std([r["we_mean"] for r in entries])),
                "n_sequences": len(entries),
            }

    summary["_meta"] = {
        "elapsed_s": elapsed,
        "max_pairs_per_seq": args.max_pairs,
        "limit": args.limit,
        "device": str(device),
        "xiph_dir": str(args.xiph_dir),
        "prealigned_dir": str(pa_dir),
    }

    # tOF deviation from GT (already computed pairwise during eval)
    for m in methods:
        if m == "gt" or m not in summary:
            continue
        entries = [r for r in all_rows if r["method"] == m]
        if entries:
            summary[m]["tof_deviation"] = float(np.mean([r["tof_dev_mean"] for r in entries]))

    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final table
    print(f"\n{'='*75}")
    print("Temporal Consistency Summary")
    print("  tOF = mean optical flow magnitude between consecutive interpolated frames")
    print("  tOF_dev = |tOF_method - tOF_GT| (closer to GT = better temporal fidelity)")
    print("  WE = warping error (lower = better)")
    print(f"{'='*75}")
    print(f"{'Method':<12} {'tOF':>8} {'tOF_dev':>10} {'WE':>8} {'Seqs':>5}")
    print("-" * 48)
    for m in methods:
        if m in summary and m != "_meta":
            s = summary[m]
            dev = s.get("tof_deviation", 0.0)
            label = "(GT ref)" if m == "gt" else ""
            print(f"{m:<12} {s['tof_mean']:8.3f} {dev:10.3f} "
                  f"{s['we_mean']:8.4f} {s['n_sequences']:5d}  {label}")

    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Results: {output_dir}/")


if __name__ == "__main__":
    main()
