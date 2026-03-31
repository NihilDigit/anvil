#!/usr/bin/env python3
"""Generate Figure 6: Visual quality comparison on Xiph 1080p sequences.

3 rows (scenarios) x 3 columns (ANVIL-M, RIFE HDv3, Ground Truth)
with zoom insets and per-crop PSNR annotations.

Usage:
    pixi run python scripts/gen_fig_visual_comparison.py \
        --xiph-dir data/xiph_1080p \
        --anvil-ckpt artifacts/checkpoints/D-unet-v3bm-nomv/best.pt \
        --rife-repo vendor/ECCV2022-RIFE \
        --output-dir artifacts/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.patheffects
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "text.usetex": False,
})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (seq, triplet, row_slice, col_slice, subplot_label, zoom_slice)
# zoom_slice: (row_slice, col_slice) within the already-cropped 270x480 image
SEQUENCES = [
    ("old_town_cross", "00120", slice(350, 620), slice(400, 880),
     "(a) ANVIL wins", (slice(80, 160), slice(80, 160))),
    ("tractor",        "00150", slice(250, 520), slice(800, 1280),
     "(b) Different failures", (slice(160, 240), slice(400, 480))),
    ("riverbed",       "00050", slice(300, 570), slice(400, 880),
     "(c) Both fail", (slice(80, 160), slice(320, 400))),
]

COL_NAMES = ["ANVIL-M", "RIFE HDv3", "Ground Truth"]


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_img(path: Path) -> np.ndarray:
    """Load image as RGB uint8 HWC."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to load {path}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def psnr_crop(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute PSNR between two uint8 HWC crops."""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(255.0 ** 2 / mse)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_anvil_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Load ANVIL-M model from checkpoint."""
    from anvil_exp01.models.conv_vfi import build_model

    model = build_model("D-unet-v3bm-nomv")
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model


def load_rife_model(rife_repo: Path, weights_dir: Path) -> object:
    """Load RIFE HDv3 model from the cloned repository.

    Falls back to model/RIFE.py if HDv3 files are not found in weights_dir.
    """
    repo_str = str(rife_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    hdv3_py = weights_dir / "RIFE_HDv3.py"
    if hdv3_py.exists():
        from train_log.RIFE_HDv3 import Model
    else:
        from model.RIFE import Model

    rife = Model()
    rife.load_model(str(weights_dir), -1)
    rife.eval()
    return rife


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_anvil(model: torch.nn.Module, im1_aligned: np.ndarray,
              im3_aligned: np.ndarray) -> np.ndarray:
    """Run ANVIL-M inference. Returns RGB uint8 HWC."""
    im1_t = torch.from_numpy(im1_aligned.copy()).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    im3_t = torch.from_numpy(im3_aligned.copy()).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0

    with torch.no_grad():
        inp = torch.cat([im1_t, im3_t], dim=1)
        blend = (im1_t + im3_t) / 2.0
        residual = model(inp)
        output = (blend + residual).clamp(0, 1)

    return (output[0].cpu().permute(1, 2, 0) * 255).round().byte().numpy()


def run_rife(model: object, im1: np.ndarray, im3: np.ndarray) -> np.ndarray:
    """Run RIFE HDv3 inference. Returns RGB uint8 HWC."""
    h, w = im1.shape[:2]

    t0 = torch.from_numpy(im1.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t1 = torch.from_numpy(im3.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t0, t1 = t0.to(DEVICE), t1.to(DEVICE)

    # Pad to multiple of 32
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    pad = (0, pw - w, 0, ph - h)
    t0 = F.pad(t0, pad)
    t1 = F.pad(t1, pad)

    with torch.no_grad():
        mid = model.inference(t0, t1)

    mid = mid[:, :, :h, :w]
    return (mid[0].clamp(0, 1) * 255).round().byte().cpu().permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 6: Visual quality comparison on Xiph 1080p sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--xiph-dir", type=Path, required=True,
        help="Xiph 1080p root (containing sequences/ and prealigned_v2/).",
    )
    parser.add_argument(
        "--prealigned-dir", type=Path, default=None,
        help="Prealigned v2 directory. Default: <xiph-dir>/prealigned_v2.",
    )
    parser.add_argument(
        "--anvil-ckpt", type=Path, required=True,
        help="ANVIL-M checkpoint path (best.pt).",
    )
    parser.add_argument(
        "--rife-repo", type=Path, required=True,
        help="Path to cloned ECCV2022-RIFE repository.",
    )
    parser.add_argument(
        "--rife-weights", type=Path, default=None,
        help="RIFE weights directory (containing flownet.pkl). Default: <rife-repo>/train_log.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/figures"),
        help="Output directory for the figure (default: artifacts/figures).",
    )
    parser.add_argument(
        "--format", choices=["pdf", "png"], default="pdf",
        help="Output format (default: pdf).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="If >0, only process first N scenarios (for smoke test).",
    )
    args = parser.parse_args()

    # Resolve defaults
    seq_dir = args.xiph_dir / "sequences"
    prealign_dir = args.prealigned_dir if args.prealigned_dir else args.xiph_dir / "prealigned_v2"
    rife_weights_dir = args.rife_weights if args.rife_weights else args.rife_repo / "train_log"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight checks
    if not seq_dir.is_dir():
        print(f"ERROR: Xiph sequences not found at {seq_dir}")
        sys.exit(1)
    if not prealign_dir.is_dir():
        print(f"ERROR: Prealigned v2 directory not found at {prealign_dir}")
        sys.exit(1)
    if not args.anvil_ckpt.is_file():
        print(f"ERROR: ANVIL checkpoint not found at {args.anvil_ckpt}")
        sys.exit(1)
    if not args.rife_repo.is_dir():
        print(f"ERROR: RIFE repo not found at {args.rife_repo}")
        sys.exit(1)

    sequences = SEQUENCES
    if args.limit > 0:
        sequences = sequences[:args.limit]

    # Load models
    print("Loading ANVIL-M model...")
    anvil_model = load_anvil_model(args.anvil_ckpt, DEVICE)
    print("Loading RIFE HDv3 model...")
    rife_model = load_rife_model(args.rife_repo, rife_weights_dir)

    n_rows = len(sequences)
    n_cols = len(COL_NAMES)

    # Collect crops: [row][col] = (crop_uint8, psnr_or_None)
    all_crops: list[list[tuple[np.ndarray, float | None]]] = []

    for seq, triplet, rslice, cslice, _label, _zoom in sequences:
        print(f"Processing {seq}/{triplet}...")
        row_crops: list[tuple[np.ndarray, float | None]] = []

        # Load raw frames
        im1 = load_img(seq_dir / seq / triplet / "im1.png")
        im2 = load_img(seq_dir / seq / triplet / "im2.png")  # GT
        im3 = load_img(seq_dir / seq / triplet / "im3.png")

        # Load prealigned v2 frames
        im1_aligned = load_img(prealign_dir / seq / triplet / "im1_aligned.png")
        im3_aligned = load_img(prealign_dir / seq / triplet / "im3_aligned.png")

        gt_crop = im2[rslice, cslice]

        # Col 0: ANVIL-M
        print("  Running ANVIL-M inference...")
        anvil_out = run_anvil(anvil_model, im1_aligned, im3_aligned)
        anvil_crop = anvil_out[rslice, cslice]
        row_crops.append((anvil_crop, psnr_crop(anvil_crop, gt_crop)))

        # Col 1: RIFE HDv3
        print("  Running RIFE HDv3 inference...")
        rife_out = run_rife(rife_model, im1, im3)
        rife_crop = rife_out[rslice, cslice]
        row_crops.append((rife_crop, psnr_crop(rife_crop, gt_crop)))

        # Col 2: Ground Truth
        row_crops.append((gt_crop, None))

        all_crops.append(row_crops)

    # Print PSNR summary
    print("\nPSNR summary (crop regions):")
    for i, (seq, triplet, _, _, _, _) in enumerate(sequences):
        vals = [f"{all_crops[i][j][1]:.2f}" if all_crops[i][j][1] is not None else "GT"
                for j in range(n_cols)]
        print(f"  {seq}/{triplet}: {vals}")

    # Compute figure height from crop aspect ratio
    # All crops are 270h x 480w
    crop_h, crop_w = 270, 480
    cell_w = 7.16 / n_cols
    cell_h = cell_w * (crop_h / crop_w)
    fig_h = cell_h * n_rows + 0.5  # extra space for titles

    print(f"\nGenerating figure ({7.16:.2f} x {fig_h:.2f} inches)...")
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.16, fig_h),
        gridspec_kw={"wspace": 0.03, "hspace": 0.06},
    )

    # Handle single-row case (axes is 1D when n_rows=1)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            crop, psnr_val = all_crops[r][c]
            ax.imshow(crop)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color("0.5")

            # Column titles on first row only
            if r == 0:
                ax.set_title(COL_NAMES[c], fontsize=8, fontweight="bold", pad=4)

            # Subplot label in top-left corner (e.g., "(a) ANVIL wins")
            if c == 0:
                label = sequences[r][4]
                ax.text(
                    0.03, 0.96, label,
                    transform=ax.transAxes,
                    fontsize=7, fontweight="bold",
                    color="white", ha="left", va="top",
                    path_effects=[
                        matplotlib.patheffects.withStroke(linewidth=2, foreground="black"),
                    ],
                )

            # PSNR overlay (bottom-right, white text + black outline)
            # Skip GT column
            if psnr_val is not None:
                txt = f"{psnr_val:.2f} dB"
                ax.text(
                    0.97, 0.04, txt,
                    transform=ax.transAxes,
                    fontsize=7,
                    color="white",
                    ha="right", va="bottom",
                    fontweight="bold",
                    path_effects=[
                        matplotlib.patheffects.withStroke(linewidth=2, foreground="black"),
                    ],
                )

            # Zoom-in inset
            zoom_rslice, zoom_cslice = sequences[r][5]
            zoom_patch = crop[zoom_rslice, zoom_cslice]
            zoom_color = "#FFD700"

            # Draw rectangle on main image showing zoom region
            y0, y1 = zoom_rslice.start, zoom_rslice.stop
            x0, x1 = zoom_cslice.start, zoom_cslice.stop
            rect = mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1.5, edgecolor=zoom_color, facecolor="none",
            )
            ax.add_patch(rect)

            # Create inset axes (top-right of cell)
            inset = ax.inset_axes([0.55, 0.55, 0.43, 0.43])
            inset.imshow(zoom_patch, interpolation="nearest")
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color(zoom_color)

    outpath = output_dir / f"fig_visual_comparison.{args.format}"
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
