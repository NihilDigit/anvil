"""Experiment 2: Route A vs D comparison with neural networks.

Evaluates 6 trained models on the Vimeo90K test set:
  A-small, A-large           (raw frames, 6ch)
  D-small, D-large           (prealigned + MV, 8ch)
  D-small-nomv, D-large-nomv (prealigned only, 6ch)

Reports PSNR / SSIM / LPIPS overall and per motion-magnitude bin.

Usage:
    pixi run python -m anvil_exp01.experiments.exp2_route_comparison \
        --data-dir data/vimeo_triplet \
        --mv-flow-dir data/mv_dense_flow \
        --prealigned-dir data/prealigned \
        --motion-csv data/motion_labels.csv \
        --checkpoint-dir checkpoints \
        --output-dir results/exp2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from anvil_exp01.data.dataset import Vimeo90KDataset
from anvil_exp01.eval.metrics import compute_lpips_batch, compute_psnr, compute_ssim
from anvil_exp01.models.conv_vfi import (
    build_model,
    count_parameters,
    infer_route,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP2_MODELS = [
    "A-small", "A-large",
    "D-small", "D-large",
    "D-small-nomv", "D-large-nomv",
]

MOTION_BINS = ["small", "medium", "large"]

_COL_HEADERS = [
    "Model", "Params",
    "Overall PSNR", "Overall SSIM", "Overall LPIPS",
    "Small PSNR", "Medium PSNR", "Large PSNR",
    "Small LPIPS", "Medium LPIPS", "Large LPIPS",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_motion_labels(csv_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["triplet_id"].strip()] = row["motion_bin"].strip()
    return labels


def evaluate_model(
    model_id: str,
    checkpoint_dir: Path,
    data_dir: Path,
    mv_flow_dir: Path | None,
    prealigned_dir: Path | None,
    device: torch.device,
    batch_size: int = 32,
    lpips_pair_batch: int = 64,
) -> tuple[list[str], list[float], list[float], list[float], int]:
    """Evaluate a single model. Returns (triplet_ids, psnrs, ssims, lpipss, n_params)."""
    route = infer_route(model_id)

    # Load model
    model = build_model(model_id).to(device)
    n_params = count_parameters(model)
    ckpt_path = checkpoint_dir / model_id / "best.pt"
    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return [], [], [], [], n_params

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Dataset (full resolution, no augmentation)
    ds = Vimeo90KDataset(
        data_dir=data_dir, split="test", route=route,
        mv_flow_dir=mv_flow_dir, prealigned_dir=prealigned_dir,
        crop_size=0, augment=False,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    triplet_ids: list[str] = []
    psnr_list: list[float] = []
    ssim_list: list[float] = []
    lpips_list: list[float] = []

    # Process in chunks to avoid buffering all images in RAM.
    chunk_preds: list[np.ndarray] = []
    chunk_gts: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {model_id}", unit="batch", leave=False):
            inp = batch["input"].to(device)
            blend = batch["blend"].to(device)
            gt_t = batch["gt"]

            residual = model(inp)
            pred_t = (blend + residual).clamp(0, 1).cpu()

            pred_u8 = (pred_t * 255).round().clamp(0, 255).byte().numpy()
            gt_u8 = (gt_t * 255).round().clamp(0, 255).byte().numpy()

            for i in range(pred_u8.shape[0]):
                p = pred_u8[i].transpose(1, 2, 0)  # CHW → HWC
                g = gt_u8[i].transpose(1, 2, 0)
                chunk_preds.append(p)
                chunk_gts.append(g)
                psnr_list.append(compute_psnr(p, g))
                ssim_list.append(compute_ssim(p, g))

            triplet_ids.extend(batch["triplet_id"])

            # Flush LPIPS chunk when large enough
            if len(chunk_preds) >= lpips_pair_batch:
                lpips_list.extend(compute_lpips_batch(
                    chunk_preds, chunk_gts,
                    device=str(device), pair_batch_size=lpips_pair_batch,
                ))
                chunk_preds.clear()
                chunk_gts.clear()

    # Flush remaining
    if chunk_preds:
        lpips_list.extend(compute_lpips_batch(
            chunk_preds, chunk_gts,
            device=str(device), pair_batch_size=lpips_pair_batch,
        ))

    return triplet_ids, psnr_list, ssim_list, lpips_list, n_params


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _build_summary_row(
    model_id: str,
    n_params: int,
    psnr_list: list[float],
    ssim_list: list[float],
    lpips_list: list[float],
    triplet_ids: list[str],
    motion_labels: dict[str, str],
) -> dict[str, str]:
    bin_indices: dict[str, list[int]] = {b: [] for b in MOTION_BINS}
    for idx, tid in enumerate(triplet_ids):
        b = motion_labels.get(tid, "")
        if b in bin_indices:
            bin_indices[b].append(idx)

    row: dict[str, str] = {
        "Model": model_id,
        "Params": f"{n_params:,}",
        "Overall PSNR": f"{np.mean(psnr_list):.4f}",
        "Overall SSIM": f"{np.mean(ssim_list):.6f}",
        "Overall LPIPS": f"{np.mean(lpips_list):.6f}",
    }
    for bname in MOTION_BINS:
        idxs = bin_indices[bname]
        if idxs:
            row[f"{bname.capitalize()} PSNR"] = f"{np.mean([psnr_list[i] for i in idxs]):.4f}"
            row[f"{bname.capitalize()} LPIPS"] = f"{np.mean([lpips_list[i] for i in idxs]):.6f}"
        else:
            row[f"{bname.capitalize()} PSNR"] = "N/A"
            row[f"{bname.capitalize()} LPIPS"] = "N/A"
    return row


def _print_table(rows: list[dict[str, str]], headers: list[str]) -> None:
    widths = {col: len(col) for col in headers}
    for row in rows:
        for col in headers:
            widths[col] = max(widths[col], len(row.get(col, "")))
    print(" | ".join(col.ljust(widths[col]) for col in headers))
    print("-+-".join("-" * widths[col] for col in headers))
    for row in rows:
        print(" | ".join(row.get(col, "").ljust(widths[col]) for col in headers))


def _save_summary_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COL_HEADERS)
        w.writeheader()
        w.writerows(rows)
    print(f"Summary saved to {path}")


def _save_per_triplet_csv(
    models_data: dict[str, dict],
    path: Path,
    motion_labels: dict[str, str],
) -> None:
    """Save per-triplet metrics for all evaluated models."""
    # Use triplet_ids from the first model (all share the same test set order)
    first = next(iter(models_data.values()))
    triplet_ids = first["triplet_ids"]

    fieldnames = ["triplet_id", "motion_bin"]
    for mid in models_data:
        fieldnames.extend([f"{mid}_PSNR", f"{mid}_SSIM", f"{mid}_LPIPS"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for idx, tid in enumerate(triplet_ids):
            row: dict[str, str] = {
                "triplet_id": tid,
                "motion_bin": motion_labels.get(tid, ""),
            }
            for mid, d in models_data.items():
                if idx < len(d["psnr"]):
                    row[f"{mid}_PSNR"] = f"{d['psnr'][idx]:.4f}"
                    row[f"{mid}_SSIM"] = f"{d['ssim'][idx]:.6f}"
                    row[f"{mid}_LPIPS"] = f"{d['lpips'][idx]:.6f}"
            w.writerow(row)
    print(f"Per-triplet results saved to {path}")


def _load_exp1_rows(exp1_csv: Path) -> list[dict[str, str]]:
    """Load exp1 summary rows and adapt column names.

    Exp1 uses 'Method' as the name column; remap to 'Model' so merged
    rows display correctly in the exp2 summary table.
    """
    rows: list[dict[str, str]] = []
    with open(exp1_csv, newline="") as f:
        for row in csv.DictReader(f):
            adapted: dict[str, str] = {"Params": "-"}
            for k, v in row.items():
                if k == "Method":
                    adapted["Model"] = v
                else:
                    adapted[k] = v
            rows.append(adapted)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 2: Route A vs D neural network comparison.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mv-flow-dir", type=Path, default=None)
    parser.add_argument("--prealigned-dir", type=Path, default=None)
    parser.add_argument("--motion-csv", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints"),
        help="Base checkpoint directory (default: checkpoints/).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/exp2"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lpips-pair-batch", type=int, default=64)
    parser.add_argument(
        "--models", nargs="+", default=EXP2_MODELS,
        help=f"Models to evaluate (default: {EXP2_MODELS}).",
    )
    parser.add_argument(
        "--exp1-results", type=Path, default=None,
        help="Path to exp1 summary.csv to include baseline rows.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    motion_labels = _load_motion_labels(args.motion_csv.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(args.models)} models on test set ...")
    t_start = time.time()

    summary_rows: list[dict[str, str]] = []
    models_data: dict[str, dict] = {}

    for model_id in args.models:
        print(f"\n[{model_id}]")
        tids, psnrs, ssims, lpipss, n_params = evaluate_model(
            model_id=model_id,
            checkpoint_dir=args.checkpoint_dir.resolve(),
            data_dir=args.data_dir.resolve(),
            mv_flow_dir=args.mv_flow_dir.resolve() if args.mv_flow_dir else None,
            prealigned_dir=args.prealigned_dir.resolve() if args.prealigned_dir else None,
            device=device,
            batch_size=args.batch_size,
            lpips_pair_batch=args.lpips_pair_batch,
        )
        if not tids:
            print(f"  Skipped (no checkpoint).")
            continue

        models_data[model_id] = {
            "triplet_ids": tids, "psnr": psnrs, "ssim": ssims, "lpips": lpipss,
        }
        summary_rows.append(_build_summary_row(
            model_id, n_params, psnrs, ssims, lpipss, tids, motion_labels,
        ))

    elapsed = time.time() - t_start
    print(f"\nEvaluation completed in {elapsed:.1f}s\n")

    # Merge exp1 baselines if provided
    if args.exp1_results and args.exp1_results.exists():
        exp1_rows = _load_exp1_rows(args.exp1_results.resolve())
        summary_rows = exp1_rows + summary_rows
        print(f"Merged {len(exp1_rows)} baseline rows from exp1.\n")

    if summary_rows:
        _print_table(summary_rows, _COL_HEADERS)
        print()
        _save_summary_csv(summary_rows, output_dir / "summary.csv")

    if models_data:
        _save_per_triplet_csv(models_data, output_dir / "per_triplet.csv", motion_labels)


if __name__ == "__main__":
    main()
