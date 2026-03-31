from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from anvil_exp01.data.dataset import Vimeo90KDataset
from anvil_exp01.eval.metrics import compute_lpips_batch, compute_psnr, compute_ssim
from anvil_exp01.models.conv_vfi import build_model, count_parameters, infer_route

MOTION_BINS = ["small", "medium", "large"]


def load_motion_labels(csv_path: Path | None) -> dict[str, str]:
    if csv_path is None or not csv_path.exists():
        return {}
    labels: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["triplet_id"].strip()] = row["motion_bin"].strip()
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ANVIL on Vimeo90K test set.")
    parser.add_argument("--model", required=True, help="Model ID from registry.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Vimeo90K root directory.")
    parser.add_argument("--prealigned-dir", type=Path, required=True, help="Prealigned frame root.")
    parser.add_argument("--motion-csv", type=Path, default=None, help="Optional motion_labels.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/vimeo"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N triplets (0=all, for smoke testing).")
    args = parser.parse_args()

    device = torch.device(args.device)
    route = infer_route(args.model)
    model = build_model(args.model).to(device)
    model_returns_frame = getattr(model, "returns_frame", False)
    n_params = count_parameters(model)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    dataset = Vimeo90KDataset(
        data_dir=args.data_dir,
        split="test",
        route=route,
        prealigned_dir=args.prealigned_dir,
        crop_size=0,
        augment=False,
    )
    if args.limit > 0:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    motion_labels = load_motion_labels(args.motion_csv)

    triplet_ids: list[str] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    lpipss: list[float] = []
    lpips_preds: list[np.ndarray] = []
    lpips_gts: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=args.model, unit="batch"):
            inp = batch["input"].to(device)
            blend = batch["blend"].to(device)
            gt = batch["gt"]

            output = model(inp)
            pred = output.cpu() if model_returns_frame else (blend + output).clamp(0, 1).cpu()
            pred_u8 = (pred * 255).round().clamp(0, 255).byte().numpy()
            gt_u8 = (gt * 255).round().clamp(0, 255).byte().numpy()

            for i in range(pred_u8.shape[0]):
                pred_img = pred_u8[i].transpose(1, 2, 0)
                gt_img = gt_u8[i].transpose(1, 2, 0)
                triplet_ids.append(batch["triplet_id"][i])
                psnrs.append(compute_psnr(pred_img, gt_img))
                ssims.append(compute_ssim(pred_img, gt_img))
                lpips_preds.append(pred_img)
                lpips_gts.append(gt_img)

            if len(lpips_preds) >= max(1, args.batch_size):
                lpipss.extend(
                    compute_lpips_batch(
                        lpips_preds,
                        lpips_gts,
                        device=str(device),
                        pair_batch_size=max(1, args.batch_size),
                    )
                )
                lpips_preds.clear()
                lpips_gts.clear()

    if lpips_preds:
        lpipss.extend(
            compute_lpips_batch(
                lpips_preds,
                lpips_gts,
                device=str(device),
                pair_batch_size=max(1, args.batch_size),
            )
        )

    row = {
        "Model": args.model,
        "Params": n_params,
        "Overall PSNR": float(np.mean(psnrs)),
        "Overall SSIM": float(np.mean(ssims)),
        "Overall LPIPS": float(np.mean(lpipss)),
    }
    for motion_bin in MOTION_BINS:
        indices = [i for i, tid in enumerate(triplet_ids) if motion_labels.get(tid) == motion_bin]
        row[f"{motion_bin.capitalize()} PSNR"] = float(np.mean([psnrs[i] for i in indices])) if indices else float("nan")
        row[f"{motion_bin.capitalize()} LPIPS"] = float(np.mean([lpipss[i] for i in indices])) if indices else float("nan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_triplet = args.output_dir / f"{args.model}_per_triplet.csv"
    with open(per_triplet, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["triplet_id", "motion_bin", "psnr", "ssim", "lpips"])
        for i, tid in enumerate(triplet_ids):
            writer.writerow([tid, motion_labels.get(tid, ""), f"{psnrs[i]:.4f}", f"{ssims[i]:.6f}", f"{lpipss[i]:.6f}"])

    summary = args.output_dir / "summary.csv"
    fieldnames = ["Model", "Params", "Overall PSNR", "Overall SSIM", "Overall LPIPS"]
    for motion_bin in MOTION_BINS:
        fieldnames.extend([f"{motion_bin.capitalize()} PSNR", f"{motion_bin.capitalize()} LPIPS"])
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        formatted = {}
        for key, value in row.items():
            if isinstance(value, float):
                formatted[key] = f"{value:.4f}" if "PSNR" in key or "SSIM" in key else f"{value:.6f}"
            else:
                formatted[key] = str(value)
        writer.writerow(formatted)

    print(f"Model: {args.model} ({n_params:,} params)")
    print(f"Overall PSNR: {row['Overall PSNR']:.4f}")
    print(f"Overall SSIM: {row['Overall SSIM']:.6f}")
    print(f"Overall LPIPS: {row['Overall LPIPS']:.6f}")
    print(f"Summary saved to {summary}")


if __name__ == "__main__":
    main()
