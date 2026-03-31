from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ANVIL on Xiph 1080p.")
    parser.add_argument("--model", required=True, help="Model ID from registry.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Xiph root directory.")
    parser.add_argument("--prealigned-dir", type=Path, required=True, help="Prealigned frame root.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/xiph"))
    parser.add_argument("--route", default=None, choices=["A", "D", "D-nomv", "FR"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N triplets (0=all, for smoke testing).")
    args = parser.parse_args()

    device = torch.device(args.device)
    route = args.route or infer_route(args.model)

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
        data_dir=str(args.data_dir),
        split="test",
        route=route,
        mv_flow_dir=str(args.data_dir / "dense_flow") if route in ("D", "FR") else None,
        prealigned_dir=str(args.prealigned_dir) if route in ("D", "D-nomv") else None,
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

    triplet_ids: list[str] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    lpipss: list[float] = []
    lpips_preds: list[np.ndarray] = []
    lpips_gts: list[np.ndarray] = []
    pending_triplet_ids: list[str] = []
    seq_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"psnr": [], "ssim": [], "lpips": []})

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
                triplet_id = batch["triplet_id"][i]
                seq = triplet_id.split("/")[0]
                psnr = compute_psnr(pred_img, gt_img)
                ssim = compute_ssim(pred_img, gt_img)
                triplet_ids.append(triplet_id)
                pending_triplet_ids.append(triplet_id)
                psnrs.append(psnr)
                ssims.append(ssim)
                lpips_preds.append(pred_img)
                lpips_gts.append(gt_img)
                seq_metrics[seq]["psnr"].append(psnr)
                seq_metrics[seq]["ssim"].append(ssim)

            if len(lpips_preds) >= max(1, args.batch_size):
                batch_lpips = compute_lpips_batch(
                    lpips_preds,
                    lpips_gts,
                    device=str(device),
                    pair_batch_size=max(1, args.batch_size),
                )
                lpipss.extend(batch_lpips)
                for triplet_id, lpips_value in zip(pending_triplet_ids, batch_lpips):
                    seq = triplet_id.split("/")[0]
                    seq_metrics[seq]["lpips"].append(lpips_value)
                lpips_preds.clear()
                lpips_gts.clear()
                pending_triplet_ids.clear()

    if lpips_preds:
        batch_lpips = compute_lpips_batch(
            lpips_preds,
            lpips_gts,
            device=str(device),
            pair_batch_size=max(1, args.batch_size),
        )
        lpipss.extend(batch_lpips)
        for triplet_id, lpips_value in zip(pending_triplet_ids, batch_lpips):
            seq = triplet_id.split("/")[0]
            seq_metrics[seq]["lpips"].append(lpips_value)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dataset", "n_triplets", "params", "psnr", "ssim", "lpips"])
        writer.writerow([
            args.model,
            "xiph_1080p",
            len(dataset),
            n_params,
            f"{float(np.mean(psnrs)):.4f}",
            f"{float(np.mean(ssims)):.6f}",
            f"{float(np.mean(lpipss)):.6f}",
        ])

    per_sequence = args.output_dir / "per_sequence.csv"
    with open(per_sequence, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "n_triplets", "psnr", "ssim", "lpips"])
        for seq in sorted(seq_metrics):
            metrics = seq_metrics[seq]
            writer.writerow([
                seq,
                len(metrics["psnr"]),
                f"{float(np.mean(metrics['psnr'])):.4f}",
                f"{float(np.mean(metrics['ssim'])):.6f}",
                f"{float(np.mean(metrics['lpips'])):.6f}",
            ])

    per_triplet = args.output_dir / "per_triplet.csv"
    with open(per_triplet, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["triplet_id", "psnr", "ssim", "lpips"])
        for triplet_id, psnr, ssim, lpips in zip(triplet_ids, psnrs, ssims, lpipss):
            writer.writerow([triplet_id, f"{psnr:.4f}", f"{ssim:.6f}", f"{lpips:.6f}"])

    print(f"Model: {args.model} ({n_params:,} params)")
    print(f"Overall PSNR: {float(np.mean(psnrs)):.4f}")
    print(f"Overall SSIM: {float(np.mean(ssims)):.6f}")
    print(f"Overall LPIPS: {float(np.mean(lpipss)):.6f}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
