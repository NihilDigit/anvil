"""Unified training script for ANVIL VFI models.

Trains residual-learning VFI models with L1 loss.
Supports all model IDs in the registry and auto-infers data route.

Usage:
    pixi run python -m anvil_exp01.train \
        --model D-small --data-dir data/vimeo_triplet \
        --mv-flow-dir data/mv_dense_flow \
        --prealigned-dir data/prealigned \
        --epochs 100 --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import signal
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*", module="torchvision")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*", module="torchvision")
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

log = logging.getLogger("anvil.train")

from anvil_exp01.data.dataset import Vimeo90KDataset
from anvil_exp01.models.conv_vfi import (
    MODEL_REGISTRY,
    NAFNetVFI,
    build_model,
    count_parameters,
    infer_route,
    load_nafnet_pretrained,
)


def _setup_logging(log_file: Path) -> None:
    """Configure logging to both console and a file."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File (append so resumes keep history)
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def _edge_map(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a simple differentiable high-frequency map.

    Args:
        x: (B, C, H, W) in [0, 1]
        mode: 'laplacian' or 'sobel'
    """
    if mode not in {"laplacian", "sobel"}:
        raise ValueError(f"Unknown edge loss mode: {mode}")

    gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

    if mode == "laplacian":
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, 3, 3)
        return F.conv2d(gray, kernel, padding=1)

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-6)


def _edge_loss(pred: torch.Tensor, gt: torch.Tensor, mode: str) -> torch.Tensor:
    return F.l1_loss(_edge_map(pred, mode), _edge_map(gt, mode))


def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    drop_last: bool,
    prefetch_factor: int,
) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


class _CudaPrefetchLoader:
    """Overlap host->device copies with the current step on a side CUDA stream."""

    def __init__(self, loader: DataLoader, device: torch.device, channels_last: bool = False):
        self.loader = loader
        self.device = device
        self.channels_last = channels_last
        self.stream = torch.cuda.Stream(device=device)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        first = True
        next_batch = None
        for batch in self.loader:
            with torch.cuda.stream(self.stream):
                next_batch = {
                    "input": batch["input"].to(self.device, non_blocking=True),
                    "gt": batch["gt"].to(self.device, non_blocking=True),
                    "blend": batch["blend"].to(self.device, non_blocking=True),
                    "triplet_id": batch["triplet_id"],
                }
                if self.channels_last:
                    next_batch["input"] = next_batch["input"].contiguous(memory_format=torch.channels_last)
                    next_batch["gt"] = next_batch["gt"].contiguous(memory_format=torch.channels_last)
                    next_batch["blend"] = next_batch["blend"].contiguous(memory_format=torch.channels_last)

            if not first:
                yield current_batch
            else:
                first = False

            torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
            current_batch = next_batch

        if not first:
            yield current_batch


def _iter_batches(loader: DataLoader, device: torch.device, channels_last: bool):
    if device.type == "cuda":
        return _CudaPrefetchLoader(loader, device, channels_last=channels_last)
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    scaler: torch.amp.GradScaler | None = None,
    edge_loss_weight: float = 0.0,
    edge_loss_mode: str = "laplacian",
    channels_last: bool = False,
    returns_frame: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(_iter_batches(loader, device, channels_last), total=len(loader), desc="  train", unit="batch", leave=False)
    for batch in pbar:
        inp = batch["input"]
        gt = batch["gt"]
        blend = batch["blend"]

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            output = model(inp)
            pred = output if returns_frame else (blend + output).clamp(0, 1)
            l1_loss = F.l1_loss(pred, gt)
            loss = l1_loss

            if edge_loss_weight > 0:
                loss = loss + edge_loss_weight * _edge_loss(pred, gt, edge_loss_mode)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    channels_last: bool = False,
    returns_frame: bool = False,
) -> tuple[float, float]:
    """Run validation, return (mean_l1_loss, mean_psnr).

    BatchNorm layers stay in train mode (use batch stats) to avoid
    numerical instability from stale running stats early in training.
    Running stats (running_mean, running_var, num_batches_tracked) are
    saved before the validation loop and restored afterwards so that
    validation data does not contaminate the BN statistics used at
    deploy time (e.g. fuse_for_deploy()).
    """
    model.eval()
    # Keep BN in train mode: running stats may diverge from batch stats
    # early in training, causing NaN with SimpleGate/SCA multiplicative ops.
    bn_modules = [
        m for m in model.modules()
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
    ]
    # Snapshot running stats before validation so they are not contaminated.
    bn_snapshots = []
    for m in bn_modules:
        bn_snapshots.append((
            m.running_mean.clone() if m.running_mean is not None else None,
            m.running_var.clone() if m.running_var is not None else None,
            m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None,
        ))
        m.train()

    total_l1 = 0.0
    total_psnr = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in _iter_batches(loader, device, channels_last):
            inp = batch["input"]
            gt = batch["gt"]
            blend = batch["blend"]
            bs = inp.size(0)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                output = model(inp)
                pred = output if returns_frame else (blend + output).clamp(0, 1)

            total_l1 += F.l1_loss(pred, gt).item() * bs

            for i in range(bs):
                mse = F.mse_loss(pred[i], gt[i])
                if mse > 0 and not torch.isnan(mse):
                    total_psnr += (10 * torch.log10(1.0 / mse)).item()
                elif torch.isnan(mse):
                    total_psnr += 0.0  # NaN → 0 dB (worst case)
                else:
                    total_psnr += 100.0

            n_samples += bs

    # Restore BN running stats so validation data doesn't leak into them.
    for m, (rm, rv, nbt) in zip(bn_modules, bn_snapshots):
        if rm is not None:
            m.running_mean.copy_(rm)
        if rv is not None:
            m.running_var.copy_(rv)
        if nbt is not None:
            m.num_batches_tracked.copy_(nbt)

    return total_l1 / max(n_samples, 1), total_psnr / max(n_samples, 1)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_psnr: float,
    ckpt_dir: Path,
    is_best: bool,
    save_epoch: bool = True,
    no_improve_count: int = 0,
    es_ref_psnr: float = 0.0,
) -> None:
    # Unwrap torch.compile wrapper to save clean state_dict keys
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    state = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_psnr": val_psnr,
        "no_improve_count": no_improve_count,
        "es_ref_psnr": es_ref_psnr,
    }
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(state, ckpt_dir / "latest.pt")

    if save_epoch:
        torch.save(state, ckpt_dir / f"epoch{epoch:03d}.pt")
        # Keep only last 3 epoch checkpoints
        epoch_files = sorted(ckpt_dir.glob("epoch*.pt"))
        while len(epoch_files) > 3:
            epoch_files.pop(0).unlink()

    if is_best:
        torch.save(state, ckpt_dir / "best.pt")


def _dataset_triplet_ids(dataset) -> list[str]:
    """Return the concrete triplet IDs covered by a dataset/subset."""
    if isinstance(dataset, Vimeo90KDataset):
        return list(dataset.triplet_ids)
    if isinstance(dataset, Subset) and isinstance(dataset.dataset, Vimeo90KDataset):
        base = dataset.dataset
        return [base.triplet_ids[i] for i in dataset.indices]
    return []


def _validate_route_artifacts(
    route: str,
    dataset,
    mv_flow_dir: Path | None,
    prealigned_dir: Path | None,
    split_name: str,
    max_examples: int = 8,
) -> None:
    """Fail fast when Route D/D-nomv/FR inputs are incomplete."""
    if route == "A":
        return

    triplet_ids = _dataset_triplet_ids(dataset)
    if not triplet_ids:
        return

    missing: list[Path] = []
    for tid in triplet_ids:
        seq_id, trip_id = tid.split("/")

        if route in ("D", "D-nomv"):
            if prealigned_dir is None:
                raise ValueError("prealigned_dir is required for route D / D-nomv")
            for name in ("im1_aligned.png", "im3_aligned.png"):
                path = prealigned_dir / seq_id / trip_id / name
                if not path.exists():
                    missing.append(path)
                    if len(missing) >= max_examples:
                        break

        if route in ("D", "FR"):
            if mv_flow_dir is None:
                raise ValueError("mv_flow_dir is required for route D / FR")
            path = mv_flow_dir / seq_id / f"{trip_id}.npy"
            if not path.exists():
                missing.append(path)

        if len(missing) >= max_examples:
            break

    if missing:
        details = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing precomputed artifacts for split '{split_name}' (route {route}). "
            f"First missing paths:\n{details}"
        )


def _install_sigterm_handler() -> None:
    """Re-raise SIGTERM as SystemExit so DataLoader workers get cleaned up."""
    def _handler(signum, frame):
        raise SystemExit(f"Received signal {signum}, exiting gracefully.")
    signal.signal(signal.SIGTERM, _handler)


def main() -> None:
    _install_sigterm_handler()
    parser = argparse.ArgumentParser(description="Train ANVIL VFI models.")
    parser.add_argument(
        "--model", required=True, choices=sorted(MODEL_REGISTRY.keys()),
        help="Model ID from registry.",
    )
    parser.add_argument(
        "--route", default=None, choices=["A", "D", "D-nomv", "FR"],
        help="Data route (auto-inferred from model ID if omitted).",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mv-flow-dir", type=Path, default=None)
    parser.add_argument("--prealigned-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--prefetch-factor", type=int, default=2,
        help="DataLoader prefetch factor per worker (workers>0 only).",
    )
    parser.add_argument("--val-interval", type=int, default=3)
    parser.add_argument(
        "--patience", type=int, default=7,
        help="Early stopping: halt after this many val rounds without PSNR improvement. "
             "Actual epochs = patience * val_interval. 0 = disabled.",
    )
    parser.add_argument(
        "--min-delta", type=float, default=0.10,
        help="Minimum PSNR improvement (dB) to count as progress for early stopping. "
             "Prevents tiny fluctuations from resetting the patience counter.",
    )
    parser.add_argument(
        "--edge-loss-weight", type=float, default=0.0,
        help="Optional L1 loss on high-frequency maps (default: 0.0 = disabled).",
    )
    parser.add_argument(
        "--edge-loss-mode", choices=["laplacian", "sobel"], default="laplacian",
        help="High-frequency map used by --edge-loss-weight.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--channels-last", action="store_true",
        help="Use NHWC/channels_last tensors on CUDA conv models for better throughput.",
    )
    parser.add_argument(
        "--pretrained", type=Path, default=None,
        help="Pretrained weights to initialize from (e.g. NAFNet GoPro checkpoint). "
             "Only used when not resuming.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Checkpoint path to resume from. If 'auto', resumes from latest.pt if it exists.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("checkpoints"),
        help="Base directory for checkpoints (default: checkpoints/).",
    )
    parser.add_argument(
        "--subset", type=float, default=1.0,
        help="Fraction of training data to use (e.g. 0.25 for 1/4). Default: 1.0 (all).",
    )
    parser.add_argument(
        "--val-subset", type=float, default=1.0,
        help="Fraction of validation data to use for quick ablations. Default: 1.0 (all).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)  # reserve 15% VRAM for desktop/driver

    # Infer route from model ID
    route = args.route or infer_route(args.model)

    # Validate route vs model input channels
    _, model_kwargs = MODEL_REGISTRY[args.model]
    expected_ch = model_kwargs.get("in_ch", 8)
    route_ch = 8 if route in ("D", "FR") else 6
    if expected_ch != route_ch:
        log.error(
            "model '%s' expects %dch input but route '%s' provides %dch.",
            args.model, expected_ch, route, route_ch,
        )
        sys.exit(1)

    # Build model
    model = build_model(args.model).to(device)
    model_returns_frame = getattr(model, "returns_frame", False)

    # Load pretrained weights (before resume, which takes priority)
    if args.pretrained and isinstance(model, NAFNetVFI):
        loaded, skipped = load_nafnet_pretrained(model, str(args.pretrained))
        log.info("Pretrained: loaded %d keys, skipped %d from %s", loaded, skipped, args.pretrained)

    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    n_params = count_parameters(model)

    # Checkpoint dir (needed early for logging setup)
    ckpt_dir = args.output_dir / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging: console + file
    _setup_logging(ckpt_dir / "train.log")

    log.info("Model: %s  Route: %s  Params: %s", args.model, route, f"{n_params:,}")

    # Datasets
    ds_kwargs = dict(
        data_dir=args.data_dir,
        route=route,
        mv_flow_dir=args.mv_flow_dir,
        prealigned_dir=args.prealigned_dir,
    )
    train_ds = Vimeo90KDataset(split="train", crop_size=256, augment=True, **ds_kwargs)
    val_ds = Vimeo90KDataset(split="val", crop_size=0, augment=False, **ds_kwargs)

    if args.subset < 1.0:
        full_len = len(train_ds)
        n = int(full_len * args.subset)
        gen = torch.Generator().manual_seed(42)
        indices = torch.randperm(full_len, generator=gen)[:n].tolist()
        train_ds = torch.utils.data.Subset(train_ds, indices)
        log.info("Subset: %d / %d training samples (%.0f%%)", n, full_len, args.subset * 100)

    if args.val_subset < 1.0:
        full_len = len(val_ds)
        n = int(full_len * args.val_subset)
        gen = torch.Generator().manual_seed(123)
        indices = torch.randperm(full_len, generator=gen)[:n].tolist()
        val_ds = torch.utils.data.Subset(val_ds, indices)
        log.info("Val subset: %d / %d validation samples (%.0f%%)", n, full_len, args.val_subset * 100)

    _validate_route_artifacts(route, train_ds, args.mv_flow_dir, args.prealigned_dir, "train")
    _validate_route_artifacts(route, val_ds, args.mv_flow_dir, args.prealigned_dir, "val")

    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    # Performance: AMP + cudnn benchmark + compile
    torch.backends.cudnn.benchmark = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # GradScaler only needed for float16; bfloat16 has sufficient dynamic range
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    torch.set_float32_matmul_precision("high")

    # Optimizer, scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # Resume (--resume auto looks for latest.pt automatically)
    start_epoch = 1
    best_psnr = 0.0
    resumed_no_improve = 0
    resumed_es_ref = 0.0
    resume_path = args.resume
    if resume_path is not None and str(resume_path) == "auto":
        candidate = ckpt_dir / "latest.pt"
        resume_path = candidate if candidate.exists() else None
    if resume_path and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        # Handle both clean keys and legacy _orig_mod. prefixed keys
        state_dict = ckpt["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("val_psnr", 0.0)
        resumed_no_improve = ckpt.get("no_improve_count", 0)
        resumed_es_ref = ckpt.get("es_ref_psnr", best_psnr)
        log.info("Resumed from epoch %d, best PSNR: %.4f", ckpt["epoch"], best_psnr)

    train_loader = _make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.workers,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = _make_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
        drop_last=False,
        prefetch_factor=args.prefetch_factor,
    )
    log.info(
        "AMP dtype: %s  cudnn.benchmark: True  workers=%d  batch=%d  channels_last=%s  edge_loss=%s@%.3f",
        amp_dtype, args.workers, args.batch_size, args.channels_last,
        args.edge_loss_mode, args.edge_loss_weight,
    )

    # Compile after resume so checkpoint keys are clean
    model = torch.compile(model)
    log.info("torch.compile: model compiled")

    # CSV metrics log
    csv_path = ckpt_dir / "training_log.csv"
    csv_exists = csv_path.exists() and resume_path is not None
    csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "lr"])

    # Early stopping state (restored from checkpoint if resuming)
    patience = args.patience
    min_delta = args.min_delta
    no_improve_count = resumed_no_improve
    es_ref_psnr = resumed_es_ref if resumed_es_ref > 0 else best_psnr

    # Training loop
    log.info(
        "Training for epochs %d–%d (patience=%d, min_delta=%.2f, edge=%s@%.3f) ...",
        start_epoch, args.epochs, patience, min_delta,
        args.edge_loss_mode, args.edge_loss_weight,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            amp_dtype, scaler,
            edge_loss_weight=args.edge_loss_weight, edge_loss_mode=args.edge_loss_mode,
            channels_last=args.channels_last, returns_frame=model_returns_frame,
        )

        do_val = (epoch % args.val_interval == 0) or (epoch == args.epochs)
        val_loss, val_psnr = (-1.0, -1.0)
        is_best = False

        if do_val:
            val_loss, val_psnr = validate(
                model, val_loader, device, amp_dtype=amp_dtype,
                channels_last=args.channels_last, returns_frame=model_returns_frame,
            )
            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
            # Early stopping: only reset counter on meaningful improvement
            if val_psnr > es_ref_psnr + min_delta:
                es_ref_psnr = val_psnr
                no_improve_count = 0
            else:
                no_improve_count += 1
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_psnr,
                ckpt_dir, is_best, save_epoch=True,
                no_improve_count=no_improve_count, es_ref_psnr=es_ref_psnr,
            )
        else:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_psnr,
                ckpt_dir, is_best=False, save_epoch=False,
                no_improve_count=no_improve_count, es_ref_psnr=es_ref_psnr,
            )

        scheduler.step()
        elapsed = time.time() - t0

        # CSV log
        csv_writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}" if val_loss >= 0 else "",
            f"{val_psnr:.4f}" if val_psnr >= 0 else "",
            f"{lr:.2e}",
        ])
        csv_file.flush()

        status = f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f}"
        if val_psnr >= 0:
            status += f" | val_loss={val_loss:.4f} | val_psnr={val_psnr:.2f}"
            if is_best:
                status += " *"
            status += f" | no_improve={no_improve_count}/{patience}"
        status += f" | lr={lr:.2e} | {elapsed:.1f}s"
        log.info(status)

        # Early stopping check
        if patience > 0 and no_improve_count >= patience:
            log.info("Early stopping at epoch %d: no improvement for %d val rounds.", epoch, patience)
            break

    csv_file.close()
    log.info("Done. Best PSNR: %.4f", best_psnr)
    log.info("Checkpoints: %s", ckpt_dir)


if __name__ == "__main__":
    main()
