"""Distillation training for ANVIL VFI models.

Train a student model with guidance from a frozen teacher model.
Loss = L1(pred, gt) + α·L1(student_res, teacher_res) [+ LPIPS].

Usage:
    pixi run python -m anvil_exp01.train_distill \
        --student D-nafnet-bn-s-nomv \
        --teacher D-nafnet-nomv \
        --teacher-checkpoint checkpoints_phase4/D-nafnet-nomv/best.pt \
        --data-dir data/vimeo_triplet \
        --prealigned-dir data/prealigned \
        --epochs 100 --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*", module="torchvision")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*", module="torchvision")
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

log = logging.getLogger("anvil.distill")

from anvil_exp01.data.dataset import Vimeo90KDataset
from anvil_exp01.models.conv_vfi import (
    MODEL_REGISTRY,
    build_model,
    count_parameters,
    infer_route,
)
from anvil_exp01.train import (
    _chunked_lpips,
    _get_lpips_loss,
    _iter_batches,
    _make_loader,
    _setup_logging,
    _validate_route_artifacts,
    save_checkpoint,
    validate,
)


def _install_sigterm_handler() -> None:
    def _handler(signum, frame):
        raise SystemExit(f"Received signal {signum}, exiting gracefully.")
    signal.signal(signal.SIGTERM, _handler)


def _load_teacher(model_id: str, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Load teacher model from checkpoint, frozen in eval mode."""
    model = build_model(model_id).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("Teacher %s loaded from %s (%.4f dB)", model_id, checkpoint, ckpt.get("val_psnr", 0.0))
    return model


def train_one_epoch_distill(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lpips_fn,
    device: torch.device,
    distill_alpha: float = 1.0,
    amp_dtype: torch.dtype = torch.bfloat16,
    scaler: torch.amp.GradScaler | None = None,
    use_lpips: bool = True,
    lpips_every_k: int = 4,
    channels_last: bool = False,
) -> float:
    student.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(_iter_batches(loader, device, channels_last), total=len(loader), desc="  train", unit="batch", leave=False)
    for batch in pbar:
        inp = batch["input"]
        gt = batch["gt"]
        blend = batch["blend"]

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            # Student forward
            student_res = student(inp)
            student_pred = (blend + student_res).clamp(0, 1)

            # Teacher forward (no grad, already frozen)
            with torch.no_grad():
                teacher_res = teacher(inp)

            # L1 task loss
            l1_loss = F.l1_loss(student_pred, gt)

            # Distillation loss: match teacher's residual
            distill_loss = F.l1_loss(student_res, teacher_res)

            loss = l1_loss + distill_alpha * distill_loss

            # LPIPS (sparse, on final pred vs gt)
            if use_lpips and n_batches % lpips_every_k == 0:
                lpips_loss = _chunked_lpips(lpips_fn, student_pred, gt)
                loss = loss + 0.1 * lpips_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / max(n_batches, 1)


def main() -> None:
    _install_sigterm_handler()
    parser = argparse.ArgumentParser(description="Distillation training for ANVIL VFI.")
    parser.add_argument("--student", required=True, choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--teacher", required=True, choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--distill-alpha", type=float, default=1.0,
                        help="Weight for distillation loss (default: 1.0).")
    parser.add_argument("--route", default=None, choices=["A", "D", "D-nomv"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mv-flow-dir", type=Path, default=None)
    parser.add_argument("--prealigned-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--val-interval", type=int, default=3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=0.10)
    parser.add_argument("--lpips-warmup", type=int, default=20)
    parser.add_argument("--lpips-every-k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument(
        "--compile-student", action="store_true",
        help="Compile the student with torch.compile. Disabled by default because "
             "Phase 5 changes train batch size after warmup, which has triggered "
             "CUDA graph / illegal-instruction failures on resume and late epochs.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints_phase5"))
    parser.add_argument("--subset", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)

    # Route: student and teacher must use the same route
    route = args.route or infer_route(args.student)
    teacher_route = infer_route(args.teacher)
    if route != teacher_route:
        log.error("Student route '%s' != teacher route '%s'", route, teacher_route)
        sys.exit(1)

    # Validate channels
    _, student_kwargs = MODEL_REGISTRY[args.student]
    expected_ch = student_kwargs.get("in_ch", 8)
    route_ch = 8 if route == "D" else 6
    if expected_ch != route_ch:
        log.error("Student '%s' expects %dch but route '%s' provides %dch.",
                  args.student, expected_ch, route, route_ch)
        sys.exit(1)

    # Build student
    student = build_model(args.student).to(device)
    n_params = count_parameters(student)

    # Load teacher (frozen)
    teacher = _load_teacher(args.teacher, args.teacher_checkpoint, device)
    teacher_params = count_parameters(teacher)

    # Checkpoint dir
    ckpt_dir = args.output_dir / args.student
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(ckpt_dir / "train.log")

    log.info("Student: %s (%s params)  Teacher: %s (%s params)  Route: %s  alpha=%.2f",
             args.student, f"{n_params:,}", args.teacher, f"{teacher_params:,}",
             route, args.distill_alpha)

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

    _validate_route_artifacts(route, train_ds, args.mv_flow_dir, args.prealigned_dir, "train")
    _validate_route_artifacts(route, val_ds, args.mv_flow_dir, args.prealigned_dir, "val")
    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    # AMP + cudnn
    torch.backends.cudnn.benchmark = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    torch.set_float32_matmul_precision("high")

    # Loss, optimizer, scheduler
    lpips_fn = _get_lpips_loss(args.device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
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
        state_dict = ckpt["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        student.load_state_dict(state_dict)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_psnr = ckpt.get("val_psnr", 0.0)
        resumed_no_improve = ckpt.get("no_improve_count", 0)
        resumed_es_ref = ckpt.get("es_ref_psnr", best_psnr)
        log.info("Resumed from epoch %d, best PSNR: %.4f", ckpt.get("epoch", 0), best_psnr)

    warmup_batch_size = args.warmup_batch_size if args.warmup_batch_size > 0 else args.batch_size
    train_batch_size = warmup_batch_size if start_epoch <= args.lpips_warmup else args.batch_size
    train_loader = _make_loader(
        train_ds, batch_size=train_batch_size, shuffle=True,
        workers=args.workers, drop_last=True, prefetch_factor=args.prefetch_factor,
    )
    val_loader = _make_loader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        workers=args.workers, drop_last=False, prefetch_factor=args.prefetch_factor,
    )
    log.info("AMP dtype: %s  cudnn.benchmark: True  workers=%d  train_batch=%d  warmup_batch=%d",
             amp_dtype, args.workers, args.batch_size, warmup_batch_size)

    # Distillation already pays for a full teacher forward every step, so the
    # stability trade-off is not worth forcing torch.compile here. In practice
    # Phase 5 switches train batch size after warmup (16 -> 8), and the first
    # compiled backward at the new size has hit CUDA illegal-instruction errors.
    if args.compile_student:
        student = torch.compile(student)
        log.info("torch.compile: student compiled")
    else:
        log.info("torch.compile: disabled for student")

    # CSV log
    csv_path = ckpt_dir / "training_log.csv"
    csv_exists = csv_path.exists() and resume_path is not None
    csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "lr"])

    # Early stopping
    patience = args.patience
    min_delta = args.min_delta
    no_improve_count = resumed_no_improve
    es_ref_psnr = resumed_es_ref if resumed_es_ref > 0 else best_psnr

    log.info("Training for epochs %d–%d (patience=%d, min_delta=%.2f, lpips_warmup=%d, "
             "lpips_every_k=%d, distill_alpha=%.2f) ...",
             start_epoch, args.epochs, patience, min_delta, args.lpips_warmup,
             args.lpips_every_k, args.distill_alpha)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        desired_batch_size = warmup_batch_size if epoch <= args.lpips_warmup else args.batch_size
        if desired_batch_size != train_batch_size:
            train_batch_size = desired_batch_size
            train_loader = _make_loader(
                train_ds, batch_size=train_batch_size, shuffle=True,
                workers=args.workers, drop_last=True, prefetch_factor=args.prefetch_factor,
            )
            log.info("Switched train batch size to %d at epoch %d", train_batch_size, epoch)

        use_lpips = epoch > args.lpips_warmup
        train_loss = train_one_epoch_distill(
            student, teacher, train_loader, optimizer, lpips_fn, device,
            distill_alpha=args.distill_alpha, amp_dtype=amp_dtype, scaler=scaler,
            use_lpips=use_lpips, lpips_every_k=args.lpips_every_k,
            channels_last=args.channels_last,
        )

        do_val = (epoch % args.val_interval == 0) or (epoch == args.epochs)
        val_loss, val_psnr = (-1.0, -1.0)
        is_best = False

        if do_val:
            val_loss, val_psnr = validate(
                student, val_loader, device, amp_dtype=amp_dtype,
                channels_last=args.channels_last, returns_frame=False,
            )
            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
            if val_psnr > es_ref_psnr + min_delta:
                es_ref_psnr = val_psnr
                no_improve_count = 0
            else:
                no_improve_count += 1
            save_checkpoint(
                student, optimizer, scheduler, epoch, val_psnr,
                ckpt_dir, is_best, save_epoch=True,
                no_improve_count=no_improve_count, es_ref_psnr=es_ref_psnr,
            )
        else:
            save_checkpoint(
                student, optimizer, scheduler, epoch, best_psnr,
                ckpt_dir, is_best=False, save_epoch=False,
                no_improve_count=no_improve_count, es_ref_psnr=es_ref_psnr,
            )

        scheduler.step()
        elapsed = time.time() - t0

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

        if patience > 0 and no_improve_count >= patience:
            log.info("Early stopping at epoch %d: no improvement for %d val rounds.", epoch, patience)
            break

    csv_file.close()
    log.info("Done. Best PSNR: %.4f", best_psnr)
    log.info("Checkpoints: %s", ckpt_dir)


if __name__ == "__main__":
    main()
