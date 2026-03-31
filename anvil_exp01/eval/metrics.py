"""Unified evaluation metrics for VFI experiments.

Provides PSNR, SSIM, and LPIPS computation on uint8 HWC numpy images.

Usage:
    from anvil_exp01.eval.metrics import compute_all, compute_batch

    scores = compute_all(pred, gt, device="cuda")
    # {'psnr': 32.1, 'ssim': 0.95, 'lpips': 0.032}

    batch_scores = compute_batch(preds, gts, device="cuda")
    # {'psnr': [...], 'ssim': [...], 'lpips': [...],
    #  'mean_psnr': 32.1, 'mean_ssim': 0.95, 'mean_lpips': 0.032}
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

# Module-level cache for LPIPS models, keyed by (net, device).
_lpips_cache: dict[tuple[str, str], object] = {}


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two uint8 HWC images.

    Parameters
    ----------
    pred : np.ndarray
        Predicted image, uint8, shape (H, W, C).
    gt : np.ndarray
        Ground truth image, uint8, shape (H, W, C).

    Returns
    -------
    float
        PSNR in dB. Returns 100.0 if images are identical (MSE == 0).
    """
    pred_f = pred.astype(np.float64)
    gt_f = gt.astype(np.float64)
    mse = np.mean((pred_f - gt_f) ** 2)
    if mse == 0.0:
        return 100.0
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Structural Similarity Index between two uint8 HWC images.

    Uses Gaussian weighting (win_size=11, sigma=1.5), matching the standard
    Wang et al. SSIM definition and the GPU-accelerated compute_ssim_batch().

    Parameters
    ----------
    pred : np.ndarray
        Predicted image, uint8, shape (H, W, C).
    gt : np.ndarray
        Ground truth image, uint8, shape (H, W, C).

    Returns
    -------
    float
        SSIM value in [0, 1].
    """
    return float(
        structural_similarity(
            pred,
            gt,
            channel_axis=2,
            data_range=255,
            gaussian_weights=True,
            win_size=11,
            sigma=1.5,
        )
    )


_ssim_kernel_cache: dict[tuple[int, int, str], torch.Tensor] = {}


def _get_ssim_kernel(window_size: int, channels: int, device: str) -> torch.Tensor:
    key = (window_size, channels, device)
    if key not in _ssim_kernel_cache:
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords.pow(2) / (2 * sigma**2))
        g = g / g.sum()
        kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
        kernel = (
            kernel_2d.unsqueeze(0)
            .unsqueeze(0)
            .expand(channels, 1, -1, -1)
            .contiguous()
            .to(device)
        )
        _ssim_kernel_cache[key] = kernel
    return _ssim_kernel_cache[key]


def compute_ssim_batch(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    device: str = "cuda",
    batch_size: int = 16,
) -> list[float]:
    """GPU-accelerated SSIM for multiple uint8 HWC image pairs.

    Uses gaussian kernel (window=11, sigma=1.5) with reflect padding,
    matching compute_ssim(). ~10-50x faster than skimage on 1080p.
    """
    if not preds:
        return []
    if len(preds) != len(gts):
        raise ValueError(
            f"preds and gts length mismatch: {len(preds)} vs {len(gts)}"
        )

    window_size = 11
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    pad = window_size // 2

    results: list[float] = []

    for start in range(0, len(preds), batch_size):
        end = min(start + batch_size, len(preds))

        pred_np = np.stack(preds[start:end]).astype(np.float32)
        gt_np = np.stack(gts[start:end]).astype(np.float32)

        pred_t = torch.from_numpy(pred_np).permute(0, 3, 1, 2).to(device)
        gt_t = torch.from_numpy(gt_np).permute(0, 3, 1, 2).to(device)

        channels = pred_t.shape[1]
        kernel = _get_ssim_kernel(window_size, channels, device)

        pred_p = F.pad(pred_t, [pad, pad, pad, pad], mode="reflect")
        gt_p = F.pad(gt_t, [pad, pad, pad, pad], mode="reflect")

        mu1 = F.conv2d(pred_p, kernel, groups=channels)
        mu2 = F.conv2d(gt_p, kernel, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu12 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(pred_p * pred_p, kernel, groups=channels) - mu1_sq
        )
        sigma2_sq = F.conv2d(gt_p * gt_p, kernel, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred_p * gt_p, kernel, groups=channels) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        ssim_per_image = ssim_map.mean(dim=[1, 2, 3])
        results.extend(float(x) for x in ssim_per_image.cpu().tolist())

        del pred_t, gt_t, pred_p, gt_p, mu1, mu2, mu1_sq, mu2_sq, mu12
        del sigma1_sq, sigma2_sq, sigma12, ssim_map, ssim_per_image

    return results


def _get_lpips_model(net: str, device: str) -> object:
    """Return a cached LPIPS model instance.

    The import is deferred so that ``lpips`` is only required when actually
    computing LPIPS, not at module import time.
    """
    key = (net, device)
    if key not in _lpips_cache:
        import lpips as lpips_lib

        model = lpips_lib.LPIPS(net=net, verbose=False).eval().to(device)
        _lpips_cache[key] = model
    return _lpips_cache[key]


def compute_lpips(
    pred: np.ndarray,
    gt: np.ndarray,
    net: str = "alex",
    device: str = "cuda",
) -> float:
    """Compute LPIPS perceptual distance between two uint8 HWC images.

    Parameters
    ----------
    pred : np.ndarray
        Predicted image, uint8, shape (H, W, C).
    gt : np.ndarray
        Ground truth image, uint8, shape (H, W, C).
    net : str
        Backbone network for LPIPS ('alex', 'vgg', 'squeeze'). Default: 'alex'.
    device : str
        Torch device string. Default: 'cuda'.

    Returns
    -------
    float
        LPIPS distance (lower is better).
    """
    model = _get_lpips_model(net, device)

    # uint8 HWC [0, 255] -> float32 CHW [-1, 1]
    pred_t = torch.from_numpy(pred.astype(np.float32)).permute(2, 0, 1) / 127.5 - 1.0
    gt_t = torch.from_numpy(gt.astype(np.float32)).permute(2, 0, 1) / 127.5 - 1.0

    pred_t = pred_t.unsqueeze(0).to(device)
    gt_t = gt_t.unsqueeze(0).to(device)

    with torch.no_grad():
        dist = model(pred_t, gt_t)

    return float(dist.item())


def compute_lpips_batch(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    net: str = "alex",
    device: str = "cuda",
    pair_batch_size: int | None = 64,
) -> list[float]:
    """Compute LPIPS for multiple image pairs in a single batched forward pass.

    Much faster than calling compute_lpips() in a loop because it avoids
    repeated CPU→GPU transfers and kernel launches.

    Parameters
    ----------
    preds : list[np.ndarray]
        Predicted images (uint8 HWC).
    gts : list[np.ndarray]
        Ground-truth images (uint8 HWC), same length as preds.
    net : str
        LPIPS backbone ('alex', 'vgg', 'squeeze').
    device : str
        Torch device string.
    pair_batch_size : int | None
        Number of image pairs per LPIPS forward pass. Use smaller values to
        reduce VRAM usage and larger values to improve throughput.
    """
    if not preds:
        return []
    if len(preds) != len(gts):
        raise ValueError(
            f"preds and gts must have the same length, got {len(preds)} and {len(gts)}"
        )

    model = _get_lpips_model(net, device)
    n = len(preds)
    if pair_batch_size is None or pair_batch_size <= 0:
        pair_batch_size = n

    out: list[float] = []
    start = 0

    while start < n:
        end = min(start + pair_batch_size, n)

        pred_np = np.stack(preds[start:end], axis=0).astype(np.float32)
        gt_np = np.stack(gts[start:end], axis=0).astype(np.float32)

        pred_batch = torch.from_numpy(pred_np).permute(0, 3, 1, 2) / 127.5 - 1.0
        gt_batch = torch.from_numpy(gt_np).permute(0, 3, 1, 2) / 127.5 - 1.0

        if device.startswith("cuda"):
            pred_batch = pred_batch.pin_memory()
            gt_batch = gt_batch.pin_memory()
            pred_batch = pred_batch.to(device, non_blocking=True)
            gt_batch = gt_batch.to(device, non_blocking=True)
        else:
            pred_batch = pred_batch.to(device)
            gt_batch = gt_batch.to(device)

        try:
            with torch.no_grad():
                dists = model(pred_batch, gt_batch).flatten()
        except torch.OutOfMemoryError:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            del pred_batch, gt_batch
            if pair_batch_size <= 1:
                raise RuntimeError(
                    "LPIPS OOM even at pair_batch_size=1. "
                    "Free GPU memory or use CPU for LPIPS."
                ) from None
            pair_batch_size = max(1, pair_batch_size // 2)
            continue

        out.extend(float(x) for x in dists.detach().cpu().tolist())
        del pred_batch, gt_batch, dists
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        start = end

    return out


def compute_all(
    pred: np.ndarray,
    gt: np.ndarray,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute PSNR, SSIM, and LPIPS for a single image pair.

    Parameters
    ----------
    pred : np.ndarray
        Predicted image, uint8, shape (H, W, C).
    gt : np.ndarray
        Ground truth image, uint8, shape (H, W, C).
    device : str
        Torch device for LPIPS. Default: 'cuda'.

    Returns
    -------
    dict[str, float]
        Keys: 'psnr', 'ssim', 'lpips'.
    """
    return {
        "psnr": compute_psnr(pred, gt),
        "ssim": compute_ssim(pred, gt),
        "lpips": compute_lpips(pred, gt, device=device),
    }


def compute_batch(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    device: str = "cuda",
) -> dict[str, list[float] | float]:
    """Compute metrics for a batch of image pairs.

    Parameters
    ----------
    preds : list[np.ndarray]
        List of predicted images, each uint8 HWC.
    gts : list[np.ndarray]
        List of ground truth images, each uint8 HWC.
    device : str
        Torch device for LPIPS. Default: 'cuda'.

    Returns
    -------
    dict
        Keys: 'psnr', 'ssim', 'lpips' (lists of per-pair values),
        plus 'mean_psnr', 'mean_ssim', 'mean_lpips'.

    Raises
    ------
    ValueError
        If preds and gts have different lengths.
    """
    if len(preds) != len(gts):
        raise ValueError(
            f"preds and gts must have the same length, got {len(preds)} and {len(gts)}"
        )

    psnr_list: list[float] = []
    ssim_list: list[float] = []
    lpips_list: list[float] = []

    for pred, gt in zip(preds, gts):
        psnr_list.append(compute_psnr(pred, gt))
        ssim_list.append(compute_ssim(pred, gt))
        lpips_list.append(compute_lpips(pred, gt, device=device))

    return {
        "psnr": psnr_list,
        "ssim": ssim_list,
        "lpips": lpips_list,
        "mean_psnr": float(np.mean(psnr_list)),
        "mean_ssim": float(np.mean(ssim_list)),
        "mean_lpips": float(np.mean(lpips_list)),
    }
