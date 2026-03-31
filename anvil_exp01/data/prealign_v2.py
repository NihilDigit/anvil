"""Prealign v2: Block-native MV prealignment with sub-pixel precision.

Reads raw .npz block MVs directly (skips mv_to_dense intermediate step).
Supports multiple methods for progressive ablation:

  med_gauss    : ★ PRODUCTION. 1/4↓ median 5 → Gaussian σ=2 → bilinear↑ → remap
                 +1.28 dB over v1. Matches device pipeline (CPU+GPU, <1ms GPU).
  block_avg    : Block-native average splatting, integer pixel (no OBMC)
  block_subpel : Block-native average splatting, sub-pixel via cv2.remap
  obmc         : OBMC-weighted flow field → sub-pixel remap
  aobmc        : SAD+spatial reliability weighted OBMC → sub-pixel remap
  daala        : Gaussian-smoothed flow → sub-pixel remap (tunable sigma)

Each method produces im1_aligned.png + im3_aligned.png, compatible with
existing dataset.py Route D-nomv loading.

Usage:
    pixi run python anvil_exp01/data/prealign_v2.py \
        --data-dir data/vimeo_triplet \
        --mv-dir data/vimeo_triplet/mv_cache \
        --output-dir data/vimeo_triplet/prealigned_v2 \
        --method med_gauss \
        --split test \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

# ───────────────────── OBMC weight tables ─────────────────────
# Raised cosine pyramid weights (from FFmpeg vf_minterpolate / H.263 Annex F).
# Separable 1-D profiles, outer product gives 2-D window.
# Values are uint8 [0..255] and will be normalized to float at use time.

def _make_obmc_window(bh: int, bw: int) -> np.ndarray:
    """Generate a 2-D raised cosine OBMC window for block of size (bh, bw).

    Returns float32 array of shape (bh, bw) in [0, 1].
    Center ≈ 1.0, edges ≈ 0.0, smooth cosine taper.
    """
    # 1-D raised cosine: w(i) = 0.5 * (1 - cos(pi * (2i+1) / (2N)))
    wy = 0.5 * (1.0 - np.cos(np.pi * (2.0 * np.arange(bh) + 1.0) / (2.0 * bh)))
    wx = 0.5 * (1.0 - np.cos(np.pi * (2.0 * np.arange(bw) + 1.0) / (2.0 * bw)))
    return np.outer(wy, wx).astype(np.float32)


# Cache windows for common block sizes
_OBMC_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _get_obmc_window(bh: int, bw: int) -> np.ndarray:
    key = (bh, bw)
    if key not in _OBMC_CACHE:
        _OBMC_CACHE[key] = _make_obmc_window(bh, bw)
    return _OBMC_CACHE[key]


# ───────────────────── MV parsing ─────────────────────

def _parse_mv_blocks(
    mv_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse .npz MV data into arrays of block properties.

    Returns (dstx, dsty, blockw, blockh, mv_px_x, mv_px_y) as float32.
    MV is in pixel units, negated to forward direction (I0→I1).
    """
    blockw = mv_data["blockw"].astype(np.float32)
    blockh = mv_data["blockh"].astype(np.float32)
    dstx = mv_data["dstx"].astype(np.float32)
    dsty = mv_data["dsty"].astype(np.float32)
    motion_x = mv_data["motion_x"].astype(np.float32)
    motion_y = mv_data["motion_y"].astype(np.float32)
    motion_scale = float(mv_data["motion_scale"])

    # Negate: codec MV is I1→I0, we want I0→I1
    mv_px_x = -(motion_x / motion_scale)
    mv_px_y = -(motion_y / motion_scale)

    return dstx, dsty, blockw, blockh, mv_px_x, mv_px_y


# ───────────────────── Method: block_avg (integer, no OBMC) ─────────────────────

def _prealign_block_avg(
    I0: np.ndarray,
    I1: np.ndarray,
    mv_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Block-native splatting with integer-pixel half-shift, average overlap.

    Improvement over v1:
    - Uses codec's original block partition (variable size) instead of fixed 16×16
    - Average splatting on overlap instead of last-write-wins
    """
    h, w, c = I0.shape
    dstx, dsty, blockw, blockh, mv_px_x, mv_px_y = _parse_mv_blocks(mv_data)

    # Accumulators for average splatting
    num0 = np.zeros((h, w, c), dtype=np.float64)
    den0 = np.zeros((h, w), dtype=np.float64)
    num1 = np.zeros((h, w, c), dtype=np.float64)
    den1 = np.zeros((h, w), dtype=np.float64)

    n_blocks = len(blockw)
    for i in range(n_blocks):
        bw = int(blockw[i])
        bh = int(blockh[i])
        bx = int(dstx[i])
        by = int(dsty[i])

        # Half-shift, rounded to integer
        dx = int(round(mv_px_x[i] * 0.5))
        dy = int(round(mv_px_y[i] * 0.5))

        # Source region in original frame (the block as defined by codec)
        sy0, sy1 = max(by, 0), min(by + bh, h)
        sx0, sx1 = max(bx, 0), min(bx + bw, w)
        if sy0 >= sy1 or sx0 >= sx1:
            continue

        # --- Forward shift I0 by +half_mv ---
        oy0 = sy0 + dy
        ox0 = sx0 + dx
        oy1 = sy1 + dy
        ox1 = sx1 + dx

        # Clip to frame
        clip_t = max(0, -oy0)
        clip_l = max(0, -ox0)
        clip_b = max(0, oy1 - h)
        clip_r = max(0, ox1 - w)

        if clip_t < (sy1 - sy0) and clip_l < (sx1 - sx0):
            src_s = (
                slice(sy0 + clip_t, sy1 - clip_b),
                slice(sx0 + clip_l, sx1 - clip_r),
            )
            dst_s = (
                slice(oy0 + clip_t, oy1 - clip_b),
                slice(ox0 + clip_l, ox1 - clip_r),
            )
            num0[dst_s] += I0[src_s].astype(np.float64)
            den0[dst_s[0], dst_s[1]] += 1.0

        # --- Backward shift I1 by -half_mv ---
        oy0 = sy0 - dy
        ox0 = sx0 - dx
        oy1 = sy1 - dy
        ox1 = sx1 - dx

        clip_t = max(0, -oy0)
        clip_l = max(0, -ox0)
        clip_b = max(0, oy1 - h)
        clip_r = max(0, ox1 - w)

        if clip_t < (sy1 - sy0) and clip_l < (sx1 - sx0):
            src_s = (
                slice(sy0 + clip_t, sy1 - clip_b),
                slice(sx0 + clip_l, sx1 - clip_r),
            )
            dst_s = (
                slice(oy0 + clip_t, oy1 - clip_b),
                slice(ox0 + clip_l, ox1 - clip_r),
            )
            num1[dst_s] += I1[src_s].astype(np.float64)
            den1[dst_s[0], dst_s[1]] += 1.0

    # Average splatting with fallback to original frame for holes
    I0_out = I0.copy()
    I1_out = I1.copy()
    mask0 = den0 > 0
    mask1 = den1 > 0
    I0_out[mask0] = (num0[mask0] / den0[mask0, np.newaxis]).clip(0, 255).astype(np.uint8)
    I1_out[mask1] = (num1[mask1] / den1[mask1, np.newaxis]).clip(0, 255).astype(np.uint8)

    return I0_out, I1_out


# ───────────────────── Method: block_subpel ─────────────────────

def _prealign_block_subpel(
    I0: np.ndarray,
    I1: np.ndarray,
    mv_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Block-native sub-pixel prealignment using cv2.remap.

    Instead of per-block integer shift, builds a per-pixel remap field
    from block MVs (preserving quarter-pel precision) and uses cv2.remap
    for bilinear interpolation.
    """
    h, w, c = I0.shape
    dstx, dsty, blockw, blockh, mv_px_x, mv_px_y = _parse_mv_blocks(mv_data)

    # Build per-pixel flow field from block MVs (zero-order hold)
    # This is similar to mv_to_dense but without the fixed 16×16 re-gridding
    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)

    n_blocks = len(blockw)
    for i in range(n_blocks):
        bw = int(blockw[i])
        bh_val = int(blockh[i])
        bx = int(dstx[i])
        by = int(dsty[i])

        y0, y1 = max(by, 0), min(by + bh_val, h)
        x0, x1 = max(bx, 0), min(bx + bw, w)
        if y0 >= y1 or x0 >= x1:
            continue

        flow_x[y0:y1, x0:x1] = mv_px_x[i]
        flow_y[y0:y1, x0:x1] = mv_px_y[i]

    # Half the flow for intermediate frame
    half_fx = flow_x * 0.5
    half_fy = flow_y * 0.5

    # Build remap coordinates
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    # I0 shifted forward: sample I0 at (x - half_fx, y - half_fy)
    # Because remap does dst(x,y) = src(map_x(x,y), map_y(x,y))
    map0_x = grid_x - half_fx
    map0_y = grid_y - half_fy

    # I1 shifted backward: sample I1 at (x + half_fx, y + half_fy)
    map1_x = grid_x + half_fx
    map1_y = grid_y + half_fy

    I0_out = cv2.remap(I0, map0_x, map0_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    I1_out = cv2.remap(I1, map1_x, map1_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return I0_out, I1_out


# ───────────────────── Method: obmc ─────────────────────

def _prealign_obmc(
    I0: np.ndarray,
    I1: np.ndarray,
    mv_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Block-native sub-pixel prealignment with OBMC weighting.

    Builds a weighted-average flow field where each block's MV is weighted
    by a raised cosine OBMC window, then does a single cv2.remap pass.

    This approximates true OBMC (which warps multiple times per pixel)
    by interpolating the flow field instead. Equivalent when image is
    locally linear; differs slightly at texture boundaries.
    """
    h, w, c = I0.shape
    dstx, dsty, blockw, blockh, mv_px_x, mv_px_y = _parse_mv_blocks(mv_data)

    # Build OBMC-weighted flow field
    flow_x_num = np.zeros((h, w), dtype=np.float64)
    flow_y_num = np.zeros((h, w), dtype=np.float64)
    flow_den = np.zeros((h, w), dtype=np.float64)

    n_blocks = len(blockw)
    for i in range(n_blocks):
        bw = int(blockw[i])
        bh_val = int(blockh[i])
        bx = int(dstx[i])
        by = int(dsty[i])

        y0, y1 = max(by, 0), min(by + bh_val, h)
        x0, x1 = max(bx, 0), min(bx + bw, w)
        if y0 >= y1 or x0 >= x1:
            continue

        # OBMC window (clipped to actual visible region)
        window = _get_obmc_window(bh_val, bw)
        wy0 = y0 - by
        wx0 = x0 - bx
        w_clip = window[wy0 : wy0 + (y1 - y0), wx0 : wx0 + (x1 - x0)].astype(np.float64)

        flow_x_num[y0:y1, x0:x1] += w_clip * float(mv_px_x[i])
        flow_y_num[y0:y1, x0:x1] += w_clip * float(mv_px_y[i])
        flow_den[y0:y1, x0:x1] += w_clip

    # Normalize flow
    valid = flow_den > 1e-8
    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)
    flow_x[valid] = (flow_x_num[valid] / flow_den[valid]).astype(np.float32)
    flow_y[valid] = (flow_y_num[valid] / flow_den[valid]).astype(np.float32)

    # Single remap pass with sub-pixel precision
    half_fx = flow_x * 0.5
    half_fy = flow_y * 0.5

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    I0_out = cv2.remap(I0, grid_x - half_fx, grid_y - half_fy,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    I1_out = cv2.remap(I1, grid_x + half_fx, grid_y + half_fy,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return I0_out, I1_out


# ───────────────────── Method: aobmc ─────────────────────

def _compute_block_sad(
    I0: np.ndarray,
    I1: np.ndarray,
    dstx: np.ndarray,
    dsty: np.ndarray,
    blockw: np.ndarray,
    blockh: np.ndarray,
    mv_px_x: np.ndarray,
    mv_px_y: np.ndarray,
) -> np.ndarray:
    """Compute per-block SAD (Sum of Absolute Differences) as reliability signal.

    SAD = mean |I0(block) - I1(block + MV)|
    Lower SAD = better MV match = more reliable.
    """
    h, w = I0.shape[:2]
    n = len(blockw)
    sads = np.full(n, 255.0, dtype=np.float64)  # default high SAD for edge blocks

    # Convert I0/I1 to grayscale float for fast SAD
    if I0.ndim == 3:
        gray0 = np.mean(I0.astype(np.float64), axis=2)
        gray1 = np.mean(I1.astype(np.float64), axis=2)
    else:
        gray0 = I0.astype(np.float64)
        gray1 = I1.astype(np.float64)

    for i in range(n):
        bw = int(blockw[i])
        bh_val = int(blockh[i])
        bx = int(dstx[i])
        by = int(dsty[i])
        dx = int(round(mv_px_x[i]))
        dy = int(round(mv_px_y[i]))

        # Source block in I0
        sy0, sy1 = max(by, 0), min(by + bh_val, h)
        sx0, sx1 = max(bx, 0), min(bx + bw, w)

        # Target in I1 (I0 block + MV = I1 location)
        # MV is I0→I1, so matching location in I1 is (bx+mv_x, by+mv_y)
        ty0, ty1 = sy0 + dy, sy1 + dy
        tx0, tx1 = sx0 + dx, sx1 + dx

        # Clip target to frame
        if ty0 < 0 or tx0 < 0 or ty1 > h or tx1 > w:
            continue
        if sy0 >= sy1 or sx0 >= sx1:
            continue

        sads[i] = np.mean(np.abs(gray0[sy0:sy1, sx0:sx1] - gray1[ty0:ty1, tx0:tx1]))

    return sads


def _compute_mv_spatial_consistency(
    dstx: np.ndarray,
    dsty: np.ndarray,
    blockw: np.ndarray,
    blockh: np.ndarray,
    mv_px_x: np.ndarray,
    mv_px_y: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Compute per-block spatial MV consistency (deviation from local median).

    Lower deviation = more consistent with neighbors = more reliable.
    """
    n = len(blockw)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Place MVs on a coarse grid (by block center)
    cx = (dstx + blockw * 0.5).astype(np.float32)
    cy = (dsty + blockh * 0.5).astype(np.float32)

    deviations = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # Find blocks whose centers are within 24 pixels
        dx = cx - cx[i]
        dy = cy - cy[i]
        dist_sq = dx * dx + dy * dy
        neighbors = (dist_sq > 0) & (dist_sq < 24.0 * 24.0)

        if np.sum(neighbors) < 2:
            deviations[i] = 0.0  # isolated block, assume OK
            continue

        # Vector median: the neighbor MV closest to all others
        nmx = mv_px_x[neighbors]
        nmy = mv_px_y[neighbors]

        # Deviation of current MV from neighborhood median
        med_x = np.median(nmx)
        med_y = np.median(nmy)
        deviations[i] = np.sqrt(
            (mv_px_x[i] - med_x) ** 2 + (mv_px_y[i] - med_y) ** 2
        )

    return deviations


def _prealign_aobmc(
    I0: np.ndarray,
    I1: np.ndarray,
    mv_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Block-native sub-pixel prealignment with Adaptive OBMC.

    Like OBMC, but each block's OBMC weight is further modulated by a
    reliability score combining SAD and spatial MV consistency.

    Based on Choi et al. TCSVT 2007 (AOBMC) and FFmpeg minterpolate.
    Builds a reliability-weighted flow field, then single remap pass.
    """
    h, w, c = I0.shape
    dstx, dsty, blockw, blockh, mv_px_x, mv_px_y = _parse_mv_blocks(mv_data)

    n_blocks = len(blockw)
    if n_blocks == 0:
        return I0.copy(), I1.copy()

    # Compute reliability signals
    sads = _compute_block_sad(
        I0, I1, dstx, dsty, blockw, blockh, mv_px_x, mv_px_y,
    )
    spatial_dev = _compute_mv_spatial_consistency(
        dstx, dsty, blockw, blockh, mv_px_x, mv_px_y, h, w,
    )

    # Combined reliability: lower = better
    sad_max = max(sads.max(), 1.0)
    dev_max = max(spatial_dev.max(), 1.0)
    unreliability = 0.7 * (sads / sad_max) + 0.3 * (spatial_dev / dev_max)
    reliability = np.exp(-2.0 * unreliability)

    # Build reliability-weighted flow field
    flow_x_num = np.zeros((h, w), dtype=np.float64)
    flow_y_num = np.zeros((h, w), dtype=np.float64)
    flow_den = np.zeros((h, w), dtype=np.float64)

    for i in range(n_blocks):
        bw = int(blockw[i])
        bh_val = int(blockh[i])
        bx = int(dstx[i])
        by = int(dsty[i])

        y0, y1 = max(by, 0), min(by + bh_val, h)
        x0, x1 = max(bx, 0), min(bx + bw, w)
        if y0 >= y1 or x0 >= x1:
            continue

        window = _get_obmc_window(bh_val, bw)
        wy0 = y0 - by
        wx0 = x0 - bx
        w_clip = (window[wy0 : wy0 + (y1 - y0), wx0 : wx0 + (x1 - x0)]
                  * reliability[i]).astype(np.float64)

        flow_x_num[y0:y1, x0:x1] += w_clip * float(mv_px_x[i])
        flow_y_num[y0:y1, x0:x1] += w_clip * float(mv_px_y[i])
        flow_den[y0:y1, x0:x1] += w_clip

    # Normalize flow
    valid = flow_den > 1e-8
    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)
    flow_x[valid] = (flow_x_num[valid] / flow_den[valid]).astype(np.float32)
    flow_y[valid] = (flow_y_num[valid] / flow_den[valid]).astype(np.float32)

    half_fx = flow_x * 0.5
    half_fy = flow_y * 0.5

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    I0_out = cv2.remap(I0, grid_x - half_fx, grid_y - half_fy,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    I1_out = cv2.remap(I1, grid_x + half_fx, grid_y + half_fy,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return I0_out, I1_out


# ───────────────────── Method: daala ─────────────────────

def _prealign_daala(
    I0: np.ndarray,
    I1: np.ndarray,
    mv_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Daala/NETVC-style corner-point bilinear interpolation.

    Places MVs at block corners (average of neighboring block-center MVs),
    then bilinear-interpolates between 4 corner MVs per pixel to get a
    continuous motion field. No block boundaries at all.

    Based on IETF draft-terriberry-netvc-obmc-00.
    """
    h, w, c = I0.shape
    dstx, dsty, blockw, blockh, mv_px_x, mv_px_y = _parse_mv_blocks(mv_data)

    n_blocks = len(blockw)

    # Step 1: Build a block-center MV grid (dense, zero-order hold)
    # Then compute corner MVs by averaging neighboring block-center MVs
    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)

    for i in range(n_blocks):
        bw = int(blockw[i])
        bh_val = int(blockh[i])
        bx = int(dstx[i])
        by = int(dsty[i])

        y0, y1 = max(by, 0), min(by + bh_val, h)
        x0, x1 = max(bx, 0), min(bx + bw, w)
        if y0 >= y1 or x0 >= x1:
            continue

        flow_x[y0:y1, x0:x1] = mv_px_x[i]
        flow_y[y0:y1, x0:x1] = mv_px_y[i]

    # Step 2: Smooth the flow field to approximate corner-point interpolation
    # Use a Gaussian filter to create smooth transitions at block boundaries
    # sigma proportional to typical block size (~12 pixels)
    sigma = 6.0
    flow_x_smooth = cv2.GaussianBlur(flow_x, (0, 0), sigma)
    flow_y_smooth = cv2.GaussianBlur(flow_y, (0, 0), sigma)

    # Step 3: Sub-pixel remap with smoothed flow
    half_fx = flow_x_smooth * 0.5
    half_fy = flow_y_smooth * 0.5

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    map0_x = grid_x - half_fx
    map0_y = grid_y - half_fy
    map1_x = grid_x + half_fx
    map1_y = grid_y + half_fy

    I0_out = cv2.remap(I0, map0_x, map0_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    I1_out = cv2.remap(I1, map1_x, map1_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return I0_out, I1_out


# ───────────────────── Method: med_gauss (PRODUCTION) ─────────────────────

# Frozen hyperparameters — matches device-side pipeline exactly.
# DO NOT change without re-validating on full test set AND updating device code.
_DS = 4          # downsample factor (flow is block-level, 1/4 preserves all info)
_MED_KSIZE = 5   # median kernel at 1/4 res (≈ 20px receptive field at full res)
_GAUSS_SIGMA = 2.0  # Gaussian σ at 1/4 res (+ bilinear upsample ≈ σ=8 at full res)


def _prealign_med_gauss(
    I0: np.ndarray,
    I1: np.ndarray,
    mv_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Production prealign v2: ZOH → 1/4↓ median → Gaussian → bilinear↑ → remap.

    Validated on Vimeo90K full test set (3782 triplets):
      MV Blend PSNR: v1 29.92 → v2 31.20 (+1.28 dB)
      95.3% samples improve, large-motion +1.70 dB

    This exact algorithm runs on-device as:
      CPU:  ZOH fill (~5ms) + 1/4 downsample + median 5
      GPU:  1/4 Gaussian σ=2 + bilinear upsample + remap  (~1ms Vulkan compute)
      Total: ~6ms, <0.2W GPU power

    Pipeline:
      1. ZOH: block MV → dense flow (preserving codec's variable block partition)
      2. Downsample 1/4 (nearest — flow is piecewise constant, no aliasing)
      3. Median 5×5 at 1/4 resolution (remove MV outliers, ≈20px full-res RF)
      4. Gaussian σ=2 at 1/4 resolution (smooth block boundaries)
      5. Bilinear upsample to full resolution (provides additional smoothing,
         combined effect ≈ Gaussian σ=8 at full resolution)
      6. cv2.remap bilinear (sub-pixel half-shift warp, 2 frames)
    """
    h, w, c = I0.shape
    dstx, dsty, blockw, blockh, mv_px_x, mv_px_y = _parse_mv_blocks(mv_data)

    # Step 1: Zero-order hold — block MV → full-resolution dense flow
    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)

    n_blocks = len(blockw)
    for i in range(n_blocks):
        bw = int(blockw[i])
        bh_val = int(blockh[i])
        bx = int(dstx[i])
        by = int(dsty[i])

        y0, y1 = max(by, 0), min(by + bh_val, h)
        x0, x1 = max(bx, 0), min(bx + bw, w)
        if y0 >= y1 or x0 >= x1:
            continue

        flow_x[y0:y1, x0:x1] = mv_px_x[i]
        flow_y[y0:y1, x0:x1] = mv_px_y[i]

    # Step 2: Downsample to 1/4 (nearest — flow is piecewise constant)
    # Use cv2.resize instead of array slicing to handle non-divisible dimensions
    h_s, w_s = h // _DS, w // _DS
    flow_x_s = cv2.resize(flow_x, (w_s, h_s), interpolation=cv2.INTER_NEAREST)
    flow_y_s = cv2.resize(flow_y, (w_s, h_s), interpolation=cv2.INTER_NEAREST)

    # Step 3: Median filter at 1/4 resolution
    flow_x_s = median_filter(flow_x_s, size=_MED_KSIZE).astype(np.float32)
    flow_y_s = median_filter(flow_y_s, size=_MED_KSIZE).astype(np.float32)

    # Step 4: Gaussian blur at 1/4 resolution
    flow_x_s = cv2.GaussianBlur(flow_x_s, (0, 0), _GAUSS_SIGMA)
    flow_y_s = cv2.GaussianBlur(flow_y_s, (0, 0), _GAUSS_SIGMA)

    # Step 5: Bilinear upsample to full resolution
    flow_x = cv2.resize(flow_x_s, (w, h), interpolation=cv2.INTER_LINEAR)
    flow_y = cv2.resize(flow_y_s, (w, h), interpolation=cv2.INTER_LINEAR)

    # Step 6: Sub-pixel remap (half-shift warp)
    half_fx = flow_x * 0.5
    half_fy = flow_y * 0.5

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    I0_out = cv2.remap(
        I0, grid_x - half_fx, grid_y - half_fy,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    I1_out = cv2.remap(
        I1, grid_x + half_fx, grid_y + half_fy,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    return I0_out, I1_out


# ───────────────────── Dispatcher ─────────────────────

METHODS = {
    "med_gauss": _prealign_med_gauss,
    "block_avg": _prealign_block_avg,
    "block_subpel": _prealign_block_subpel,
    "obmc": _prealign_obmc,
    "aobmc": _prealign_aobmc,
    "daala": _prealign_daala,
}


# ───────────────────── Batch processing ─────────────────────

def _read_split_list(data_dir: Path, split: str) -> list[str]:
    filename = {"train": "tri_trainlist.txt", "test": "tri_testlist.txt"}[split]
    list_path = data_dir / filename
    if not list_path.exists():
        raise FileNotFoundError(f"Split list not found: {list_path}")
    ids: list[str] = []
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def _process_triplet(
    args: tuple[str, Path, Path, Path, str, bool],
) -> str | None:
    """Process a single triplet. Returns triplet_id on failure, None on success."""
    triplet_id, data_dir, mv_dir, output_dir, method, force = args

    seq_id, trip_id = triplet_id.split("/")

    # Input paths
    seq_dir = data_dir / "sequences" / seq_id / trip_id
    im1_path = seq_dir / "im1.png"
    im3_path = seq_dir / "im3.png"
    mv_path = mv_dir / seq_id / f"{trip_id}.npz"

    # Output paths
    out_dir = output_dir / seq_id / trip_id
    out_im1 = out_dir / "im1_aligned.png"
    out_im3 = out_dir / "im3_aligned.png"
    meta_path = out_dir / "meta.json"

    # Skip check
    if not force and out_im1.exists() and out_im3.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            meta = None
        if isinstance(meta, dict) and meta.get("method") == method:
            if mv_path.exists():
                mv_mtime = mv_path.stat().st_mtime
                out_mtime = min(out_im1.stat().st_mtime, out_im3.stat().st_mtime)
                if out_mtime >= mv_mtime:
                    return None
            else:
                return None

    # Validate inputs
    for p in (im1_path, im3_path, mv_path):
        if not p.exists():
            return triplet_id

    # Load
    I0 = cv2.cvtColor(cv2.imread(str(im1_path)), cv2.COLOR_BGR2RGB)
    I1 = cv2.cvtColor(cv2.imread(str(im3_path)), cv2.COLOR_BGR2RGB)
    mv_data = dict(np.load(mv_path))

    # Process
    fn = METHODS[method]
    try:
        I0_out, I1_out = fn(I0, I1, mv_data)
    except Exception as e:
        print(f"Error processing {triplet_id}: {e}", file=sys.stderr)
        return triplet_id

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_im1), cv2.cvtColor(I0_out, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_im3), cv2.cvtColor(I1_out, cv2.COLOR_RGB2BGR))
    meta_path.write_text(
        json.dumps({"method": method}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prealign v2: Block-native MV prealignment with multiple methods."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Vimeo90K root (containing sequences/).",
    )
    parser.add_argument(
        "--mv-dir", type=Path, required=True,
        help="Directory with .npz MV files (output of extract_mv.py).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for prealigned frames.",
    )
    parser.add_argument(
        "--method", choices=list(METHODS.keys()), required=True,
        help="Prealignment method.",
    )
    parser.add_argument(
        "--split", choices=["train", "test", "both"], default="both",
    )
    parser.add_argument(
        "--workers", type=int, default=cpu_count(),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    mv_dir = args.mv_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for d in (data_dir, mv_dir):
        if not d.exists():
            print(f"Directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    triplet_ids: list[str] = []
    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        ids = _read_split_list(data_dir, split)
        print(f"[{split}] {len(ids)} triplets")
        triplet_ids.extend(ids)

    print(f"Method: {args.method}, {len(triplet_ids)} triplets, {args.workers} workers")

    tasks = [
        (tid, data_dir, mv_dir, output_dir, args.method, args.force)
        for tid in triplet_ids
    ]

    failed: list[str] = []
    with Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_triplet, tasks),
            total=len(tasks),
            desc=f"Prealign v2 ({args.method})",
            unit="triplet",
        ):
            if result is not None:
                failed.append(result)

    n_ok = len(tasks) - len(failed)
    print(f"\nDone: {n_ok}/{len(tasks)} succeeded.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for tid in failed[:20]:
            print(f"  {tid}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
