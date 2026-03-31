"""Codec parameter robustness sweep for ANVIL prealignment.

Tests how encoding parameters affect MV-based prealignment quality:
  - preset: ultrafast / medium / veryslow (MV estimation thoroughness)
  - CRF: 18 / 28 (quality vs compression)
  - bframes: 0 / 3 (GOP structure)

Evaluates MV Blend PSNR on Xiph 1080p representative sequences.
Baseline: bframes=0, crf=18, preset=medium (default encoding parameters).

Usage:
    pixi run python scripts/eval_codec_robustness.py --xiph-dir data/xiph_1080p
    pixi run python scripts/eval_codec_robustness.py --xiph-dir data/xiph_1080p --sequences 4 --limit 30
    pixi run python scripts/eval_codec_robustness.py --xiph-dir data/xiph_1080p --full
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

# ---------------------------------------------------------------------------
# Paths (overridden by CLI args)
# ---------------------------------------------------------------------------
XIPH_DIR = Path("data/xiph_1080p")
RESULTS_DIR = Path("artifacts/eval/codec_robustness")
TMP_DIR = Path("artifacts/cache/codec_robustness_tmp")

SEQUENCES = [
    "station2", "sunflower", "rush_hour", "blue_sky",
    "old_town_cross", "in_to_tree", "tractor", "pedestrian_area",
    "crowd_run", "ducks_take_off", "park_joy", "riverbed",
]

# Default sweep grid
DEFAULT_PRESETS = ["ultrafast", "medium", "veryslow"]
DEFAULT_CRFS = [18, 28]
DEFAULT_BFRAMES = [0, 3]

# ---------------------------------------------------------------------------
# Frame I/O (from eval_bframe_fallback.py)
# ---------------------------------------------------------------------------
def get_triplet_dirs(seq: str) -> list[Path]:
    seq_dir = XIPH_DIR / "sequences" / seq
    return sorted(d for d in seq_dir.iterdir() if d.is_dir())


def reconstruct_source_frames(triplets: list[Path]) -> tuple[list[Path], list[Path]]:
    source = [t / "im1.png" for t in triplets]
    source.append(triplets[-1] / "im3.png")
    gt = [t / "im2.png" for t in triplets]
    return source, gt


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode_frames_as_h264(
    frame_paths: list[Path],
    output_mp4: Path,
    bframes: int = 0,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    concat_path = TMP_DIR / "concat_list.txt"
    with open(concat_path, "w") as f:
        for p in frame_paths:
            f.write(f"file '{p.resolve()}'\n")
            f.write("duration 0.04\n")  # 25fps

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-x264-params",
        f"keyint=250:min-keyint=250:bframes={bframes}:b-adapt=0:b-pyramid=none:ref=1",
        "-an",
        str(output_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {r.stderr.decode(errors='replace')[-500:]}")


# ---------------------------------------------------------------------------
# MV extraction
# ---------------------------------------------------------------------------
def extract_mvs_display_order(video_path: Path) -> list[dict]:
    from mvextractor.videocap import VideoCap

    cap = VideoCap()
    if not cap.open(str(video_path)):
        raise RuntimeError(f"Cannot open {video_path}")

    results = []
    while True:
        ret, frame, mv, frame_type = cap.read()
        if not ret:
            break
        if mv is not None and len(mv) > 0:
            mv_arr = np.array(mv, dtype=np.float64)
        else:
            mv_arr = np.empty((0, 10), dtype=np.float64)
        results.append({"type": frame_type, "mvs": mv_arr})

    cap.release()
    return results


def compute_ref_distances(frame_types: list[str]) -> list[tuple[int, int]]:
    n = len(frame_types)
    l0_dists = []
    last_ip_pos = 0
    for i, ft in enumerate(frame_types):
        if ft == "I":
            l0_dists.append(0)
            last_ip_pos = i
        elif ft == "P":
            l0_dists.append(i - last_ip_pos)
            last_ip_pos = i
        else:
            l0_dists.append(i - last_ip_pos)

    l1_dists = [0] * n
    next_ip_pos = n
    for i in range(n - 1, -1, -1):
        ft = frame_types[i]
        if ft in ("I", "P"):
            next_ip_pos = i
        else:
            if next_ip_pos < n:
                l1_dists[i] = next_ip_pos - i
    return list(zip(l0_dists, l1_dists))


# ---------------------------------------------------------------------------
# MV → Dense Flow → Prealign v2
# ---------------------------------------------------------------------------
def zoh_fill(mv_data, H, W, source_filter=-1, negate=True):
    fx = np.zeros((H, W), dtype=np.float32)
    fy = np.zeros((H, W), dtype=np.float32)
    if len(mv_data) == 0:
        return fx, fy

    n_cols = mv_data.shape[1]
    if n_cols >= 11:
        src_col, bw_col, bh_col = 0, 1, 2
        dx_col, dy_col = 5, 6
        mx_col, my_col, ms_col = 8, 9, 10
    else:
        src_col, bw_col, bh_col = 0, 1, 2
        dx_col, dy_col = 5, 6
        mx_col, my_col, ms_col = 7, 8, 9

    sign = -1.0 if negate else 1.0
    for row in mv_data:
        if int(row[src_col]) != source_filter:
            continue
        scale = float(row[ms_col]) if row[ms_col] > 0 else 1.0
        mx = sign * float(row[mx_col]) / scale
        my = sign * float(row[my_col]) / scale
        bw, bh = int(row[bw_col]), int(row[bh_col])
        cx, cy = int(row[dx_col]), int(row[dy_col])
        x0, y0 = max(0, cx - bw // 2), max(0, cy - bh // 2)
        x1, y1 = min(W, x0 + bw), min(H, y0 + bh)
        fx[y0:y1, x0:x1] = mx
        fy[y0:y1, x0:x1] = my
    return fx, fy


def prealign_v2_smooth(fx, fy):
    H, W = fx.shape
    ds = 4
    h_ds, w_ds = H // ds, W // ds
    fx_ds = cv2.resize(fx, (w_ds, h_ds), interpolation=cv2.INTER_AREA)
    fy_ds = cv2.resize(fy, (w_ds, h_ds), interpolation=cv2.INTER_AREA)
    fx_ds = median_filter(fx_ds, size=5).astype(np.float32)
    fy_ds = median_filter(fy_ds, size=5).astype(np.float32)
    fx_ds = gaussian_filter(fx_ds, sigma=2.0).astype(np.float32)
    fy_ds = gaussian_filter(fy_ds, sigma=2.0).astype(np.float32)
    fx_up = cv2.resize(fx_ds, (W, H), interpolation=cv2.INTER_LINEAR)
    fy_up = cv2.resize(fy_ds, (W, H), interpolation=cv2.INTER_LINEAR)
    return fx_up, fy_up


def mv_blend(I0, I1, fx, fy):
    H, W = I0.shape[:2]
    hfx, hfy = fx * 0.5, fy * 0.5
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                          np.arange(H, dtype=np.float32))
    I0_w = cv2.remap(I0, gx - hfx, gy - hfy,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    I1_w = cv2.remap(I1, gx + hfx, gy + hfy,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    blend = (I0_w.astype(np.float32) + I1_w.astype(np.float32)) * 0.5
    return blend.clip(0, 255).astype(np.uint8)


def naive_blend(I0, I1):
    return ((I0.astype(np.float32) + I1.astype(np.float32)) * 0.5).clip(0, 255).astype(np.uint8)


def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(255.0**2 / mse))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_config(
    source_frames: list[Path],
    gt_frames: list[Path],
    n_triplets: int,
    bframes: int,
    crf: int,
    preset: str,
    seq: str,
) -> list[dict]:
    """Encode, extract MVs, evaluate MV Blend for one config."""
    mp4_path = TMP_DIR / f"{seq}_bf{bframes}_crf{crf}_{preset}.mp4"
    src_subset = source_frames[:n_triplets + 1]

    encode_frames_as_h264(src_subset, mp4_path, bframes=bframes,
                          crf=crf, preset=preset)

    frame_data = extract_mvs_display_order(mp4_path)
    frame_types = [f["type"] for f in frame_data]
    ref_dists = compute_ref_distances(frame_types)

    results = []
    n_extracted = len(frame_data)

    for t in range(n_triplets):
        enc_idx = t + 1
        if enc_idx >= n_extracted:
            break

        I0 = cv2.imread(str(source_frames[t]))
        I1 = cv2.imread(str(source_frames[t + 1]))
        GT = cv2.imread(str(gt_frames[t]))
        H, W = I0.shape[:2]

        ft = frame_data[enc_idx]["type"]
        mv_data = frame_data[enc_idx]["mvs"]
        l0_dist, l1_dist = ref_dists[enc_idx]

        if ft == "I" or (l0_dist == 0 and l1_dist == 0):
            blend = naive_blend(I0, I1)
            psnr = compute_psnr(blend, GT)
            results.append({"psnr": psnr, "frame_type": "I", "ref_dist": 0})
            continue

        # Use L0 for P-frames, best_ref for B-frames
        use_l1 = False
        dist = l0_dist
        if ft == "B" and l1_dist > 0 and l1_dist < l0_dist:
            use_l1 = True
            dist = l1_dist

        if use_l1:
            fx, fy = zoh_fill(mv_data, H, W, source_filter=1, negate=False)
        else:
            fx, fy = zoh_fill(mv_data, H, W, source_filter=-1, negate=True)

        if np.abs(fx).max() < 1e-6 and np.abs(fy).max() < 1e-6:
            blend = naive_blend(I0, I1)
            psnr = compute_psnr(blend, GT)
            results.append({"psnr": psnr, "frame_type": "I", "ref_dist": 0})
            continue

        fx, fy = prealign_v2_smooth(fx, fy)
        if dist > 1:
            fx, fy = fx / dist, fy / dist

        blend = mv_blend(I0, I1, fx, fy)
        psnr = compute_psnr(blend, GT)
        results.append({"psnr": psnr, "frame_type": ft, "ref_dist": dist})

    mp4_path.unlink(missing_ok=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Codec parameter robustness sweep (preset × CRF × bframes)")
    parser.add_argument("--sequences", type=int, default=4,
                        help="Number of sequences (default: 4 representative)")
    parser.add_argument("--limit", type=int, default=30,
                        help="Max triplets per sequence (default: 30, 0=all)")
    parser.add_argument("--full", action="store_true",
                        help="Full sweep: all 12 sequences, all triplets")
    parser.add_argument("--xiph-dir", type=Path, default=XIPH_DIR,
                        help="Xiph 1080p dataset directory")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR,
                        help="Output directory for results")
    parser.add_argument("--tmp-dir", type=Path, default=TMP_DIR,
                        help="Temporary directory for encoded videos")
    parser.add_argument("--presets", type=str,
                        default=",".join(DEFAULT_PRESETS))
    parser.add_argument("--crfs", type=str,
                        default=",".join(str(c) for c in DEFAULT_CRFS))
    parser.add_argument("--bframes", type=str,
                        default=",".join(str(b) for b in DEFAULT_BFRAMES))
    args = parser.parse_args()

    # Override globals with CLI args
    global XIPH_DIR, RESULTS_DIR, TMP_DIR
    XIPH_DIR = args.xiph_dir
    RESULTS_DIR = args.output_dir
    TMP_DIR = args.tmp_dir

    if args.full:
        args.sequences = 12
        args.limit = 0

    presets = args.presets.split(",")
    crfs = [int(c) for c in args.crfs.split(",")]
    bframe_list = [int(b) for b in args.bframes.split(",")]

    # Representative sequences: static, medium, large motion, texture
    repr_seqs = ["station2", "old_town_cross", "tractor", "crowd_run"]
    seqs = repr_seqs[:args.sequences] if args.sequences <= 4 else SEQUENCES[:args.sequences]

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_configs = len(presets) * len(crfs) * len(bframe_list)
    print(f"Codec robustness sweep: {len(seqs)} sequences × {n_configs} configs")
    print(f"  presets: {presets}")
    print(f"  CRFs: {crfs}")
    print(f"  bframes: {bframe_list}")
    print(f"  limit: {args.limit or 'all'} triplets/seq")
    print()

    all_rows = []

    for seq in seqs:
        print(f"{'='*60}")
        print(f"Sequence: {seq}")
        triplet_dirs = get_triplet_dirs(seq)
        source_frames, gt_frames = reconstruct_source_frames(triplet_dirs)
        n_triplets = len(gt_frames)
        if args.limit > 0:
            n_triplets = min(n_triplets, args.limit)
        print(f"  Triplets: {n_triplets}")

        for preset in presets:
            for crf in crfs:
                for bf in bframe_list:
                    label = f"preset={preset:10s} crf={crf:2d} bf={bf}"
                    sys.stdout.write(f"  {label} ... ")
                    sys.stdout.flush()

                    try:
                        results = evaluate_config(
                            source_frames, gt_frames, n_triplets,
                            bframes=bf, crf=crf, preset=preset, seq=seq,
                        )
                        psnrs = [r["psnr"] for r in results]
                        mean_psnr = float(np.mean(psnrs))
                        n_eval = len(psnrs)

                        # Count frame types
                        n_P = sum(1 for r in results if r["frame_type"] == "P")
                        n_B = sum(1 for r in results if r["frame_type"] == "B")
                        n_I = sum(1 for r in results if r["frame_type"] == "I")

                        print(f"PSNR={mean_psnr:.2f} dB  "
                              f"(P={n_P} B={n_B} I={n_I}, n={n_eval})")

                        all_rows.append({
                            "sequence": seq,
                            "preset": preset,
                            "crf": crf,
                            "bframes": bf,
                            "psnr": round(mean_psnr, 4),
                            "n_triplets": n_eval,
                            "n_P": n_P, "n_B": n_B, "n_I": n_I,
                        })
                    except Exception as e:
                        print(f"FAILED: {e}")
                        all_rows.append({
                            "sequence": seq,
                            "preset": preset,
                            "crf": crf,
                            "bframes": bf,
                            "psnr": float("nan"),
                            "n_triplets": 0,
                            "n_P": 0, "n_B": 0, "n_I": 0,
                        })

    # Save per-config results
    csv_path = RESULTS_DIR / "per_config.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sequence", "preset", "crf", "bframes",
            "psnr", "n_triplets", "n_P", "n_B", "n_I",
        ])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-config results: {csv_path}")

    # ---------------------------------------------------------------------------
    # Summary: average across sequences, grouped by (preset, crf, bframes)
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY: Codec Robustness (MV Blend PSNR, averaged across sequences)")
    print(f"{'='*70}")

    # Baseline: medium, crf=18, bf=0
    baseline_vals = [r["psnr"] for r in all_rows
                     if r["preset"] == "medium" and r["crf"] == 18
                     and r["bframes"] == 0 and not np.isnan(r["psnr"])]
    baseline = float(np.mean(baseline_vals)) if baseline_vals else 0.0

    summary = []
    configs = sorted(set((r["preset"], r["crf"], r["bframes"]) for r in all_rows))
    print(f"\n{'Preset':<12} {'CRF':>4} {'BF':>3} {'PSNR':>7} {'Δ':>7}  Note")
    print("-" * 55)

    for preset, crf, bf in configs:
        vals = [r["psnr"] for r in all_rows
                if r["preset"] == preset and r["crf"] == crf
                and r["bframes"] == bf and not np.isnan(r["psnr"])]
        if not vals:
            continue
        mean = float(np.mean(vals))
        delta = mean - baseline
        is_baseline = (preset == "medium" and crf == 18 and bf == 0)
        note = "← baseline" if is_baseline else ""
        print(f"{preset:<12} {crf:>4} {bf:>3} {mean:>7.2f} {delta:>+7.2f}  {note}")
        summary.append({
            "preset": preset, "crf": crf, "bframes": bf,
            "psnr": round(mean, 4), "delta": round(delta, 4),
        })

    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print key findings
    print(f"\n--- Key findings ---")
    if len(summary) > 1:
        psnrs = [s["psnr"] for s in summary if not np.isnan(s["psnr"])]
        print(f"Range: {min(psnrs):.2f} — {max(psnrs):.2f} dB "
              f"(spread: {max(psnrs)-min(psnrs):.2f} dB)")

        # Preset effect (bf=0, crf=18)
        preset_vals = {s["preset"]: s["psnr"] for s in summary
                       if s["crf"] == 18 and s["bframes"] == 0}
        if len(preset_vals) > 1:
            print(f"Preset effect (bf=0, crf=18): "
                  + ", ".join(f"{p}={v:.2f}" for p, v in sorted(preset_vals.items())))

        # CRF effect (medium, bf=0)
        crf_vals = {s["crf"]: s["psnr"] for s in summary
                    if s["preset"] == "medium" and s["bframes"] == 0}
        if len(crf_vals) > 1:
            print(f"CRF effect (medium, bf=0): "
                  + ", ".join(f"crf{c}={v:.2f}" for c, v in sorted(crf_vals.items())))

        # B-frame effect (medium, crf=18)
        bf_vals = {s["bframes"]: s["psnr"] for s in summary
                   if s["preset"] == "medium" and s["crf"] == 18}
        if len(bf_vals) > 1:
            print(f"B-frame effect (medium, crf=18): "
                  + ", ".join(f"bf{b}={v:.2f}" for b, v in sorted(bf_vals.items())))

    print(f"\nResults: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
