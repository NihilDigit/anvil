"""Vimeo90K PyTorch Dataset for ANVIL VFI training.

Supports four data routes:
  - Route A: raw frames (im1.png + im3.png → 6ch)
  - Route D: prealigned frames + MV flow (im1_aligned + im3_aligned + flow → 8ch)
  - Route D-nomv: prealigned frames only (im1_aligned + im3_aligned → 6ch)
  - Route FR: raw frames + MV flow (im1.png + im3.png + flow → 8ch, for FlowRefineNet)

Usage:
    from anvil_exp01.data.dataset import Vimeo90KDataset

    ds = Vimeo90KDataset("data/vimeo_triplet", split="train", route="D",
                         mv_flow_dir="data/mv_dense_flow",
                         prealigned_dir="data/prealigned",
                         crop_size=256, augment=True)
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# OpenCV's internal threading interacts poorly with multi-worker DataLoaders.
# Keep decoding single-threaded per worker and let PyTorch handle parallelism.
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Vimeo90KDataset(Dataset):
    """Vimeo90K triplet dataset for video frame interpolation.

    Args:
        data_dir: Root directory of Vimeo90K (containing sequences/, tri_*list.txt).
        split: 'train' or 'test'.
        route: 'A' (raw frames), 'D' (prealigned + MV), or 'D-nomv' (prealigned only).
        mv_flow_dir: Directory with dense MV flow .npy files (required for route D).
        prealigned_dir: Directory with prealigned frames (required for routes D, D-nomv).
        crop_size: Random crop size (0 = no crop). Default 256.
        augment: Apply random augmentation (hflip, vflip, temporal swap). Default False.

    Returns per sample:
        dict with keys:
            input:      (C, H, W) float32 [0,1] for images, raw pixels for flow.
                        C=6 for routes A/D-nomv, C=8 for route D.
            gt:         (3, H, W) float32 [0,1]
            blend:      (3, H, W) float32 [0,1]
            triplet_id: str like '00001/0001'
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        route: str,
        mv_flow_dir: str | Path | None = None,
        prealigned_dir: str | Path | None = None,
        crop_size: int = 256,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.seq_dir = self.data_dir / "sequences"
        self.route = route
        self.mv_flow_dir = Path(mv_flow_dir) if mv_flow_dir else None
        self.prealigned_dir = Path(prealigned_dir) if prealigned_dir else None
        self.crop_size = crop_size
        self.augment = augment

        if route in ("D", "FR") and mv_flow_dir is None:
            raise ValueError("mv_flow_dir is required for route D / FR")
        if route in ("D", "D-nomv") and prealigned_dir is None:
            raise ValueError("prealigned_dir is required for route D / D-nomv")

        if split in ("train", "val"):
            list_path = self.data_dir / "tri_trainlist.txt"
        elif split == "test":
            list_path = self.data_dir / "tri_testlist.txt"
        else:
            raise ValueError(f"Unknown split: {split!r}. Use 'train', 'val', or 'test'.")

        all_ids: list[str] = []
        with open(list_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_ids.append(line)

        if split in ("train", "val"):
            # Sequence-level 90/10 split: group triplets by sequence
            # (first 5 chars of "SSSSS/NNNN"), shuffle sequences with a
            # fixed seed, then assign the last 10% of sequences to val.
            # This guarantees no sequence appears in both splits.
            seq_to_ids: dict[str, list[str]] = {}
            for tid in all_ids:
                seq = tid[:5]
                seq_to_ids.setdefault(seq, []).append(tid)
            sequences = sorted(seq_to_ids.keys())
            random.Random(42).shuffle(sequences)
            n_val_seq = max(1, len(sequences) // 10)
            val_sequences = set(sequences[-n_val_seq:])
            if split == "val":
                self.triplet_ids = [
                    tid for seq in sequences if seq in val_sequences
                    for tid in sorted(seq_to_ids[seq])
                ]
            else:
                self.triplet_ids = [
                    tid for seq in sequences if seq not in val_sequences
                    for tid in sorted(seq_to_ids[seq])
                ]
        else:
            self.triplet_ids = all_ids

    def __len__(self) -> int:
        return len(self.triplet_ids)

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        """Load image as float32 HWC in [0, 1]."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> dict:
        tid = self.triplet_ids[idx]
        seq_id, trip_id = tid.split("/")
        triplet_dir = self.seq_dir / seq_id / trip_id

        gt = self._load_image(triplet_dir / "im2.png")

        if self.route in ("A", "FR"):
            i0 = self._load_image(triplet_dir / "im1.png")
            i1 = self._load_image(triplet_dir / "im3.png")
            if self.route == "FR":
                flow = np.load(
                    str(self.mv_flow_dir / seq_id / f"{trip_id}.npy")
                )  # (H, W, 2) float32
                h, w = flow.shape[:2]
                flow = flow / np.array([w, h], dtype=np.float32)
            else:
                flow = None
        else:
            pre_dir = self.prealigned_dir / seq_id / trip_id
            i0 = self._load_image(pre_dir / "im1_aligned.png")
            i1 = self._load_image(pre_dir / "im3_aligned.png")
            if self.route == "D":
                flow = np.load(
                    str(self.mv_flow_dir / seq_id / f"{trip_id}.npy")
                )  # (H, W, 2) float32
                h, w = flow.shape[:2]
                flow = flow / np.array([w, h], dtype=np.float32)
            else:
                flow = None

        blend = (i0 + i1) * 0.5

        # ----- Augmentation (defer contiguous copies to the end) -----
        if self.augment:
            do_hflip = random.random() < 0.5
            do_vflip = random.random() < 0.5
            do_tswap = random.random() < 0.5

            if do_hflip:
                i0, i1, gt, blend = i0[:, ::-1], i1[:, ::-1], gt[:, ::-1], blend[:, ::-1]
                if flow is not None:
                    flow = flow[:, ::-1]

            if do_vflip:
                i0, i1, gt, blend = i0[::-1], i1[::-1], gt[::-1], blend[::-1]
                if flow is not None:
                    flow = flow[::-1]

            if do_tswap:
                i0, i1 = i1, i0

            if flow is not None and (do_hflip or do_vflip or do_tswap):
                flow = flow.copy()
                if do_hflip:
                    flow[:, :, 0] *= -1
                if do_vflip:
                    flow[:, :, 1] *= -1
                if do_tswap:
                    flow *= -1

        # ----- Random crop -----
        if self.crop_size > 0:
            h, w = i0.shape[:2]
            if h > self.crop_size or w > self.crop_size:
                max_y = max(0, h - self.crop_size)
                max_x = max(0, w - self.crop_size)
                y = random.randint(0, max_y) if max_y > 0 else 0
                x = random.randint(0, max_x) if max_x > 0 else 0
                s = (slice(y, y + self.crop_size), slice(x, x + self.crop_size))
                i0, i1, gt, blend = i0[s], i1[s], gt[s], blend[s]
                if flow is not None:
                    flow = flow[s]

        # ----- Assemble input tensor -----
        if flow is not None:
            inp = np.concatenate([i0, i1, flow], axis=2)  # (H, W, 8)
        else:
            inp = np.concatenate([i0, i1], axis=2)  # (H, W, 6)

        return {
            "input": torch.from_numpy(np.ascontiguousarray(inp.transpose(2, 0, 1))),
            "gt": torch.from_numpy(np.ascontiguousarray(gt.transpose(2, 0, 1))),
            "blend": torch.from_numpy(np.ascontiguousarray(blend.transpose(2, 0, 1))),
            "triplet_id": tid,
        }
