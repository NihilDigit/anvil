#!/usr/bin/env python3
"""Export IFRNet to ONNX (frame and flow variants at 256x448).

This script exports the IFRNet baseline used in the paper's INT8 comparison.
It produces:

  - ifrnet_frame.onnx : input (img0, img1) -> frame (1,3,256,448)
  - ifrnet_flow.onnx  : input (img0, img1) -> flow  (1,4,256,448)

The release bundle does not vendor IFRNet itself. Clone the AMT repository,
which includes `networks/IFRNet.py`, and download the official
`IFRNet_Vimeo90K.pth` checkpoint.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

H = 256
W = 448


def load_ifrnet_model(amt_repo: Path, weights: Path) -> nn.Module:
    """Load IFRNet from the AMT repository with pretrained weights."""
    sys.path.insert(0, str(amt_repo.resolve()))
    try:
        from networks.IFRNet import Model
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import IFRNet from {amt_repo}. "
            "Clone https://github.com/MCG-NKU/AMT into vendor/AMT."
        ) from exc

    model = Model().cpu().eval()
    checkpoint = torch.load(str(weights), map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: missing keys while loading IFRNet: {len(missing)}")
    if unexpected:
        print(f"WARNING: unexpected keys while loading IFRNet: {len(unexpected)}")
    return model


class IFRNetFrameWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        embt = img0.new_full((img0.shape[0], 1, 1, 1), 0.5)
        out = self.model(img0, img1, embt, scale_factor=1.0, eval=True)
        return out["imgt_pred"] if isinstance(out, dict) else out


class IFRNetFlowWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        embt = img0.new_full((img0.shape[0], 1, 1, 1), 0.5)
        out = self.model(img0, img1, embt, scale_factor=1.0, eval=False)
        flow0 = out["flow0_pred"][0]
        flow1 = out["flow1_pred"][0]
        return torch.cat([flow0, flow1], dim=1)


def export_variant(model: nn.Module, mode: str, output_dir: Path) -> Path:
    """Export one IFRNet ONNX variant."""
    if mode == "frame":
        wrapper = IFRNetFrameWrapper(model).cpu().eval()
        filename = "ifrnet_frame.onnx"
        output_names = ["output"]
    elif mode == "flow":
        wrapper = IFRNetFlowWrapper(model).cpu().eval()
        filename = "ifrnet_flow.onnx"
        output_names = ["flow"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    onnx_path = output_dir / filename
    dummy0 = torch.randn(1, 3, H, W)
    dummy1 = torch.randn(1, 3, H, W)

    print(f"  Exporting {filename} input=(1,3,{H},{W})")
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (dummy0, dummy1),
        str(onnx_path),
        opset_version=16,
        input_names=["img0", "img1"],
        output_names=output_names,
        do_constant_folding=True,
    )
    elapsed = time.time() - t0

    import onnx as _onnx

    ext_data = Path(str(onnx_path) + ".data")
    if ext_data.exists():
        model_proto = _onnx.load(str(onnx_path), load_external_data=True)
        _onnx.save(model_proto, str(onnx_path), save_as_external_data=False)
        ext_data.unlink()
        print("    inlined external weights")

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"    -> {onnx_path.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return onnx_path


def verify_onnx(onnx_path: Path, mode: str) -> None:
    """Quick sanity check with onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("    [skip verification: onnxruntime not installed]")
        return

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    img0 = np.random.randn(1, 3, H, W).astype(np.float32)
    img1 = np.random.randn(1, 3, H, W).astype(np.float32)
    outputs = sess.run(None, {"img0": img0, "img1": img1})

    if mode == "frame":
        assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
        assert outputs[0].shape == (1, 3, H, W), f"Frame shape mismatch: {outputs[0].shape}"
    else:
        assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
        assert outputs[0].shape == (1, 4, H, W), f"Flow shape mismatch: {outputs[0].shape}"

    print("    verification OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export IFRNet to ONNX (frame / flow variants).")
    parser.add_argument(
        "--amt-repo",
        type=Path,
        default=Path("vendor/AMT"),
        help="Path to cloned AMT repository containing networks/IFRNet.py.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to IFRNet_Vimeo90K.pth. Default: <amt-repo>/../IFRNet_Vimeo90K.pth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/onnx/ifrnet"),
        help="Output directory for ONNX files.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["flow", "frame"],
        default=["flow", "frame"],
        help="Which variants to export (default: both).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX verification with onnxruntime.",
    )
    args = parser.parse_args()

    weights = args.weights or (args.amt_repo.parent / "IFRNet_Vimeo90K.pth")

    if not args.amt_repo.is_dir():
        print(f"ERROR: AMT repo not found at {args.amt_repo}")
        print("Clone it: git clone https://github.com/MCG-NKU/AMT vendor/AMT")
        sys.exit(1)
    if not weights.is_file():
        print(f"ERROR: IFRNet weights not found at {weights}")
        print("Download IFRNet_Vimeo90K.pth and place it at vendor/IFRNet_Vimeo90K.pth")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading IFRNet from {args.amt_repo}")
    print(f"Weights: {weights}")
    model = load_ifrnet_model(args.amt_repo, weights)

    for mode in args.modes:
        onnx_path = export_variant(model, mode, args.output_dir)
        if not args.no_verify:
            verify_onnx(onnx_path, mode)

    print("")
    print("Done.")
    for mode in args.modes:
        print(f"  {args.output_dir / f'ifrnet_{mode}.onnx'}")


if __name__ == "__main__":
    main()
