"""Export RIFE HDv3 to ONNX in 4 variants (360p/480p x flow/frame).

This script exports the RIFE HDv3 model into ONNX format at two resolutions
and two output modes, producing the files needed by eval_rife_reduced_res.py
and the QNN/HTP deployment pipeline.

Output files:
  - rife_flow_360p.onnx  : input (1,6,384,640)  -> flow (1,4,384,640) + mask (1,1,384,640)
  - rife_frame_360p.onnx : input (1,6,384,640)  -> frame (1,3,384,640)
  - rife_flow_480p.onnx  : input (1,6,512,864)  -> flow (1,4,512,864) + mask (1,1,512,864)
  - rife_frame_480p.onnx : input (1,6,512,864)  -> frame (1,3,512,864)

Flow mode returns the intermediate 4-channel bidirectional flow and 1-channel
blending mask before final warping.  This allows the caller to upsample the
flow field to full resolution and perform CPU-side warping at 1080p, which is
the preferred deployment strategy (flow-upsample).

Frame mode returns the final blended 3-channel RGB output at the model's
native resolution.  The caller bilinearly upsamples this to full resolution
(frame-upsample), which is simpler but lower quality.

Prerequisites:
  1. Clone the official RIFE repository:
       git clone https://github.com/megvii-research/ECCV2022-RIFE /path/to/ECCV2022-RIFE

  2. Download the HDv3 pretrained weights.  The weights file (flownet.pkl)
     should be placed at: /path/to/ECCV2022-RIFE/train_log/flownet.pkl
     Download link: https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view

  3. The RIFE repository must contain train_log/RIFE_HDv3.py and
     train_log/IFNet_HDv3.py (shipped with the HDv3 weights download).

Usage:
    python scripts/export_rife_onnx.py \\
        --rife-repo /path/to/ECCV2022-RIFE \\
        --output-dir artifacts/onnx/rife

    # Export only 360p variants:
    python scripts/export_rife_onnx.py \\
        --rife-repo /path/to/ECCV2022-RIFE \\
        --output-dir artifacts/onnx/rife \\
        --resolutions 360p

Architecture notes:
    RIFE HDv3 uses IFNet with 3-stage iterative flow refinement.  The network
    takes a 6-channel input (two RGB frames concatenated along the channel
    dimension, values in [0, 1]) and produces:
      - flow: 4 channels (flow_0->t_x, flow_0->t_y, flow_1->t_x, flow_1->t_y)
      - mask: 1 channel (soft blending weight for warped frame 0, sigmoid output)

    The final frame is computed by warping both input frames with the flow
    fields and blending with the mask.  In flow mode we export before the
    warping step; in frame mode we export the complete pipeline.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Resolution definitions
# ---------------------------------------------------------------------------

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "360p": (384, 640),
    "480p": (512, 864),
}


# ---------------------------------------------------------------------------
# RIFE model loading
# ---------------------------------------------------------------------------

def load_rife_flownet(rife_repo: Path, weights: Path) -> nn.Module:
    """Import and load the RIFE HDv3 flownet from the cloned repository.

    Returns the raw flownet (IFNet) module on CPU with loaded weights.

    Important: this function must be called with ``CUDA_VISIBLE_DEVICES=""``
    set in the environment.  RIFE's warplayer caches grid tensors on a
    module-level ``device`` variable; hiding CUDA ensures this resolves to
    CPU, producing a clean ONNX graph without spurious Cast nodes.
    """
    if torch.cuda.is_available():
        print(
            "WARNING: CUDA is visible. Export may produce Cast/ConstantOfShape "
            "nodes that differ from the reference ONNX. Set CUDA_VISIBLE_DEVICES='' "
            "to match the reference export."
        )

    repo_str = str(rife_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    hdv3_ifnet = rife_repo / "train_log" / "IFNet_HDv3.py"
    if not hdv3_ifnet.exists():
        raise FileNotFoundError(
            f"IFNet_HDv3.py not found at {hdv3_ifnet}.\n"
            "Make sure the HDv3 model files (RIFE_HDv3.py, IFNet_HDv3.py) "
            "are in the train_log/ directory of the RIFE repo."
        )

    # Clear any cached warp grids from prior runs so the tracer
    # sees fresh grid creation on the correct device.
    import model.warplayer as wp
    wp.backwarp_tenGrid = {}

    from train_log.IFNet_HDv3 import IFNet

    flownet = IFNet()
    state_dict = torch.load(str(weights), map_location="cpu", weights_only=True)

    cleaned = {}
    for k, v in state_dict.items():
        key = k
        for prefix in ("module.", "flownet."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = v

    flownet.load_state_dict(cleaned)
    flownet.eval()
    return flownet


# ---------------------------------------------------------------------------
# Wrapper modules for ONNX export
# ---------------------------------------------------------------------------

class RIFEFlowExporter(nn.Module):
    """Wraps IFNet to output (flow, mask) without final warping.

    Input:  (B, 6, H, W) float32 in [0, 1]  (two RGB frames concatenated)
    Output: flow (B, 4, H, W), mask (B, 1, H, W)

    The 4-channel flow contains [f01_x, f01_y, f10_x, f10_y] in pixel units
    at the model's native resolution.  The mask is the sigmoid blending
    weight returned by IFNet (already sigmoid-applied inside the network).
    """

    def __init__(self, flownet: nn.Module) -> None:
        super().__init__()
        self.flownet = flownet

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # IFNet.forward returns (flow_list, mask_single_tensor, merged_list).
        # mask is already sigmoid-applied (IFNet line 112).
        flow_list, mask, _ = self.flownet(x, scale_list=[4, 2, 1])
        return flow_list[2], mask  # (B,4,H,W), (B,1,H,W)


class RIFEFrameExporter(nn.Module):
    """Wraps IFNet to output the final blended frame.

    Input:  (B, 6, H, W) float32 in [0, 1]
    Output: frame (B, 3, H, W) float32 in [0, 1]

    This is the complete RIFE inference pipeline including warping and
    blending at the model's native resolution.
    """

    def __init__(self, flownet: nn.Module) -> None:
        super().__init__()
        self.flownet = flownet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, merged = self.flownet(x, scale_list=[4, 2, 1])
        return merged[2]  # (B, 3, H, W)


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------

def export_one(
    wrapper: nn.Module,
    mode: str,
    res_name: str,
    hw: tuple[int, int],
    output_dir: Path,
) -> Path:
    """Export a single ONNX variant.

    Args:
        wrapper: RIFEFlowExporter or RIFEFrameExporter
        mode: "flow" or "frame"
        res_name: "360p" or "480p"
        hw: (height, width)
        output_dir: directory to write the ONNX file

    Returns:
        Path to the exported ONNX file.
    """
    h, w = hw
    filename = f"rife_{mode}_{res_name}.onnx"
    onnx_path = output_dir / filename

    if mode == "flow":
        output_names = ["flow", "mask"]
    else:
        output_names = ["output"]

    # Clear warp grid cache before each export to ensure fresh tracing.
    import model.warplayer as wp
    wp.backwarp_tenGrid = {}

    # All-CPU export: RIFE's warplayer uses a module-level `device` variable
    # for grid creation.  With CUDA_VISIBLE_DEVICES="" this resolves to CPU,
    # producing a clean graph without Cast/ConstantOfShape artifacts.
    dummy = torch.randn(1, 6, h, w)

    print(f"  Exporting {filename}  input=(1, 6, {h}, {w})  outputs={output_names}")
    t0 = time.time()

    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        opset_version=16,
        input_names=["input"],
        output_names=output_names,
        do_constant_folding=True,
        # dynamo exporter (default in PyTorch >=2.9) produces cleaner graphs
        # with fewer spurious nodes than the legacy TorchScript path.
    )

    elapsed = time.time() - t0

    # Dynamo exporter may store weights as external data (.onnx.data file).
    # Inline them into the protobuf for QNN converter compatibility.
    import onnx as _onnx

    ext_data = Path(str(onnx_path) + ".data")
    if ext_data.exists():
        model = _onnx.load(str(onnx_path), load_external_data=True)
        _onnx.save(model, str(onnx_path), save_as_external_data=False)
        ext_data.unlink()
        print("    inlined external weights")

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"    -> {onnx_path.name}  ({size_mb:.1f} MB, {elapsed:.1f}s)")

    return onnx_path


def verify_onnx(onnx_path: Path, mode: str, hw: tuple[int, int]) -> None:
    """Quick sanity check: load the ONNX model and run one inference."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("    [skip verification: onnxruntime not installed]")
        return

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    h, w = hw
    dummy = np.random.randn(1, 6, h, w).astype(np.float32)
    outputs = sess.run(None, {"input": dummy})

    if mode == "flow":
        assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"
        assert outputs[0].shape == (1, 4, h, w), f"Flow shape mismatch: {outputs[0].shape}"
        assert outputs[1].shape == (1, 1, h, w), f"Mask shape mismatch: {outputs[1].shape}"
    else:
        assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
        assert outputs[0].shape == (1, 3, h, w), f"Frame shape mismatch: {outputs[0].shape}"

    print("    [verified: shapes OK]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RIFE HDv3 to ONNX (360p/480p x flow/frame).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/export_rife_onnx.py \\\n"
            "      --rife-repo /path/to/ECCV2022-RIFE \\\n"
            "      --weights /path/to/ECCV2022-RIFE/train_log/flownet.pkl \\\n"
            "      --output-dir artifacts/onnx/rife\n"
        ),
    )
    parser.add_argument(
        "--rife-repo", type=Path, required=True,
        help="Path to cloned ECCV2022-RIFE repository.",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help=(
            "Path to flownet.pkl (RIFE HDv3 pretrained weights). "
            "Default: <rife-repo>/train_log/flownet.pkl."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/onnx/rife"),
        help="Output directory for ONNX files (default: artifacts/onnx/rife).",
    )
    parser.add_argument(
        "--resolutions", nargs="+", choices=list(RESOLUTIONS.keys()),
        default=list(RESOLUTIONS.keys()),
        help="Which resolutions to export (default: all).",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=["flow", "frame"],
        default=["flow", "frame"],
        help="Which output modes to export (default: both).",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip ONNX verification with onnxruntime.",
    )
    args = parser.parse_args()

    # Pre-flight checks
    if not args.rife_repo.is_dir():
        print(f"ERROR: RIFE repo not found at {args.rife_repo}")
        print("Clone it: git clone https://github.com/megvii-research/ECCV2022-RIFE")
        sys.exit(1)

    weights = args.weights or (args.rife_repo / "train_log" / "flownet.pkl")

    if not weights.is_file():
        print(f"ERROR: Weights not found at {weights}")
        print("Download HDv3 weights from:")
        print("  https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading RIFE HDv3 from {args.rife_repo}")
    print(f"Weights: {weights}")
    flownet = load_rife_flownet(args.rife_repo, weights)

    total_params = sum(p.numel() for p in flownet.parameters())
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Build wrapper modules
    flow_wrapper = RIFEFlowExporter(flownet)
    flow_wrapper.eval()
    frame_wrapper = RIFEFrameExporter(flownet)
    frame_wrapper.eval()

    # Export all requested variants
    print(f"\nExporting to {args.output_dir}/")
    exported = []

    with torch.no_grad():
        for res_name in args.resolutions:
            hw = RESOLUTIONS[res_name]
            for mode in args.modes:
                wrapper = flow_wrapper if mode == "flow" else frame_wrapper
                onnx_path = export_one(wrapper, mode, res_name, hw, args.output_dir)
                exported.append(onnx_path)

                if not args.no_verify:
                    verify_onnx(onnx_path, mode, hw)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Exported {len(exported)} ONNX models:")
    for p in exported:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:<30} {size_mb:>6.1f} MB")
    print(f"{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
