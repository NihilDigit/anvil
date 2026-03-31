"""Verify deploy-time BN fusion is mathematically exact.

Reproduces paper claim:
  - Max absolute difference between fused and unfused outputs: < 1.5e-5 (FP32 rounding)

Usage:
    pixi run python scripts/eval_deploy_fusion.py \
        --model D-unet-v3bs-nomv \
        --checkpoint artifacts/checkpoints/D-unet-v3bs-nomv/best.pt

    pixi run python scripts/eval_deploy_fusion.py \
        --model D-unet-v3bm-nomv \
        --checkpoint artifacts/checkpoints/D-unet-v3bm-nomv/best.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import onnx
import torch

from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from anvil_exp01.models.conv_vfi import build_model, count_parameters


RESOLUTIONS = {
    "vimeo": (256, 448),
    "1080p": (1080, 1920),
}


def count_onnx_ops(onnx_path: str) -> dict[str, int]:
    """Load an ONNX model and return a dict of {op_type: count}."""
    model = onnx.load(onnx_path)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def verify_fusion_equivalence(
    model_id: str,
    checkpoint: Path,
    n_inputs: int,
    resolution: str,
) -> dict:
    """Run unfused and fused models on identical inputs, compare outputs and ONNX ops.

    Returns a dict with max_abs_diff, unfused_ops, fused_ops, and computed conv ratios.
    """
    device = torch.device("cpu")
    H, W = RESOLUTIONS[resolution]

    # --- Build and load model ---
    model = build_model(model_id).to(device)
    n_params = count_parameters(model)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # --- Generate random inputs ---
    torch.manual_seed(42)
    inputs = [torch.randn(1, 6, H, W, device=device) for _ in range(n_inputs)]

    # --- Unfused forward pass ---
    unfused_outputs = []
    with torch.no_grad():
        for inp in inputs:
            unfused_outputs.append(model(inp).clone())

    # --- Export unfused ONNX ---
    dummy = torch.randn(1, 6, H, W, device=device)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as f:
        torch.onnx.export(
            model, dummy, f.name,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
        )
        unfused_ops = count_onnx_ops(f.name)

    # --- Fuse and forward ---
    n_fused_blocks = model.fuse_for_deploy()

    fused_outputs = []
    with torch.no_grad():
        for inp in inputs:
            fused_outputs.append(model(inp).clone())

    # --- Export fused ONNX ---
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as f:
        torch.onnx.export(
            model, dummy, f.name,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
        )
        fused_ops = count_onnx_ops(f.name)

    # --- Compute max abs diff ---
    max_diff = 0.0
    per_input_diffs = []
    for u, f_ in zip(unfused_outputs, fused_outputs):
        d = (u - f_).abs().max().item()
        per_input_diffs.append(d)
        max_diff = max(max_diff, d)

    # --- Compute conv ratios ---
    conv_ops = ["Conv", "ConvTranspose"]

    unfused_conv_count = sum(unfused_ops.get(op, 0) for op in conv_ops)
    unfused_total = sum(unfused_ops.values())
    unfused_conv_ratio = unfused_conv_count / unfused_total if unfused_total > 0 else 0.0

    fused_conv_count = sum(fused_ops.get(op, 0) for op in conv_ops)
    fused_total = sum(fused_ops.values())
    fused_conv_ratio = fused_conv_count / fused_total if fused_total > 0 else 0.0

    return {
        "model_id": model_id,
        "n_params": n_params,
        "resolution": f"{H}x{W}",
        "n_inputs": n_inputs,
        "n_fused_blocks": n_fused_blocks,
        "max_abs_diff": max_diff,
        "per_input_diffs": per_input_diffs,
        "unfused_ops": unfused_ops,
        "unfused_total_ops": unfused_total,
        "unfused_conv_count": unfused_conv_count,
        "unfused_conv_ratio": unfused_conv_ratio,
        "fused_ops": fused_ops,
        "fused_total_ops": fused_total,
        "fused_conv_count": fused_conv_count,
        "fused_conv_ratio": fused_conv_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify deploy-time BN fusion is mathematically exact."
    )
    parser.add_argument("--model", required=True, help="Model ID from registry.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--n-inputs", type=int, default=10, help="Number of random inputs.")
    parser.add_argument(
        "--resolution", default="vimeo", choices=list(RESOLUTIONS.keys()),
        help="Input resolution.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/eval/deploy_fusion"),
        help="Directory for output JSON.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Alias for --n-inputs (smoke test compatibility). Overrides --n-inputs if > 0.",
    )
    args = parser.parse_args()

    n_inputs = args.limit if args.limit > 0 else args.n_inputs

    result = verify_fusion_equivalence(
        model_id=args.model,
        checkpoint=args.checkpoint,
        n_inputs=n_inputs,
        resolution=args.resolution,
    )

    # --- Print results ---
    passed = result["max_abs_diff"] < 1.5e-5
    status = "PASS" if passed else "FAIL"

    print(f"Model: {result['model_id']} ({result['n_params']:,} params)")
    print(f"Resolution: {result['resolution']}, N={result['n_inputs']} random inputs")
    print(f"Fused blocks: {result['n_fused_blocks']}")
    print(f"Max absolute diff: {result['max_abs_diff']:.2e}")
    print(
        f"Unfused: Conv={result['unfused_conv_ratio']:.1%} "
        f"({result['unfused_conv_count']}/{result['unfused_total_ops']} ops), "
        f"BN={result['unfused_ops'].get('BatchNormalization', 0)} ops, "
        f"Add={result['unfused_ops'].get('Add', 0)} ops"
    )
    print(
        f"Fused:   Conv={result['fused_conv_ratio']:.1%} "
        f"({result['fused_conv_count']}/{result['fused_total_ops']} ops), "
        f"BN={result['fused_ops'].get('BatchNormalization', 0)} ops, "
        f"Add={result['fused_ops'].get('Add', 0)} ops"
    )
    print(f"{status}: fusion is {'mathematically exact (< 1.5e-5)' if passed else 'NOT exact (>= 1.5e-5)'}")

    # --- Save JSON ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"

    summary = {
        "model_id": result["model_id"],
        "n_params": result["n_params"],
        "resolution": result["resolution"],
        "n_inputs": result["n_inputs"],
        "n_fused_blocks": result["n_fused_blocks"],
        "max_abs_diff": result["max_abs_diff"],
        "per_input_max_diffs": result["per_input_diffs"],
        "unfused": {
            "total_ops": result["unfused_total_ops"],
            "conv_count": result["unfused_conv_count"],
            "conv_ratio": round(result["unfused_conv_ratio"], 4),
            "ops": result["unfused_ops"],
        },
        "fused": {
            "total_ops": result["fused_total_ops"],
            "conv_count": result["fused_conv_count"],
            "conv_ratio": round(result["fused_conv_ratio"], 4),
            "ops": result["fused_ops"],
        },
        "passed": passed,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_path}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
