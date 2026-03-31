"""Export ANVIL D-nomv models to ONNX for NPU benchmark.

Supports two modes:
  - Random weights (default): for latency-only benchmarks (graph structure determines latency)
  - Trained weights (--checkpoint-dir): for INT8 quantization (needs real weight distributions)

Usage:
    pixi run python anvil_exp01/export_onnx_npu.py
    pixi run python anvil_exp01/export_onnx_npu.py --checkpoint-dir checkpoints_dnomv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper, shape_inference

# Add parent to path for model imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.conv_vfi import MODEL_REGISTRY, build_model, count_parameters

INT64_MAX = 9223372036854775807

MODELS = [
    "D-tiny-nomv",
    "D-mini-nomv",
    "D-mid-nomv",
    "D-unet-s-nomv",
    "D-unet-l-nomv",
    "D-nafnet-nomv",
    "D-nafnet-bn-s-nomv",
    "D-nafnet-ln-s-nomv",
    "D-unet-v2-s-nomv",
    "D-unet-v2-m-nomv",
    "D-unet-v2-l-nomv",
    "D-unet-v3-s-nomv",
    "D-unet-v3-m-nomv",
    "D-unet-v3-l-nomv",
    "D-unet-v3xs-nomv",
    "D-unet-v3t-nomv",
    "D-unet-v3tt-nomv",
    "D-unet-v3s-nomv",
    "D-unet-v3m-nomv",
    "D-unet-v3l-nomv",
    "D-unet-v3bs-nomv",
    "D-unet-v3bm-nomv",
]

RESOLUTIONS = {
    "vimeo": (256, 448),
    "540p": (540, 960),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}


def fix_slice_int64max(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace INT64_MAX in Slice 'ends' with actual dimension sizes."""
    model = shape_inference.infer_shapes(model)
    shapes: dict[str, list[int]] = {}
    for vi in list(model.graph.value_info) + list(model.graph.input):
        if vi.type.tensor_type.HasField("shape"):
            shapes[vi.name] = [
                d.dim_value for d in vi.type.tensor_type.shape.dim
            ]
    consts: dict[str, np.ndarray] = {}
    for i in model.graph.initializer:
        consts[i.name] = numpy_helper.to_array(i)
    const_nodes: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    consts[node.output[0]] = numpy_helper.to_array(attr.t)
                    const_nodes[node.output[0]] = node
    patched = 0
    for node in model.graph.node:
        if node.op_type != "Slice" or len(node.input) < 3:
            continue
        ends_name = node.input[2]
        if ends_name not in consts:
            continue
        ends = consts[ends_name]
        if not any(v >= INT64_MAX for v in ends.flat):
            continue
        axes = consts.get(node.input[3]) if len(node.input) > 3 else None
        data_shape = shapes.get(node.input[0])
        if data_shape is None:
            continue
        new_ends = ends.copy()
        for i, v in enumerate(new_ends.flat):
            if v >= INT64_MAX:
                ax = int(axes.flat[i]) if axes is not None else i
                if 0 <= ax < len(data_shape) and data_shape[ax] > 0:
                    new_ends.flat[i] = data_shape[ax]
                    patched += 1
        if ends_name in const_nodes:
            for attr in const_nodes[ends_name].attribute:
                if attr.name == "value":
                    attr.t.CopyFrom(numpy_helper.from_array(new_ends))
        else:
            for init in model.graph.initializer:
                if init.name == ends_name:
                    init.CopyFrom(numpy_helper.from_array(new_ends, ends_name))
                    break
    if patched:
        print(f"    Fixed {patched} INT64_MAX Slice ends")
    return model


def export_one(
    model: torch.nn.Module,
    model_id: str,
    in_ch: int,
    h: int,
    w: int,
    res_name: str,
    out_dir: Path,
) -> Path:
    """Export a single model at a given resolution."""
    safe_name = model_id.replace("-", "_")
    onnx_path = out_dir / f"{safe_name}_{res_name}.onnx"

    dummy = torch.randn(1, in_ch, h, w)
    with torch.no_grad():
        pt_out = model(dummy)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=16,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx_model = fix_slice_int64max(onnx_model)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(onnx_path))

    # ORT numerical verification
    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    ort_out = sess.run(None, {"input": dummy.numpy()})[0]
    diff = np.abs(pt_out.numpy() - ort_out).max()
    size_kb = onnx_path.stat().st_size / 1024
    print(f"    {res_name}: {size_kb:.0f}KB, ORT diff={diff:.2e}")
    if diff > 5e-3:
        print(f"    WARNING: large diff {diff:.2e}")

    return onnx_path


def generate_dummy_input(in_ch: int, h: int, w: int, out_path: Path) -> None:
    """Generate a random float32 raw input file for qnn-net-run."""
    data = np.random.randn(1, in_ch, h, w).astype(np.float32)
    data.tofile(out_path)


def load_trained_weights(model: torch.nn.Module, model_id: str, ckpt_dir: Path) -> bool:
    """Load trained weights from checkpoint directory. Returns True if loaded."""
    best_pt = ckpt_dir / model_id / "best.pt"
    if not best_pt.exists():
        return False
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("npu_bench/onnx"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Load trained weights from this directory (required for INT8)",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS, choices=MODELS
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["vimeo", "1080p"],
        choices=list(RESOLUTIONS),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    using_trained = args.checkpoint_dir is not None
    print("ANVIL NPU ONNX Export")
    print(f"  Weights: {'trained' if using_trained else 'random (latency-only)'}")
    print("=" * 60)

    for model_id in args.models:
        _, kwargs = MODEL_REGISTRY[model_id]
        in_ch = kwargs["in_ch"]
        model = build_model(model_id)

        if using_trained:
            if not load_trained_weights(model, model_id, args.checkpoint_dir):
                print(f"\n{model_id}: SKIP — no checkpoint at {args.checkpoint_dir}/{model_id}/best.pt")
                continue
            print(f"\n{model_id}: loaded trained weights")

        model.eval()

        # Deploy-time fusion for v3b models: fold BN into Conv weights
        if hasattr(model, "fuse_for_deploy"):
            model.fuse_for_deploy()
            print(f"\n{model_id}: fuse_for_deploy() applied (BN fusion)")

        params = count_parameters(model)
        print(f"  {params:,} params, {in_ch}ch input")

        for res_name in args.resolutions:
            h, w = RESOLUTIONS[res_name]
            export_one(model, model_id, in_ch, h, w, res_name, args.out_dir)

            # Generate dummy input for benchmark
            raw_path = args.out_dir / f"{model_id.replace('-', '_')}_{res_name}.raw"
            generate_dummy_input(in_ch, h, w, raw_path)

    # Write input list files (one per model×resolution)
    for model_id in args.models:
        safe_name = model_id.replace("-", "_")
        for res_name in args.resolutions:
            raw_name = f"{safe_name}_{res_name}.raw"
            list_path = args.out_dir / f"input_list_{safe_name}_{res_name}.txt"
            list_path.write_text(f"{raw_name}\n")

    print(f"\nDone. ONNX + raw files in {args.out_dir}/")


if __name__ == "__main__":
    main()
