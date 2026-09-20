#!/usr/bin/env python3
"""Compare one real Qwen3 ANE prefix with its original BF16 GPU calculation.

Random hidden states test a single partition, not encoder/image quality or
end-to-end performance. Run separately from timing campaigns.
"""
import argparse
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

from benchmark_coreml_ffn_bridge import CoreMLFFN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coreml"))
from export_qwen3 import QwenSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-scale", type=float, default=0.25)
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    manifest = json.loads(args.manifest.read_text())
    shape = manifest["shape"]
    hidden, width = int(shape["K"]), int(shape["ane_mlp_end"])
    output_scale = float(shape.get("output_scale", 1))
    if int(shape["ane_mlp_start"]) != 0 or args.rows not in shape["buckets"]:
        raise ValueError("requires a prefix partition and an exact token bucket")
    if args.block < 0 or str(args.block) not in manifest["artifacts"]:
        raise ValueError("block is not present in the compiled manifest")
    if not np.isfinite(args.input_scale) or args.input_scale <= 0:
        raise ValueError("input scale must be positive and finite")
    source = QwenSource(args.model)
    prefix = manifest.get("export_identity", {}).get("tensor_prefix", "model")
    with source:
        matrices = []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            value = source.raw(f"{prefix}.layers.{args.block}.mlp.{projection}.weight")
            if value.dtype == np.uint32:
                raise ValueError("this oracle requires original BF16/F16 weights")
            value = value[:, :width] if projection == "down_proj" else value[:width]
            tensor = mx.array(value, dtype=mx.bfloat16)
            mx.eval(tensor)
            matrices.append(tensor)
    gate, up, down = matrices
    if gate.shape != (width, hidden) or up.shape != gate.shape or down.shape != (hidden, width):
        raise ValueError("checkpoint geometry does not match the partition")
    rng = np.random.default_rng(args.seed)
    inputs = (rng.standard_normal((1, args.rows, hidden), dtype=np.float32) * args.input_scale)
    x = mx.array(inputs, dtype=mx.bfloat16)
    mx.eval(x)
    # Both paths see exactly the same BF16-rounded values. FP16 transport here
    # preserves these values; the Core ML artifact uses FP16 arithmetic and its exported weight precision.
    transported = np.asarray(x.astype(mx.float32)).astype(np.float16)
    projected = x @ gate.T
    reference = ((projected * mx.sigmoid(projected)) * (x @ up.T)) @ down.T
    mx.eval(reference)
    expected = np.asarray(reference.astype(mx.float32))
    bridge = CoreMLFFN(args.library, args.manifest, source.checkpoint, args.rows, 1)
    try:
        actual, _ = bridge.predict(args.block, transported)
        # Native Qwen3 restores this scale after the raw Core ML FFN call.
        actual = actual * np.float16(output_scale)
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError("partition produced a non-finite value")
        actual32 = actual.astype(np.float32)
        difference = actual32 - expected
        norm = float(np.linalg.norm(expected.reshape(-1)))
        actual_norm = float(np.linalg.norm(actual32.reshape(-1)))
        peak = float(np.max(np.abs(expected)))
        if norm == 0 or actual_norm == 0 or peak == 0:
            raise ValueError("degenerate oracle output")
        report = {
            "schema_version": 1,
            "scope": "One real-weight MLP prefix on random hidden states; no full encoder/image quality or performance claim",
            "model": str(args.model.resolve()), "manifest": str(args.manifest.resolve()),
            "block": args.block, "rows": args.rows, "hidden": hidden, "prefix_width": width,
            "seed": args.seed, "input_scale": args.input_scale,
            "reference_device": str(mx.default_device()), "output_scale": output_scale,
            "reference": "original weights and MLP on MLX GPU in BF16",
            "candidate": "native Core ML FFN bridge; compiled artifact; CPU+NE configuration",
            "relative_l2": float(np.linalg.norm(difference.reshape(-1))) / norm,
            "cosine": float(np.dot(actual32.reshape(-1), expected.reshape(-1))) / (norm * actual_norm),
            "max_abs_error": float(np.max(np.abs(difference))),
            "relative_max_abs": float(np.max(np.abs(difference))) / peak,
            "all_finite": True, "native_metrics": bridge.metrics(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
