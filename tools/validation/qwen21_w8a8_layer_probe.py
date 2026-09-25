"""Offline Qwen21 FFN W8A8 layer check on a real cached-decode activation.

The Core ML C ABI uses CPU_AND_NE; this is not proof of runtime ANE placement
or end-to-end image quality. Only block 0 is supported by this first probe.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native"))
from benchmark_coreml_ffn_bridge import CoreMLFFN


def metrics(actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite FFN output")
    difference = actual - expected
    norm = float(np.sum(expected.astype(np.float64) ** 2))
    dot = float(np.sum(actual.astype(np.float64) * expected.astype(np.float64)))
    actual_norm = float(np.sum(actual.astype(np.float64) ** 2))
    if norm <= 0 or actual_norm <= 0:
        raise ValueError("zero-energy FFN output")
    return {"relative_rmse": float(np.sqrt(np.sum(difference.astype(np.float64) ** 2) / norm)),
            "cosine": float(dot / np.sqrt(actual_norm * norm)),
            "max_abs": float(np.max(np.abs(difference)))}


def validate_manifest(manifest, weight_only_control=False):
    identity = manifest.get("export_identity", {})
    shape = manifest.get("shape", {})
    buckets = shape.get("buckets")
    width = identity.get("ane_mlp_end")
    expected_precision = "fp16" if weight_only_control else "int8"
    if (identity.get("tensor_layout") != "qwen21" or
            identity.get("activation_precision", "fp16") != expected_precision or
            type(width) is not int or width < 32 or width >= 12288 or width % 32 or
            shape.get("ane_mlp_start") != 0 or shape.get("ane_mlp_end") != width or
            shape.get("K") != 4096 or shape.get("mlp_width") != 12288 or
            sorted(manifest.get("artifacts", {})) != ["0"] or
            buckets not in ([1024], [4096])):
        raise ValueError("expected a compiled Qwen21 block-0 1024/4096-row manifest with a valid FFN partition")
    return buckets[0], width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("build/native/libturbocider.dylib"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--weight-only-control", action="store_true",
                        help="accept a Qwen21 W8A16 control manifest without activation Q/DQ")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected_precision = "fp16" if args.weight_only_control else "int8"
    rows, width = validate_manifest(manifest, args.weight_only_control)
    if not 1 <= args.iterations <= 100:
        parser.error("iterations must be 1...100")
    x = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if x.shape != (rows, 4096) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError(f"expected finite real Qwen21 [{rows},4096] FP32 activation")
    x16 = np.ascontiguousarray(x[None].astype(np.float16))
    weights = mx.load(str(args.checkpoint))
    prefix = "transformer_blocks.0.img_mlp."
    fused = weights[prefix + "gate_up.weight"]
    down = weights[prefix + "out.weight"]
    activation = mx.array(np.asarray(x)[None]).astype(mx.bfloat16)

    def ffn(gate_up, projection):
        gate, up = mx.split(mx.matmul(activation, gate_up.T), 2, axis=-1)
        return mx.matmul(gate * mx.sigmoid(gate) * up, projection.T)

    total = 12288
    ane_weights = mx.concatenate([fused[:width], fused[total:total + width]])
    gpu_weights = mx.concatenate([fused[width:total], fused[total + width:]])
    ane_down, gpu_down = down[:, :width], down[:, width:]
    reference = ffn(fused, down)
    branch_reference = ffn(ane_weights, ane_down)
    gpu_suffix = ffn(gpu_weights, gpu_down)
    mx.eval(reference, branch_reference, gpu_suffix)
    bridge = CoreMLFFN(args.library, args.manifest, args.checkpoint, rows, 0)
    try:
        prediction = None
        durations = []
        for _ in range(args.iterations + 2):
            prediction, seconds = bridge.predict(0, x16)
            durations.append(seconds)
        hybrid = (np.array(gpu_suffix.astype(mx.float32)) + prediction.astype(np.float32)).astype(np.float16)
        full = np.array(reference.astype(mx.float32))
        report = {
            "scope": "one cached-decode FFN block on one real prompt; not e2e or ANE residency",
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "activation_precision": expected_precision,
            "ane_mlp_width": width,
            "ane_branch": metrics(prediction, np.array(branch_reference.astype(mx.float32))),
            "hybrid": metrics(hybrid, full),
            "coreml_prediction_seconds": durations[2:],
            "coreml_prediction_median": statistics.median(durations[2:]),
            "cpu_and_ne_policy": True,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
