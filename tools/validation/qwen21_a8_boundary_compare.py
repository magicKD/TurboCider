"""Compare Core ML CPU and CPU_AND_NE on identical Qwen21 FFN activations.

Any disagreement is a backend/graph deployment issue, not a BF16 oracle
accuracy claim. Boundary-isolated exports are research artifacts, not W8A8.
"""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np


def compare(source, value, reference=None):
    manifest = json.loads(source.read_text())
    identity = manifest["export_identity"]
    if identity.get("tensor_layout") != "qwen21" or set(manifest["artifacts"]) != {"0"}:
        raise ValueError("expected single-block Qwen21 manifest")
    path = source.parent / manifest["artifacts"]["0"]["int8_pc"]
    outputs = {}
    for label, units in (("cpu", ct.ComputeUnit.CPU_ONLY),
                         ("cpu_ne", ct.ComputeUnit.CPU_AND_NE)):
        model = ct.models.MLModel(str(path), compute_units=units)
        outputs[label] = model.predict({"x": value})["y"].astype(np.float32)
    a, b = outputs["cpu"], outputs["cpu_ne"]
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite or mismatched Core ML output")
    numerator = np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    denominator = np.sum(a.astype(np.float64) ** 2)
    ne_energy = np.sum(b.astype(np.float64) ** 2)
    if denominator <= 0 or ne_energy <= 0:
        raise ValueError("zero-energy Core ML output")
    # An amplitude mismatch and a changed output direction require different
    # remedies. This least-squares scale is diagnostic, not a runtime fix.
    gain = float(np.sum(a.astype(np.float64) * b.astype(np.float64)) / ne_energy)
    report = {"boundary": identity.get("a8_graph", "weight_only_or_fp16"),
              "cpu_std": float(a.std()), "cpu_ne_std": float(b.std()),
              "cpu_ne_vs_cpu_rrmse": float(np.sqrt(numerator / denominator)),
              "cpu_ne_least_squares_gain": gain,
              "cpu_ne_gain_corrected_rrmse": float(np.sqrt(
                  np.sum((a.astype(np.float64) - gain * b.astype(np.float64)) ** 2)
                  / denominator))}
    if reference is not None:
        target = reference.astype(np.float64)
        target_norm = np.sum(target ** 2)
        report["cpu_vs_bf16_rrmse"] = float(np.sqrt(np.sum((a.astype(np.float64) - target) ** 2) / target_norm))
        report["cpu_ne_vs_bf16_rrmse"] = float(np.sqrt(np.sum((b.astype(np.float64) - target) ** 2) / target_norm))
    return report


def bf16_reference(checkpoint, inputs):
    import mlx.core as mx
    weights = mx.load(str(checkpoint))
    prefix = "transformer_blocks.0.img_mlp."
    activation = mx.array(np.array(inputs).transpose(0, 3, 2, 1).reshape(1, 1024, 4096)).astype(mx.bfloat16)
    gate_up = weights[prefix + "gate_up.weight"]
    gate, up = mx.split(mx.matmul(activation, mx.concatenate(
        [gate_up[:4096], gate_up[12288:16384]]).T), 2, axis=-1)
    expected = mx.matmul(gate * mx.sigmoid(gate) * up,
                         weights[prefix + "out.weight"][:, :4096].T)
    return np.array(expected.astype(mx.float32)).reshape(1, 1024, 4096).transpose(0, 2, 1)[:, :, None, :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="optional true BF16 MLP branch reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if value.shape != (1024, 4096) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("expected finite [1024,4096] FP32 Qwen21 FFN input")
    provided = np.ascontiguousarray(value.astype(np.float16).T[None, :, None, :])
    reference = bf16_reference(args.checkpoint, provided) if args.checkpoint else None
    report = {source.parent.name: compare(source, provided, reference) for source in args.source}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
