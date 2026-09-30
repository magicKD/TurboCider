"""Offline dynamic-weight Q/DQ MatMul candidate; not an App runtime backend.

The public graph boundary remains FP16. Both matrix operands are runtime
inputs. Fixed-scale INT8 Q/DQ is a compiler candidate, not proof of integer
execution or Neural Engine placement. No checkpoint is read by this exporter.
Use the matching FP16 control before any model-sized experiment.
"""

import argparse
import json
from pathlib import Path

from export_runtime_ane import export, geometry


def specification(rows, hidden, width, precision, hadamard_group=1):
    if precision not in ("fp16", "qdq_int8"):
        raise ValueError("precision must be fp16 or qdq_int8")
    if (type(hadamard_group) is not int or hadamard_group < 1
            or hadamard_group > 256 or hadamard_group & (hadamard_group - 1)):
        raise ValueError("Hadamard group must be a power of two from 1 through 256")
    spec = geometry("matmul", rows, hidden, width, hidden, width)
    if hidden % hadamard_group:
        raise ValueError("hidden must be divisible by Hadamard group")
    spec.update(
        backend="research_dynamic_weight_projection", graph_version=1,
        precision=precision, interface_dtype="fp16", arithmetic_verified=False,
        runtime_compatible=False, hadamard_group=hadamard_group,
        hadamard_order="normalized_sylvester", normalization="per_matrix_row_absmax",
        restore_dtype="fp32_then_fp16",
        scale_restore="y = matmul(x_normalized, w_normalized.T) * x_scale * w_scale.T",
        scope="single_projection_candidate_not_full_ffn",
    )
    spec["inputs"].update(x_scale=[rows, 1], w_scale=[width, 1])
    if precision == "qdq_int8":
        spec["quantization"] = {"dtype": "int8", "zero_point": 0,
                                "scale": "fp16(1/127)", "operands": ["x", "w"]}
    return spec


def prepare_inputs(x, w, spec):
    """CPU reference preparation only; include this cost in any wall-time report.

    A future host/GPU implementation must preserve this transform, operand
    ordering, normalization and FP16 boundary. This does not accept already
    rotated ConvRot checkpoint codes or perform a production weight conversion.
    """
    import numpy as np

    group = spec["hadamard_group"]
    # Only a small group matrix is allocated; never construct hidden x hidden H.
    rotation = np.ones((1, 1), dtype=np.float32)
    while rotation.shape[0] < group:
        rotation = np.block([[rotation, rotation], [rotation, -rotation]])
    rotation *= np.float32(1 / np.sqrt(group))

    def prepare(value, expected):
        value = np.asarray(value)
        if (list(value.shape) != expected or value.dtype.kind != "f"
                or not np.isfinite(value).all()):
            raise ValueError("operand must be a finite floating-point matrix of the declared shape")
        value = value.astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError("operand exceeds FP32 reference-preparation range")
        if group != 1:
            value = (value.reshape(value.shape[0], -1, group) @ rotation).reshape(value.shape)
        maximum = np.max(np.abs(value), axis=1, keepdims=True)
        # Scale is an FP16 model input. Reject rather than hide clipping or
        # underflow in the normalization/restoration protocol.
        if not np.isfinite(maximum).all() or np.any(maximum > np.finfo(np.float16).max):
            raise ValueError("rotated operand exceeds the FP16 scale range")
        scale = np.where(maximum == 0, 1, maximum).astype(np.float16)
        if np.any(scale == 0):
            raise ValueError("nonzero operand scale underflows FP16")
        normalized = (value / scale.astype(np.float32)).astype(np.float16)
        if not np.isfinite(normalized).all():
            raise ValueError("normalization produced nonfinite values")
        return np.ascontiguousarray(normalized), np.ascontiguousarray(scale)

    x, sx = prepare(x, spec["inputs"]["x"])
    w, sw = prepare(w, spec["inputs"]["w"])
    return {"x": x, "w": w, "x_scale": sx, "w_scale": sw}


def make_program(spec):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    def operand(value):
        if spec["precision"] == "fp16":
            return value
        scale = np.float16(1 / 127)
        return mb.dequantize(input=mb.quantize(input=value, scale=scale,
                                              output_dtype="int8"), scale=scale)

    signatures = [mb.TensorSpec(shape=tuple(shape), dtype=types.fp16)
                  for shape in spec["inputs"].values()]

    @mb.program(input_specs=signatures, opset_version=ct.target.macOS15)
    def program(x, w, x_scale, w_scale):
        product = mb.matmul(x=operand(x), y=operand(w), transpose_y=True)
        # A large x scale and small w scale can have a finite final output
        # while a sequential FP16 intermediate overflows. Restore in FP32;
        # its placement/cost is part of the candidate, not excluded overhead.
        restored = mb.mul(x=mb.cast(x=product, dtype="fp32"),
                          y=mb.cast(x=x_scale, dtype="fp32"))
        restored = mb.mul(x=restored, y=mb.transpose(
            x=mb.cast(x=w_scale, dtype="fp32"), perm=[1, 0]))
        return mb.identity(x=mb.cast(x=restored, dtype="fp16"), name="y")

    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--precision", choices=("fp16", "qdq_int8"), required=True)
    parser.add_argument("--hadamard-group", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = specification(args.rows, args.hidden, args.width, args.precision, args.hadamard_group)
    import coremltools as ct
    # Explicit offline export compiles an artifact, but never predicts or
    # loads a user checkpoint. The research backend is rejected by RuntimeGraph.
    # FLOAT32 disables the converter's automatic FP16 rewrite; explicit FP16
    # matrices/QDQ remain typed FP16 while restoration remains typed FP32.
    print(json.dumps(export(args.output, spec, program_factory=make_program,
                            compute_precision=ct.precision.FLOAT32), sort_keys=True))


if __name__ == "__main__":
    main()
