"""Isolated dynamic-weight SwiGLU/Hadamard Q/DQ research graph.

No checkpoint, production backend or Core ML prediction is used on import.
Gate/up LoRA corrections are prepared from original X before SiLU. Down LoRA
reads the corrected hidden in its original basis, never its rotated/quantized
copy. The FP16 control and Q/DQ candidate have identical interfaces and FP32
nonlinearity/restoration. Q/DQ does not prove integer execution/ANE placement.
"""
import argparse
import json
import math
import numbers
from pathlib import Path

from export_runtime_ane import export, geometry
from runtime_ane_int8_candidate import specification as projection_specification


def specification(rows, hidden, width, precision, hadamard_group=1, lora_rank=4):
    # Reuse the single projection's checked precision/group/dimension contract.
    projection_specification(rows, hidden, width, precision, hadamard_group)
    if width % hadamard_group:
        raise ValueError("width must be divisible by Hadamard group for down projection")
    if type(lora_rank) is not int or not 1 <= lora_rank <= min(hidden, width, 256):
        raise ValueError("LoRA rank must be an integer in [1,min(hidden,width,256)]")
    spec = geometry("swiglu", rows, hidden, width, hidden, width, "exp", True)
    spec.update(backend="research_dynamic_weight_swiglu", graph_version=1,
                precision=precision, interface_dtype="fp16", runtime_compatible=False,
                arithmetic_verified=False, hardware_int8_verified=False,
                production_speedup_verified=False, hadamard_group=hadamard_group,
                hadamard_order="normalized_sylvester", lora_rank=lora_rank,
                normalization="per_matrix_row_absmax",
                hidden_scale_dtype="fp32", nonlinear_dtype="fp32_then_fp16_hidden",
                restore_dtype="fp32_then_fp16",
                lora_boundary="original_X_gate_up_deltas_before_SiLU; original_h_down_A_B",
                lora_scale_semantics="explicit_multiplier_no_implicit_alpha_over_rank",
                scope="complete_swiglu_cpu_gate_up_lora_preparation_and_dynamic_down_lora")
    # Keep a fixed argument order for the MIL signature below. Base requests
    # use zero corrections and zero down factors, not merged model weights.
    spec["inputs"] = {"x": [rows, hidden], "wg": [width, hidden],
                      "wu": [width, hidden], "wd": [hidden, width],
                      "x_scale": [rows, 1], "wg_scale": [width, 1],
                      "wu_scale": [width, 1], "wd_scale": [hidden, 1],
                      "dg": [rows, width], "du": [rows, width],
                      "down_a": [lora_rank, width], "down_b": [hidden, lora_rank],
                      "down_scale": [1]}
    if precision == "qdq_int8":
        spec["quantization"] = {"dtype": "int8", "zero_point": 0,
                                "scale": "fp16(1/127)",
                                "operands": ["x", "wg", "wu", "rotated_h", "wd"],
                                "lora_quantized": False}
    spec["outputs"] = {"y": [rows, hidden], "h": [rows, width]}
    spec["input_bytes"] = 2 * sum(math.prod(s) for s in spec["inputs"].values())
    return spec


def _matrix(value, shape, label):
    import numpy as np
    value = np.asarray(value)
    if list(value.shape) != list(shape) or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite floating-point array of shape {shape}")
    value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{label} exceeds FP32 range")
    return value


def hadamard(value, group):
    """Normalized Sylvester FWHT over contiguous groups of the last dimension."""
    import numpy as np
    value = np.asarray(value)
    if (value.ndim != 2 or value.dtype.kind != "f" or not np.isfinite(value).all()
            or type(group) is not int or group < 1 or group > 256 or group & (group - 1)
            or value.shape[1] % group):
        raise ValueError("invalid finite FWHT matrix/group")
    value = value.astype(np.float32, copy=True)
    stride = 1
    with np.errstate(over="ignore", invalid="ignore"):
        while stride < group:
            pairs = value.reshape(value.shape[0], -1, 2 * stride)
            left, right = pairs[..., :stride].copy(), pairs[..., stride:].copy()
            pairs[..., :stride], pairs[..., stride:] = left + right, left - right
            stride *= 2
        value *= np.float32(1 / np.sqrt(group))
    if not np.isfinite(value).all():
        raise ValueError("FWHT exceeds FP32 range")
    return value


def _fp16(value, label):
    import numpy as np
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value).astype(np.float16)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} exceeds finite FP16 boundary")
    return np.ascontiguousarray(result)


def _normalize(value, *, fp16_scale):
    import numpy as np
    maximum = np.max(np.abs(value), axis=1, keepdims=True)
    scale = np.where(maximum == 0, np.float32(1), maximum)
    if fp16_scale:
        scale = _fp16(scale, "operand scale")
        if np.any(scale == 0):
            raise ValueError("nonzero operand scale underflows FP16")
    normalized = _fp16(value / scale.astype(np.float32), "normalized operand")
    return normalized, np.ascontiguousarray(scale)


def lora_delta(value, a, b, scale):
    """Unquantized CPU FP32 low-rank branch; explicit scale, [out,in] factors."""
    import numpy as np
    value = np.asarray(value)
    a, b = np.asarray(a), np.asarray(b)
    if (value.ndim != 2 or a.ndim != 2 or b.ndim != 2 or
            a.shape[1] != value.shape[1] or b.shape[1] != a.shape[0] or a.shape[0] < 1 or
            isinstance(scale, (bool, np.bool_)) or not isinstance(scale, numbers.Real) or not np.isfinite(scale)):
        raise ValueError("invalid LoRA shapes or explicit scale")
    value = _matrix(value, value.shape, "LoRA input")
    a, b = _matrix(a, a.shape, "LoRA A"), _matrix(b, b.shape, "LoRA B")
    with np.errstate(over="ignore", invalid="ignore"):
        # Reference butterflies and reductions do not depend on a host BLAS
        # library's ordering or floating-point exception-state side effects.
        low = np.einsum("ik,rk->ir", value, a, optimize=False)
        delta = np.einsum("ir,or->io", low, b, optimize=False) * np.float32(scale)
    if not np.isfinite(delta).all():
        raise ValueError("LoRA branch exceeds finite FP32 range")
    return delta


def prepare_inputs(x, wg, wu, wd, spec, *, dg=None, du=None,
                   down_a=None, down_b=None, down_scale=1):
    """CPU reference staging. Its rotation/delta/conversion cost is not excluded.

    dg/du must be computed with original X (e.g. lora_delta), not the rotated
    graph input. Weight transforms remain dynamic; no LoRA is merged into W.
    """
    import numpy as np
    inputs, group = spec["inputs"], spec["hadamard_group"]
    if isinstance(down_scale, (bool, np.bool_)) or not isinstance(down_scale, numbers.Real) or not np.isfinite(down_scale):
        raise ValueError("down_scale must be an explicit finite numeric multiplier")
    result = {}
    for name, value in (("x", x), ("wg", wg), ("wu", wu), ("wd", wd)):
        rotated = hadamard(_matrix(value, inputs[name], name), group)
        result[name], result[name + "_scale"] = _normalize(rotated, fp16_scale=True)
    for name, value in (("dg", dg), ("du", du), ("down_a", down_a), ("down_b", down_b)):
        if value is None:
            value = np.zeros(inputs[name], dtype=np.float32)
        result[name] = _fp16(_matrix(value, inputs[name], name), name)
    result["down_scale"] = _fp16(_matrix(np.asarray([down_scale], dtype=np.float64), [1],
                                        "down_scale"), "down_scale")
    return {name: result[name] for name in inputs}


def _qdq(value, precision):
    import numpy as np
    value = np.asarray(value).astype(np.float32)
    if precision == "fp16":
        return value
    scale = np.float32(np.float16(1 / 127))
    codes = np.clip(np.rint(value / scale), -128, 127).astype(np.int8)
    # MIL dequantize retains the FP16 operand type.
    return (codes.astype(np.float32) * scale).astype(np.float16).astype(np.float32)


def reference(inputs, spec):
    """CPU graph oracle (FP32 matmul accumulation, declared FP16 boundaries).

    Backend reduction ordering is not claimed byte-identical. This reference
    fails nonfinite boundaries; a probe must likewise inspect both h and y.
    """
    import numpy as np
    if set(inputs) != set(spec["inputs"]):
        raise ValueError("candidate input names do not match specification")
    for name, shape in spec["inputs"].items():
        value = np.asarray(inputs[name])
        if value.dtype != np.float16 or list(value.shape) != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid prepared FP16 input {name}")
    for name in ("x_scale", "wg_scale", "wu_scale", "wd_scale"):
        if np.any(inputs[name] <= 0):
            raise ValueError("normalization scales must be positive")

    def projection(x, w, sx, sw):
        product = _fp16(np.einsum("ik,jk->ij", _qdq(x, spec["precision"]),
                                  _qdq(w, spec["precision"]), optimize=False),
                        "normalized matmul")
        return _fp16(product.astype(np.float32) * sx.astype(np.float32) * sw.T.astype(np.float32),
                     "restored projection")

    gate = projection(inputs["x"], inputs["wg"], inputs["x_scale"], inputs["wg_scale"])
    up = projection(inputs["x"], inputs["wu"], inputs["x_scale"], inputs["wu_scale"])
    gate, up = gate.astype(np.float32) + inputs["dg"], up.astype(np.float32) + inputs["du"]
    with np.errstate(over="ignore", invalid="ignore"):
        hidden = _fp16((gate / (1 + np.exp(-gate))) * up, "corrected hidden")
    rotated, scale = _normalize(hadamard(hidden, spec["hadamard_group"]), fp16_scale=False)
    base = projection(rotated, inputs["wd"], scale, inputs["wd_scale"])
    delta = lora_delta(hidden, inputs["down_a"], inputs["down_b"], inputs["down_scale"][0])
    return {"y": _fp16(base.astype(np.float32) + delta, "complete SwiGLU output"), "h": hidden}


def make_program(spec):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    def fp32(value):
        return mb.cast(x=value, dtype="fp32")

    def operand(value):
        if spec["precision"] == "fp16":
            return value
        scale = np.float16(1 / 127)
        return mb.dequantize(input=mb.quantize(input=value, scale=scale, output_dtype="int8"), scale=scale)

    def projection(x, w, sx, sw, *, prepared_x=False):
        product = mb.matmul(x=x if prepared_x else operand(x), y=operand(w), transpose_y=True)
        restored = mb.mul(x=fp32(product), y=fp32(sx))
        restored = mb.mul(x=restored, y=mb.transpose(x=fp32(sw), perm=[1, 0]))
        return mb.cast(x=restored, dtype="fp16")

    def rotate_hidden(hidden):
        value = fp32(hidden)
        stride, rows, width = 1, spec["rows"], spec["width"]
        while stride < spec["hadamard_group"]:
            value = mb.reshape(x=value, shape=[rows, width // (2 * stride), 2 * stride])
            left = mb.slice_by_index(x=value, begin=[0, 0, 0], end=[rows, width // (2 * stride), stride])
            right = mb.slice_by_index(x=value, begin=[0, 0, stride], end=[rows, width // (2 * stride), 2 * stride])
            value = mb.concat(values=[mb.add(x=left, y=right), mb.sub(x=left, y=right)], axis=2)
            stride *= 2
        value = mb.reshape(x=value, shape=[rows, width])
        return mb.mul(x=value, y=np.float32(1 / np.sqrt(spec["hadamard_group"])))

    signatures = [mb.TensorSpec(shape=tuple(shape), dtype=types.fp16)
                  for shape in spec["inputs"].values()]

    @mb.program(input_specs=signatures, opset_version=ct.target.macOS15)
    def program(x, wg, wu, wd, x_scale, wg_scale, wu_scale, wd_scale,
                dg, du, down_a, down_b, down_scale):
        shared_x = operand(x)
        gate = mb.add(x=fp32(projection(shared_x, wg, x_scale, wg_scale, prepared_x=True)), y=fp32(dg))
        up = mb.add(x=fp32(projection(shared_x, wu, x_scale, wu_scale, prepared_x=True)), y=fp32(du))
        activated = mb.real_div(x=gate, y=mb.add(x=np.float32(1), y=mb.exp(x=mb.mul(x=gate, y=np.float32(-1)))))
        hidden = mb.cast(x=mb.mul(x=activated, y=up), dtype="fp16")
        rotated = rotate_hidden(hidden)
        maximum = mb.reduce_max(x=mb.abs(x=rotated), axes=[1], keep_dims=True)
        scale = mb.select(cond=mb.equal(x=maximum, y=np.float32(0)),
                          a=mb.fill(shape=[spec["rows"], 1], value=np.float32(1)), b=maximum)
        normalized = mb.cast(x=mb.real_div(x=rotated, y=scale), dtype="fp16")
        base = projection(normalized, wd, scale, wd_scale)
        # Down adapter uses original hidden, FP32 factors/accumulation, and no
        # Hadamard/QDQ. Quantizing this branch would change the LoRA contract.
        delta = mb.matmul(x=fp32(hidden), y=fp32(down_a), transpose_y=True)
        delta = mb.matmul(x=delta, y=fp32(down_b), transpose_y=True)
        delta = mb.mul(x=delta, y=fp32(down_scale))
        output = mb.cast(x=mb.add(x=fp32(base), y=delta), dtype="fp16")
        return mb.identity(x=output, name="y"), mb.identity(x=hidden, name="h")

    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--precision", choices=("fp16", "qdq_int8"), required=True)
    parser.add_argument("--hadamard-group", type=int, default=32)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = specification(args.rows, args.hidden, args.width, args.precision,
                         args.hadamard_group, args.lora_rank)
    import coremltools as ct
    print(json.dumps(export(args.output, spec, program_factory=make_program,
                            compute_precision=ct.precision.FLOAT32), sort_keys=True))


if __name__ == "__main__":
    main()
