"""Portable GPU full-SwiGLU research controls; never a production backend.

fp16_control: normalized FP16 GEMM, FP16 product/restore boundaries.
a8w8_dequant_fp16: INT8 storage, FP16 dequant/GEMM (floating simulation).
a8w8_integer_alu: signed-byte loads, scalar int32 multiply/accumulate in Metal,
then FP32 restoration. It is NOT native matrix/TensorOps INT8 acceleration.

The integer and dequant variants deliberately have different rounding
boundaries. Each has its own CPU oracle. Runtime LoRA is FP32 in original X/h
bases. No MLX, Metal or Core ML import/evaluation occurs on module import.
"""
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/coreml"))
from runtime_ane_swiglu_int8_candidate import hadamard

VARIANTS = ("fp16_control", "a8w8_dequant_fp16", "a8w8_integer_alu")
INTEGER_SOURCE = r"""
const uint col = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
if (row < M && col < N) {
    int accumulator = 0;
    for (uint k = 0; k < K; ++k) {
        // MLX int8 input pointers are signed char; widen BEFORE multiplying.
        accumulator += int(x[row * K + k]) * int(w[col * K + k]);
    }
    out[row * N + col] = accumulator;
}
"""


def geometry(size):
    if size not in ("tiny", "medium"):
        raise ValueError("size must be tiny or medium")
    rows, hidden, width, group, rank = ((32, 64, 96, 32, 4) if size == "tiny"
                                      else (128, 512, 768, 64, 8))
    return {"rows": rows, "hidden": hidden, "width": width, "group": group, "rank": rank,
            "variant_arithmetic": {
                "fp16_control": "FP32 FWHT/row-scale; FP16 normalized GEMM/product; FP32 restore -> FP16",
                "a8w8_dequant_fp16": "round-even symmetric INT8 storage; code/127 -> FP16; floating FP16 GEMM/product; FP32 restore -> FP16",
                "a8w8_integer_alu": "round-even symmetric INT8 storage; signed INT8 -> int32 scalar integer dot; FP32 /127^2 and restore -> FP16"},
            "integer_accumulator_max_abs": max(hidden, width) * 127 * 127,
            "native_int8_matrix_acceleration": False, "production_speedup_verified": False}


def _fp16(value, label):
    import numpy as np
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value).astype(np.float16)
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite FP16 boundary: {label}")
    return result


def quantize_reference(value, group):
    """CPU FWHT/row-absmax reference; scales stay FP32, codes are [-127,127]."""
    import numpy as np
    rotated = hadamard(value, group)
    maximum = np.max(np.abs(rotated), axis=1, keepdims=True)
    scale = np.where(maximum == 0, np.float32(1), maximum).astype(np.float32)
    normalized = rotated / scale
    codes = np.clip(np.rint(normalized * np.float32(127)), -127, 127).astype(np.int8)
    return normalized.astype(np.float16), codes, scale


def integer_dot_reference(x, w):
    """Int64 oracle for the integer Metal dot, with int32 overflow admission."""
    import numpy as np
    if (x.ndim != 2 or w.ndim != 2 or x.dtype != np.int8 or w.dtype != np.int8 or
            x.shape[1] != w.shape[1] or min(*x.shape, *w.shape) < 1 or
            x.shape[1] * 128 * 128 > np.iinfo(np.int32).max):
        raise ValueError("invalid signed INT8 matrices or unsafe int32 reduction")
    return np.einsum("ik,jk->ij", x.astype(np.int64), w.astype(np.int64), optimize=False).astype(np.int32)


def _lowrank(value, factors):
    import numpy as np
    a, b, scale = factors
    low = np.einsum("ik,rk->ir", value.astype(np.float32), a.astype(np.float32), optimize=False)
    return np.einsum("ir,or->io", low, b.astype(np.float32), optimize=False) * np.float32(scale)


def reference(case, config, variant):
    import numpy as np
    if variant not in VARIANTS:
        raise ValueError("unknown arithmetic variant")
    _, x, weights, adapters = case
    prepared = [quantize_reference(value, config["group"]) for value in (x, *weights)]

    def project(left, right):
        a, qa, sa = left
        b, qb, sb = right
        if variant == "a8w8_integer_alu":
            product = integer_dot_reference(qa, qb).astype(np.float32) / np.float32(127 * 127)
        else:
            if variant == "a8w8_dequant_fp16":
                a = (qa.astype(np.float32) / np.float32(127)).astype(np.float16)
                b = (qb.astype(np.float32) / np.float32(127)).astype(np.float16)
            # Explicit float64 oracle accumulation, followed by the declared
            # FP16 output boundary. GPU reduction ordering can differ.
            product = _fp16(np.einsum("ik,jk->ij", a.astype(np.float64), b.astype(np.float64), optimize=False),
                            "normalized floating GEMM").astype(np.float32)
        return _fp16(product * sa * sb.T, "projection")

    gate = project(prepared[0], prepared[1]).astype(np.float32) + _lowrank(x, adapters[0])
    up = project(prepared[0], prepared[2]).astype(np.float32) + _lowrank(x, adapters[1])
    with np.errstate(over="ignore", invalid="ignore"):
        hidden = _fp16(gate / (1 + np.exp(-gate)) * up, "original corrected hidden")
    down = project(quantize_reference(hidden, config["group"]), prepared[3])
    return {"y": _fp16(down.astype(np.float32) + _lowrank(hidden, adapters[2]), "complete output"), "h": hidden}


class GpuCandidate:
    """Lazy MLX implementation; construction/evaluation only by explicit probe."""
    def __init__(self, config):
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError("Metal GPU is unavailable; CPU fallback is not this experiment")
        mx.set_default_device(mx.gpu)
        self.mx, self.config = mx, config
        self.integer_kernel = mx.fast.metal_kernel(name="tc_research_a8w8_integer_alu",
            input_names=["x", "w"], output_names=["out"], source=INTEGER_SOURCE,
            ensure_row_contiguous=True)

    def enter(self, case):
        _, x, weights, adapters = case
        mx = self.mx
        return (mx.array(x, dtype=mx.float32),
                [mx.array(value, dtype=mx.float32) for value in weights],
                [(mx.array(a, dtype=mx.float32), mx.array(b, dtype=mx.float32), float(scale))
                 for a, b, scale in adapters])

    def rotate(self, value):
        mx, group = self.mx, self.config["group"]
        original = value.shape
        value = value.astype(mx.float32)
        if group != 1:
            value = mx.hadamard_transform(mx.reshape(value, (-1, group)), scale=1 / math.sqrt(group))
        return mx.reshape(value, original)

    def prepare(self, value, variant):
        mx = self.mx
        rotated = self.rotate(value)
        maximum = mx.max(mx.abs(rotated), axis=1, keepdims=True)
        scale = mx.where(maximum == 0, mx.ones_like(maximum), maximum)
        normalized = rotated / scale
        if variant == "fp16_control":
            return normalized.astype(mx.float16), scale
        return mx.clip(mx.round(normalized * 127.0), -127, 127).astype(mx.int8), scale

    def lowrank(self, value, factors):
        mx = self.mx
        a, b, scale = factors
        return ((value.astype(mx.float32) @ a.T) @ b.T) * scale

    def prepare_request(self, entered, variant):
        if variant not in VARIANTS:
            raise ValueError("unknown arithmetic variant")
        x, weights, adapters = entered
        # Gate and up share one prepared X. All three weights are prepared
        # anew for each request; a cache cannot hide their rotation/quant cost.
        prepared = [self.prepare(value, variant) for value in (x, *weights)]
        return prepared, self.lowrank(x, adapters[0]), self.lowrank(x, adapters[1]), adapters[2]

    def integer_dot(self, x, w):
        mx = self.mx
        if x.dtype != mx.int8 or w.dtype != mx.int8 or x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
            raise ValueError("integer kernel requires two signed INT8 [row,reduction] matrices")
        # The public helper also accepts externally constructed -128 codes;
        # do not use the narrower internal quantizer's [-127,127] bound here.
        if x.shape[1] * 128 * 128 > 2147483647:
            raise ValueError("int32 accumulation bound exceeded")
        return self.integer_kernel(inputs=[x, w], output_shapes=[(x.shape[0], w.shape[0])],
            output_dtypes=[mx.int32], grid=(w.shape[0], x.shape[0], 1), threadgroup=(8, 8, 1),
            template=[("M", x.shape[0]), ("N", w.shape[0]), ("K", x.shape[1])])[0]

    def project(self, left, right, variant):
        mx = self.mx
        x, sx = left
        w, sw = right
        if variant == "a8w8_integer_alu":
            product = self.integer_dot(x, w).astype(mx.float32) / float(127 * 127)
        else:
            if variant == "a8w8_dequant_fp16":
                x = (x.astype(mx.float32) / 127.0).astype(mx.float16)
                w = (w.astype(mx.float32) / 127.0).astype(mx.float16)
            product = (x @ w.T).astype(mx.float32)
        return (product * sx * sw.T).astype(mx.float16)

    def forward(self, prepared, variant):
        mx = self.mx
        values, dg, du, down_adapter = prepared
        gate = self.project(values[0], values[1], variant).astype(mx.float32) + dg
        up = self.project(values[0], values[2], variant).astype(mx.float32) + du
        hidden = (gate / (1.0 + mx.exp(-gate)) * up).astype(mx.float16)
        # The FWHT happens only after SiLU/up. Down LoRA reads the unrotated
        # hidden object; neither adapter branch consumes INT8 codes.
        down = self.project(self.prepare(hidden, variant), values[3], variant)
        return {"y": (down.astype(mx.float32) + self.lowrank(hidden, down_adapter)).astype(mx.float16), "h": hidden}
