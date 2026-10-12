"""Independent NumPy oracles for ConvRot A8 and normalized runtime INT8.

No model route or hardware claim. Original signed weight codes are unchanged.
"""
from __future__ import annotations

import numpy as np


def hadamard256():
    h4 = np.asarray([[1, 1, 1, -1], [1, 1, -1, 1],
                     [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float32) * np.float32(.5)
    h16 = np.kron(h4, h4)
    return np.kron(h16, h16)


def rotate(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim < 1 or not x.shape[-1] or x.shape[-1] % 256 or not np.all(np.isfinite(x)):
        raise ValueError("rotation requires finite complete H256 groups")
    output = np.einsum("ij,jk->ik", x.reshape(-1, 256), hadamard256(),
                       optimize=False).reshape(x.shape)
    if not np.all(np.isfinite(output)):
        raise OverflowError("rotation overflow")
    return output


def quantize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("invalid activation")
    maximum = np.max(np.abs(x), axis=1)
    scale = np.where(maximum == 0, np.float32(1), maximum / np.float32(127)).astype(np.float32)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("activation scale underflow/overflow")
    codes = np.clip(np.rint(x / scale[:, None]), -127, 127).astype(np.int8)
    return codes, scale


def integer_dot(x, w, fast=False):
    x = np.asarray(x)
    w = np.asarray(w)
    if (x.dtype != np.int8 or w.dtype != np.int8 or x.ndim != 2 or w.ndim != 2
            or x.shape[1] != w.shape[1] or not x.size or not w.size):
        raise ValueError("signed I8 matrices required")
    if np.any(x == -128):
        raise ValueError("activation codes must be [-127,127]")
    if x.shape[1] * 127 * 128 > np.iinfo(np.int32).max:
        raise OverflowError("INT32 worst-case dot bound exceeded")
    if fast:
        # Every multiply and partial sum is exact in FP64 (absolute bound
        # <2^31). Some Apple BLAS builds leave spurious FP exception flags;
        # inspect every output and independently check fixed I64 samples.
        with np.errstate(all="ignore"):
            result = x.astype(np.float64) @ w.astype(np.float64).T
        if not np.all(np.isfinite(result)) or not np.all(result == np.rint(result)):
            raise ArithmeticError("noninteger reference result")
        rows = np.unique(np.linspace(0, x.shape[0] - 1, min(4, x.shape[0]), dtype=int))
        columns = np.unique(np.linspace(0, w.shape[0] - 1, min(8, w.shape[0]), dtype=int))
        exact = x[rows].astype(np.int64) @ w[columns].astype(np.int64).T
        if not np.array_equal(result[np.ix_(rows, columns)], exact):
            raise ArithmeticError("BLAS/I64 oracle mismatch")
        result = result.astype(np.int64)
    else:
        result = x.astype(np.int64) @ w.astype(np.int64).T
    return result.astype(np.int32)


def normalize(codes):
    codes = np.asarray(codes)
    if codes.dtype != np.int8:
        raise ValueError("I8 codes required")
    return codes.astype(np.float16) * np.float16(1 / 128)


def normalized_output(dot):
    dot = np.asarray(dot)
    if dot.dtype != np.int32:
        raise ValueError("INT32 dot required")
    with np.errstate(over="ignore"):
        output = (dot.astype(np.float32) * np.float32(1 / 16384)).astype(np.float16)
    if not np.all(np.isfinite(output)):
        raise OverflowError("normalized FP16 output overflow")
    return output


def restore_normalized(output, activation_scales, weight_scales, bias=None):
    output = np.asarray(output, dtype=np.float32)
    sx = np.asarray(activation_scales, dtype=np.float32)
    sw = np.asarray(weight_scales, dtype=np.float32)
    if output.ndim != 2 or sx.shape != (output.shape[0],) or sw.shape != (output.shape[1],):
        raise ValueError("scale geometry mismatch")
    if (not np.all(np.isfinite(output)) or not np.all(np.isfinite(sx))
            or not np.all(np.isfinite(sw)) or np.any(sx <= 0)):
        raise ValueError("nonfinite/invalid scales")
    with np.errstate(over="ignore", invalid="ignore"):
        result = output * ((sx[:, None] * sw[None, :]) * np.float32(16384))
        if bias is not None:
            bias = np.asarray(bias, dtype=np.float32)
            if bias.shape != (output.shape[1],) or not np.all(np.isfinite(bias)):
                raise ValueError("invalid bias")
            result = result + bias
    if not np.all(np.isfinite(result)):
        raise OverflowError("scale restoration overflow")
    return result


def integer_epilogue(dot, activation_scales, weight_scales, bias=None):
    """GPU recipe: I32 -> F32 -> multiply sx -> sw -> bias, in order.

    Deliberately different from the normalized FP16 Core ML exit recipe.
    """
    dot = np.asarray(dot)
    sx = np.asarray(activation_scales, dtype=np.float32)
    sw = np.asarray(weight_scales, dtype=np.float32)
    if dot.dtype != np.int32 or dot.ndim != 2 or not dot.size:
        raise ValueError("nonempty INT32 matrix required")
    if sx.shape != (dot.shape[0],) or sw.shape != (dot.shape[1],):
        raise ValueError("scale geometry mismatch")
    if not np.all(np.isfinite(sx)) or np.any(sx <= 0) or not np.all(np.isfinite(sw)):
        raise ValueError("invalid/nonfinite scales")
    with np.errstate(over="ignore", invalid="ignore"):
        result = (dot.astype(np.float32) * sx[:, None]) * sw[None, :]
        if bias is not None:
            bias = np.asarray(bias, dtype=np.float32)
            if bias.shape != (dot.shape[1],) or not np.all(np.isfinite(bias)):
                raise ValueError("invalid bias")
            result = result + bias
    if not np.all(np.isfinite(result)):
        raise OverflowError("integer epilogue overflow")
    return result


def group_integer_projection(x, w, activation_scales, weight_scales, group_size):
    """Per-group A8: restore each partial dot before FP32 accumulation."""
    x = np.asarray(x)
    w = np.asarray(w)
    if type(group_size) is not int or group_size <= 0 or x.ndim != 2 or w.ndim != 2:
        raise ValueError("invalid group geometry")
    if x.shape[1] != w.shape[1] or not x.size or not w.size or x.shape[1] % group_size:
        raise ValueError("group-aligned equal reduction required")
    sx = np.asarray(activation_scales, dtype=np.float32)
    if sx.shape != (x.shape[0], x.shape[1] // group_size):
        raise ValueError("group scale geometry mismatch")
    result = np.zeros((x.shape[0], w.shape[0]), dtype=np.float32)
    for index, begin in enumerate(range(0, x.shape[1], group_size)):
        dot = integer_dot(x[:, begin:begin + group_size], w[:, begin:begin + group_size])
        result = result + integer_epilogue(dot, sx[:, index], weight_scales)
        if not np.all(np.isfinite(result)):
            raise OverflowError("group accumulation overflow")
    return result


def fp16_ulp_distance(a, b):
    a = np.asarray(a, dtype=np.float16)
    b = np.asarray(b, dtype=np.float16)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("finite equal shapes required")

    def ordered(x):
        bits = x.view(np.uint16).astype(np.int32)
        return np.where(bits & 0x8000, 0x8000 - (bits & 0x7fff), 0x8000 + bits)

    return np.abs(ordered(a) - ordered(b))
