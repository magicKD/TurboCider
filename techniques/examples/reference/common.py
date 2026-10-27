"""Small, dependency-free arithmetic helpers for readable teaching examples."""
import math
import struct


def matrix(rows, cols, rng, scale=1.0):
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def transpose(x):
    return [list(row) for row in zip(*x)]


def matmul(x, y):
    """Ordinary [M,K]@[K,N], Python float accumulation (not a device recipe)."""
    if not x or not y or not y[0] or any(len(row) != len(y) for row in x) or any(len(row) != len(y[0]) for row in y):
        raise ValueError("matmul shape mismatch")
    return [[math.fsum(a * b for a, b in zip(row, col)) for col in zip(*y)] for row in x]


def linear(x, w):
    return matmul(x, transpose(w))


def add(a, b):
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        raise ValueError("add shape mismatch")
    return [[x + y for x, y in zip(row, other)] for row, other in zip(a, b)]


def silu(x):
    if x >= 0:
        return x / (1 + math.exp(-x))
    value = math.exp(x)
    return x * value / (1 + value)


def swiglu(x, wg, wu, wd):
    gate, up = linear(x, wg), linear(x, wu)
    h = [[silu(g) * u for g, u in zip(gs, us)] for gs, us in zip(gate, up)]
    return linear(h, wd), h


def flat(value):
    if isinstance(value, (list, tuple)):
        return [item for row in value for item in flat(row)]
    return [float(value)]


def metrics(reference, candidate):
    a, b = flat(reference), flat(candidate)
    if not a or len(a) != len(b) or not all(math.isfinite(x) for x in a + b):
        raise ValueError("metrics require equal nonempty finite arrays")
    energy_a = math.fsum(x * x for x in a)
    energy_b = math.fsum(x * x for x in b)
    error = math.fsum((x - y) ** 2 for x, y in zip(a, b))
    cosine = (math.fsum(x * y for x, y in zip(a, b)) / math.sqrt(energy_a * energy_b)
              if energy_a and energy_b else (1.0 if a == b else 0.0))
    return {"max_abs": max(abs(x - y) for x, y in zip(a, b)),
            "relative_l2": math.sqrt(error / energy_a) if energy_a else None,
            "cosine": cosine, "finite": True, "elements": len(a)}


def fp16(value):
    if not math.isfinite(value):
        raise ValueError("nonfinite source")
    try:
        result = struct.unpack("<e", struct.pack("<e", float(value)))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("FP16 overflow: needs headroom or a different recipe") from error
    if not math.isfinite(result):
        raise ValueError("FP16 overflow")
    return result


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0x7f800000 == 0x7f800000:
        raise ValueError("nonfinite F32/BF16 source")
    bits = ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffffffff) & 0xffff0000
    result = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(result):
        raise ValueError("BF16 overflow")
    return result


def round_even_int8(value, signed_min=-127):
    if not math.isfinite(value) or signed_min not in (-127, -128):
        raise ValueError("invalid quantizer input/range")
    return max(signed_min, min(127, round(value)))


def quantize_rows(rows, *, normalize=False, half_scales=False):
    """RNE per-row symmetric Q8. Zero rows keep a safe nonzero scale."""
    codes, scales = [], []
    for row in rows:
        if not row or not all(math.isfinite(x) for x in row):
            raise ValueError("nonempty finite quantizer rows required")
        peak = max(abs(x) for x in row)
        scale = peak / 127 if peak else 1.0
        if normalize:
            scale *= 128
        if half_scales:
            scale = fp16(max(scale, 2 ** -24))
        multiplier = 128 if normalize else 1
        codes.append([round_even_int8(value / scale * multiplier) for value in row])
        scales.append(scale)
    return codes, scales


def dequantize_rows(codes, scales, *, normalize=False):
    if len(codes) != len(scales):
        raise ValueError("scale row mismatch")
    denominator = 128 if normalize else 1
    return [[code * scale / denominator for code in row] for row, scale in zip(codes, scales)]


def fwht(row):
    """Orthonormal Sylvester transform, no sign scrambling."""
    n = len(row)
    if not n or n & (n - 1) or not all(math.isfinite(x) for x in row):
        raise ValueError("FWHT requires finite power-of-two length")
    values, stride = list(row), 1
    while stride < n:
        for first in range(0, n, stride * 2):
            for offset in range(stride):
                a, b = values[first + offset], values[first + offset + stride]
                values[first + offset], values[first + offset + stride] = a + b, a - b
        stride *= 2
    return [value / math.sqrt(n) for value in values]


def seed_sign(index, seed=20260930):
    mask = (1 << 64) - 1
    value = (seed + index * 0x9e3779b97f4a7c15) & mask
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
    return -1 if (value ^ (value >> 31)) & 1 else 1


def rotate(row, block, *, seed=20260930, inverse=False):
    """Forward X D H/sqrt(B); inverse multiplies H then D, not D then H."""
    if type(block) is not int or block <= 0 or block & (block - 1) or len(row) % block:
        raise ValueError("rotation requires a full power-of-two block")
    output = []
    for start in range(0, len(row), block):
        chunk = row[start:start + block]
        if inverse:
            chunk = [x * seed_sign(i, seed) for i, x in enumerate(fwht(chunk))]
        else:
            chunk = fwht([x * seed_sign(i, seed) for i, x in enumerate(chunk)])
        output.extend(chunk)
    return output


def report(lab_id, details, *, scope="synthetic CPU reference math", backend="python_cpu", synthetic=True):
    """Return an in-memory teaching summary; never write evidence or receipts."""
    return {"example": lab_id, "scope": scope, "backend": backend, "synthetic": synthetic,
            "not_device_or_model_result": True,
            "details": details}
