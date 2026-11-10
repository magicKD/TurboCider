"""Diagnostic BF16 bias selection for repeated cross-attention padding."""
import itertools
import math
import struct


def padding_biases(padding_rows, copies):
    """Approximate the original duplicate mass with BF16 additive biases.

    This is not exact attention: finite-precision softmax reduction and
    BF16 output rounding can still differ even if the mass error is small.
    """
    if (type(padding_rows) is not int or type(copies) is not int or
            not 1 <= padding_rows <= 1024 or not 1 <= copies <= min(64, padding_rows)):
        raise ValueError("invalid padding row or retained-copy count")
    target = math.log(padding_rows / copies)
    bits = struct.unpack(">I", struct.pack(">f", target))[0] >> 16

    def value(bf16_bits):
        return struct.unpack(">f", struct.pack(">I", bf16_bits << 16))[0]

    candidates = [(value(item), math.exp(value(item))) for item in range(max(0, bits - 3), bits + 4)]
    choices = ((abs((a * low_mass + b * mid_mass + (copies - a - b) * high_mass) / padding_rows - 1),
                a, b, low, mid, high)
               for (low, low_mass), (mid, mid_mass), (high, high_mass) in
               itertools.combinations_with_replacement(candidates, 3)
               for a in range(copies + 1) for b in range(copies - a + 1))
    error, a, b, low, mid, high = min(choices)
    return [low] * a + [mid] * b + [high] * (copies - a - b), error
