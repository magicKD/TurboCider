"""Stable SiLU algebra and legal paired-channel slices, with no device imports."""
import math

from common import report


def stable_silu(value):
    if not math.isfinite(value):
        raise ValueError("finite source required")
    small = math.exp(-abs(value))
    return (value if value >= 0 else value * small) / (1 + small)


def channel_slices(width, share):
    if type(width) is not int or type(share) is not int or not 0 < share < width:
        raise ValueError("nonempty complementary intermediate-channel partitions required")
    return {"gate": [0, share], "up": [width, width + share], "down_columns": [0, share],
            "gpu_gate": [share, width], "gpu_up": [width + share, 2 * width], "gpu_down_columns": [share, width]}


def domain(seed=17):
    points = [-1000., -100., -12., -1., 0., 1., 12., 100., 1000.]
    values = [stable_silu(point) for point in points]
    assert all(math.isfinite(value) for value in values)
    assert values[4] == 0. and values[-1] == 1000.
    return report("silu-domain", {"points": points, "stable_silu": values, "seed": seed,
                  "paired_example": channel_slices(12, 4), "device_numerics_verified": False},
                  scope="CPU real-arithmetic domain and legal slice demonstration; no FP16/backend qualification")
