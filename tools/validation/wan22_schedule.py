"""FastWan2.2 TI2V DMD schedule, matching the shifted training table."""
import math


def warped_sigmas(steps=(1000, 757, 522), shift=5.0):
    if not math.isfinite(shift) or shift <= 0 or not steps:
        raise ValueError("positive finite shift and nonempty steps required")
    if any(type(step) is not int or not 1 <= step <= 1000 for step in steps):
        raise ValueError("training step indices must be integers in 1...1000")
    if any(left <= right for left, right in zip(steps, steps[1:])):
        raise ValueError("training step indices must strictly decrease")
    return [shift * (step / 1000.0) / (1.0 + (shift - 1.0) * (step / 1000.0)) for step in steps]
