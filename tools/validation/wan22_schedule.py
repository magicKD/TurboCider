"""Diagnostic FastWan2.2 DMD timestep and sigma conventions."""
import math


def warped_sigmas(steps=(1000, 757, 522), shift=5.0):
    if not math.isfinite(shift) or shift <= 0 or not steps:
        raise ValueError("positive finite shift and nonempty steps required")
    if any(type(step) is not int or not 1 <= step <= 1000 for step in steps):
        raise ValueError("training step indices must be integers in 1...1000")
    if any(left <= right for left, right in zip(steps, steps[1:])):
        raise ValueError("training step indices must strictly decrease")
    return [shift * (step / 1000.0) / (1.0 + (shift - 1.0) * (step / 1000.0)) for step in steps]


def cuda_raw_sigmas(steps=(1000, 757, 522), shift=8.0):
    """Nearest training-table sigmas for FastVideo's dense CUDA DMD path.

    Model timesteps remain the raw input steps; these returned values are
    only the sigmas used for prediction and re-noising. This is distinct
    from mapping step indices through the shifted table before inference.
    """
    import numpy as np

    # Reuse the common input validation without reusing the warped result.
    warped_sigmas(steps, shift)
    if shift > float(np.finfo(np.float32).max):
        raise ValueError("shift exceeds the float32 scheduler range")
    base = np.arange(1000, 0, -1, dtype=np.float32) / np.float32(1000)
    shifted = np.float32(shift) * base / (np.float32(1) + np.float32(shift - 1) * base)
    requested = np.asarray(steps, dtype=np.float32)
    indices = np.abs(shifted[:, None] * np.float32(1000) - requested[None, :]).argmin(axis=0)
    return [float(value) for value in shifted[indices]]
