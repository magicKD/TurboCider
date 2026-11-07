"""Shared BF16 metrics for Gemma MLP and encoder benchmarks; thresholds stay in each driver.

NumPy is loaded only when reading tensors, so CLI help and dry-run stay usable
without optional numerical packages.
"""
from __future__ import annotations

import math
from pathlib import Path


INVALID_METRIC = 1e30


def bf16(path: Path):
    import numpy as np

    words = np.fromfile(path, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def tensor_quality(candidate_path: Path, reference_path: Path) -> dict:
    import numpy as np

    candidate = bf16(candidate_path).astype(np.float64)
    reference = bf16(reference_path).astype(np.float64)
    return array_quality(candidate, reference)


def array_quality(candidate, reference) -> dict:
    """Compare decoded float64 tensors; callers own file/error-report policy."""
    import numpy as np

    if candidate.shape != reference.shape:
        return {"shape_match": False, "relative_l2": INVALID_METRIC,
                "cosine": -1.0, "max_abs": INVALID_METRIC,
                "relative_max_abs": INVALID_METRIC}
    finite = bool(np.isfinite(candidate).all() and np.isfinite(reference).all())
    if not finite:
        return {"shape_match": True, "all_finite": False,
                "relative_l2": INVALID_METRIC, "cosine": -1.0,
                "max_abs": INVALID_METRIC,
                "relative_max_abs": INVALID_METRIC}
    delta = candidate - reference
    delta2 = float(np.dot(delta, delta))
    candidate2 = float(np.dot(candidate, candidate))
    reference2 = float(np.dot(reference, reference))
    denominator = math.sqrt(candidate2 * reference2)
    maximum = float(np.max(np.abs(delta), initial=0.0))
    amplitude = float(np.max(np.abs(reference), initial=0.0))
    return {
        "shape_match": True,
        "all_finite": True,
        "relative_l2": math.sqrt(delta2 / reference2)
        if reference2 > 0 else (0.0 if delta2 == 0 else INVALID_METRIC),
        "cosine": max(-1.0, min(
            1.0, float(np.dot(candidate, reference)) / denominator
            if denominator > 0 else
            (1.0 if candidate2 == reference2 == 0 else 0.0),
        )),
        "max_abs": maximum,
        "relative_max_abs": maximum / amplitude
        if amplitude > 0 else (0.0 if maximum == 0 else INVALID_METRIC),
    }
