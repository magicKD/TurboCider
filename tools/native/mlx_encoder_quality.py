"""Shared MLX tensor metrics for H3 and Qwen3 encoder benchmarks.

Load MLX only when comparing safetensors, so CLI help and dry-run need no
optional numerical packages. Drivers select their own tensor key and thresholds.
"""
from __future__ import annotations

import math
from pathlib import Path


INVALID_METRIC = 1e30


def tensor_quality(candidate_path: Path, reference_path: Path, tensor_name: str) -> dict:
    import mlx.core as mx

    candidate = mx.load(str(candidate_path))[tensor_name].astype(mx.float32)
    reference = mx.load(str(reference_path))[tensor_name].astype(mx.float32)
    if candidate.shape != reference.shape:
        return {
            "shape_match": False,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
            "relative_l2": INVALID_METRIC,
            "cosine": -1.0,
            "max_abs": INVALID_METRIC,
            "relative_max_abs": INVALID_METRIC,
        }
    delta = candidate - reference
    scale = mx.maximum(
        mx.maximum(mx.max(mx.abs(candidate)), mx.max(mx.abs(reference))),
        mx.array(1.0, dtype=mx.float32),
    )
    scaled_candidate = candidate / scale
    scaled_reference = reference / scale
    scaled_delta = delta / scale
    delta2 = mx.sum(scaled_delta * scaled_delta)
    candidate2 = mx.sum(scaled_candidate * scaled_candidate)
    reference2 = mx.sum(scaled_reference * scaled_reference)
    dot = mx.sum(scaled_candidate * scaled_reference)
    max_abs = mx.max(mx.abs(delta))
    reference_max = mx.max(mx.abs(reference))
    finite = mx.logical_and(
        mx.all(mx.isfinite(candidate)), mx.all(mx.isfinite(reference))
    )
    mx.eval(delta2, candidate2, reference2, dot, max_abs, reference_max, finite)
    if not bool(finite.item()):
        return {
            "shape_match": True, "all_finite": False,
            "relative_l2": INVALID_METRIC, "cosine": -1.0,
            "max_abs": INVALID_METRIC, "relative_max_abs": INVALID_METRIC,
        }
    delta_energy = float(delta2.item())
    candidate_energy = float(candidate2.item())
    reference_energy = float(reference2.item())
    absolute = float(max_abs.item())
    amplitude = float(reference_max.item())
    denominator = math.sqrt(candidate_energy * reference_energy)
    return {
        "shape_match": True,
        "all_finite": True,
        "relative_l2": math.sqrt(delta_energy / reference_energy)
        if reference_energy > 0 else (0.0 if delta_energy == 0 else INVALID_METRIC),
        "cosine": max(-1.0, min(1.0, float(dot.item()) / denominator
        if denominator > 0 else (1.0 if candidate_energy == reference_energy == 0
                                 else 0.0))),
        "max_abs": absolute,
        "relative_max_abs": absolute / amplitude
        if amplitude > 0 else (0.0 if absolute == 0 else INVALID_METRIC),
    }
