"""Numerical comparison shared by H3 audio and video VAE probes."""

from __future__ import annotations

import numpy as np


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    left = reference.astype(np.float64).reshape(-1)
    right = candidate.astype(np.float64).reshape(-1)
    diff = left - right
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": float(np.dot(left, right) / denominator) if denominator else 1.0,
    }
