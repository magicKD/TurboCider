#!/usr/bin/env python3
"""Compare FastVideo and TurboCider shared-noise E2E tensor dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    left = reference.astype(np.float64).reshape(-1)
    right = candidate.astype(np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    difference = left - right
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "cosine": (float(np.dot(left, right) / denominator)
                   if denominator else float(np.array_equal(left, right))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--latent-max-abs", type=float, default=2e-2)
    parser.add_argument("--latent-mean-abs", type=float, default=2e-3)
    parser.add_argument("--condition-max-abs", type=float, default=1e-3)
    parser.add_argument("--condition-min-cosine", type=float, default=0.99999)
    args = parser.parse_args()

    reference = load_file(str(args.reference))
    candidate = load_file(str(args.candidate))
    required = ("conditioning", "token_tags", "video_rows", "audio_rows")
    for name in required:
        if name not in reference or name not in candidate:
            raise KeyError(f"missing required tensor: {name}")

    report = {
        name: metrics(np.asarray(reference[name]), np.asarray(candidate[name]))
        for name in required
    }
    report["token_tags_exact"] = bool(np.array_equal(
        np.asarray(reference["token_tags"]), np.asarray(candidate["token_tags"])))
    condition_ok = (
        report["token_tags_exact"] and
        report["conditioning"]["max_abs"] <= args.condition_max_abs and
        report["conditioning"]["cosine"] >= args.condition_min_cosine
    )
    latent_ok = all(
        report[name]["max_abs"] <= args.latent_max_abs and
        report[name]["mean_abs"] <= args.latent_mean_abs
        for name in ("video_rows", "audio_rows")
    )
    report["passed"] = condition_ok and latent_ok
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
