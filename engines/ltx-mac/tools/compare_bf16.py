#!/usr/bin/env python3
"""Compare matching raw BF16 tensors with generative-model metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def bf16(path: Path) -> np.ndarray:
    words = np.fromfile(path, dtype=np.uint16)
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference = bf16(args.reference).astype(np.float64)
    candidate = bf16(args.candidate).astype(np.float64)
    if reference.shape != candidate.shape:
        raise SystemExit(
            f"shape mismatch: {reference.shape} != {candidate.shape}"
        )
    finite = np.isfinite(reference) & np.isfinite(candidate)
    difference = candidate[finite] - reference[finite]
    reference_finite = reference[finite]
    candidate_finite = candidate[finite]
    reference2 = float(np.dot(reference_finite, reference_finite))
    candidate2 = float(np.dot(candidate_finite, candidate_finite))
    difference2 = float(np.dot(difference, difference))
    dot = float(np.dot(reference_finite, candidate_finite))
    rel_l2 = np.sqrt(difference2 / max(reference2, 1e-300))
    cosine = dot / np.sqrt(max(reference2 * candidate2, 1e-300))
    rms_ratio = np.sqrt(candidate2 / max(reference2, 1e-300))
    print(
        f"elements={reference.size} finite={int(finite.sum())} "
        f"nonfinite={int((~finite).sum())} "
        f"rel_l2={rel_l2:.9g} cosine={cosine:.9g} "
        f"rms_ratio={rms_ratio:.9g} "
        f"rmse={np.sqrt(difference2 / max(int(finite.sum()), 1)):.9g} "
        f"max_abs={np.max(np.abs(difference)):.9g}"
    )


if __name__ == "__main__":
    main()
