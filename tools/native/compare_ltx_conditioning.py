"""Compare raw BF16 LTX conditioning directories on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_bf16(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    return (raw.astype(np.uint32) << 16).view(np.float32)


def load(directory: Path) -> dict[str, np.ndarray]:
    mask = read_bf16(directory / "raw_text_mask.bf16")
    if mask.size == 0:
        raise ValueError(f"empty mask in {directory}")
    video = read_bf16(directory / "raw_video_context.bf16")
    audio = read_bf16(directory / "raw_audio_context.bf16")
    if video.size % mask.size or audio.size % mask.size:
        raise ValueError(f"conditioning geometry mismatch in {directory}")
    return {
        "video": video.reshape(mask.size, video.size // mask.size),
        "audio": audio.reshape(mask.size, audio.size // mask.size),
        "mask": mask,
    }


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        return {
            "shape_mismatch": [list(reference.shape), list(candidate.shape)]
        }
    difference = reference.astype(np.float64) - candidate.astype(np.float64)
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    reference_norm = np.linalg.norm(reference64)
    candidate_norm = np.linalg.norm(candidate64)
    relative_l2 = (np.linalg.norm(difference) / reference_norm
                   if reference_norm else 0.0 if not np.any(difference)
                   else float("inf"))
    cosine_similarity = (
        np.vdot(reference64, candidate64).real /
        (reference_norm * candidate_norm)
        if reference_norm and candidate_norm else
        1.0 if not reference_norm and not candidate_norm else 0.0
    )
    return {
        "shape": list(reference.shape),
        "finite": bool(np.isfinite(reference).all() and
                       np.isfinite(candidate).all()),
        "identical": bool(np.array_equal(reference, candidate)),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "relative_l2": float(relative_l2),
        "cosine_similarity": float(cosine_similarity),
        "reference_rms": float(np.sqrt(np.mean(reference64 ** 2))),
        "candidate_rms": float(np.sqrt(np.mean(candidate64 ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    reference = load(args.reference)
    candidate = load(args.candidate)
    report = {
        name: compare(reference[name], candidate[name])
        for name in ("video", "audio", "mask")
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and not all(
            value.get("identical") and value.get("finite")
            for value in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
