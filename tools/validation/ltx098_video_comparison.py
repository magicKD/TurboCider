#!/usr/bin/env python3
"""Record all-frame RGB/motion diagnostics for a native/Comfy video pair."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
from video_quality_gate import compare, contract_passes, passes


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, candidate = args.reference.resolve(strict=True), args.candidate.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error("output must not already exist")
    metrics = compare(reference, candidate)
    result = {"schema": "ltx098-video-comparison-v1", "reference": str(reference),
              "candidate": str(candidate), "reference_sha256": digest(reference),
              "candidate_sha256": digest(candidate), "metrics": metrics,
              "contract_passed": contract_passes(metrics), "aligned_rgb_gate_passed": passes(metrics),
              "scope": "All decoded RGB frames and frame-to-frame motion energy; different decoder noise and MP4 encoders; no perceptual or multi-prompt quality claim."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("contract_passed", "aligned_rgb_gate_passed")}, indent=2))
    print(json.dumps({key: metrics[key] for key in ("frame_count_compared", "mean_correlation", "minimum_correlation",
                      "mean_cosine", "mean_mae_255", "maximum_motion_relative_error")}, indent=2))


if __name__ == "__main__":
    main()
