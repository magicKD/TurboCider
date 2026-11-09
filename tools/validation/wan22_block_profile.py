#!/usr/bin/env python3
"""Record synchronous diagnostic timings from one real Wan2.2 block.

The 128-pixel captured inputs are tiled to the requested row count. Each
trace point synchronizes the GPU, so the phase timings are diagnostic and
must not be summed as a prediction of end-to-end runtime.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=6400)
    parser.add_argument("--first-frame-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be a new file in an existing directory")
    paths = [path.resolve(strict=True) for path in (args.probe, args.checkpoint, args.capture)]
    result = subprocess.run([str(path) for path in paths] + [str(args.rows), str(args.first_frame_tokens)],
                            capture_output=True, text=True, check=True, timeout=300)
    timings = {}
    for line in result.stdout.splitlines():
        name, index, seconds = line.split(",")
        samples = timings.setdefault(name, {})
        if index in samples:
            raise ValueError(f"duplicate timing: {name} {index}")
        samples[index] = float(seconds)
    if not timings or any(set(samples) != {"0", "1", "2"} for samples in timings.values()):
        raise ValueError("incomplete block profile")
    report = {"schema": "wan22-block-profile-v1", "rows": args.rows, "first_frame_tokens": args.first_frame_tokens,
              "scope": "real first-block weights; tiled small captured inputs; per-phase GPU synchronization; no cross-KV reuse",
              "samples": {name: [samples[str(i)] for i in range(3)] for name, samples in timings.items()},
              "medians": {name: statistics.median(samples.values()) for name, samples in timings.items()}}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
