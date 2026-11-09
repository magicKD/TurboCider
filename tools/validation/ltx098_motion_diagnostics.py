#!/usr/bin/env python3
"""Locate per-transition RGB motion-energy differences in two videos."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
from video_quality_gate import _command, _decode_process, _probe, _read_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    reference, candidate = args.reference.resolve(strict=True), args.candidate.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error("output must not already exist")
    ffmpeg, ffprobe = _command("ffmpeg", None), _command("ffprobe", None)
    left, right = _probe(reference, ffprobe), _probe(candidate, ffprobe)
    if (left["width"], left["height"], left["fps"]) != (right["width"], right["height"], right["fps"]):
        raise ValueError("video geometry differs")
    frame_bytes = left["width"] * left["height"] * 3
    processes = [_decode_process(path, ffmpeg) for path in (reference, candidate)]
    previous, rows = None, []
    try:
        while True:
            data = [_read_frame(process.stdout, frame_bytes, path)
                    for process, path in zip(processes, (reference, candidate))]
            if all(value is None for value in data):
                break
            if any(value is None for value in data):
                raise ValueError("frame counts differ")
            frames = [np.frombuffer(value, np.uint8).astype(np.int16) for value in data]
            if previous is not None:
                first, second = [float(np.abs(current - old).mean()) for current, old in zip(frames, previous)]
                rows.append({"from_frame": len(rows), "reference_energy": first, "candidate_energy": second,
                             "relative_difference": abs(first - second) / max(first, second, 1.0)})
            previous = frames
        for process in processes:
            stderr = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise RuntimeError(stderr)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            process.stdout.close()
            process.stderr.close()
    result = {"schema": "ltx098-motion-diagnostics-v1", "reference": str(reference),
              "candidate": str(candidate), "transitions": len(rows), "rows": rows,
              "largest": sorted(rows, key=lambda row: row["relative_difference"], reverse=True)[:10]}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["largest"], indent=2))


if __name__ == "__main__":
    main()
