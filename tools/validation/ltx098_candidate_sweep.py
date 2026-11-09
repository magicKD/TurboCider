#!/usr/bin/env python3
"""ABBA-screen a candidate versus baseline on captured full-size DiT input."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

from ltx098_comfy_reference import MODEL_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    checkpoint, data, baseline, candidate = (p.resolve(strict=True) for p in
        (args.checkpoint, args.input, args.baseline, args.candidate))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the pinned source")
    import torch
    from safetensors.torch import load_file

    output.mkdir(parents=True)
    report = {"schema": "ltx098-candidate-sweep-v1", "status": "running", "checkpoint_sha256": MODEL_SHA256,
              "input_sha256": sha256(data), "baseline_sha256": sha256(baseline), "candidate_sha256": sha256(candidate),
              "scope": "512x512x121 full transformer, captured conditioning; ABBA; excludes model load, text projection and end-to-end decoding",
              "runs": []}
    reference = None
    try:
        for index, (name, binary) in enumerate((("baseline", baseline), ("candidate", candidate),
                                                ("candidate", candidate), ("baseline", baseline))):
            path = output / f"prediction-{index}.safetensors"
            command = [str(binary), str(checkpoint), str(data), str(path), "1", str(args.warmups), str(args.repeats)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(f"{name}: {result.stderr[-4000:]}")
            row = json.loads(result.stdout)
            value = load_file(str(path))["output"].float()
            if reference is None:
                reference = value
            difference = value - reference
            row.update(index=index, variant=name, median_seconds=statistics.median(row["seconds"]),
                       exact=bool(torch.equal(value, reference)),
                       relative_l2=(difference.norm() / reference.norm().clamp_min(1e-30)).item())
            report["runs"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        for name in ("baseline", "candidate"):
            report[name + "_median_seconds"] = statistics.median(
                sample for row in report["runs"] if row["variant"] == name for sample in row["seconds"])
        report["speedup"] = report["baseline_median_seconds"] / report["candidate_median_seconds"]
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
