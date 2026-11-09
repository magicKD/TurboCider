#!/usr/bin/env python3
"""ABBA-screen block evaluation intervals using one captured real-model input.

This is a transformer microbenchmark, not an end-to-end performance claim.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

from ltx098_comfy_reference import MODEL_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--native-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intervals", default="2,4")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    checkpoint, probe, trace = (p.resolve(strict=True) for p in (args.checkpoint, args.probe, args.native_trace))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    intervals = [int(value) for value in args.intervals.split(",")]
    if not intervals or len(set(intervals)) != len(intervals) or any(value < 0 or value > 28 or value == 1 for value in intervals):
        parser.error("intervals must be unique values in [0,28] excluding the baseline 1")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the pinned source")

    import torch
    from safetensors.torch import load_file, save_file

    def tensor(name):
        return load_file(str(trace / f"{name}.safetensors"))["value"]

    latents = tensor("initial_noise").clone()
    latents[:, :1] = tensor("first_frame").float()
    latents = latents.to(torch.bfloat16)
    text = tensor("text")
    # This captured input corresponds to the reference lighthouse prompt.
    if tuple(latents.shape) != (1, 16, 16, 16, 128) or tuple(text.shape) != (1, 128, 4096):
        raise ValueError("the sweep requires the 512x512x121 reference capture")
    mask = torch.full((1, 1, 1, 128), -10000, dtype=torch.bfloat16)
    mask[:, :, :, :45] = 0
    output.mkdir(parents=True)
    input_path = output / "input.safetensors"
    save_file({"latents": latents.contiguous(), "text": text.contiguous(), "text_mask": mask}, str(input_path))
    report = {"schema": "ltx098-eval-sweep-v1", "status": "running", "checkpoint_sha256": MODEL_SHA256,
              "probe_sha256": sha256(probe), "input_sha256": sha256(input_path),
              "scope": "512x512x121 transformer-only, captured native text and first-frame conditioning; separate processes; ABBA; excludes model load and text preparation",
              "runs": []}
    order = [item for candidate in intervals for item in (1, candidate, candidate, 1)]
    baseline = None
    try:
        for index, interval in enumerate(order):
            prediction = output / f"prediction-{index}.safetensors"
            command = [str(probe), str(checkpoint), str(input_path), str(prediction), str(interval),
                       str(args.warmups), str(args.repeats)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(f"interval {interval} exited {result.returncode}: {result.stderr[-4000:]}")
            row = json.loads(result.stdout)
            value = load_file(str(prediction))["output"].float()
            if baseline is None:
                baseline = value
            delta = value - baseline
            row.update(index=index, median_seconds=statistics.median(row["seconds"]),
                       exact=bool(torch.equal(value, baseline)),
                       relative_l2=(delta.norm() / baseline.norm().clamp_min(1e-30)).item(),
                       max_abs=delta.abs().max().item())
            report["runs"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
