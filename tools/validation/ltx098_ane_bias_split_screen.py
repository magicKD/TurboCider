#!/usr/bin/env python3
"""Native IOSurface Core ML/GPU split screen with real LTX FFN+bias/input.

This is a component feasibility screen. CPU_AND_NE may fall back to CPU; no
physical ANE placement, model-quality or end-to-end speed claim is made.
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compute", choices=("cpu", "ne"), default="ne")
    parser.add_argument("--block", type=int, choices=range(28), default=0)
    args = parser.parse_args()
    checkpoint, manifest, activation, probe = (path.resolve(strict=True) for path in
        (args.checkpoint, args.manifest, args.activation, args.probe))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the pinned source")
    spec = json.loads(manifest.read_text())
    if (spec.get("backend"), spec.get("biases"), spec.get("rows"), spec.get("hidden"), spec.get("width")) != (
            "ltx098_experimental_bias_fp16", True, 256, 2048, 8192):
        raise ValueError("expected experimental bias-inclusive 256x2048x8192 graph")
    compiled = manifest.parent / "graph.mlmodelc"
    for relative, expected in spec["files"].items():
        if relative.startswith("graph.mlmodelc/") and sha256(manifest.parent / relative) != expected:
            raise ValueError(f"compiled graph hash mismatch: {relative}")
    output.mkdir(parents=True)
    command = [str(probe), str(compiled), str(checkpoint), str(activation), str(output / "outputs.safetensors"),
               str(args.chunks), str(args.repeats), args.compute, str(args.block)]
    report = {"schema": "ltx098-ane-bias-split-screen-v1", "status": "running", "checkpoint_sha256": MODEL_SHA256,
              "manifest_sha256": sha256(manifest), "activation_sha256": sha256(activation), "probe_sha256": sha256(probe),
              "command": command, "compute_units": "CPU_AND_NE" if args.compute == "ne" else "CPU_ONLY",
              "scope": f"Native IOSurface runtime, captured Comfy block-{args.block} FFN input, real BF16 weights and biases; CPU_AND_NE placement unknown; GPU/NE row split is component-only"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=1200)
        (output / "stdout.txt").write_text(run.stdout)
        (output / "stderr.txt").write_text(run.stderr)
        if run.returncode:
            raise RuntimeError(f"native probe exited {run.returncode}: {run.stderr[-4000:]}")
        import torch
        from safetensors.torch import load_file
        measurements = json.loads(run.stdout)
        if measurements["block"] != args.block:
            raise ValueError("reported block differs from requested block")
        values = load_file(str(output / "outputs.safetensors"))
        reference, candidate = values["gpu"].float(), values["split"].float()
        if reference.shape != (4096, 2048) or candidate.shape != reference.shape:
            raise ValueError("unexpected output geometry")
        if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
            raise ValueError("nonfinite output")
        head = measurements["gpu_rows"]

        def difference(a, b):
            delta = a - b
            return {"relative_l2": (delta.norm() / a.norm().clamp_min(1e-30)).item(),
                    "max_abs": delta.abs().max().item(), "exact": bool(torch.equal(a, b))}

        gpu = statistics.median(measurements["gpu_seconds"])
        split = statistics.median(measurements["split_seconds"])
        staged = statistics.median(measurements["split_with_stage_seconds"])
        report.update(status="complete", measurements=measurements, gpu_median_seconds=gpu,
            split_median_seconds=split, split_with_stage_median_seconds=staged,
            speedup_without_stage=gpu / split, speedup_with_stage=gpu / staged,
            error_all=difference(reference, candidate), error_gpu_head=difference(reference[:head], candidate[:head]),
            error_coreml_tail=difference(reference[head:], candidate[head:]))
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("command", "measurements")}, indent=2))


if __name__ == "__main__":
    main()
