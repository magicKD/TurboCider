#!/usr/bin/env python3
"""Screen a compiled RoPE candidate for exact BF16 output and GPU timing."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

from ltx098_comfy_reference import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    probe, output = args.probe.resolve(strict=True), args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    output.mkdir(parents=True)
    run = subprocess.run([str(probe), str(output / "outputs.safetensors"), str(args.tokens), str(args.repeats)],
                         capture_output=True, text=True, timeout=600)
    (output / "stdout.txt").write_text(run.stdout)
    (output / "stderr.txt").write_text(run.stderr)
    report = {"schema": "ltx098-rope-screen-v1", "probe_sha256": sha256(probe), "returncode": run.returncode,
              "scope": "Synthetic BF16 Q/K and frequencies, 32 heads x 64 dimensions, GPU-only ABBA; not end-to-end"}
    if run.returncode:
        report.update(status="failed", error=run.stderr[-4000:])
    else:
        import torch
        from safetensors.torch import load_file
        values = load_file(str(output / "outputs.safetensors"))
        measurement = json.loads(run.stdout)
        if [row["compiled"] for row in measurement["runs"]] != [False, True, True, False]:
            raise ValueError("ABBA order differs")
        errors = {}
        for name in ("q", "k"):
            reference, candidate = values[f"reference_{name}"].float(), values[f"candidate_{name}"].float()
            delta = reference - candidate
            errors[name] = {"exact": bool(torch.equal(reference, candidate)),
                            "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
                            "max_abs": delta.abs().max().item()}
        baseline = statistics.median(v for row in measurement["runs"] if not row["compiled"] for v in row["seconds"])
        candidate = statistics.median(v for row in measurement["runs"] if row["compiled"] for v in row["seconds"])
        report.update(status="complete", measurements=measurement, errors=errors,
                      baseline_median_seconds=baseline, candidate_median_seconds=candidate, speedup=baseline / candidate)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "measurements"}, indent=2))
    if run.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
