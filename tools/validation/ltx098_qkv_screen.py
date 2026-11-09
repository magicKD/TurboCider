#!/usr/bin/env python3
"""ABBA-screen packed QKV on real LTX weights and synthetic 4096-row input."""
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    checkpoint, probe = (p.resolve(strict=True) for p in (args.checkpoint, args.probe))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the pinned source")
    output.mkdir(parents=True)
    run = subprocess.run([str(probe), str(checkpoint), str(output / "outputs.safetensors"), str(args.rows), str(args.repeats)],
                         capture_output=True, text=True, timeout=600)
    (output / "stdout.txt").write_text(run.stdout)
    (output / "stderr.txt").write_text(run.stderr)
    report = {"schema": "ltx098-qkv-screen-v1", "checkpoint_sha256": MODEL_SHA256, "probe_sha256": sha256(probe),
              "returncode": run.returncode,
              "scope": "Block 0 real QKV weights and biases, synthetic input; GPU-only ABBA component screen; packed and original weights both resident; not end-to-end"}
    if run.returncode:
        report.update(status="failed", error=run.stderr[-4000:])
    else:
        import torch
        from safetensors.torch import load_file
        values = load_file(str(output / "outputs.safetensors"))
        measurement = json.loads(run.stdout)
        if [row["packed"] for row in measurement["runs"]] != [False, True, True, False]:
            raise ValueError("ABBA order differs")
        errors = {}
        for name in ("q", "k", "v"):
            left, right = values[f"ref_{name}"].float(), values[f"packed_{name}"].float()
            delta = left - right
            errors[name] = {"exact": bool(torch.equal(left, right)),
                            "relative_l2": (delta.norm() / left.norm().clamp_min(1e-30)).item(),
                            "max_abs": delta.abs().max().item()}
        base = statistics.median(v for row in measurement["runs"] if not row["packed"] for v in row["seconds"])
        packed = statistics.median(v for row in measurement["runs"] if row["packed"] for v in row["seconds"])
        report.update(status="complete", measurements=measurement, errors=errors,
                      separate_median_seconds=base, packed_median_seconds=packed, speedup=base / packed)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if report["returncode"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
