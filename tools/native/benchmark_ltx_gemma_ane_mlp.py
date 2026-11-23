#!/usr/bin/env python3
"""Qualify a Gemma GPU+ANE gated-MLP split against the GPU reference."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


# Keep direct CLI invocation and file-based tooling imports working.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltx_gemma_mlp_benchmark import (  # noqa: E402
    DEFAULT_ROWS, GPU_PROBE_FORMAT, passes_quality_gate, run_probe,
    validate_limits, write_report,
)
from ltx_gemma_quality import INVALID_METRIC, bf16, tensor_quality  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-probe", type=Path,
                        default=Path("build/native/ltx-gemma-mlp-probe"))
    parser.add_argument("--hybrid-probe", type=Path,
                        default=Path("build/native/ltx-gemma-ane-mlp-probe"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shader", type=Path,
                        default=Path("build/native/ltx_shaders.metal"))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=DEFAULT_ROWS)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-speedup", type=float, default=1.1)
    parser.add_argument("--max-relative-l2", type=float, default=0.025)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-relative-abs", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.layer < 0 or args.layer >= 48:
        parser.error("--layer must be in 0...47")
    if (not args.rows or len(set(args.rows)) != len(args.rows) or
            any(row not in DEFAULT_ROWS for row in args.rows)):
        parser.error(f"--rows values must be unique members of {DEFAULT_ROWS}")
    validate_limits(parser, args)
    return args


def command_for(args: argparse.Namespace, variant: str, rows: int,
                folder: Path) -> list[str]:
    if variant == "gpu":
        return [str(args.gpu_probe), str(args.checkpoint), str(args.shader),
                str(rows), str(args.runs), str(folder)]
    return [str(args.hybrid_probe), str(args.manifest), str(args.checkpoint),
            str(args.shader), str(args.layer), str(rows), str(args.runs),
            str(folder)]


def run_case(command: list[str], folder: Path, timeout: int,
             expected_format: str) -> dict:
    # The native probes create their artifact directory directly.  Ensure its
    # parent exists, but leave the leaf absent as required by the probe.
    folder.parent.mkdir(parents=True, exist_ok=True)
    return run_probe(
        command, folder, timeout, environment=os.environ.copy(), expected_format=expected_format,
        failure_label=f"Gemma {expected_format}", invalid_label="Gemma",
    )


def main() -> int:
    args = arguments()
    for name in ("gpu_probe", "hybrid_probe", "manifest", "checkpoint",
                 "shader", "output"):
        setattr(args, name, getattr(args, name).resolve())
    planned = []
    for index, rows in enumerate(args.rows):
        order = ["gpu", "hybrid"] if index % 2 == 0 else ["hybrid", "gpu"]
        planned.append({
            "rows": rows,
            "order": order,
            "commands": {
                variant: command_for(
                    args, variant, rows,
                    args.output / str(rows) / variant / "artifacts",
                )
                for variant in order
            },
        })
    if args.dry_run:
        print(json.dumps({
            "format": "turbocider-ltx-gemma-ane-mlp-plan-v1",
            "planned": planned,
        }, indent=2, sort_keys=True))
        return 0

    for path, label in (
        (args.gpu_probe, "--gpu-probe"),
        (args.hybrid_probe, "--hybrid-probe"),
        (args.manifest, "--manifest"),
        (args.checkpoint, "--checkpoint"),
        (args.shader, "--shader"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if not os.access(args.gpu_probe, os.X_OK):
        raise ValueError("--gpu-probe must be executable")
    if not os.access(args.hybrid_probe, os.X_OK):
        raise ValueError("--hybrid-probe must be executable")
    if args.output.exists():
        raise ValueError("--output must be a new path")

    args.output.mkdir(parents=True)
    report = {
        "format": "turbocider-ltx-gemma-ane-mlp-qualification-v1",
        "hardware": {
            "machine": platform.machine(),
            "chip": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=False,
            ).stdout.strip(),
        },
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "shader": str(args.shader),
        "layer": args.layer,
        "method": (
            "separate GPU-reference and GPU+ANE processes; deterministic "
            "BF16 input; process order reverses by row count; first pass and "
            "warm median separate; final MLP output compared after timing"
        ),
        "gates": {
            "min_warm_speedup": args.min_speedup,
            "max_relative_l2": args.max_relative_l2,
            "min_cosine": args.min_cosine,
            "max_relative_abs": args.max_relative_abs,
        },
        "cases": [],
    }
    report_path = args.output / "report.json"
    formats = {
        "gpu": GPU_PROBE_FORMAT,
        "hybrid": "turbocider-ltx-gemma-ane-mlp-probe-v1",
    }
    for item in planned:
        results = {}
        for variant in item["order"]:
            results[variant] = run_case(
                item["commands"][variant],
                args.output / str(item["rows"]) / variant / "artifacts",
                args.timeout, formats[variant],
            )
        quality = tensor_quality(results["hybrid"]["output"],
                                 results["gpu"]["output"])
        speedup = (results["gpu"]["warm_median_seconds"] /
                   results["hybrid"]["warm_median_seconds"])
        quality_passed = passes_quality_gate(quality, args)
        for result in results.values():
            result["output"] = str(result["output"])
        case = {
            "rows": item["rows"],
            "order": item["order"],
            "gpu": results["gpu"],
            "hybrid": results["hybrid"],
            "warm_speedup": speedup,
            "quality": quality,
            "quality_passed": quality_passed,
            "speed_passed": speedup >= args.min_speedup,
            "retain": quality_passed and speedup >= args.min_speedup,
        }
        report["cases"].append(case)
        write_report(report_path, report)
        print(json.dumps(case, sort_keys=True), flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
