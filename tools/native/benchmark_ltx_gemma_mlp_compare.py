#!/usr/bin/env python3
"""Compare staged and fused Gemma gated-MLP GPU graphs by token rows."""

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
    parser.add_argument("--probe", type=Path,
                        default=Path("build/native/ltx-gemma-mlp-probe"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shader", type=Path,
                        default=Path("build/native/ltx_shaders.metal"))
    parser.add_argument("--rows", type=int, nargs="+", default=DEFAULT_ROWS)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-speedup", type=float, default=1.1)
    parser.add_argument("--max-relative-l2", type=float, default=0.0005)
    parser.add_argument("--min-cosine", type=float, default=0.99999)
    parser.add_argument("--max-relative-abs", type=float, default=0.005)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (not args.rows or len(set(args.rows)) != len(args.rows) or
            any(row < 1 or row > 1024 for row in args.rows)):
        parser.error("--rows values must be unique and in 1...1024")
    validate_limits(parser, args)
    return args


def command_for(args: argparse.Namespace, rows: int, folder: Path) -> list[str]:
    return [str(args.probe), str(args.checkpoint), str(args.shader),
            str(rows), str(args.runs), str(folder)]


def run_case(command: list[str], folder: Path, timeout: int,
             fused: bool) -> dict:
    environment = os.environ.copy()
    environment.pop("TURBOCIDER_LTX_GEMMA_FUSED_MLP", None)
    if fused:
        environment["TURBOCIDER_LTX_GEMMA_FUSED_MLP"] = "1"
    return run_probe(
        command, folder, timeout, environment=environment, expected_format=GPU_PROBE_FORMAT,
        failure_label="Gemma MLP", invalid_label="Gemma MLP",
    )


def main() -> int:
    args = arguments()
    args.probe = args.probe.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.shader = args.shader.resolve()
    args.output = args.output.resolve()
    for path, label in ((args.probe, "--probe"),
                        (args.checkpoint, "--checkpoint"),
                        (args.shader, "--shader")):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if not os.access(args.probe, os.X_OK):
        raise ValueError("--probe must be executable")
    if args.output.exists():
        raise ValueError("--output must be a new path")

    planned = []
    for index, rows in enumerate(args.rows):
        order = ["staged", "fused"] if index % 2 == 0 else ["fused", "staged"]
        planned.append({
            "rows": rows, "order": order,
            "commands": {variant: command_for(
                args, rows, args.output / str(rows) / variant / "artifacts")
                for variant in order},
        })
    if args.dry_run:
        print(json.dumps({
            "format": "turbocider-ltx-gemma-mlp-compare-plan-v1",
            "planned": planned,
        }, indent=2, sort_keys=True))
        return 0

    args.output.mkdir(parents=True)
    report = {
        "format": "turbocider-ltx-gemma-mlp-compare-v1",
        "hardware": {
            "machine": platform.machine(),
            "chip": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=False,
            ).stdout.strip(),
        },
        "checkpoint": str(args.checkpoint), "shader": str(args.shader),
        "method": (
            "separate staged/fused processes; deterministic BF16 input; "
            "order reverses by row count; first pass and warm median separate; "
            "final MLP output compared after timing"
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
    for item in planned:
        results = {}
        for variant in item["order"]:
            results[variant] = run_case(
                item["commands"][variant],
                args.output / str(item["rows"]) / variant / "artifacts",
                args.timeout, variant == "fused",
            )
        quality = tensor_quality(results["fused"]["output"],
                                 results["staged"]["output"])
        speedup = (results["staged"]["warm_median_seconds"] /
                   results["fused"]["warm_median_seconds"])
        quality_passed = passes_quality_gate(quality, args)
        for result in results.values():
            result["output"] = str(result["output"])
        case = {
            "rows": item["rows"], "order": item["order"],
            "staged": results["staged"], "fused": results["fused"],
            "warm_speedup": speedup, "quality": quality,
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
