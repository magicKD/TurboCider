#!/usr/bin/env python3
"""Compare staged and fused Gemma gated-MLP GPU graphs by token rows."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path


DEFAULT_ROWS = (64, 128, 256, 512, 1024)
INVALID_METRIC = 1e30


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
    if args.runs < 1 or args.runs > 50:
        parser.error("--runs must be in 1...50")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if not math.isfinite(args.min_speedup) or args.min_speedup <= 0:
        parser.error("--min-speedup must be finite and positive")
    if not math.isfinite(args.max_relative_l2) or args.max_relative_l2 < 0:
        parser.error("--max-relative-l2 must be finite and nonnegative")
    if not math.isfinite(args.min_cosine) or not -1 <= args.min_cosine <= 1:
        parser.error("--min-cosine must be in [-1, 1]")
    if not math.isfinite(args.max_relative_abs) or args.max_relative_abs < 0:
        parser.error("--max-relative-abs must be finite and nonnegative")
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
    process = subprocess.run(
        ["/usr/bin/time", "-l", *command], capture_output=True, text=True,
        check=False, timeout=timeout, env=environment,
    )
    folder.parent.mkdir(parents=True, exist_ok=True)
    (folder.parent / "stdout.log").write_text(process.stdout)
    (folder.parent / "stderr.log").write_text(process.stderr)
    peak = re.search(r"(\d+)\s+maximum resident set size", process.stderr)
    record = {
        "command": command,
        "returncode": process.returncode,
        "process_peak_rss_bytes": int(peak.group(1)) if peak else None,
    }
    (folder.parent / "process.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    if process.returncode:
        raise RuntimeError(
            f"Gemma MLP probe failed in {folder.parent}:\n{process.stderr[-4000:]}"
        )
    payload = json.loads(process.stdout)
    warm = payload.get("warm_seconds")
    if (payload.get("format") != "turbocider-ltx-gemma-mlp-gpu-probe-v1" or
            payload.get("output_all_finite") is not True or
            not isinstance(warm, list) or len(warm) != payload.get("warm_runs")):
        raise ValueError(f"invalid Gemma MLP probe output in {folder.parent}")
    return {
        "first_seconds": payload["first_seconds"],
        "warm_seconds": warm,
        "warm_median_seconds": statistics.median(warm),
        "resident_weight_bytes": payload["resident_weight_bytes"],
        "workspace_bytes": payload["workspace_bytes"],
        "process_peak_rss_bytes": record["process_peak_rss_bytes"],
        "output": folder / "output.bf16",
    }


def bf16(path: Path):
    import numpy as np

    words = np.fromfile(path, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def tensor_quality(candidate_path: Path, reference_path: Path) -> dict:
    import numpy as np

    candidate = bf16(candidate_path).astype(np.float64)
    reference = bf16(reference_path).astype(np.float64)
    if candidate.shape != reference.shape:
        return {"shape_match": False, "relative_l2": INVALID_METRIC,
                "cosine": -1.0, "max_abs": INVALID_METRIC,
                "relative_max_abs": INVALID_METRIC}
    finite = bool(np.isfinite(candidate).all() and np.isfinite(reference).all())
    if not finite:
        return {"shape_match": True, "all_finite": False,
                "relative_l2": INVALID_METRIC, "cosine": -1.0,
                "max_abs": INVALID_METRIC,
                "relative_max_abs": INVALID_METRIC}
    delta = candidate - reference
    delta2 = float(np.dot(delta, delta))
    candidate2 = float(np.dot(candidate, candidate))
    reference2 = float(np.dot(reference, reference))
    denominator = math.sqrt(candidate2 * reference2)
    maximum = float(np.max(np.abs(delta), initial=0.0))
    amplitude = float(np.max(np.abs(reference), initial=0.0))
    return {
        "shape_match": True,
        "all_finite": True,
        "relative_l2": math.sqrt(delta2 / reference2)
        if reference2 > 0 else (0.0 if delta2 == 0 else INVALID_METRIC),
        "cosine": max(-1.0, min(1.0, float(np.dot(candidate, reference)) /
                                 denominator if denominator > 0 else
                                 (1.0 if candidate2 == reference2 == 0 else 0.0))),
        "max_abs": maximum,
        "relative_max_abs": maximum / amplitude
        if amplitude > 0 else (0.0 if maximum == 0 else INVALID_METRIC),
    }


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
        quality_passed = (
            quality.get("shape_match", False) and quality.get("all_finite", False) and
            quality["relative_l2"] <= args.max_relative_l2 and
            quality["cosine"] >= args.min_cosine and
            quality["relative_max_abs"] <= args.max_relative_abs
        )
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
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(report_path)
        print(json.dumps(case, sort_keys=True), flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
