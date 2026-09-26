#!/usr/bin/env python3
"""Measure the isolated Gemma4 layer-0 gated MLP GPU baseline by token rows.

This is a synthetic-input diagnostic. It deliberately measures only the
resident ConvRot INT8 gate/up -> GELU/product -> down branch and is not an
end-to-end Gemma quality or ANE qualification report.
"""

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
HIDDEN = 3840
INTERMEDIATE = 15360


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
    parser.add_argument(
        "--fused-mlp", action="store_true",
        help="use the fused MPSGraph gated-MLP candidate",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (not args.rows or len(set(args.rows)) != len(args.rows) or
            any(row < 1 or row > 1024 for row in args.rows)):
        parser.error("--rows values must be unique and in 1...1024")
    if args.runs < 1 or args.runs > 50:
        parser.error("--runs must be in 1...50")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    return args


def parse_peak_rss(stderr: str):
    match = re.search(r"(\d+)\s+maximum resident set size", stderr)
    return int(match.group(1)) if match else None


def run_case(args: argparse.Namespace, rows: int, folder: Path) -> dict:
    command = [
        str(args.probe), str(args.checkpoint), str(args.shader),
        str(rows), str(args.runs), str(folder),
    ]
    environment = os.environ.copy()
    environment.pop("TURBOCIDER_LTX_GEMMA_FUSED_MLP", None)
    if args.fused_mlp:
        environment["TURBOCIDER_LTX_GEMMA_FUSED_MLP"] = "1"
    process = subprocess.run(
        ["/usr/bin/time", "-l", *command],
        capture_output=True, text=True, check=False, timeout=args.timeout,
        env=environment,
    )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "stdout.log").write_text(process.stdout)
    (folder / "stderr.log").write_text(process.stderr)
    record = {
        "command": command,
        "returncode": process.returncode,
        "process_peak_rss_bytes": parse_peak_rss(process.stderr),
    }
    (folder / "process.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    if process.returncode:
        raise RuntimeError(
            f"Gemma MLP probe failed for rows={rows}:\n{process.stderr[-4000:]}"
        )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"probe produced unexpected stdout for rows={rows}")
    payload = json.loads(lines[0])
    if payload.get("format") != "turbocider-ltx-gemma-mlp-gpu-probe-v1":
        raise ValueError(f"probe format mismatch for rows={rows}")
    if (payload.get("rows") != rows or payload.get("hidden") != HIDDEN or
            payload.get("intermediate") != INTERMEDIATE or
            payload.get("layer") != 0 or
            payload.get("warm_runs") != args.runs or
            payload.get("synthetic_input") is not True or
            payload.get("output_all_finite") is not True):
        raise ValueError(f"probe geometry/finite-output mismatch for rows={rows}")
    warm = payload.get("warm_seconds")
    if not isinstance(warm, list) or len(warm) != args.runs:
        raise ValueError(f"probe warm sample count mismatch for rows={rows}")
    if (not all(isinstance(sample, (int, float)) and math.isfinite(sample) and
                sample > 0 for sample in warm) or
            not isinstance(payload.get("first_seconds"), (int, float)) or
            not math.isfinite(payload["first_seconds"]) or
            payload["first_seconds"] <= 0):
        raise ValueError(f"probe timing sample mismatch for rows={rows}")
    payload["warm_median_seconds"] = statistics.median(warm)
    payload["process_peak_rss_bytes"] = record["process_peak_rss_bytes"]
    return payload


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
        raise ValueError("--output must be a new path so prior evidence is preserved")

    planned = [
        {"rows": rows, "command": [
            str(args.probe), str(args.checkpoint), str(args.shader),
            str(rows), str(args.runs), str(args.output / str(rows)),
        ]}
        for rows in args.rows
    ]
    if args.dry_run:
        print(json.dumps({
            "format": "turbocider-ltx-gemma-mlp-sweep-plan-v1",
            "checkpoint": str(args.checkpoint),
            "shader": str(args.shader),
            "layer": 0,
            "rows": args.rows,
            "runs": args.runs,
            "fused_mlp": args.fused_mlp,
            "planned": planned,
        }, indent=2, sort_keys=True))
        return 0

    args.output.mkdir(parents=True)
    report = {
        "format": "turbocider-ltx-gemma-mlp-gpu-sweep-v1",
        "hardware": {
            "machine": platform.machine(),
            "chip": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=False,
            ).stdout.strip(),
        },
        "checkpoint": str(args.checkpoint),
        "shader": str(args.shader),
        "layer": 0,
        "rows": args.rows,
        "runs": args.runs,
        "synthetic_input": True,
        "fused_mlp": args.fused_mlp,
        "quality": "not_applicable_isolated_gpu_baseline",
        "cases": [],
    }
    report_path = args.output / "report.json"
    for rows in args.rows:
        payload = run_case(args, rows, args.output / str(rows))
        report["cases"].append(payload)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True), flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
