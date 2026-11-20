"""Common probe results and qualification gates for the Gemma MLP benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


DEFAULT_ROWS = (64, 128, 256, 512, 1024)
GPU_PROBE_FORMAT = "turbocider-ltx-gemma-mlp-gpu-probe-v1"


def validate_limits(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
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


def capture_probe(
    command: list[str], log_directory: Path, timeout: int,
    *, environment: dict[str, str],
) -> tuple[subprocess.CompletedProcess[str], int | None]:
    process = subprocess.run(
        ["/usr/bin/time", "-l", *command], capture_output=True, text=True,
        check=False, timeout=timeout, env=environment,
    )
    log_directory.mkdir(parents=True, exist_ok=True)
    (log_directory / "stdout.log").write_text(process.stdout)
    (log_directory / "stderr.log").write_text(process.stderr)
    peak = re.search(r"(\d+)\s+maximum resident set size", process.stderr)
    peak_rss_bytes = int(peak.group(1)) if peak else None
    process_record = {
        "command": command,
        "returncode": process.returncode,
        "process_peak_rss_bytes": peak_rss_bytes,
    }
    (log_directory / "process.json").write_text(
        json.dumps(process_record, indent=2, sort_keys=True) + "\n"
    )
    return process, peak_rss_bytes


def run_probe(command: list[str], folder: Path, timeout: int, *,
              environment: dict[str, str], expected_format: str,
              failure_label: str, invalid_label: str) -> dict:
    process, peak_rss_bytes = capture_probe(
        command, folder.parent, timeout, environment=environment
    )
    if process.returncode:
        raise RuntimeError(
            f"{failure_label} probe failed in {folder.parent}:\n{process.stderr[-4000:]}"
        )
    payload = json.loads(process.stdout)
    warm = payload.get("warm_seconds")
    if (payload.get("format") != expected_format or
            payload.get("output_all_finite") is not True or
            not isinstance(warm, list) or len(warm) != payload.get("warm_runs")):
        raise ValueError(f"invalid {invalid_label} probe output in {folder.parent}")
    return {
        "first_seconds": payload["first_seconds"],
        "warm_seconds": warm,
        "warm_median_seconds": statistics.median(warm),
        "resident_weight_bytes": payload["resident_weight_bytes"],
        "workspace_bytes": payload["workspace_bytes"],
        "process_peak_rss_bytes": peak_rss_bytes,
        "output": folder / "output.bf16",
    }


def passes_quality_gate(quality: dict, args: argparse.Namespace) -> bool:
    return (
        quality.get("shape_match", False) and quality.get("all_finite", False) and
        quality["relative_l2"] <= args.max_relative_l2 and
        quality["cosine"] >= args.min_cosine and
        quality["relative_max_abs"] <= args.max_relative_abs
    )


def write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
