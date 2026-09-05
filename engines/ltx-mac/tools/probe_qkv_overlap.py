#!/usr/bin/env python3
"""Measure simultaneous Metal-GPU and Core ML/ANE QKV throughput."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gpu-bench", type=Path, default=Path("build/bench_qkv"))
    parser.add_argument(
        "--prefix",
        default="model.diffusion_model.transformer_blocks.0.attn1",
    )
    parser.add_argument("--ane-rows", type=int, default=1500)
    parser.add_argument("--gpu-rows", type=int, default=2504)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def main() -> None:
    args = parse_args()
    if args.ane_rows <= 0 or args.gpu_rows <= 0:
        raise SystemExit("row counts must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("invalid warmup/iterations")

    import coremltools as ct
    import numpy as np

    runtime = ct.models.MLModel(
        str(args.model), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    rng = np.random.default_rng(42)
    features = {
        "x": (rng.standard_normal((1, 4096, 1, args.ane_rows)) * 0.1)
        .astype(np.float16)
    }
    result = None
    for _ in range(args.warmup + 1):
        result = runtime.predict(features)

    environment = os.environ.copy()
    environment.pop("LTX_QKV_BACKEND", None)
    command = [
        str(args.gpu_bench),
        str(args.checkpoint),
        args.prefix,
        str(args.gpu_rows),
        str(args.warmup),
        str(args.iterations),
    ]
    overlap_started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    ane_timings = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        result = runtime.predict(features)
        ane_timings.append((time.perf_counter() - started) * 1000.0)
    stdout, stderr = process.communicate()
    overlap_ms = (time.perf_counter() - overlap_started) * 1000.0
    if process.returncode:
        raise SystemExit(
            f"GPU benchmark failed ({process.returncode})\n{stdout}\n{stderr}"
        )
    match = re.search(r"p50_ms=([0-9.]+)", stdout)
    if not match:
        raise SystemExit(f"cannot parse GPU timing\n{stdout}")
    gpu_p50 = float(match.group(1))
    ane_p50 = statistics.median(ane_timings)
    ane_p95 = percentile(ane_timings, 0.95)
    assert result is not None
    finite = all(
        bool(np.isfinite(np.asarray(value)).all()) for value in result.values()
    )
    print(stdout, end="")
    print(
        f"overlap=metal-gpu+coreml-ane gpu_rows={args.gpu_rows} "
        f"ane_rows={args.ane_rows} iterations={args.iterations}"
    )
    print(
        f"contended_gpu_p50_ms={gpu_p50:.3f} "
        f"contended_ane_p50_ms={ane_p50:.3f} "
        f"contended_ane_p95_ms={ane_p95:.3f} "
        f"process_overlap_wall_ms={overlap_ms:.3f} "
        f"amortized_pair_wall_ms={overlap_ms / args.iterations:.3f} "
        f"finite={int(finite)}"
    )
    if not finite:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
