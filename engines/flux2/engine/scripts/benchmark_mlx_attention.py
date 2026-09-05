#!/usr/bin/env python3
"""Benchmark MLX fused SDPA with the same tensor shape as the Core ML probe."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=1088)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.sequence, args.heads, args.head_dim, args.iterations) <= 0:
        parser.error("shape and iteration values must be positive")

    shape = (1, args.heads, args.sequence, args.head_dim)
    mx.random.seed(args.seed)
    query = (mx.random.normal(shape) * 0.1).astype(mx.float16)
    key = (mx.random.normal(shape) * 0.1).astype(mx.float16)
    value = (mx.random.normal(shape) * 0.1).astype(mx.float16)
    mx.eval(query, key, value)
    scale = args.head_dim**-0.5

    def attention(q, k, v):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    operation = mx.compile(attention) if args.compiled else attention
    for _ in range(args.warmup):
        mx.eval(operation(query, key, value))

    elapsed = []
    output = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        output = operation(query, key, value)
        mx.eval(output)
        elapsed.append((time.perf_counter() - started) * 1000)

    output_array = np.asarray(output)
    report = {
        "operator": "mlx.fast.scaled_dot_product_attention",
        "device": "Metal",
        "compiled": args.compiled,
        "shape": list(shape),
        "resident_input_output_bytes": int(4 * np.prod(shape) * 2),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "elapsed_ms": elapsed,
        "median_ms": statistics.median(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "output_finite": bool(np.isfinite(output_array).all()),
        "output_rms": float(np.sqrt(np.mean(np.square(output_array.astype(np.float32))))),
        "scope": "Standalone MLX fused SDPA with evaluated inputs and synchronized output",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
