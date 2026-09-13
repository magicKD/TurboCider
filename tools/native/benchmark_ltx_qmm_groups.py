#!/usr/bin/env python3
"""Measure MLX affine-Q8 group layouts on the real LTX video GEMM shapes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx


SHAPES = (
    (1001, 4096, 4096, "stage1_video_projection"),
    (1001, 4096, 16384, "stage1_video_ffn_in"),
    (1001, 16384, 4096, "stage1_video_ffn_out"),
    (4004, 4096, 4096, "stage2_video_projection"),
    (4004, 4096, 16384, "stage2_video_ffn_in"),
    (4004, 16384, 4096, "stage2_video_ffn_out"),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--groups", type=int, nargs="+", default=(32, 64, 128))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    if any(group not in (32, 64, 128) for group in args.groups):
        parser.error("groups must be selected from 32, 64, and 128")
    return args


def packed_weight(rows: int, columns: int) -> mx.array:
    # The value pattern is deterministic and spans all uint8 values. ConvRot
    # checkpoints already store one signed INT8 scale per output row; the MLX
    # affine representation shifts q by +128 and repeats that same scale for
    # every group, so the packed bytes are independent of group size.
    values = mx.arange(rows * columns, dtype=mx.uint32)
    values = ((values * 73 + 19) & 255).astype(mx.uint8)
    return values.reshape(rows, columns).view(mx.uint32)


def one_shape(m: int, k: int, n: int, label: str,
              groups: list[int], warmup: int, iterations: int) -> dict:
    x = mx.random.normal((m, k)).astype(mx.bfloat16)
    weight = packed_weight(n, k)
    row_scale = mx.linspace(0.0005, 0.01, n, dtype=mx.float32).reshape(n, 1)
    mx.eval(x, weight, row_scale)
    rows = {}
    reference = None
    for group in groups:
        scales = mx.repeat(row_scale, k // group, axis=1)
        biases = scales * mx.array(-128.0, dtype=mx.float32)
        mx.eval(scales, biases)
        for _ in range(warmup):
            value = mx.quantized_matmul(
                x, weight, scales, biases, transpose=True,
                group_size=group, bits=8, mode="affine")
            mx.eval(value)
        timings = []
        value = None
        for _ in range(iterations):
            started = time.perf_counter()
            value = mx.quantized_matmul(
                x, weight, scales, biases, transpose=True,
                group_size=group, bits=8, mode="affine")
            mx.eval(value)
            timings.append(time.perf_counter() - started)
        assert value is not None
        digest_value = mx.sum(value.astype(mx.float32))
        mx.eval(digest_value)
        checksum = float(digest_value.item())
        if reference is None:
            reference = value
            max_abs = 0.0
        else:
            difference = mx.max(mx.abs(
                value.astype(mx.float32) - reference.astype(mx.float32)))
            mx.eval(difference)
            max_abs = float(difference.item())
        rows[str(group)] = {
            "seconds": timings,
            "median_seconds": statistics.median(timings),
            "checksum": checksum,
            "max_abs_vs_first_group": max_abs,
        }
        del scales, biases, value
        mx.clear_cache()
    fastest = min(rows, key=lambda group: rows[group]["median_seconds"])
    return {
        "label": label,
        "shape": [m, k, n],
        "groups": rows,
        "fastest_group": int(fastest),
    }


def main() -> int:
    args = arguments()
    mx.set_default_device(mx.gpu)
    report = {
        "format": "turbocider-ltx-qmm-group-sweep-v1",
        "hardware": {
            "machine": platform.machine(),
            "chip": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=False,
            ).stdout.strip(),
        },
        "mlx_device": str(mx.default_device()),
        "mlx_version": importlib.metadata.version("mlx"),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": [],
    }
    for shape in SHAPES:
        result = one_shape(*shape, list(args.groups), args.warmup, args.iterations)
        report["results"].append(result)
        print(json.dumps(result), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
