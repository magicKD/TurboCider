#!/usr/bin/env python3
"""Benchmark the public Core ML iOS18 SDPA op as an ANE feasibility gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=1088)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--verify-queries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.sequence, args.heads, args.head_dim, args.iterations) <= 0:
        parser.error("shape and iteration values must be positive")
    if not 0 <= args.verify_queries <= args.sequence:
        parser.error("verify-queries must be between zero and sequence")

    import coremltools as ct
    from coremltools.converters.mil.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    shape = (1, args.heads, args.sequence, args.head_dim)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=shape, dtype=types.fp16),
            mb.TensorSpec(shape=shape, dtype=types.fp16),
            mb.TensorSpec(shape=shape, dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def program(query, key, value):
        return mb.scaled_dot_product_attention(query=query, key=key, value=value)

    converted = ct.convert(
        program,
        source="milinternal",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    if args.save_package is not None:
        args.save_package.parent.mkdir(parents=True, exist_ok=True)
        converted.save(args.save_package)

    generator = np.random.default_rng(args.seed)
    inputs = {
        "query": generator.normal(0, 0.1, shape).astype(np.float16),
        "key": generator.normal(0, 0.1, shape).astype(np.float16),
        "value": generator.normal(0, 0.1, shape).astype(np.float16),
    }
    for _ in range(args.warmup):
        converted.predict(inputs)
    elapsed = []
    output = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        output = converted.predict(inputs)
        elapsed.append((time.perf_counter() - started) * 1000)
    output_array = next(iter(output.values()))
    reference_mae = None
    reference_max_error = None
    reference_indices = []
    if args.verify_queries:
        reference_indices = np.linspace(0, args.sequence - 1, args.verify_queries, dtype=np.int64).tolist()
        query = np.take(inputs["query"], reference_indices, axis=2).astype(np.float32)
        key = inputs["key"].astype(np.float32)
        value = inputs["value"].astype(np.float32)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scores = np.matmul(query, np.swapaxes(key, -1, -2)) / math.sqrt(args.head_dim)
            scores -= np.max(scores, axis=-1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
            reference = np.matmul(probabilities, value)
        actual = np.take(output_array, reference_indices, axis=2).astype(np.float32)
        difference = actual - reference
        reference_mae = float(np.mean(np.abs(difference)))
        reference_max_error = float(np.max(np.abs(difference)))
    report = {
        "operator": "coreml.scaled_dot_product_attention",
        "compute_units": "CPUAndNeuralEngine",
        "shape": list(shape),
        "input_output_bytes": int(4 * np.prod(shape) * 2),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "elapsed_ms": elapsed,
        "median_ms": statistics.median(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "output_finite": bool(np.isfinite(output_array).all()),
        "output_rms": float(np.sqrt(np.mean(np.square(output_array.astype(np.float32))))),
        "output_abs_max": float(np.max(np.abs(output_array.astype(np.float32)))),
        "reference_queries": args.verify_queries,
        "reference_query_indices": reference_indices,
        "reference_mae": reference_mae,
        "reference_max_error": reference_max_error,
        "scope": "Core ML Python provider path; device attribution still requires Instruments",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
