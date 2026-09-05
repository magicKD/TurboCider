#!/usr/bin/env python3
"""Probe fixed-shape FP16 SDPA on CPU+ANE without GPU fallback.

This is a placement/performance gate for heterogeneous attention head
parallelism.  It intentionally contains only QK^T, scale, softmax and AV so a
slow result rejects full Core ML self-attention before exporting block weights.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import statistics
import tempfile
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4004)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--full-partial", action="store_true",
        help="include QKV, norm, gate and partial output projection",
    )
    parser.add_argument("--keep", type=Path)
    return parser.parse_args()


def build_model(rows: int, heads: int, head_dim: int):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    shape = (1, heads, rows, head_dim)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=shape, dtype=types.fp16),
            mb.TensorSpec(shape=shape, dtype=types.fp16),
            mb.TensorSpec(shape=shape, dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def program(query, key, value):
        scores = mb.matmul(
            x=query, y=key, transpose_y=True, name="attention_scores"
        )
        scores = mb.mul(
            x=scores,
            y=np.float16(1.0 / math.sqrt(head_dim)),
            name="scaled_scores",
        )
        probabilities = mb.softmax(x=scores, axis=-1, name="probabilities")
        return mb.matmul(x=probabilities, y=value, name="output")

    return ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )


def build_partial_model(rows: int, heads: int, head_dim: int):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    hidden = 4096
    shard = heads * head_dim
    rng = np.random.default_rng(7)

    def weight(output: int, input_: int, scale: float = 0.01):
        return np.ascontiguousarray(
            (rng.standard_normal((output, input_, 1, 1)) * scale)
            .astype(np.float16)
        )

    query_weight = weight(shard, hidden)
    key_weight = weight(shard, hidden)
    value_weight = weight(shard, hidden)
    output_weight = weight(hidden, shard)
    gate_weight = weight(heads, hidden)
    query_norm = np.ones((1, 1, heads, head_dim), dtype=np.float16)
    key_norm = np.ones((1, 1, heads, head_dim), dtype=np.float16)
    cosine = np.ones((1, heads, rows, head_dim // 2), dtype=np.float16)
    sine = np.zeros((1, heads, rows, head_dim // 2), dtype=np.float16)

    def token_heads(x, name: str):
        x = mb.squeeze(x=x, axes=[2], name=f"{name}_squeeze")
        x = mb.transpose(x=x, perm=[0, 2, 1], name=f"{name}_tokens")
        return mb.reshape(
            x=x, shape=(1, rows, heads, head_dim), name=f"{name}_heads"
        )

    def rms_norm(x, norm, name: str):
        square = mb.mul(x=x, y=x, name=f"{name}_square")
        mean = mb.reduce_mean(
            x=square, axes=[-1], keep_dims=True, name=f"{name}_mean"
        )
        inverse = mb.rsqrt(
            x=mb.add(x=mean, y=np.float16(1.0e-6)),
            name=f"{name}_inverse",
        )
        return mb.mul(
            x=mb.mul(x=x, y=inverse, name=f"{name}_normalized"),
            y=norm,
            name=name,
        )

    def rope(x, name: str):
        x = mb.transpose(x=x, perm=[0, 2, 1, 3], name=f"{name}_major")
        first, second = mb.split(
            x=x,
            split_sizes=[head_dim // 2, head_dim // 2],
            axis=3,
            name=f"{name}_split",
        )
        return mb.concat(
            values=[
                mb.sub(
                    x=mb.mul(x=first, y=cosine),
                    y=mb.mul(x=second, y=sine),
                ),
                mb.add(
                    x=mb.mul(x=first, y=sine),
                    y=mb.mul(x=second, y=cosine),
                ),
            ],
            axis=3,
            name=name,
        )

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, hidden, 1, rows), dtype=types.fp16)
        ],
        opset_version=ct.target.macOS15,
    )
    def program(x):
        query = token_heads(
            mb.conv(x=x, weight=query_weight, name="query_projection"),
            "query",
        )
        key = token_heads(
            mb.conv(x=x, weight=key_weight, name="key_projection"),
            "key",
        )
        value = token_heads(
            mb.conv(x=x, weight=value_weight, name="value_projection"),
            "value",
        )
        query = rope(rms_norm(query, query_norm, "query_norm"), "query_rope")
        key = rope(rms_norm(key, key_norm, "key_norm"), "key_rope")
        value = mb.transpose(x=value, perm=[0, 2, 1, 3], name="value_major")
        scores = mb.matmul(
            x=query, y=key, transpose_y=True, name="attention_scores"
        )
        probabilities = mb.softmax(
            x=mb.mul(
                x=scores,
                y=np.float16(1.0 / math.sqrt(head_dim)),
                name="scaled_scores",
            ),
            axis=-1,
            name="probabilities",
        )
        attended = mb.matmul(x=probabilities, y=value, name="attended")
        gate = mb.conv(x=x, weight=gate_weight, name="gate_logits")
        gate = mb.mul(
            x=mb.sigmoid(x=gate), y=np.float16(2.0), name="gate"
        )
        gate = mb.transpose(x=gate, perm=[0, 1, 3, 2], name="gate_major")
        attended = mb.mul(x=attended, y=gate, name="gated_attention")
        attended = mb.transpose(
            x=attended, perm=[0, 2, 1, 3], name="row_major_heads"
        )
        attended = mb.reshape(
            x=attended, shape=(1, rows, shard), name="row_major"
        )
        attended = mb.transpose(
            x=attended, perm=[0, 2, 1], name="channels_first"
        )
        attended = mb.expand_dims(
            x=attended, axes=[2], name="channels_first_4d"
        )
        return mb.conv(
            x=attended, weight=output_weight, name="partial_output"
        )

    return ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.heads <= 0 or args.head_dim <= 0:
        raise SystemExit("rows, heads and head-dim must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("warmup must be nonnegative and iterations positive")

    import coremltools as ct
    import numpy as np

    temporary = Path(tempfile.mkdtemp(prefix="ltx_ane_sdpa_"))
    package = temporary / "sdpa.mlpackage"
    try:
        build_started = time.perf_counter()
        builder = build_partial_model if args.full_partial else build_model
        builder(args.rows, args.heads, args.head_dim).save(package)
        build_seconds = time.perf_counter() - build_started

        load_started = time.perf_counter()
        model = ct.models.MLModel(
            str(package), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        load_seconds = time.perf_counter() - load_started

        rng = np.random.default_rng(42)
        if args.full_partial:
            shape = (1, 4096, 1, args.rows)
            features = {
                "x": (rng.standard_normal(shape) * 0.1).astype(np.float16)
            }
        else:
            shape = (1, args.heads, args.rows, args.head_dim)
            query = (rng.standard_normal(shape) * 0.1).astype(np.float16)
            key = (rng.standard_normal(shape) * 0.1).astype(np.float16)
            value = rng.standard_normal(shape).astype(np.float16)
            features = {"query": query, "key": key, "value": value}

        for _ in range(args.warmup):
            result = model.predict(features)

        timings = []
        result = None
        for _ in range(args.iterations):
            started = time.perf_counter()
            result = model.predict(features)
            timings.append((time.perf_counter() - started) * 1000.0)

        assert result is not None
        output = np.asarray(result[next(iter(result))])
        finite = bool(np.isfinite(output).all())
        timings.sort()
        p50 = statistics.median(timings)
        p95 = timings[min(len(timings) - 1, math.ceil(0.95 * len(timings)) - 1)]
        print(
            f"backend=coreml-cpu-and-ne graph="
            f"{'partial-self-attention' if args.full_partial else 'sdpa'} "
            f"rows={args.rows} heads={args.heads} "
            f"head_dim={args.head_dim} build_s={build_seconds:.3f} "
            f"load_s={load_seconds:.3f}"
        )
        print(
            f"warmup={args.warmup} iterations={args.iterations} "
            f"p50_ms={p50:.3f} p95_ms={p95:.3f} "
            f"finite={int(finite)} rms={float(np.sqrt(np.mean(output.astype(np.float32) ** 2))):.9g}"
        )
        if args.keep:
            if args.keep.exists():
                shutil.rmtree(args.keep)
            shutil.copytree(package, args.keep)
            print(f"artifact={args.keep}")
        if not finite:
            raise SystemExit(1)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
