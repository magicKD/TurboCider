#!/usr/bin/env python3
"""Compare Wan2.2 dense SDPA with interleaved and contiguous head layouts."""
import argparse
import json
import math
from pathlib import Path
import time

import mlx.core as mx


def measure(operation, warmup, repeats):
    for _ in range(warmup):
        mx.eval(operation())
    samples, output = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        output = operation()
        mx.eval(output)
        samples.append(time.perf_counter() - started)
    return output, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=6400)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir() or min(args.tokens, args.heads, args.head_dim, args.repeats) <= 0 or args.warmup < 0:
        parser.error("output must be new and dimensions/repetitions valid")
    mx.random.seed(22)
    shape = (1, args.tokens, args.heads, args.head_dim)
    values = [(mx.random.normal(shape) * scale).astype(mx.bfloat16) for scale in (1., 1., .1)]
    mx.eval(*values)
    q, k, v = [x.transpose(0, 2, 1, 3) for x in values]
    mx.eval(q, k, v)
    scale = 1 / math.sqrt(args.head_dim)
    def attention(a, b, c):
        return mx.fast.scaled_dot_product_attention(a, b, c, scale=scale)
    reference, strided_times = measure(lambda: attention(q, k, v), args.warmup, args.repeats)
    started = time.perf_counter()
    qc, kc, vc = [mx.contiguous(x) for x in (q, k, v)]
    mx.eval(qc, kc, vc)
    preparation = time.perf_counter() - started
    contiguous, contiguous_times = measure(lambda: attention(qc, kc, vc), args.warmup, args.repeats)
    copied, copied_times = measure(lambda: attention(mx.contiguous(q), mx.contiguous(k), mx.contiguous(v)),
                                   args.warmup, args.repeats)
    results = {}
    for name, output, samples in (("precontiguous", contiguous, contiguous_times), ("copy_each_call", copied, copied_times)):
        diff = output.astype(mx.float32) - reference.astype(mx.float32)
        relative = mx.sqrt(mx.sum(diff * diff) / mx.maximum(mx.sum(reference.astype(mx.float32) ** 2), 1e-30))
        maximum = mx.max(mx.abs(diff))
        equal = mx.all(output == reference)
        mx.eval(relative, maximum, equal)
        results[name] = {"seconds": samples, "relative_l2": relative.item(), "max_abs": maximum.item(), "equal": bool(equal.item())}
    report = {"schema": "wan22-sdpa-layout-benchmark-v1", "tokens": args.tokens, "heads": args.heads,
              "head_dim": args.head_dim, "dtype": "bf16", "strided_seconds": strided_times,
              "preparation_seconds": preparation, "results": results,
              "scope": "warm isolated dense MLX SDPA; synthetic tensors; copy_each_call includes layout copies"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
