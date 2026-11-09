#!/usr/bin/env python3
"""Compare a real Wan2.2 projection with experimental MLX affine W8/W4.

The benchmark measures warm GPU operations only. Quantization time, numerical
error, and memory are reported separately; it does not alter a checkpoint or
claim end-to-end video quality.
"""
import argparse
import json
from pathlib import Path
import time

import mlx.core as mx


def timed(operation, warmup, repeats):
    for _ in range(warmup):
        mx.eval(operation())
    samples = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = operation()
        mx.eval(output)
        samples.append(time.perf_counter() - started)
    return output, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefix", default="blocks.0.ffn.net.0.proj")
    parser.add_argument("--rows", type=int, default=6400)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=64, choices=(32, 64, 128))
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be a new file in an existing directory")
    if args.rows <= 0 or args.warmup < 0 or args.repeats < 1:
        parser.error("invalid benchmark repetition or row count")
    checkpoint = args.checkpoint.resolve(strict=True)
    tensors = mx.load(str(checkpoint))
    weight = tensors[args.prefix + ".weight"].astype(mx.bfloat16)
    if weight.ndim != 2 or weight.shape[1] % args.group_size:
        raise ValueError("projection shape is incompatible with affine quantization")
    bias = tensors[args.prefix + ".bias"].astype(mx.bfloat16)
    mx.random.seed(22)
    x = (mx.random.normal((1, args.rows, weight.shape[1])) * .1).astype(mx.bfloat16)
    mx.eval(weight, bias, x)
    reference, dense_times = timed(lambda: mx.addmm(bias, x, weight.T), args.warmup, args.repeats)
    report = {"schema": "wan22-linear-benchmark-v1", "prefix": args.prefix,
              "shape": {"rows": args.rows, "in": weight.shape[1], "out": weight.shape[0]},
              "dtype": "bf16", "warmup": args.warmup, "repeats": args.repeats,
              "scope": "warm isolated GPU projection; excludes loading, quantization and video generation",
              "dense_seconds": dense_times, "dense_weight_bytes": weight.nbytes,
              "quantized": {}}
    for bits in (8, 4):
        started = time.perf_counter()
        packed = mx.quantize(weight.astype(mx.float16), group_size=args.group_size, bits=bits, mode="affine")
        mx.eval(*packed)
        quantize_seconds = time.perf_counter() - started
        def project():
            product = mx.quantized_matmul(x, packed[0], packed[1], packed[2], transpose=True,
                                          group_size=args.group_size, bits=bits, mode="affine")
            return product + bias
        output, samples = timed(project, args.warmup, args.repeats)
        diff = output.astype(mx.float32) - reference.astype(mx.float32)
        error = mx.sqrt(mx.sum(diff * diff) / mx.maximum(mx.sum(reference.astype(mx.float32) ** 2), 1e-30))
        maximum = mx.max(mx.abs(diff))
        mx.eval(error, maximum)
        report["quantized"][str(bits)] = {"seconds": samples, "quantize_seconds": quantize_seconds,
            "weight_bytes": sum(item.nbytes for item in packed), "relative_l2": error.item(), "max_abs": maximum.item()}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
