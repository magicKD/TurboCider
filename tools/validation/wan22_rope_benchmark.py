#!/usr/bin/env python3
"""Measure Wan2.2 rotary-frequency construction on MLX GPU."""
import argparse
import json
from pathlib import Path
import time

import mlx.core as mx


def frequencies(frames, height, width, head_dim):
    spatial = 2 * (head_dim // 6)
    dims = (head_dim - 2 * spatial, spatial, spatial)
    positions = mx.arange(frames * height * width, dtype=mx.int32)
    cosines, sines = [], []
    for dim, inner, extent in zip(dims, (height * width, width, 1), (frames, height, width)):
        coordinate = mx.remainder(mx.floor_divide(positions, inner), extent).astype(mx.float32)
        inverse = 1 / mx.power(mx.array(10000., dtype=mx.float32), mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        angles = mx.expand_dims(coordinate, 1) * inverse
        cosines.append(mx.repeat(mx.cos(angles), 2, axis=1))
        sines.append(mx.repeat(mx.sin(angles), 2, axis=1))
    return mx.concatenate(cosines, axis=1), mx.concatenate(sines, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir() or min(args.frames, args.height, args.width, args.head_dim, args.repeats) <= 0:
        parser.error("output must be new and dimensions/repeats must be positive")
    spatial = 2 * (args.head_dim // 6)
    if args.head_dim % 2 or spatial == 0 or args.head_dim - 2 * spatial <= 0:
        parser.error("head dimension must have positive even temporal and spatial components")
    for _ in range(2):
        mx.eval(*frequencies(args.frames, args.height, args.width, args.head_dim))
    samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        cosine, sine = frequencies(args.frames, args.height, args.width, args.head_dim)
        mx.eval(cosine, sine)
        samples.append(time.perf_counter() - started)
    report = {"schema": "wan22-rope-benchmark-v1", "frames": args.frames, "height": args.height,
              "width": args.width, "head_dim": args.head_dim, "seconds": samples,
              "output_bytes": cosine.nbytes + sine.nbytes,
              "scope": "warm isolated MLX GPU frequency construction; not end-to-end generation"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
