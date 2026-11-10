#!/usr/bin/env python3
"""Prototype paired Wan2.2 RoPE plus head-major packing on MLX Metal."""
import argparse
import json
import math
from pathlib import Path
import time

import mlx.core as mx


SOURCE = r"""
#pragma clang fp contract(off)
uint pair = thread_position_in_grid.x;
if (pair >= PAIRS) return;
uint d = (pair % (D / 2)) * 2;
uint temp = pair / (D / 2);
uint token = temp % N;
uint head = temp / N;
uint src = (token * H + head) * D + d;
uint dst = (head * N + token) * D + d;
float c0 = cosine[token * D + d], c1 = cosine[token * D + d + 1];
float s0 = sine[token * D + d], s1 = sine[token * D + d + 1];
float qa = float(q[src]), qb = float(q[src + 1]);
float ka = float(k[src]), kb = float(k[src + 1]);
qo[dst] = T(qa * c0 - qb * s0);
qo[dst + 1] = T(qb * c1 + qa * s1);
ko[dst] = T(ka * c0 - kb * s0);
ko[dst + 1] = T(kb * c1 + ka * s1);
vo[dst] = v[src];
vo[dst + 1] = v[src + 1];
"""


def measure(operation, warmup, repeats):
    for _ in range(warmup):
        value = operation()
        mx.eval(*value) if isinstance(value, tuple) else mx.eval(value)
    samples, output = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        output = operation()
        mx.eval(*output) if isinstance(output, tuple) else mx.eval(output)
        samples.append(time.perf_counter() - started)
    return output, samples


def metrics(reference, candidate):
    difference = reference.astype(mx.float32) - candidate.astype(mx.float32)
    relative = mx.sqrt(mx.sum(difference * difference) /
                       mx.maximum(mx.sum(reference.astype(mx.float32) ** 2), 1e-30))
    maximum = mx.max(mx.abs(difference))
    changed = mx.sum(difference != 0)
    mx.eval(relative, maximum, changed)
    return {"relative_l2": relative.item(), "max_abs": maximum.item(), "changed_values": changed.item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=6400)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir() or min(args.tokens, args.heads, args.head_dim, args.repeats) <= 0 or args.head_dim % 2 or args.warmup < 0:
        parser.error("invalid output, geometry or repetitions")
    n, h, d = args.tokens, args.heads, args.head_dim
    mx.random.seed(22)
    q, k, v = [mx.random.normal((1, n, h * d)).astype(mx.bfloat16) for _ in range(3)]
    angle = mx.random.normal((n, d // 2))
    cosine, sine = mx.repeat(mx.cos(angle), 2, axis=1), mx.repeat(mx.sin(angle), 2, axis=1)
    mx.eval(q, k, v, cosine, sine)

    def rotate(value):
        x = value.astype(mx.float32).reshape(1, n, h, d)
        pair = x.reshape(1, n, h, d // 2, 2)
        even, odd = mx.split(pair, 2, axis=-1)
        rotated = mx.concatenate((-odd, even), axis=-1).reshape(1, n, h, d)
        c, s = cosine.reshape(1, n, 1, d), sine.reshape(1, n, 1, d)
        return (x * c + rotated * s).astype(mx.bfloat16).transpose(0, 2, 1, 3)

    def ordinary():
        return rotate(q), rotate(k), v.reshape(1, n, h, d).transpose(0, 2, 1, 3)

    kernel = mx.fast.metal_kernel(name="tc_wan22_rope_qkv_profile", input_names=["q", "k", "v", "cosine", "sine"],
                                  output_names=["qo", "ko", "vo"], source=SOURCE)
    pairs = n * h * (d // 2)
    def fused():
        return tuple(kernel(inputs=[q, k, v, cosine, sine], template=[("T", mx.bfloat16), ("N", n), ("H", h), ("D", d), ("PAIRS", pairs)],
                            grid=(pairs, 1, 1), threadgroup=(256, 1, 1),
                            output_shapes=[(1, h, n, d)] * 3, output_dtypes=[mx.bfloat16] * 3))

    ordinary_values, ordinary_times = measure(ordinary, args.warmup, args.repeats)
    fused_values, fused_times = measure(fused, args.warmup, args.repeats)
    scale = 1 / math.sqrt(d)
    def attention(values):
        return mx.fast.scaled_dot_product_attention(*values, scale=scale)
    baseline, baseline_times = measure(lambda: attention(ordinary()), args.warmup, args.repeats)
    optimized, optimized_times = measure(lambda: attention(fused()), args.warmup, args.repeats)
    report = {"schema": "wan22-rope-fusion-benchmark-v1", "tokens": n, "heads": h, "head_dim": d,
              "scope": "warm synthetic BF16 Q/K/V; paired RoPE and packing; optional dense SDPA",
              "ordinary_rope_seconds": ordinary_times, "fused_rope_seconds": fused_times,
              "ordinary_with_sdpa_seconds": baseline_times, "fused_with_sdpa_seconds": optimized_times,
              "q_error": metrics(ordinary_values[0], fused_values[0]),
              "k_error": metrics(ordinary_values[1], fused_values[1]),
              "v_error": metrics(ordinary_values[2], fused_values[2]),
              "attention_error": metrics(baseline, optimized)}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
