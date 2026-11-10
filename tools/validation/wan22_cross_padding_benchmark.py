#!/usr/bin/env python3
"""Measure whether collapsing identical padded cross-attention rows helps.

This is an isolated, approximate experiment, not a production executor. It
uses captured first-block Q/K/V, repeats Q to the requested token count, and
only proceeds if all discarded K/V rows are bit-identical to the retained
padding row. Softmax reduction and mask rounding may still change outputs.
"""
import argparse
import json
import math
from pathlib import Path
import time

import mlx.core as mx
from wan22_cross_padding import padding_biases


def measure(operation, warmup, repeats):
    for _ in range(warmup):
        mx.eval(operation())
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        mx.eval(result)
        values.append(time.perf_counter() - start)
    return result, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=6400)
    parser.add_argument("--valid-rows", type=int, default=53)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--mask-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--pad-copies", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be a new file in an existing directory")
    if min(args.tokens, args.valid_rows, args.heads, args.repeats, args.pad_copies) <= 0 or args.warmup < 0 or args.pad_copies > 64:
        parser.error("invalid geometry or repetitions")
    captured = mx.load(str(args.capture.resolve(strict=True)))
    q, k, v = (captured[name] for name in ("cross_q_norm", "cross_k_norm", "cross_v"))
    if (q.ndim != 3 or q.shape[0] != 1 or k.ndim != 3 or k.shape != v.shape or
            k.shape[0] != 1 or q.shape[-1] != k.shape[-1] or q.shape[-1] % args.heads or
            q.dtype != mx.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype or
            args.valid_rows >= k.shape[1] or args.pad_copies > k.shape[1] - args.valid_rows):
        raise ValueError("unexpected captured cross-attention geometry or dtype")
    tail_key, tail_value = k[:, args.valid_rows:, :], v[:, args.valid_rows:, :]
    same_key = mx.all(tail_key == tail_key[:, :1, :])
    same_value = mx.all(tail_value == tail_value[:, :1, :])
    mx.eval(same_key, same_value)
    if not same_key.item() or not same_value.item():
        raise ValueError("discarded padding rows are not all identical")
    rows, heads, dim = args.tokens, args.heads, q.shape[-1] // args.heads
    copies = (rows + q.shape[1] - 1) // q.shape[1]
    q = mx.tile(q, (1, copies, 1))[:, :rows, :]

    def split(x):
        return x.reshape(1, x.shape[1], heads, dim).transpose(0, 2, 1, 3)

    q, k, v = split(q), split(k), split(v)
    reduced_k = k[:, :, :args.valid_rows + args.pad_copies, :]
    reduced_v = v[:, :, :args.valid_rows + args.pad_copies, :]
    padding_count = k.shape[2] - args.valid_rows
    # For several copies, combine nearby BF16 biases to approximate the
    # original repeated rows' total mathematical softmax mass more closely.
    biases, mass_error = padding_biases(padding_count, args.pad_copies)
    if args.mask_dtype == "fp32":
        biases = [math.log(padding_count / args.pad_copies)] * args.pad_copies
        mass_error = 0.
    mask_dtype = mx.float32 if args.mask_dtype == "fp32" else mx.bfloat16
    mask = mx.array([0.] * args.valid_rows + biases, dtype=mask_dtype)
    mx.eval(q, k, v, reduced_k, reduced_v, mask)
    scale = 1. / math.sqrt(dim)
    baseline, baseline_seconds = measure(
        lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale), args.warmup, args.repeats)
    report = {"schema": "wan22-cross-padding-benchmark-v1", "tokens": rows,
              "text_rows": k.shape[2], "valid_rows": args.valid_rows,
              "padding_rows": padding_count, "retained_padding_rows": args.pad_copies,
              "theoretical_relative_mass_error": mass_error, "baseline_seconds": baseline_seconds,
              "scope": "captured block-0 Q/K/V, repeated queries, warm isolated attention; not a video test"}
    try:
        candidate, candidate_seconds = measure(
            lambda: mx.fast.scaled_dot_product_attention(q, reduced_k, reduced_v, scale=scale, mask=mask),
            args.warmup, args.repeats)
        delta = baseline.astype(mx.float32) - candidate.astype(mx.float32)
        relative = mx.sqrt(mx.sum(delta * delta) /
                           mx.maximum(mx.sum(baseline.astype(mx.float32) ** 2), 1e-30))
        maximum, changed = mx.max(mx.abs(delta)), mx.sum(delta != 0)
        mx.eval(relative, maximum, changed)
        report.update(candidate_seconds=candidate_seconds, mask_dtype=str(mask.dtype),
                      relative_l2=relative.item(), max_abs=maximum.item(), changed_values=changed.item())
    except ValueError as error:
        report["candidate_error"] = str(error)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
