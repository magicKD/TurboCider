#!/usr/bin/env python3
"""Small standard-library CPU oracle for channel/row split SwiGLU + LoRA.

Python float arithmetic only. No GPU/ANE, no quantization, no latency claims.
Weights use checkpoint [out,in] order. Down bias/residual are added once.
"""
import json
import math
import random


def linear(rows, weight):
    if not rows or not weight or not weight[0]:
        raise ValueError("nonempty matrices required")
    width = len(weight[0])
    if any(len(row) != width for row in weight + rows):
        raise ValueError("linear shape mismatch")
    return [[sum(x * w for x, w in zip(row, column)) for column in weight] for row in rows]


def add(left, right):
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right)):
        raise ValueError("add shape mismatch")
    return [[a + b for a, b in zip(x, y)] for x, y in zip(left, right)]


def delta(rows, adapter):
    a, b, scale = adapter
    return [[value * scale for value in row] for row in linear(linear(rows, a), b)]


def hidden(rows, wg, wu, gate_adapter, up_adapter):
    gate, up = add(linear(rows, wg), delta(rows, gate_adapter)), add(linear(rows, wu), delta(rows, up_adapter))
    return [[g / (1 + math.exp(-g)) * u for g, u in zip(gs, us)] for gs, us in zip(gate, up)]


def full(rows, wg, wu, wd, adapters, bias):
    h = hidden(rows, wg, wu, adapters[0], adapters[1])
    y = add(linear(h, wd), delta(h, adapters[2]))
    return add(add(y, [bias[:] for _ in rows]), rows)


def channels(rows, wg, wu, wd, adapters, bias, cuts):
    width = len(wg)
    if not cuts or any(type(cut) is not int for cut in cuts) or sorted(cuts) != list(cuts) or len(set(cuts)) != len(cuts) or cuts[0] <= 0 or cuts[-1] >= width:
        raise ValueError("cuts must be strictly increasing interior channel indices")
    boundaries = [0, *cuts, width]
    partials, ranks = [], []
    down_a, down_b, scale = adapters[2]
    for first, last in zip(boundaries, boundaries[1:]):
        ga, gb, gs = adapters[0]
        ua, ub, us = adapters[1]
        h = hidden(rows, wg[first:last], wu[first:last],
                   (ga, gb[first:last], gs), (ua, ub[first:last], us))
        # Preserve the correspondence of gate/up channels and down columns.
        partials.append(linear(h, [row[first:last] for row in wd]))
        ranks.append(linear(h, [row[first:last] for row in down_a]))
    base, rank = partials[0], ranks[0]
    for item in partials[1:]:
        base = add(base, item)
    for item in ranks[1:]:
        rank = add(rank, item)
    # Exactly ONE down B, ONE correction scale and ONE residual/bias add.
    correction = [[v * scale for v in row] for row in linear(rank, down_b)]
    return add(add(add(base, correction), [bias[:] for _ in rows]), rows)


def fixture(seed=17, rows=5, hidden_width=4, intermediate=7, rank=3, strength=0.7):
    rng = random.Random(seed)
    def matrix(out_width, in_width):
        return [[rng.uniform(-.3, .3) for _ in range(in_width)] for _ in range(out_width)]
    def adapter(out_width, in_width):
        return matrix(rank, in_width), matrix(out_width, rank), strength
    return (matrix(rows, hidden_width), matrix(intermediate, hidden_width),
            matrix(intermediate, hidden_width), matrix(hidden_width, intermediate),
            (adapter(intermediate, hidden_width), adapter(intermediate, hidden_width),
             adapter(hidden_width, intermediate)), matrix(1, hidden_width)[0])


def max_error(left, right):
    return max(abs(a - b) for x, y in zip(left, right) for a, b in zip(x, y))


def demonstration():
    args = fixture()
    reference = full(*args)
    split = channels(*args, cuts=[2, 5])  # uneven tail, 3 logical devices
    rows = args[0]
    row_split = full(rows[:2], *args[1:]) + full(rows[2:], *args[1:])
    channel_error, row_error = max_error(reference, split), max_error(reference, row_split)
    if channel_error > 1e-12 or row_error > 1e-12:
        raise AssertionError("split algebra failed")
    return {"scope": "synthetic CPU float algebra only; no hardware/precision/performance qualification",
            "channel_split_max_abs": channel_error, "row_split_max_abs": row_error,
            "lora": "gate/up before activation, sum down-A ranks then one B, bias/residual once",
            "passed": True}


if __name__ == "__main__":
    print(json.dumps(demonstration(), ensure_ascii=False, indent=2))
