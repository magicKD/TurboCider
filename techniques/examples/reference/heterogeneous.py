"""Independent gpu_ane-inspired topology, ABI and policy teaching samples.

Only Python standard library/local numerical references. No imported sibling
runtime, original weight format, private API, or hardware-performance claims.
"""
from dataclasses import dataclass
from itertools import product
import math
import random
import struct

try:
    from .common import add, bf16, linear, matrix, metrics, report, silu, swiglu
    from .operators import dense_attention
    from .systems import schedule
except ImportError:
    from common import add, bf16, linear, matrix, metrics, report, silu, swiglu
    from operators import dense_attention
    from systems import schedule


def parallel_branches(seed=17):
    rng = random.Random(seed)
    x, residual = matrix(5, 4, rng), matrix(5, 4, rng)
    q, k, v = [matrix(4, 4, rng) for _ in range(3)]
    wg, wu, wo = matrix(7, 4, rng), matrix(7, 4, rng), matrix(4, 11, rng)
    attention = dense_attention(linear(x, q), linear(x, k), linear(x, v))
    gate, up = linear(x, wg), linear(x, wu)
    hidden = [[silu(g) * u for g, u in zip(a, b)] for a, b in zip(gate, up)]
    concatenated = [a + b for a, b in zip(attention, hidden)]
    full = linear(concatenated, wo)
    attention_partial = linear(attention, [row[:4] for row in wo])
    mlp_partial = linear(hidden, [row[4:] for row in wo])
    joined = add(attention_partial, mlp_partial)
    modulation_gate = [.2, -.3, .4, .1]
    finish = lambda output: add(residual, [[value * weight for value, weight in zip(row, modulation_gate)] for row in output])
    exact = metrics(finish(full), finish(joined))
    # Sequential topology is a DIFFERENT graph, not an alternative schedule.
    wrong_input = add(x, attention)
    wrong_mlp, _ = swiglu(wrong_input, wg, wu, [row[4:] for row in wo])
    wrong = finish(add(attention_partial, wrong_mlp))
    assert exact["max_abs"] < 1e-12
    return report("parallel-branches", {
        "concat_output_projection_vs_partial_sum": exact,
        "wrong_sequential_rewrite": metrics(finish(full), wrong),
        "same_normalized_input_required": True, "residual_and_modulation_gate_applied_once": True,
        "scope_limit": "FLUX.2-like single-block algebra, not its weights/RoPE/precision or device execution. A sequential block must not be rewritten into this topology.",
    })


def attention_head_shards(seed=17):
    rng, heads, rows, dim, output = random.Random(seed), 3, 5, 4, 6
    q, k, v = [[matrix(rows, dim, rng) for _ in range(heads)] for _ in range(3)]
    wo = matrix(output, heads * dim, rng)
    attention = [dense_attention(q[i], k[i], v[i]) for i in range(heads)]
    concatenated = [sum((attention[h][row] for h in range(heads)), []) for row in range(rows)]
    reference = linear(concatenated, wo)
    partials = [linear(attention[h], [row[h * dim:(h + 1) * dim] for row in wo]) for h in range(heads)]
    candidate = add(add(partials[0], partials[1]), partials[2])
    # Forgetting one complete head still returns a correctly-shaped hidden tensor.
    missing = add(partials[0], partials[1])
    assert metrics(reference, candidate)["max_abs"] < 1e-12
    return report("attention-head-shards", {
        "complete_heads_through_output_projection": metrics(reference, candidate),
        "missing_head_despite_same_output_shape": metrics(reference, missing),
        "head_boundary": "Each shard keeps full softmax over allowed keys and matching output-projection input columns. GQA KV ownership must be mapped separately.",
    })


def strided_offsets(shape, strides):
    if not shape or len(shape) != len(strides) or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError("positive dimensions and matching strides required")
    if math.prod(shape) > 4096 or any(type(value) is not int or value < 0 for value in strides):
        raise ValueError("bounded shape and nonnegative element strides required")
    offsets = [sum(i * stride for i, stride in zip(index, strides)) for index in product(*(range(dim) for dim in shape))]
    if len(set(offsets)) != len(offsets):
        raise ValueError("overlapping logical elements")
    return offsets


def consumer_strides(seed=17):
    rows, channels = 5, 7
    shape = [1, channels, 1, rows]
    default = strided_offsets(shape, [channels * rows, rows, rows, 1])
    consumer = strided_offsets(shape, [channels * rows, 1, channels * rows, channels])
    values = [float(channel * 10 + row) for channel in range(channels) for row in range(rows)]
    def place(offsets):
        output = [0.] * (rows * channels)
        for offset, value in zip(offsets, values):
            output[offset] = value
        return [output[start:start + channels] for start in range(0, len(output), channels)]
    reference = [[float(channel * 10 + row) for channel in range(channels)] for row in range(rows)]
    exact = metrics(reference, place(consumer))
    assert exact["max_abs"] == 0
    return report("consumer-strides", {
        "logical_shape": shape, "element_strides": [channels * rows, 1, channels * rows, channels],
        "producer_written_row_major": exact,
        "default_channel_major_read_as_row_major": metrics(reference, place(default)),
        "maximum_element_offset": max(consumer), "minimum_backing_elements": max(consumer) + 1,
        "limits": "Address-map witness, not a Core ML placement or zero-copy benchmark. Actual returned backing/strides and layout premium require device validation.",
    })


def segmented_ffn(seed=17):
    rng = random.Random(seed)
    x, wg, wu, wd = matrix(5, 4, rng), matrix(7, 4, rng), matrix(7, 4, rng), matrix(4, 7, rng)
    reference, _ = swiglu(x, wg, wu, wd)
    gate, up = linear(x, wg), linear(x, wu)
    partials, wrong_parts = [], []
    segments = [(0, 2), (2, 5), (5, 7)]
    for begin, end in segments:
        # Retain paired (gate_d,up_d) segments instead of a full [gate|up] copy.
        g, u = [row[begin:end] for row in gate], [row[begin:end] for row in up]
        hidden = [[silu(a) * b for a, b in zip(gs, us)] for gs, us in zip(g, u)]
        wrong = [[silu(b) * a for a, b in zip(gs, us)] for gs, us in zip(g, u)]
        down = [row[begin:end] for row in wd]
        partials.append(linear(hidden, down))
        wrong_parts.append(linear(wrong, down))
    candidate = add(add(partials[0], partials[1]), partials[2])
    wrong = add(add(wrong_parts[0], wrong_parts[1]), wrong_parts[2])
    assert metrics(reference, candidate)["max_abs"] < 1e-12
    return report("segmented-ffn", {
        "paired_segments_through_down_partials": metrics(reference, candidate),
        "swapped_gate_up_abi": metrics(reference, wrong), "segments": segments,
        "limits": "CPU paired-layout algebra only. Segmented scatter+one full down GEMM and per-shard down GEMMs have different device costs; neither is automatically fastest.",
    })


@dataclass(frozen=True)
class DeviceCost:
    launch_ms: float
    per_channel_ms: float
    p95_factor: float = 1.

    def __post_init__(self):
        if any(not math.isfinite(v) for v in (self.launch_ms, self.per_channel_ms, self.p95_factor)):
            raise ValueError("finite cost profile required")
        if self.launch_ms < 0 or self.per_channel_ms <= 0 or self.p95_factor < 1:
            raise ValueError("invalid launch/slope/tail profile")

    def cost(self, width):
        if type(width) is not int or width < 0:
            raise ValueError("nonnegative integer shard width required")
        result = 0. if width == 0 else (self.launch_ms + width * self.per_channel_ms) * self.p95_factor
        if not math.isfinite(result):
            raise ValueError("device cost overflow")
        return result


def choose_partition(width, quantum, profiles, *, join_ms=.1, contention=.05):
    if any(type(v) is not int or v <= 0 for v in (width, quantum)) or width % quantum or width // quantum > 32:
        raise ValueError("bounded exact channel quantum required")
    if not math.isfinite(join_ms) or not math.isfinite(contention) or join_ms < 0 or contention < 0:
        raise ValueError("finite nonnegative join/contention cost required")
    if set(profiles) != {"gpu", "ane", "cpu"}:
        raise ValueError("all optional device profiles must be explicit")
    baseline = profiles["gpu"].cost(width)
    plans = []
    units = width // quantum
    for ane, cpu in product(range(units + 1), repeat=2):
        gpu = units - ane - cpu
        if gpu <= 0:
            continue  # This join adapter requires one GPU shard.
        widths = {"gpu": gpu * quantum, "ane": ane * quantum, "cpu": cpu * quantum}
        costs = {device: profiles[device].cost(size) for device, size in widths.items()}
        active = sum(size > 0 for size in widths.values())
        critical = max(costs.values())
        wall = critical * (1 + contention * (active - 1)) + join_ms * (active - 1)
        reason = None
        if ane and costs["ane"] > costs["gpu"] * 1.2:
            reason = "ane misses the assumed first-consumer deadline"
        elif cpu and costs["cpu"] > max(costs["gpu"], costs["ane"]) * 1.1:
            reason = "cpu p95 straggler"
        elif active > 1 and wall > baseline / 1.02:
            reason = "no assumed minimum gain"
        plans.append({"widths": widths, "branch_ms": costs, "wall_ms": wall, "rejected": reason})
    admitted = [plan for plan in plans if plan["rejected"] is None]
    selected = min(admitted, key=lambda plan: (plan["wall_ms"], sum(v > 0 for v in plan["widths"].values())))
    return {"selected": selected, "gpu_baseline_ms": baseline, "candidate_count": len(plans)}


def optional_devices(seed=17):
    profiles = {"gpu": DeviceCost(.1, 1.), "ane": DeviceCost(.8, .55), "cpu": DeviceCost(.4, 1.8, 1.8)}
    selected = choose_partition(8, 1, profiles)
    triple_cpu = DeviceCost(.01, .45, 1.1)
    triple = choose_partition(8, 1, {**profiles, "cpu": triple_cpu})
    gpu_only = choose_partition(8, 1, {**profiles, "ane": DeviceCost(20., 1.), "cpu": DeviceCost(20., 1.)})
    assert selected["selected"]["widths"]["cpu"] == 0
    assert all(size > 0 for size in triple["selected"]["widths"].values())
    assert gpu_only["selected"]["widths"] == {"gpu": 8, "ane": 0, "cpu": 0}
    return report("optional-devices", {"assumed_profiles": {name: vars(value) for name, value in profiles.items()},
        "optional_cpu_declined": selected, "three_way_scenario_cpu_profile": vars(triple_cpu),
        "profitable_three_way_scenario": triple,
        "all_accelerator_options_declined": gpu_only,
        "limits": "Synthetic p95/critical-path planner, no hardware fit. Real admission also needs memory, source, numerical, tail and full-request trials; these thresholds are example policy, not Apple device laws."},
        scope="synthetic optional-device planner with full-GPU fallback")


def independent_regions(a, b):
    for region in (a, b):
        if not isinstance(region["allocation"], str) or not region["allocation"] or type(region["write"]) is not bool:
            raise ValueError("actual allocation identity and explicit access mode required")
        if type(region["offset"]) is not int or type(region["bytes"]) is not int or region["offset"] < 0 or region["bytes"] <= 0:
            raise ValueError("valid physical byte intervals required")
    if not a["write"] and not b["write"]:
        return True
    return a["allocation"] != b["allocation"] or a["offset"] + a["bytes"] <= b["offset"] or b["offset"] + b["bytes"] <= a["offset"]


def resource_hazards(seed=17):
    tasks = [{"id": "a", "resource": "same-tracked-identity", "duration": 3., "deps": []},
             {"id": "b", "resource": "same-tracked-identity", "duration": 3., "deps": []}]
    separate = [{**task, "resource": "gpu-lane-" + task["id"]} for task in tasks]
    saturated = [{**task, "duration": 6.} for task in separate]
    a = {"allocation": "backing-A", "offset": 0, "bytes": 64, "write": True}
    disjoint = {**a, "offset": 64}
    alias = {**a, "offset": 32}
    assert independent_regions(a, disjoint) and not independent_regions(a, alias)
    return report("resource-hazards", {
        "coarse_identity_serial_scenario": schedule(tasks), "independent_identity_scenario": schedule(separate),
        "saturated_bandwidth_scenario_no_wall_gain": schedule(saturated), "overlapping_writes_rejected": True,
        "limits": "Synthetic hazard granularity, not an observation of this driver. Separate resource names do not prove nonaliasing; never disable Metal hazard tracking without complete alias and synchronization proofs."},
        scope="synthetic resource-identity and physical-region dependency model")


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def precision_islands(seed=17):
    a, b, state = .9997, .001, .25
    low, state_only, shadow = state, state, state
    for _ in range(256):
        state = a * state + b
        low = bf16(bf16(a) * low + bf16(b))
        state_only = f32(bf16(a) * state_only + bf16(b))
        shadow = f32(f32(a) * shadow + f32(b))
    assert abs(shadow - state) < abs(low - state)
    return report("precision-islands", {
        "steps": 256, "reference_state": state, "low_state": low,
        "fp32_state_with_already_rounded_parameters": state_only, "full_fp32_parameter_and_state_island": shadow,
        "low_state_error": abs(low - state), "state_only_error": abs(state_only - state), "shadow_error": abs(shadow - state),
        "limits": "Scalar recurrent-state witness inspired by selective GDN precision work, not a GDN implementation. Real groups need source weights, complete projections/norm/state and task-quality validation plus measured shadow-path cost.",
    })
