"""Sequence, hidden-input/output and global-softmax partition algebra witnesses."""
import math
import random

try:
    from .common import add, linear, matrix, metrics, report, silu, swiglu
    from .operators import dense_attention
except ImportError:
    from common import add, linear, matrix, metrics, report, silu, swiglu
    from operators import dense_attention


def partition_axes(seed=17):
    rng, m, h, f = random.Random(seed), 7, 8, 12
    x = matrix(m, h, rng, .7)
    wg, wu, wd = matrix(f, h, rng, .3), matrix(f, h, rng, .3), matrix(h, f, rng, .3)
    reference, hidden = swiglu(x, wg, wu, wd)
    # Hidden INPUT / reduction-K split: reduce BOTH projections before SiLU.
    gate_parts, up_parts, wrong_hidden_parts = [], [], []
    for first, last in ((0, 3), (3, h)):
        source = [row[first:last] for row in x]
        gate = linear(source, [row[first:last] for row in wg])
        up = linear(source, [row[first:last] for row in wu])
        gate_parts.append(gate); up_parts.append(up)
        wrong_hidden_parts.append([[silu(a) * b for a, b in zip(g, u)] for g, u in zip(gate, up)])
    gate, up = add(*gate_parts), add(*up_parts)
    joined_hidden = [[silu(a) * b for a, b in zip(g, u)] for g, u in zip(gate, up)]
    hidden_input_result = linear(joined_hidden, wd)
    wrong_nonlinearity = linear(add(*wrong_hidden_parts), wd)
    assert metrics(reference, hidden_input_result)["max_abs"] < 1e-12
    # Hidden OUTPUT split of down: concatenate columns; no partial sum here.
    output_parts = [linear(hidden, wd[:3]), linear(hidden, wd[3:])]
    output_result = [left + right for left, right in zip(*output_parts)]
    assert metrics(reference, output_result)["max_abs"] < 1e-12
    def rms(row):
        inverse = (math.fsum(v*v for v in row) / len(row) + 1e-6) ** -.5
        return [value * inverse for value in row]
    global_norm = [rms(row) for row in output_result]
    wrong_local_norm = [rms(left) + rms(right) for left, right in zip(*output_parts)]
    # Sequence ROW split is independent for this FFN (not for every model op).
    sequence_result = swiglu(x[:2], wg, wu, wd)[0] + swiglu(x[2:], wg, wu, wd)[0]
    assert sequence_result == reference
    q, k, v = matrix(5, 4, rng), matrix(11, 4, rng), matrix(11, 3, rng)
    dense = dense_attention(q, k, v)
    shard_stats, local_outputs = [], []
    for first, last in ((0, 3), (3, 11)):
        stats, local = [], []
        for query in q:
            scores = [math.fsum(a*b for a,b in zip(query, key)) / 2 for key in k[first:last]]
            peak = max(scores); weights = [math.exp(score - peak) for score in scores]
            denominator = math.fsum(weights)
            numerator = [math.fsum(weight * value[d] for weight, value in zip(weights, v[first:last])) for d in range(3)]
            stats.append((peak, denominator, numerator))
            local.append([value / denominator for value in numerator])
        shard_stats.append(stats); local_outputs.append(local)
    merged = []
    for left, right in zip(*shard_stats):
        peak = max(left[0], right[0])
        a, b = math.exp(left[0] - peak), math.exp(right[0] - peak)
        denominator = a * left[1] + b * right[1]
        merged.append([(a*x + b*y) / denominator for x, y in zip(left[2], right[2])])
    wrong_local_average = [[(a+b)/2 for a,b in zip(left,right)] for left,right in zip(*local_outputs)]
    assert metrics(dense, merged)["max_abs"] < 1e-12
    # Logical traffic arithmetic, not Apple DRAM/SRAM or measured bandwidth.
    gpu_bytes, ane_bytes, staging_bytes, assumed_bandwidth = 300e6, 200e6, 100e6, 100e9
    uma_floor = (gpu_bytes + ane_bytes + staging_bytes) / assumed_bandwidth
    return report("partition-axes", {"shape_M_H_F": [m,h,f],
        "hidden_input_reduce_before_activation": metrics(reference, hidden_input_result),
        "wrong_sum_of_branch_nonlinearities": metrics(reference, wrong_nonlinearity),
        "hidden_output_concat": metrics(reference, output_result),
        "wrong_per_output_shard_rms": metrics(global_norm, wrong_local_norm),
        "sequence_ffn_concat": metrics(reference, sequence_result),
        "kv_sequence_global_softmax_merge": metrics(dense, merged),
        "wrong_average_of_local_softmax_outputs": metrics(dense, wrong_local_average),
        "logical_exchange_elements": {"F_split_two_hidden_partials":2*m*h,
                                       "H_input_split_two_gate_and_up_partials":2*2*m*f},
        "bandwidth_scenario": {"assumed_gpu_bytes":gpu_bytes,"assumed_ane_bytes":ane_bytes,
                               "assumed_staging_bytes":staging_bytes,"assumed_bytes_per_second":assumed_bandwidth,
                               "uma_time_lower_bound_seconds":uma_floor,
                               "compute_only_ideal_seconds":.004,
                               "joint_lower_bound_seconds":max(.003,.004,uma_floor)},
        "limits":"Real-arithmetic CPU algebra and invented logical-byte budget. Not measured UMA bandwidth, production split-axis support or a device speedup."})
