"""Original CPU math witnesses; not reproductions of full research engines."""
import math
import random

try:
    from .common import metrics, report
except ImportError:
    from common import metrics, report


def attention(query, keys, values):
    if not keys or len(keys) != len(values):
        raise ValueError("nonempty matched KV required")
    if not query or any(len(key) != len(query) for key in keys):
        raise ValueError("query/key dimension mismatch")
    width = len(values[0])
    if not width or any(len(value) != width for value in values):
        raise ValueError("value dimension mismatch")
    if not all(math.isfinite(v) for row in [query, *keys, *values] for v in row):
        raise ValueError("finite input required")
    scale = len(query) ** -0.5
    scores = [math.fsum(a * b for a, b in zip(query, key)) * scale for key in keys]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("finite attention scores required")
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    denom = math.fsum(weights)
    return [math.fsum(w * row[d] for w, row in zip(weights, values)) / denom for d in range(width)]


def piecewise_zero(query, keys, values, block_size, exact_blocks=(), *, mass=True):
    """Block-centroid softmax reference, using log(valid_count) as mass.

    Zero-order illustrative mechanism only: no PISA first-order, Sol routing,
    learned gate, or CUDA/Metal performance claim.
    """
    attention(query, keys, values)  # shared validation, not part of a benchmark
    blocks = (len(keys) + block_size - 1) // block_size if type(block_size) is int and block_size > 0 else 0
    exact = set(exact_blocks)
    if not blocks or any(type(index) is not int or not 0 <= index < blocks for index in exact):
        raise ValueError("invalid block configuration")
    scale = len(query) ** -0.5
    terms = []
    for block in range(blocks):
        start, end = block * block_size, min((block + 1) * block_size, len(keys))
        if block in exact:
            terms.extend((math.fsum(a * b for a, b in zip(query, key)) * scale, value)
                         for key, value in zip(keys[start:end], values[start:end]))
        else:
            count = end - start
            center = [math.fsum(key[d] for key in keys[start:end]) / count for d in range(len(query))]
            mean_value = [math.fsum(value[d] for value in values[start:end]) / count for d in range(len(values[0]))]
            score = math.fsum(a * b for a, b in zip(query, center)) * scale
            terms.append((score + (math.log(count) if mass else 0.0), mean_value))
    maximum = max(score for score, _ in terms)
    weights = [math.exp(score - maximum) for score, _ in terms]
    denom = math.fsum(weights)
    return [math.fsum(w * value[d] for w, (_, value) in zip(weights, terms)) / denom
            for d in range(len(values[0]))]


def linear_forecast(previous, current, previous_time, current_time, target_time):
    if not all(math.isfinite(v) for v in (previous, current, previous_time, current_time, target_time)):
        raise ValueError("finite history required")
    if current_time <= previous_time or target_time < current_time:
        raise ValueError("ordered full-compute anchors required")
    output = current + (current - previous) * (target_time - current_time) / (current_time - previous_time)
    if not math.isfinite(output):
        raise ValueError("forecast overflow")
    return output


def algorithm_evidence(seed=17):
    rng = random.Random(seed)
    query = [0.8, -0.3]
    keys = [[0.2, 0.1]] * 4 + [[-0.4, 0.7]] * 2
    values = [[rng.uniform(-2, 2), rng.uniform(-2, 2)] for _ in keys]
    dense = attention(query, keys, values)
    zero = piecewise_zero(query, keys, values, 4)
    all_exact = piecewise_zero(query, keys, values, 4, [0, 1])
    wrong_mass = piecewise_zero(query, keys, values, 4, mass=False)
    assert metrics(dense, zero)["max_abs"] < 1e-12
    assert metrics(dense, all_exact)["max_abs"] < 1e-12
    assert metrics(dense, wrong_mass)["max_abs"] > 1e-4
    adversarial_keys, adversarial_values = [[-6.0], [6.0]], [[-1.0], [1.0]]
    dense_adversarial = attention([2.0], adversarial_keys, adversarial_values)
    zero_adversarial = piecewise_zero([2.0], adversarial_keys, adversarial_values, 2)
    assert metrics(dense_adversarial, zero_adversarial)["max_abs"] > 0.99
    linear_exact = linear_forecast(5.0, 11.0, 1.0, 4.0, 7.0)
    assert linear_exact == 17.0
    turning_true = math.sin(2.0)
    turning_forecast = linear_forecast(math.sin(0), math.sin(1), 0, 1, 2)
    assert abs(turning_forecast - turning_true) > abs(math.sin(1) - turning_true)
    center = [math.fsum(row[d] for row in keys) / len(keys) for d in range(2)]
    centered_keys = [[a - b for a, b in zip(key, center)] for key in keys]
    smoothed = attention(query, centered_keys, values)
    assert metrics(dense, smoothed)["max_abs"] < 1e-12
    return report("algorithm-evidence", {
        "constant_key_partial_tail": metrics(dense, zero),
        "all_exact": metrics(dense, all_exact),
        "missing_mass_negative": metrics(dense, wrong_mass),
        "high_variance_centroid_negative": metrics(dense_adversarial, zero_adversarial),
        "nonuniform_linear_forecast": {"actual": linear_exact, "expected": 17.0},
        "turning_forecast_negative": {"replay_error": abs(math.sin(1) - turning_true),
                                      "linear_error": abs(turning_forecast - turning_true)},
        "shared_key_mean_invariance": metrics(dense, smoothed),
        "limits": "Original CPU float math. No GPU timings, model sparsity, media quality, or full algorithm reproduction.",
    })
