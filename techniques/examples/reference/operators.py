"""Dense/online attention, norm-RoPE and exact-context VAE tile references."""
import math
import random

try:
    from .common import matrix, metrics, report
except ImportError:
    from common import matrix, metrics, report


def dense_attention(q, k, v, allowed=None):
    scale = len(q[0]) ** -.5
    output = []
    for row, query in enumerate(q):
        indices = [col for col in range(len(k)) if allowed is None or allowed(row, col)]
        if not indices:
            output.append([0.0] * len(v[0]))
            continue
        scores = [math.fsum(a * b for a, b in zip(query, k[col])) * scale for col in indices]
        peak = max(scores)
        weights = [math.exp(x - peak) for x in scores]
        denominator = math.fsum(weights)
        output.append([math.fsum(w * v[col][dim] for col, w in zip(indices, weights)) / denominator for dim in range(len(v[0]))])
    return output


def online_attention(q, k, v, block=4, allowed=None):
    if type(block) is not int or block <= 0 or len(k) != len(v):
        raise ValueError("positive block and matching KV rows required")
    output = []
    scale = len(q[0]) ** -.5
    for row, query in enumerate(q):
        peak, denominator, accumulator = -math.inf, 0.0, [0.0] * len(v[0])
        for first in range(0, len(k), block):
            selected = [col for col in range(first, min(first + block, len(k))) if allowed is None or allowed(row, col)]
            if not selected:
                continue
            scores = [math.fsum(a * b for a, b in zip(query, k[col])) * scale for col in selected]
            new_peak = max(peak, max(scores))
            correction = math.exp(peak - new_peak) if denominator else 0.0
            probabilities = [math.exp(score - new_peak) for score in scores]
            accumulator = [value * correction + math.fsum(p * v[col][dim] for col, p in zip(selected, probabilities))
                           for dim, value in enumerate(accumulator)]
            denominator = denominator * correction + math.fsum(probabilities)
            peak = new_peak
        output.append([value / denominator for value in accumulator] if denominator else accumulator)
    return output


def attention(seed=17):
    rng = random.Random(seed)
    q, k, v = matrix(7, 8, rng), matrix(11, 8, rng), matrix(11, 6, rng)
    comparisons = []
    for name, mask in (("full", None), ("causal", lambda i, j: j <= i),
                       ("empty_first_query", lambda i, j: i > 0 and j <= i + 1)):
        reference = dense_attention(q, k, v, mask)
        for block in (1, 4, 8):
            candidate = online_attention(q, k, v, block, mask)
            metric = metrics(reference, candidate)
            assert metric["max_abs"] < 1e-12
            comparisons.append({"mask": name, "block": block, "metrics": metric})
    sparse_mask = lambda i, j: j < 2 or abs(j - i) <= 2
    approximate = online_attention(q, k, v, 4, sparse_mask)
    return report("online-attention", {"shape_qkv": [[7, 8], [11, 8], [11, 6]], "exact_controls": comparisons,
                  "structured_sparse": metrics(dense_attention(q, k, v), approximate),
                  "conceptual_score_elements": {"dense": 7 * 11, "online_per_query": 4},
                  "revisit": "This single-head CPU oracle motivates an MSL tile; no sparse speed or quality promise."})


def norm_rope(seed=17):
    rng, dim = random.Random(seed), 8
    x = matrix(5, dim, rng)
    gamma = [.8 + .05 * col for col in range(dim)]
    positions = [0, 1, 7, 13, 99]
    def reference(row, position):
        inverse = (math.fsum(value * value for value in row) / dim + 1e-6) ** -.5
        normalized = [value * inverse * weight for value, weight in zip(row, gamma)]
        out = []
        for col in range(0, dim, 2):
            angle = position / (10000 ** (col / dim))
            a, b = normalized[col:col + 2]
            out.extend((a * math.cos(angle) - b * math.sin(angle), a * math.sin(angle) + b * math.cos(angle)))
        return out
    def fused(row, position):
        inverse = (math.fsum(value * value for value in row) / dim + 1e-6) ** -.5
        out = [0.] * dim
        for col in range(0, dim, 2):
            angle = position / (10000 ** (col / dim))
            a, b = row[col] * inverse * gamma[col], row[col + 1] * inverse * gamma[col + 1]
            out[col], out[col + 1] = a * math.cos(angle) - b * math.sin(angle), a * math.sin(angle) + b * math.cos(angle)
        return out
    baseline = [reference(row, pos) for row, pos in zip(x, positions)]
    candidate = [fused(row, pos) for row, pos in zip(x, positions)]
    assert baseline == candidate
    wrong_positions = [fused(row, index) for index, row in enumerate(x)]
    return report("norm-rope", {"interleaved_pairs": True, "head_dim": dim,
                  "exact": metrics(baseline, candidate), "wrong_position_control": metrics(baseline, wrong_positions),
                  "intermediate_elements_avoided": len(x) * dim,
                  "adaptation": "Define interleaved vs split-half, norm epsilon, position axes and gamma indexing before porting."})


def global_stats(seed=17):
    rng = random.Random(seed)
    image = [[value + row * 1.2 for value in values] for row, values in enumerate(matrix(7, 9, rng))]
    def normalize(values, mean, inverse):
        return [(value - mean) * inverse for value in values]
    all_values = sum(image, [])
    mean = math.fsum(all_values) / len(all_values)
    variance = math.fsum((x - mean) ** 2 for x in all_values) / len(all_values)
    inverse = (variance + 1e-5) ** -.5
    reference = [normalize(row, mean, inverse) for row in image]
    # First pass accumulates all spatial statistics; the second emits tiles.
    count = sum(len(row) for row in image)
    tile_sums = [math.fsum(sum(image[first:first + 3], [])) for first in range(0, len(image), 3)]
    mean2 = math.fsum(tile_sums) / count
    variance2 = math.fsum(math.fsum((x - mean2) ** 2 for row in image[first:first + 3] for x in row)
                          for first in range(0, len(image), 3)) / count
    exact, wrong = [], []
    for first in range(0, len(image), 3):
        tile = image[first:first + 3]
        exact.extend(normalize(row, mean2, (variance2 + 1e-5) ** -.5) for row in tile)
        local = sum(tile, [])
        local_mean = math.fsum(local) / len(local)
        local_variance = math.fsum((x - local_mean) ** 2 for x in local) / len(local)
        wrong.extend(normalize(row, local_mean, (local_variance + 1e-5) ** -.5) for row in tile)
    assert metrics(reference, exact)["max_abs"] < 1e-12
    return report("global-stat-tiles", {"global_two_pass": metrics(reference, exact),
                  "local_stat_counterexample": metrics(reference, wrong), "logical_passes": 2,
                  "revisit": "Welford/pairwise sums may improve stability; low-precision reduction needs its own witness."})


def causal_convolution(seed=17):
    rng, height, width = random.Random(seed), 7, 9
    frames = [matrix(height, width, rng) for _ in range(4)]
    kernels = [matrix(3, 3, rng, .2) for _ in range(3)]
    def value(t, row, col):
        if row < 0 or row >= height or col < 0 or col >= width:
            return 0.
        return frames[max(t, 0)][row][col]
    def conv(frame, tile=0, crop_without_halo=False):
        result = [[0.] * width for _ in range(height)]
        tile = tile or width
        for left in range(0, width, tile):
            right = min(width, left + tile)
            for row in range(height):
                for col in range(left, right):
                    terms = []
                    for tap in range(3):
                        for dy in range(3):
                            for dx in range(3):
                                y, x = row + dy - 1, col + dx - 1
                                operand = value(frame - 2 + tap, y, x)
                                if crop_without_halo and not left <= x < right:
                                    operand = 0.
                                terms.append(operand * kernels[tap][dy][dx])
                    result[row][col] = math.fsum(terms)
        return result
    reference = [conv(index) for index in range(4)]
    tiled = [conv(index, 4) for index in range(4)]
    cropped = [conv(index, 4, True) for index in range(4)]
    assert reference == tiled
    # Initial temporal taps replicate the SAME frame: fold coefficients only here.
    folded = [[math.fsum(kernels[t][dy][dx] for t in range(3)) for dx in range(3)] for dy in range(3)]
    initial = [[math.fsum(value(0, row + dy - 1, col + dx - 1) * folded[dy][dx]
                         for dy in range(3) for dx in range(3)) for col in range(width)] for row in range(height)]
    assert metrics(reference[0], initial)["max_abs"] < 1e-12
    return report("causal-halo", {"exact_tile_halo": metrics(reference, tiled),
                  "independent_crop_counterexample": metrics(reference, cropped),
                  "initial_fold": metrics(reference[0], initial),
                  "stacked_conv2d_channel_order": ["t-2", "t-1", "t"],
                  "scope_limit": "Single-channel CPU tap/halo witness. Spatial stride/upscale/multichannel/device rounding are extensions."})
