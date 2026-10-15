"""Small exactness witnesses for encoder, causal LLM, DiT and VAE boundaries.

Independent teaching implementations, not model loaders or performance claims.
All shapes are tiny; Python float math intentionally separates algebra from
FP16/INT8 lowering. See models/ for the device recipes and quality gates.
"""
import math
import random

try:
    from .common import add, linear, matrix, metrics, report, silu
    from .operators import dense_attention
except ImportError:
    from common import add, linear, matrix, metrics, report, silu
    from operators import dense_attention


def encoder_buckets(seed=17):
    rng = random.Random(seed)
    x, wq, wk, wv = (matrix(n, 4, rng) for n in (7, 4, 4, 4))

    def block(rows, valid=None):
        mask = None if valid is None else lambda i, j: j < valid
        return add(rows, dense_attention(linear(rows, wq), linear(rows, wk), linear(rows, wv), mask))

    reference = block(x)
    # Padding changes neither valid queries nor their allowed key set.
    padded = x + [[9.] * 4 for _ in range(3)]
    exact = block(padded, len(x))[:len(x)]
    wrong_padding = block(padded)[:len(x)]
    # In a bidirectional stack, layer-1 prefix states depend on the suffix.
    prefix = 3
    prefix_states, full_states = block(x[:prefix]), block(x)
    prefix_keys = linear(prefix_states, wk)
    actual_prefix_keys = linear(full_states[:prefix], wk)
    assert metrics(reference, exact)["max_abs"] < 1e-12
    return report("stage-encoder", {
        "masked_bucket": metrics(reference, exact),
        "missing_padding_mask": metrics(reference, wrong_padding),
        "bidirectional_prefix_kv_is_not_causal": metrics(actual_prefix_keys, prefix_keys),
        "valid_rows": len(x), "bucket_rows": len(padded),
        "cache_contract": ["all tokens", "mask", "positions", "source", "adapters", "normalization/pooling recipe"],
    })


def rope_heads(heads, start):
    result = []
    for head in heads:
        rotated = []
        for offset, row in enumerate(head):
            if not row or len(row) % 2:
                raise ValueError("even head dimension required")
            out = []
            for dim in range(0, len(row), 2):
                angle = (start + offset) / 10000 ** (dim / len(row))
                a, b = row[dim:dim + 2]
                out.extend((a * math.cos(angle) - b * math.sin(angle),
                            a * math.sin(angle) + b * math.cos(angle)))
            rotated.append(out)
        result.append(rotated)
    return result


def gqa(q, k, v, query_start=0):
    """[head,token,dim] GQA with a global-position causal mask, no KV repeat."""
    if not q or not k or len(k) != len(v) or len(q) % len(k):
        raise ValueError("query heads must be a multiple of nonempty KV heads")
    if type(query_start) is not int or query_start < 0 or not q[0] or not k[0]:
        raise ValueError("nonempty rows and nonnegative global position required")
    nq, nk, dim = len(q[0]), len(k[0]), len(q[0][0])
    if not dim or query_start + nq > nk or any(len(head) != nq for head in q) or any(len(head) != nk for head in k + v):
        raise ValueError("inconsistent causal attention geometry")
    if any(len(row) != dim or not all(math.isfinite(value) for value in row) for head in q + k + v for row in head):
        raise ValueError("equal finite head dimensions required")
    group = len(q) // len(k)
    return [dense_attention(head, k[index // group], v[index // group],
                            lambda i, j: j <= query_start + i)
            for index, head in enumerate(q)]


class CausalKV:
    """Single-request append-only cache; caller supplies already-positioned Q/K.

    Identity includes model/adapter/position recipe. No implicit eviction,
    prefix borrowing, device lifetime or paged allocator is hidden here.
    """
    def __init__(self, identity, kv_heads):
        if not identity or type(kv_heads) is not int or kv_heads <= 0:
            raise ValueError("nonempty identity and positive KV head count required")
        self.identity, self.length = identity, 0
        self.k, self.v = [[] for _ in range(kv_heads)], [[] for _ in range(kv_heads)]

    def append(self, q, k, v, *, identity, start):
        if identity != self.identity or type(start) is not int or start != self.length:
            raise ValueError("stale identity or noncontiguous global position")
        if not q or len(k) != len(self.k) or len(v) != len(k) or len(q) % len(k):
            raise ValueError("GQA head mismatch")
        count = len(q[0])
        arrays = list(q) + list(k) + list(v)
        dim = len(q[0][0]) if count else 0
        if not count or not dim or any(len(head) != count for head in arrays):
            raise ValueError("nonempty equal chunk lengths required")
        if any(len(row) != dim or not all(math.isfinite(value) for value in row)
               for head in arrays for row in head):
            raise ValueError("head dimensions and finite values required")
        if any(len(row) != dim for head in self.k + self.v for row in head):
            raise ValueError("cached head dimension cannot change")
        # Validate first, publish the complete append only after attention works.
        keys = [old + [list(row) for row in new] for old, new in zip(self.k, k)]
        values = [old + [list(row) for row in new] for old, new in zip(self.v, v)]
        result = gqa(q, keys, values, self.length)
        self.k, self.v, self.length = keys, values, self.length + count
        return result


def llm_prefill(seed=17):
    rng, tokens, dim, qheads, kvheads = random.Random(seed), 9, 4, 4, 2
    q = rope_heads([matrix(tokens, dim, rng) for _ in range(qheads)], 0)
    k = rope_heads([matrix(tokens, dim, rng) for _ in range(kvheads)], 0)
    v = [matrix(tokens, dim, rng) for _ in range(kvheads)]
    reference = gqa(q, k, v)
    comparisons = []
    for chunks in ((9,), (3, 4, 2), (5, 1, 1, 1, 1)):
        cache = CausalKV("source-A:adapter-A:rope-global", kvheads)
        output = [[] for _ in range(qheads)]
        for count in chunks:
            start, end = cache.length, cache.length + count
            parts = [[head[start:end] for head in value] for value in (q, k, v)]
            result = cache.append(*parts, identity=cache.identity, start=start)
            for head, rows in zip(output, result):
                head.extend(rows)
        metric = metrics(reference, output)
        assert metric["max_abs"] < 1e-12
        comparisons.append({"chunks": chunks, "metrics": metric})
    # Offsetting queries by zero for the second chunk incorrectly masks its past.
    wrong = gqa([head[5:] for head in q], k, v, query_start=0)
    return report("stage-llm", {
        "prefill_chunked_and_decode": comparisons,
        "wrong_chunk_causal_offset": metrics([head[5:] for head in reference], wrong),
        "gqa_kv_elements": 2 * kvheads * tokens * dim,
        "expanded_mha_kv_elements": 2 * qheads * tokens * dim,
        "limits": "One attention layer, no learned network, paged allocation or speculative sampler. Positions were applied before caching.",
    })


def layer_norm(rows, epsilon=1e-5):
    result = []
    for row in rows:
        mean = math.fsum(row) / len(row)
        variance = math.fsum((value - mean) ** 2 for value in row) / len(row)
        result.append([(value - mean) / math.sqrt(variance + epsilon) for value in row])
    return result


def speculative_distribution(target, draft):
    """Exact one-token accept/reject law, NOT a complete decoding engine.

    Draw i~draft, accept with min(1,target[i]/draft[i]); upon rejection draw
    from normalized positive(target-draft). Integrate all outcomes here.
    """
    if not target or len(target) != len(draft):
        raise ValueError("matching categorical distributions required")
    for values in (target, draft):
        if not all(math.isfinite(value) and value >= 0 for value in values) or not math.isclose(math.fsum(values), 1., rel_tol=0, abs_tol=1e-12):
            raise ValueError("finite normalized probabilities required")
    accepted = [min(p, q) for p, q in zip(target, draft)]
    rejection = max(0., 1 - math.fsum(accepted))
    positive = [max(p - q, 0.) for p, q in zip(target, draft)]
    mass = math.fsum(positive)
    residual = [value / mass for value in positive] if mass else [0.] * len(target)
    return {"distribution": [a + rejection * r for a, r in zip(accepted, residual)],
            "acceptance_probability": math.fsum(accepted), "residual": residual}


def llm_decode(seed=17):
    target, draft = [.1, .6, .3], [.4, .2, .4]
    exact = speculative_distribution(target, draft)
    # Accepting the draft unconditionally is faster-looking but changes the law.
    assert metrics(target, exact["distribution"])["max_abs"] < 1e-12
    return report("stage-decode", {
        "exact_accept_reject": metrics(target, exact["distribution"]),
        "always_accept_draft": metrics(target, draft),
        "acceptance_probability": exact["acceptance_probability"],
        "rejection_residual": exact["residual"],
        "limits": "Single-token probability identity only. Multi-token verification needs causal alignment, rejected KV rollback, the bonus token and independently validated RNG/sampling semantics. No speed or sequence-distribution claim.",
    })


def adaln(rows, shift, scale):
    if len(shift) != len(scale) or any(len(row) != len(scale) for row in rows):
        raise ValueError("modulation width mismatch")
    return [[value * (1 + weight) + bias for value, weight, bias in zip(row, scale, shift)]
            for row in layer_norm(rows)]


def dit_conditioning(seed=17):
    rng = random.Random(seed)
    latent, context = matrix(5, 4, rng), matrix(3, 4, rng)
    wq, wk, wv = (matrix(4, 4, rng) for _ in range(3))
    shift, scale = [.1, -.2, .3, -.1], [.2, .1, -.3, .4]
    normalized = layer_norm(latent)
    reference = [[(value * (1 + scale[i])) + shift[i] for i, value in enumerate(row)] for row in normalized]
    fused = adaln(latent, shift, scale)
    cached_k, cached_v = linear(context, wk), linear(context, wv)
    controls = []
    for step in (0, 1, 5):
        # Timestep changes Q, but NOT this separate immutable cross-attention KV.
        q = linear(adaln(latent, [value + step * .1 for value in shift], scale), wq)
        expected = dense_attention(q, linear(context, wk), linear(context, wv))
        controls.append(metrics(expected, dense_attention(q, cached_k, cached_v)))
    changed_context = [row[:] for row in context]
    changed_context[0][0] += 2
    q = linear(fused, wq)
    wrong_kv = metrics(dense_attention(q, linear(changed_context, wk), linear(changed_context, wv)),
                       dense_attention(q, cached_k, cached_v))
    # CFG combines final branch predictions, not intermediate nonlinear inputs.
    conditional, unconditional, guidance = [.3, -.7, .8], [-.4, .2, .1], 3.
    branches = [silu(u) + guidance * (silu(c) - silu(u)) for c, u in zip(conditional, unconditional)]
    wrong_cfg = [silu(u + guidance * (c - u)) for c, u in zip(conditional, unconditional)]
    assert metrics(reference, fused)["max_abs"] < 1e-12
    assert all(value["max_abs"] < 1e-12 for value in controls)
    return report("stage-dit", {
        "adaln_boundary": metrics(reference, fused), "cross_kv_hoisting": controls,
        "stale_condition_kv": wrong_kv, "guidance_before_nonlinearity": metrics(branches, wrong_cfg),
        "limits": "Cross-only architecture witness. Joint text/image self-attention, adaptive conditioning and timestep-modulated K/V need a different identity and are not hoistable by this proof.",
    })


def transpose_conv1d(x, kernel, *, stride=2, padding=1, begin=0, end=None):
    """Output-coordinate gather: disjoint output tiles retain all input taps."""
    if not x or not kernel or type(stride) is not int or stride < 1 or type(padding) is not int or padding < 0:
        raise ValueError("invalid transposed-convolution geometry")
    length = (len(x) - 1) * stride - 2 * padding + len(kernel)
    end = length if end is None else end
    if length < 1 or type(begin) is not int or type(end) is not int or not 0 <= begin < end <= length:
        raise ValueError("invalid output tile")
    result = []
    for position in range(begin, end):
        terms = []
        for tap, weight in enumerate(kernel):
            offset = position + padding - tap
            if offset % stride == 0 and 0 <= offset // stride < len(x):
                terms.append(x[offset // stride] * weight)
        result.append(math.fsum(terms))
    return result


def merge_moments(a, b):
    """Parallel Welford merge; M2 is a sum, not a local variance."""
    na, ma, m2a = a
    nb, mb, m2b = b
    if not na:
        return b
    if not nb:
        return a
    n, delta = na + nb, mb - ma
    return n, ma + delta * nb / n, m2a + m2b + delta * delta * na * nb / n


def moments(values):
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("nonempty finite group values required")
    mean = math.fsum(values) / len(values)
    return len(values), mean, math.fsum((value - mean) ** 2 for value in values)


def vae_output_tiles(seed=17):
    rng = random.Random(seed)
    x, kernel = [rng.uniform(-1, 1) for _ in range(9)], [.2, -.3, .7]
    reference = transpose_conv1d(x, kernel)
    tiled = []
    for begin in range(0, len(reference), 4):
        tiled.extend(transpose_conv1d(x, kernel, begin=begin, end=min(begin + 4, len(reference))))
    # Independent input crops omit shared boundary contributions / output phase.
    wrong = sum((transpose_conv1d(x[begin:begin + 4], kernel) for begin in range(0, len(x), 4)), [])
    wrong = (wrong + [0.] * len(reference))[:len(reference)]
    values = [value + i * .2 for i, value in enumerate(reference)]
    combined = (0, 0., 0.)
    for begin in range(0, len(values), 4):
        combined = merge_moments(combined, moments(values[begin:begin + 4]))
    n, mean, m2 = combined
    exact_mean, exact_m2 = moments(values)[1:]
    normalized = [(value - exact_mean) / math.sqrt(exact_m2 / n + 1e-5) for value in values]
    two_pass = [(value - mean) / math.sqrt(m2 / n + 1e-5) for value in values]
    assert metrics(reference, tiled)["max_abs"] < 1e-12
    assert metrics(normalized, two_pass)["max_abs"] < 1e-12
    return report("stage-vae", {
        "output_coordinate_tiles": metrics(reference, tiled),
        "independent_input_crop": metrics(reference, wrong),
        "merged_group_moments": metrics(normalized, two_pass),
        "output_rows": len(reference), "tail_tile_rows": len(reference) % 4,
        "limits": "1D scalar stride/phase and global-statistics witnesses. Full 2D/3D graph still needs receptive fields, channel groups, temporal history and device rounding validation.",
    })
