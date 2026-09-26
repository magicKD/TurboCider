#!/usr/bin/env python3
"""CPU sampled exact-attention mass diagnostic, not a GPU/video benchmark.

Compare full-block query centroids with a four-query log-mean-exp score.
The exact-mass ranking is an in-sample upper bound, not a deployable router.
All rankings omit safety regions so they use identical exact-block budgets.
Output diagnostics include actual V, BF16 summary/output rounding, gate and
pooled correction, but use FP64 attention math, not Metal's accumulation.
No safety regions are included. Mass oracle is NOT an output-error oracle.
These sampled errors are not GPU timings or video-quality evidence.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_ltx_qkv_replay import digest


def decode(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def round_bf16(value):
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16
    return decode(rounded)


def rotate(x, cosine, sine):
    """Match split RoPE: rounded FP32 product followed by FP32 fused add.

    Float64 computes the sum of the two FP32 products accurately enough to
    emulate the shader's fma before BF16 rounding for these finite fixtures.
    This is a CPU diagnostic; it does not establish bitwise Metal parity.
    """
    first, second = np.split(x, 2, axis=1)
    left = (first * cosine).astype(np.float64) - second.astype(np.float64) * sine
    right = (second * cosine).astype(np.float64) + first.astype(np.float64) * sine
    return round_bf16(np.concatenate((left, right), axis=1))


def softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def block_sums(x):
    # Sequential FP32 summation matches the shader's summary loop order.
    output = []
    for begin in range(0, len(x), 64):
        block = x[begin:begin + 64]
        total = np.zeros(x.shape[1], dtype=np.float32)
        for row in block:
            total += row
        output.append(total)
    return np.array(output)


def block_means(x):
    counts = np.minimum(64, len(x) - np.arange((len(x) + 63) // 64) * 64)
    return block_sums(x) * (1 / counts).astype(np.float32)[:, None]


def block_variances(key):
    """Population variance per key dimension (no cross-dimension covariance)."""
    return np.array([np.var(key[begin:begin + 64].astype(np.float64), axis=0)
                     for begin in range(0, len(key), 64)])


def score_routes(query_block, sampled_queries, key, key_centroids, key_variances=None):
    logits = sampled_queries.astype(np.float64) @ key.astype(np.float64).T
    logits /= np.sqrt(key.shape[1])
    probability = softmax(logits)
    mass = np.add.reduceat(probability, np.arange(0, len(key), 64), axis=1)
    centroid = block_means(query_block)[0].astype(np.float64) @ key_centroids.T
    sample_logits = sampled_queries.astype(np.float64) @ key_centroids.T
    sample_logits /= np.sqrt(key.shape[1])
    maxima = sample_logits.max(axis=0)
    multi = maxima + np.log(np.exp(sample_logits - maxima).mean(axis=0))
    if key_variances is None:
        key_variances = block_variances(key)
    # Candidate uses only full-query-block mean and per-key-block moments;
    # unlike the oracle, it never reads exact sampled attention logits.
    query_mean = block_means(query_block)[0].astype(np.float64)
    variance_score = centroid / np.sqrt(key.shape[1])
    variance_score += .5 * (query_mean ** 2 @ key_variances.T) / key.shape[1]
    scores = {"centroid": centroid, "sample_logmeanexp": multi,
              "diagonal_key_variance": variance_score,
              "in_sample_oracle": mass.mean(axis=0)}
    ranks = {name: np.argsort(-score, kind="stable") for name, score in scores.items()}
    return mass, ranks


def centroid_distribution(query_block, key_centroids, key_rows, key_variances=None):
    """Pooled logit distribution with tail-block multiplicity."""
    query = block_means(query_block)[0].astype(np.float64)
    logits = query @ key_centroids.T / np.sqrt(key_centroids.shape[1])
    if key_variances is not None:
        if (key_variances.shape != key_centroids.shape or
                not np.isfinite(key_variances).all() or np.any(key_variances < 0)):
            raise ValueError("invalid key variances")
        logits += .5 * (query ** 2 @ key_variances.T) / key_centroids.shape[1]
    counts = np.minimum(64, key_rows - np.arange(len(key_centroids)) * 64)
    if np.any(counts <= 0) or counts.sum() != key_rows:
        raise ValueError("centroid count does not match key rows")
    return softmax(logits + np.log(counts))


def mass_budget(distribution, target):
    """Select the smallest stable top-ranked prefix reaching target mass."""
    p = np.asarray(distribution, dtype=np.float64)
    if (p.ndim != 1 or not p.size or not np.isfinite(p).all() or
            np.any(p < 0) or not np.isclose(p.sum(), 1) or not 0 < target <= 1):
        raise ValueError("invalid probability distribution or target")
    rank = np.argsort(-p, kind="stable")
    keep = min(len(p), int(np.searchsorted(np.cumsum(p[rank]), target)) + 1)
    return rank[:keep]


def approximate_output(sampled_queries, key, value, key_centroids, value_sums, selected):
    """Pooled/exact numerator and denominator for selected blocks."""
    exact_logits = sampled_queries.astype(np.float64) @ key.astype(np.float64).T
    exact_logits /= np.sqrt(key.shape[1])
    pooled_logits = sampled_queries.astype(np.float64) @ key_centroids.astype(np.float64).T
    pooled_logits /= np.sqrt(key.shape[1])
    terms, vals = [], []
    selected = set(int(x) for x in selected)
    for block, begin in enumerate(range(0, len(key), 64)):
        end = min(begin + 64, len(key)); count = end - begin
        if block in selected:
            terms.append(exact_logits[:, begin:end]); vals.append(value[begin:end])
        else:
            terms.append(pooled_logits[:, block:block + 1] + np.log(count))
            vals.append(value_sums[block:block + 1].astype(np.float64) / count)
    logits = np.concatenate(terms, axis=1)
    weights = softmax(logits)
    out = np.zeros((len(sampled_queries), value.shape[1]), np.float64)
    offset = 0
    for block, begin in enumerate(range(0, len(key), 64)):
        end = min(begin + 64, len(key)); count = end - begin
        width = count if block in selected else 1
        out += weights[:, offset:offset + width] @ vals[block].astype(np.float64)
        offset += width
    return out


def gated_output_energy(reference, candidate, gate):
    reference = round_bf16(round_bf16(reference) * gate[:, None]).astype(np.float64)
    candidate = round_bf16(round_bf16(candidate) * gate[:, None]).astype(np.float64)
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("nonfinite output")
    return ((candidate - reference) ** 2).sum(axis=1), (reference ** 2).sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--query-blocks", type=int, default=8)
    parser.add_argument("--queries-per-block", type=int, default=4)
    parser.add_argument("--keep", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--mass-targets", nargs="+", type=float, default=[.9, .95, .99])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve prior evidence")
    if not 1 <= args.query_blocks <= 256 or not 1 <= args.queries_per_block <= 64:
        parser.error("sample sizes must be 1...256 blocks and 1...64 queries")
    if any(k < 1 or k > 256 for k in args.keep):
        parser.error("keep budgets must be 1...256")
    if any(not 0 < target <= 1 for target in args.mass_targets):
        parser.error("mass targets must be in (0, 1]")
    fixture = args.fixture.resolve()
    meta = json.loads((fixture / "metadata.json").read_text())
    if (meta["schema"] != "ltx-qkv-replay-v1" or meta["dtype"] != "bf16" or
            meta["qkv_layout"] != "row-head-dim" or meta["rope_layout"] != "head-row-halfdim"):
        parser.error("unsupported capture schema/layout")
    n, heads, dim = meta["rows"], meta["heads"], meta["dim"]
    if not 1 <= n <= 16384 or not 1 <= heads <= 64 or dim != 128:
        parser.error("unsupported capture dimensions")
    arrays = {}
    shapes = {"query": (n, heads, dim), "key": (n, heads, dim), "value": (n, heads, dim),
              "cosine": (heads, n, dim // 2), "sine": (heads, n, dim // 2),
              "gate": (n, heads)}
    for name, shape in shapes.items():
        path = fixture / (name + ".bf16")
        if path.stat().st_size != int(np.prod(shape)) * 2:
            parser.error(f"unexpected size: {path}")
        arrays[name] = np.memmap(path, dtype="<u2", mode="r", shape=shape)
    blocks = (n + 63) // 64
    selected = np.unique(np.linspace(0, blocks - 1, min(blocks, args.query_blocks), dtype=int))
    samples, adaptive = [], []
    for head in range(heads):
        cos = decode(arrays["cosine"][head])
        sin = decode(arrays["sine"][head])
        q = rotate(decode(arrays["query"][:, head]), cos, sin)
        k = rotate(decode(arrays["key"][:, head]), cos, sin)
        if not np.isfinite(q).all() or not np.isfinite(k).all():
            raise ValueError("nonfinite packed Q/K")
        kc = round_bf16(block_means(k))
        v = decode(arrays["value"][:, head])
        gate = decode(arrays["gate"][:, head])
        if not np.isfinite(v).all() or not np.isfinite(gate).all():
            raise ValueError("nonfinite V or gate")
        vs = round_bf16(block_sums(v))
        kv = block_variances(k)
        for qb in selected:
            query = q[qb * 64:(qb + 1) * 64]
            offsets = np.unique(np.linspace(0, len(query) - 1,
                                           min(len(query), args.queries_per_block), dtype=int))
            mass, ranks = score_routes(query, query[offsets], k, kc, kv)
            exact_out = softmax((query[offsets].astype(np.float64) @ k.astype(np.float64).T) /
                                np.sqrt(dim)) @ v.astype(np.float64)
            distributions = {"centroid_estimate": centroid_distribution(query, kc, n),
                             "diagonal_key_variance": centroid_distribution(query, kc, n, kv),
                             "in_sample_oracle": mass.mean(axis=0)}
            for policy, distribution in distributions.items():
                for target in sorted(set(args.mass_targets)):
                    chosen = mass_budget(distribution, target)
                    adaptive.append(dict(head=head, query_block=int(qb), policy=policy,
                                         target=target, keep=len(chosen),
                                         exact_block_fraction=len(chosen) / blocks,
                                         estimated_mass=float(distribution[chosen].sum()),
                                         retained_mass=mass[:, chosen].sum(axis=1).tolist()))
            for keep in sorted(set(args.keep)):
                for policy, rank in ranks.items():
                    retained = mass[:, rank[:keep]].sum(axis=1)
                    approx = approximate_output(query[offsets], k, v, kc, vs, rank[:keep])
                    error, energy = gated_output_energy(exact_out, approx, gate[qb * 64 + offsets])
                    samples.append(dict(head=head, query_block=int(qb), policy=policy,
                                        keep=keep, effective_keep=min(keep, blocks),
                                        query_rows=(qb * 64 + offsets).tolist(),
                                        retained_mass=retained.tolist(), output_squared_error=error.tolist(),
                                        output_reference_squared_norm=energy.tolist()))
        print(f"head {head + 1}/{heads}", flush=True)
    aggregates = []
    for policy in ("centroid", "sample_logmeanexp", "diagonal_key_variance", "in_sample_oracle"):
        for keep in sorted(set(args.keep)):
            values = [v for row in samples if row["policy"] == policy and row["keep"] == keep
                      for v in row["retained_mass"]]
            rows = [r for r in samples if r["policy"] == policy and r["keep"] == keep]
            errors = np.array([v for row in rows for v in row["output_squared_error"]])
            norms = np.array([v for row in rows for v in row["output_reference_squared_norm"]])
            defined = norms > 0
            ratios = np.sqrt(errors[defined] / norms[defined])
            aggregates.append(dict(policy=policy, keep=keep, samples=len(values),
                                   mean=float(np.mean(values)), minimum=float(np.min(values)),
                                   p10=float(np.percentile(values, 10)),
                                   output_squared_error=float(errors.sum()),
                                   output_reference_squared_norm=float(norms.sum()),
                                   output_relative_l2=float(np.sqrt(errors.sum() / norms.sum())) if norms.sum() else None,
                                   zero_norm_queries=int((~defined).sum()),
                                   p90_query_output_relative_l2=float(np.percentile(ratios, 90)) if len(ratios) else None))
    adaptive_aggregates = []
    for policy in ("centroid_estimate", "diagonal_key_variance", "in_sample_oracle"):
        for target in sorted(set(args.mass_targets)):
            rows = [r for r in adaptive if r["policy"] == policy and r["target"] == target]
            retained = np.array([v for r in rows for v in r["retained_mass"]])
            fractions = [r["exact_block_fraction"] for r in rows]
            adaptive_aggregates.append(dict(policy=policy, target=target,
                mean_exact_fraction=float(np.mean(fractions)),
                p90_exact_fraction=float(np.percentile(fractions, 90)),
                mean_retained_mass=float(retained.mean()),
                p10_retained_mass=float(np.percentile(retained, 10)),
                sampled_query_target_hit_rate=float(np.mean(retained >= target - 1e-12))))
    report = dict(schema="ltx-route-recall-v4", fixture=str(fixture), metadata=meta,
                  source_sha256=digest(Path(__file__)),
                  fixture_sha256={name: digest(fixture / (name + ".bf16")) for name in shapes},
                  scope=__doc__, selected_query_blocks=selected.tolist(),
                  aggregates=aggregates, samples=samples, adaptive_samples=adaptive,
                  adaptive_aggregates=adaptive_aggregates, complete=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(aggregates, indent=2))
    print(json.dumps(adaptive_aggregates, indent=2))


if __name__ == "__main__":
    main()
