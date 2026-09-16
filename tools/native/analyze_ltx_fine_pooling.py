#!/usr/bin/env python3
"""Sampled fine-remote-summary diagnostic, not Metal timing or video quality.

Route exact blocks using the unchanged 64-token centroid top-k policy, with
no safety regions. Only remote K means / V sums change granularity. Summaries
use sequential FP32 accumulation and BF16 storage; attention math is FP64.
Representation counts are a work proxy, not GPU FLOPs, latency or speedup.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_ltx_route_recall import (decode, rotate, round_bf16, block_means,
                                    gated_output_energy, softmax)
from benchmark_ltx_qkv_replay import digest


def summaries(key, value, width):
    if type(width) is not int or width not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("summary width must divide 64 and be positive")
    if len(key) == 0 or len(key) != len(value):
        raise ValueError("invalid summary input lengths")
    count = (len(key) + width - 1) // width
    counts = np.minimum(width, len(key) - np.arange(count) * width)
    def sums(x):
        out = np.zeros((count, x.shape[1]), np.float32)
        for offset in range(width):
            part = x[offset::width]
            out[:len(part)] += part
        return out
    means = sums(key) * (1 / counts).astype(np.float32)[:, None]
    return round_bf16(means), round_bf16(sums(value)), counts


def fine_output(query, key, value, selected, width, pooled=None):
    """Keep selected parent blocks exact, pool only their remote complement."""
    kc, vs, counts = summaries(key, value, width) if pooled is None else pooled
    blocks = (len(key) + 63) // 64
    selected = np.asarray(selected)
    if selected.ndim != 1 or (selected.size and
            (not np.issubdtype(selected.dtype, np.integer) or
             np.any(selected < 0) or np.any(selected >= blocks))):
        raise ValueError("invalid parent block indices")
    keep = np.zeros(blocks, dtype=bool)
    keep[selected.astype(int)] = True
    exact = keep[np.arange(len(key)) // 64]
    remote = ~keep[np.arange(len(counts)) * width // 64]
    logits = np.concatenate((query.astype(np.float64) @ key[exact].astype(np.float64).T,
                              query.astype(np.float64) @ kc[remote].astype(np.float64).T), axis=1)
    logits /= np.sqrt(key.shape[1])
    exact_count = int(exact.sum())
    logits[:, exact_count:] += np.log(counts[remote])
    values = np.concatenate((value[exact].astype(np.float64),
                              vs[remote].astype(np.float64) / counts[remote, None]), axis=0)
    return softmax(logits) @ values, dict(exact_tokens=exact_count,
        remote_summaries=int(remote.sum()), represented_keys=values.shape[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 32, 16, 8])
    parser.add_argument("--keep", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--query-blocks", type=int, default=8)
    parser.add_argument("--queries-per-block", type=int, default=4)
    args = parser.parse_args()
    if (args.output.exists() or any(w not in (1, 2, 4, 8, 16, 32, 64) for w in args.widths)
            or any(not 1 <= k <= 256 for k in args.keep) or
            not 1 <= args.query_blocks <= 256 or not 1 <= args.queries_per_block <= 64):
        parser.error("use new output and supported widths, budgets and sample counts")
    fixture = args.fixture.resolve()
    meta = json.loads((fixture / "metadata.json").read_text())
    if (meta["schema"] != "ltx-qkv-replay-v1" or meta["dtype"] != "bf16" or
            meta["qkv_layout"] != "row-head-dim" or meta["rope_layout"] != "head-row-halfdim"):
        parser.error("unsupported fixture layout")
    n, heads, dim = meta["rows"], meta["heads"], meta["dim"]
    if not 1 <= n <= 16384 or not 1 <= heads <= 64 or dim != 128:
        parser.error("unsupported fixture shape")
    shapes = {"query": (n, heads, dim), "key": (n, heads, dim), "value": (n, heads, dim),
              "cosine": (heads, n, dim // 2), "sine": (heads, n, dim // 2), "gate": (n, heads)}
    arrays = {}
    for name, shape in shapes.items():
        path = fixture / (name + ".bf16")
        if path.stat().st_size != int(np.prod(shape)) * 2:
            parser.error(f"fixture byte count mismatch: {path}")
        arrays[name] = np.memmap(path, dtype="<u2", mode="r", shape=shape)
    blocks = (n + 63) // 64
    selected_queries = np.unique(np.linspace(0, blocks - 1, min(blocks, args.query_blocks), dtype=int))
    result = dict(schema="ltx-fine-pooling-v1", complete=False, scope=__doc__, metadata=meta,
        fixture=str(fixture), source_sha256=digest(Path(__file__)),
        helper_sha256=digest(Path(__file__).with_name("analyze_ltx_route_recall.py")),
        fixture_sha256={name: digest(fixture / (name + ".bf16")) for name in shapes},
        selected_query_blocks=selected_queries.tolist(), samples=[], aggregates=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    for head in range(heads):
        cosine, sine = decode(arrays["cosine"][head]), decode(arrays["sine"][head])
        q = rotate(decode(arrays["query"][:, head]), cosine, sine)
        k = rotate(decode(arrays["key"][:, head]), cosine, sine)
        v, gate = decode(arrays["value"][:, head]), decode(arrays["gate"][:, head])
        if any(not np.isfinite(x).all() for x in (q, k, v, gate)):
            raise ValueError("nonfinite fixture")
        coarse = round_bf16(block_means(k))
        pooled = {w: summaries(k, v, w) for w in args.widths}
        for qb in selected_queries:
            query_block = q[qb * 64:(qb + 1) * 64]
            offsets = np.unique(np.linspace(0, len(query_block) - 1,
                min(args.queries_per_block, len(query_block)), dtype=int))
            queries = query_block[offsets]
            rank = np.argsort(-(block_means(query_block)[0].astype(np.float64) @ coarse.T), kind="stable")
            reference = softmax(queries.astype(np.float64) @ k.astype(np.float64).T /
                                np.sqrt(dim)) @ v.astype(np.float64)
            for keep in sorted(set(args.keep)):
                for width in sorted(set(args.widths), reverse=True):
                    output, work = fine_output(queries, k, v, rank[:keep], width, pooled[width])
                    error, norm = gated_output_energy(reference, output, gate[qb * 64 + offsets])
                    result["samples"].append(dict(head=head, query_block=int(qb), keep=keep,
                        width=width, effective_keep=min(keep, blocks), **work,
                        query_rows=(qb * 64 + offsets).tolist(),
                        output_squared_error=error.tolist(), reference_squared_norm=norm.tolist()))
        save()
        print(f"head {head + 1}/{heads}", flush=True)
    for keep in sorted(set(args.keep)):
        for width in sorted(set(args.widths), reverse=True):
            rows = [r for r in result["samples"] if r["keep"] == keep and r["width"] == width]
            errors = np.array([v for r in rows for v in r["output_squared_error"]])
            norms = np.array([v for r in rows for v in r["reference_squared_norm"]])
            defined = norms > 0
            ratios = np.sqrt(errors[defined] / norms[defined])
            result["aggregates"].append(dict(keep=keep, width=width, queries=len(errors),
                output_squared_error=float(errors.sum()), reference_squared_norm=float(norms.sum()),
                output_relative_l2=float(np.sqrt(errors.sum() / norms.sum())) if norms.sum() else None,
                p90_query_relative_l2=float(np.percentile(ratios, 90)) if len(ratios) else None,
                zero_norm_queries=int((~defined).sum()),
                mean_represented_keys=float(np.mean([r["represented_keys"] for r in rows])),
                mean_represented_fraction=float(np.mean([r["represented_keys"] / n for r in rows]))))
    result["complete"] = True
    save()
    print(json.dumps(result["aggregates"]), flush=True)


if __name__ == "__main__":
    main()
