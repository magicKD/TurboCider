#!/usr/bin/env python3
"""Isolate mode-selective summary reads without changing production kernels."""
from benchmark_ltx_query_cache import main


def selective_summary_shader(source):
    start = "kernel void ltx_sol_reduce_summaries_bf16("
    end = "kernel void ltx_sol_thresholds_diag_bf16("
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("summary kernel layout changed")
    prefix, remainder = source.split(start)
    body, suffix = remainder.split(end)
    replacements = {
        "    uint begin = block * 64u;":
            "    if (args.mode == 1u) return;\n"
            "    uint begin = block * 64u;",
        "        query_sum += ltx_bf16_to_f32(query[index]);":
            "        if (args.mode != 2u) query_sum += ltx_bf16_to_f32(query[index]);",
        "        value_sum += ltx_bf16_to_f32(value[index]);":
            "        if (args.mode != 4u) value_sum += ltx_bf16_to_f32(value[index]);",
        "    query_centroids[summary] = query_sum * inverse;":
            "    if (args.mode != 2u) query_centroids[summary] = query_sum * inverse;",
        "    value_sums[summary] = ltx_f32_to_bf16(value_sum);":
            "    if (args.mode != 4u) value_sums[summary] = ltx_f32_to_bf16(value_sum);",
    }
    for old, new in replacements.items():
        if body.count(old) != 1 or new in body:
            raise ValueError("summary kernel layout changed; review transform")
        body = body.replace(old, new)
    return prefix + start + body + end + suffix


if __name__ == "__main__":
    main(transform=selective_summary_shader, variant="selective-summaries",
         configurations=(("sol", 0, 1, 0), ("structured-drop", 1, 1, 0),
                         ("structured-pooled", 2, 1, 0), ("cider", 3, 1, 0),
                         ("topk32", 4, 0, 32), ("pooled-topk32", 5, 0, 32)))
