#!/usr/bin/env python3
"""Offline per-head keep-budget allocation from replay reports.

This is an upper-bound-style policy simulation: each head's error curve is
measured independently, then a dynamic program allocates a shared total block
budget. It does not alter the Metal API (which currently has one keep count
per dispatch), does not model route-generation cost, and is not video-quality
evidence. It is useful for deciding whether head-specific routing is worth a
new production kernel.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmark_ltx_qkv_replay import digest, validate_head_metrics


def load_curve(path, mode, label):
    data = json.loads(path.read_text())
    if data.get("schema") != "ltx-qkv-replay-matrix-v1" or data.get("complete") is not True:
        raise ValueError("source is not a completed replay")
    rows = [r for r in data.get("rows", []) if r.get("mode") == mode and r.get("label") == label]
    if not rows:
        raise ValueError(f"no {label} mode {mode} rows in {path}")
    by_keep = {}
    for row in rows:
        if (row["radius"] != 0 or row["anchor_stride"] != 0 or row["tokens_per_frame"] != 0
                or row["dim"] != 128 or row["captured_input"] is not True):
            raise ValueError("expected captured direct-top-k routes with radius/anchors/frame tokens zero")
        validate_head_metrics(row)
        keep = int(row["keep_blocks"])
        if keep in by_keep:
            raise ValueError(f"duplicate keep budget {keep}")
        by_keep[keep] = row
    keeps = sorted(by_keep)
    if not keeps or keeps[0] < 1:
        raise ValueError("missing positive budgets")
    heads = by_keep[keeps[0]]["heads"]
    n = by_keep[keeps[0]]["rows"]
    if not 1 <= n <= 16384 or not 1 <= heads <= 64:
        raise ValueError("unsupported geometry")
    if any(r["heads"] != heads or r["rows"] != n for r in by_keep.values()):
        raise ValueError("head count differs")
    errors = [[by_keep[k]["per_head"][h]["squared_error"] for k in keeps] for h in range(heads)]
    norms = [by_keep[keeps[0]]["per_head"][h]["reference_squared_norm"] for h in range(heads)]
    if any(not math.isclose(by_keep[k]["per_head"][h]["reference_squared_norm"], norms[h],
                            rel_tol=1e-9, abs_tol=1e-9) for k in keeps for h in range(heads)):
        raise ValueError("reference energies differ across budgets")
    blocks = (n + 63) // 64
    costs = []
    for h in range(heads):
        current = []
        for k in keeps:
            cells = by_keep[k]["per_head"][h]["exact_block_fraction"] * blocks * blocks
            if abs(cells - round(cells)) > 0.001:
                raise ValueError("route fraction cannot recover an integer selected-cell count")
            current.append(round(cells))
        costs.append(current)
    return data, keeps, errors, norms, costs


def optimize(keeps, errors, budget, costs=None):
    heads = len(errors)
    if (not keeps or not heads or len(set(keeps)) != len(keeps) or
            any(type(k) is not int or k <= 0 for k in keeps) or
            any(len(row) != len(keeps) or any(not math.isfinite(x) or x < 0 for x in row)
                for row in errors)):
        raise ValueError("invalid error curves")
    costs = [list(keeps) for _ in errors] if costs is None else costs
    if (len(costs) != heads or any(len(row) != len(keeps) or
            any(type(x) is not int or x <= 0 for x in row) for row in costs)):
        raise ValueError("invalid costs")
    if type(budget) is not int or not 0 <= budget <= sum(max(row) for row in costs):
        raise ValueError("budget outside supported range")
    dp = np.full(budget + 1, np.inf)
    dp[0] = 0.
    choice = np.full((heads, budget + 1), -1, dtype=np.int16)
    for h in range(heads):
        current = np.full(budget + 1, np.inf)
        for index, cost in enumerate(costs[h]):
            if cost > budget:
                continue
            candidate = dp[:-cost] + errors[h][index]
            improved = candidate < current[cost:]
            current[cost:][improved] = candidate[improved]
            choice[h, cost:][improved] = index
        dp = current
    used = int(np.argmin(dp))
    if not math.isfinite(dp[used]):
        raise ValueError("no feasible allocation")
    final_error, final_cost = float(dp[used]), used
    allocation = []
    for h in range(heads - 1, -1, -1):
        index = int(choice[h, used])
        if index < 0:
            raise ValueError("invalid optimizer backtrack")
        allocation.append(keeps[index]); used -= costs[h][index]
    return list(reversed(allocation)), final_error, final_cost


def evaluate(keeps, errors, norms, costs, allocation, uniform):
    if len(allocation) != len(errors) or any(k not in keeps for k in [uniform, *allocation]):
        raise ValueError("evaluation replay lacks allocated keep budgets")
    error = sum(errors[h][keeps.index(k)] for h, k in enumerate(allocation))
    cost = sum(costs[h][keeps.index(k)] for h, k in enumerate(allocation))
    reference_error = sum(row[keeps.index(uniform)] for row in errors)
    reference_cost = sum(row[keeps.index(uniform)] for row in costs)
    norm = sum(norms)
    if norm <= 0:
        raise ValueError("undefined relative error for zero reference energy")
    return dict(relative_l2=math.sqrt(error / norm), uniform_relative_l2=math.sqrt(reference_error / norm),
                selected_cells=cost, uniform_selected_cells=reference_cost,
                selected_cell_ratio=cost / reference_cost)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--mode", type=int, default=5)
    parser.add_argument("--label", default="pooled-topk")
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 64],
                        help="uniform keep values whose actual route-cell costs define each budget")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-report", type=Path,
                        help="evaluate frozen allocation on a different replay, without reoptimizing")
    args = parser.parse_args()
    if args.output.exists() or any(x <= 0 for x in args.budgets):
        parser.error("use a new output and positive budgets")
    data, keeps, errors, norms, costs = load_curve(args.report, args.mode, args.label)
    if any(k not in keeps for k in args.budgets):
        parser.error("each baseline keep must exist in source replay")
    evaluation = load_curve(args.evaluate_report, args.mode, args.label) if args.evaluate_report else None
    result = dict(schema="ltx-head-budget-diagnostic-v1", complete=False, scope=__doc__,
                  source_report=str(args.report.resolve()), source_sha256=digest(args.report),
                  runner_sha256=digest(Path(__file__)),
                  cost_unit="exact (query block, key block) cells summed across heads; not latency",
                  mode=args.mode, label=args.label, available_keeps=keeps,
                  metadata=data.get("metadata"), allocations=[], per_head_error_curves=errors,
                  per_head_selected_cell_costs=costs)
    if evaluation:
        result["evaluation_report"] = str(args.evaluate_report.resolve())
        result["evaluation_report_sha256"] = digest(args.evaluate_report)
        result["evaluation_metadata"] = evaluation[0].get("metadata")
    for uniform in sorted(set(args.budgets)):
        budget = sum(row[keeps.index(uniform)] for row in costs)
        allocation, error, used = optimize(keeps, errors, budget, costs)
        uniform_error = sum(errors[h][keeps.index(uniform)] for h in range(len(errors)))
        total_norm = sum(norms)
        result["allocations"].append(dict(total_budget=budget, used_budget=used,
            allocation=allocation, mean_keep=sum(allocation) / len(errors),
            optimized_relative_l2=math.sqrt(error / max(total_norm, 1e-30)),
            uniform_keep=uniform,
            uniform_relative_l2=math.sqrt(uniform_error / max(total_norm, 1e-30)),
            relative_l2_improvement=1 - math.sqrt(error / uniform_error) if uniform_error else None))
        if evaluation:
            _, eval_keeps, eval_errors, eval_norms, eval_costs = evaluation
            result["allocations"][-1]["frozen_evaluation"] = evaluate(
                eval_keeps, eval_errors, eval_norms, eval_costs, allocation, uniform)
    result["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["allocations"], indent=2))


if __name__ == "__main__":
    main()
