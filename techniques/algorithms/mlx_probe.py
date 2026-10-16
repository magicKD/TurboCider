#!/usr/bin/env python3
"""Explicit opt-in, synthetic MLX attention study. No weights or model quality.

Compare the SAME local-mask semantics implemented by full-length masked SDPA
and by sliced KV chunks, separately from approximation error against dense.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time


def utilization():
    raw = subprocess.run(["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"],
                         check=True, capture_output=True, text=True, timeout=10).stdout
    fields = {}
    for label in ("Device", "Renderer", "Tiler"):
        values = [int(value) for value in re.findall(r'"' + label + r' Utilization %"\s*=\s*(\d+)', raw)]
        if not values:
            raise RuntimeError("GPU utilization field unavailable: " + label)
        fields[label.lower()] = max(values)
    return {"unix_time": time.time(), **fields}


def idle_window():
    samples = []
    for index in range(3):
        samples.append(utilization())
        if index != 2:
            time.sleep(0.2)
    if max(s["device"] for s in samples) > 5:
        raise RuntimeError("GPU admission refused: Device utilization exceeds 5%; wait and rerun")
    return samples


def make_plan(seq, chunk, half, sink):
    plan = []
    for begin in range(0, seq, chunk):
        end = min(seq, begin + chunk)
        low, high = max(0, begin - half), min(seq, end + half)
        ranges = [(0, max(sink, high))] if low <= sink else [(0, sink), (low, high)]
        ranges = [(a, b) for a, b in ranges if a < b]
        indices = [i for a, b in ranges for i in range(a, b)]
        plan.append((begin, end, ranges, indices))
    return plan


def run_probe(args):
    # Read-only process evidence, never kill or alter another user's workload.
    processes = subprocess.run(["ps", "-axo", "pid,pcpu,comm"], check=True,
                               capture_output=True, text=True, timeout=10).stdout
    process_rows = []
    for line in processes.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) == 3 and re.search("python|comfy|turbocider|bench|llama", parts[2], re.I):
            process_rows.append({"pid": int(parts[0]), "cpu_percent": float(parts[1]),
                                 "executable": Path(parts[2]).name})
    admission = idle_window()
    import mlx.core as mx
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal unavailable")
    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    result = {"schema": "algorithm-mlx-probe-v1", "date": time.strftime("%Y-%m-%d"),
              "status": "completed", "model_weights_loaded": False,
              "scope": "synthetic component host wall including graph creation and eval; no media/model qualification",
              "environment": {"os": platform.mac_ver()[0], "mlx": importlib.metadata.version("mlx"),
                              "device": mx.metal.device_info()},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "seed": args.seed, "repetitions": args.repetitions, "rounds": args.rounds,
              "static_mask_setup_timed": False, "dtype": "bfloat16", "heads": 4, "head_dim": 64,
              "initial_admission": admission, "inference_process_snapshot": process_rows,
              "continuous_per_pid_gpu_monitoring": False, "workloads": []}

    def sdpa(q, k, v, mask=None):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask)

    def error(reference, candidate):
        a, b = reference.astype(mx.float32), candidate.astype(mx.float32)
        err = mx.sum((a - b) ** 2)
        energy = mx.sum(a ** 2)
        maximum = mx.max(mx.abs(a - b))
        finite = mx.all(mx.isfinite(b))
        mx.eval(err, energy, maximum, finite)
        return {"relative_l2": math.sqrt(float(err) / max(float(energy), 1e-30)),
                "max_abs": float(maximum), "finite": bool(finite)}

    for seq in args.sequences:
        for regime in ("independent-random", "position-clustered"):
            before = idle_window()
            shape = (1, 4, seq, 64)
            q, k, v = [mx.random.normal(shape) for _ in range(3)]
            if regime == "position-clustered":
                labels = mx.arange(seq) // ((seq + 15) // 16)
                basis = mx.take(mx.eye(64), labels, axis=0)[None, None, :, :]
                q, k = basis * 8 + 0.1 * q, basis * 8 + 0.1 * k
                centers = mx.random.normal((1, 4, 16, 64))
                v = mx.take(centers, labels, axis=2) + 0.1 * v
            q, k, v = [x.astype(mx.bfloat16) for x in (q, k, v)]
            mx.eval(q, k, v)
            half, sink, chunk = 128, 16, 128
            row_ids = mx.arange(seq)[:, None]
            col_ids = mx.arange(seq)[None, :]
            full_mask = (mx.abs(row_ids - col_ids) <= half) | (col_ids < sink)
            plans = make_plan(seq, chunk, half, sink)
            local_masks = []
            for begin, end, ranges, indices in plans:
                rows = mx.arange(begin, end)[:, None]
                columns = mx.array(indices)[None, :]
                local_masks.append((mx.abs(rows - columns) <= half) | (columns < sink))
            mx.eval(full_mask, *local_masks)

            def sliced():
                pieces = []
                for (begin, end, ranges, _), mask in zip(plans, local_masks):
                    kp = [k[:, :, a:b, :] for a, b in ranges]
                    vp = [v[:, :, a:b, :] for a, b in ranges]
                    ks = kp[0] if len(kp) == 1 else mx.concatenate(kp, axis=2)
                    vs = vp[0] if len(vp) == 1 else mx.concatenate(vp, axis=2)
                    pieces.append(sdpa(q[:, :, begin:end, :], ks, vs, mask))
                return mx.concatenate(pieces, axis=2)

            def all_exact_chunk():
                return mx.concatenate([sdpa(q[:, :, begin:min(seq, begin + chunk), :], k, v)
                                       for begin in range(0, seq, chunk)], axis=2)

            functions = {"dense": lambda: sdpa(q, k, v),
                         "full-local-mask": lambda: sdpa(q, k, v, full_mask),
                         "sliced-local-sink": sliced,
                         "all-exact-chunk": all_exact_chunk}
            oracle, local_oracle = functions["dense"](), functions["full-local-mask"]()
            mx.eval(oracle, local_oracle)
            errors = {}
            for name, function in functions.items():
                output = function()
                mx.eval(output)
                errors[name] = {"versus_dense": error(oracle, output),
                                "versus_local": error(local_oracle, output)}
                if not errors[name]["versus_dense"]["finite"]:
                    raise RuntimeError("nonfinite output: " + name)
            if errors["all-exact-chunk"]["versus_dense"]["relative_l2"] > 0.02:
                raise RuntimeError("all-exact control failed")
            if errors["sliced-local-sink"]["versus_local"]["relative_l2"] > 0.02:
                raise RuntimeError("equivalent local mask control failed")
            for function in functions.values():
                for _ in range(2):
                    mx.eval(function())
            timings = {name: [] for name in functions}
            order = list(functions)
            for round_index in range(args.rounds):
                rotated = order[round_index % len(order):] + order[:round_index % len(order)]
                if round_index % 2:
                    rotated.reverse()
                for _ in range(args.repetitions):
                    for name in rotated:
                        start = time.perf_counter()
                        output = functions[name]()
                        mx.eval(output)
                        timings[name].append((time.perf_counter() - start) * 1000)
            baseline = statistics.median(timings["dense"])
            stats = {name: {"median_ms": statistics.median(times), "min_ms": min(times),
                            "max_ms": max(times), "samples_ms": times,
                            "ratio_vs_dense": baseline / statistics.median(times)}
                     for name, times in timings.items()}
            result["workloads"].append({"seq": seq, "regime": regime, "window_half": half,
                                        "sink": sink, "query_chunk": chunk, "admission": before,
                                        "after": utilization(), "timings": stats, "errors": errors})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly enable GPU work")
    parser.add_argument("--output", type=Path, required=True, help="new report, never overwritten")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sequences", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if not args.run:
        parser.error("no GPU work without explicit --run")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be new")
    if (any(not 128 <= seq <= 8192 for seq in args.sequences)
            or len(args.sequences) > 4 or len(set(args.sequences)) != len(args.sequences)
            or not 1 <= args.repetitions <= 20 or not 1 <= args.rounds <= 8):
        parser.error("bounded sequences 128..8192 (max four), repetitions 1..20, rounds 1..8 required")
    try:
        result = run_probe(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, ImportError) as exc:
        result = {"schema": "algorithm-mlx-probe-v1", "status": "refused_or_failed",
                  "error": str(exc), "comparisons": None, "model_quality_qualified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "workloads": len(result.get("workloads", [])),
                      "error": result.get("error")}, ensure_ascii=False))
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
