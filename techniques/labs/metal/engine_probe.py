#!/usr/bin/env python3
"""Bounded MLX mechanism probes; synthetic inputs, not an engine/model ranking."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time


def summarize(samples):
    pairs = []
    if len(samples) % 4:
        raise ValueError("complete ABBA blocks required")
    for offset in range(0, len(samples), 4):
        block = samples[offset:offset + 4]
        if [x["arm"] for x in block] != ["baseline", "candidate", "candidate", "baseline"]:
            raise ValueError("invalid ABBA order")
        a = statistics.mean(x["ms"] for x in block if x["arm"] == "baseline")
        b = statistics.mean(x["ms"] for x in block if x["arm"] == "candidate")
        if not 0 < a < float("inf") or not 0 < b < float("inf"):
            raise ValueError("invalid timing")
        pairs.append(a / b)
    return {"baseline_median_ms": statistics.median(x["ms"] for x in samples if x["arm"] == "baseline"),
            "candidate_median_ms": statistics.median(x["ms"] for x in samples if x["arm"] == "candidate"),
            "paired_ratios": pairs, "paired_median_speedup": statistics.median(pairs)}


def error_metrics(np, reference, value):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(value, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("shape/nonfinite failure")
    norm = float(np.linalg.norm(a.ravel()))
    return {"max_abs": float(np.max(np.abs(a - b))),
            "relative_l2": float(np.linalg.norm((a - b).ravel()) / max(norm, 1e-30)),
            "elements": int(a.size), "finite": True}


def affine_oracle(np, q, scales, biases, bits, group_size=64):
    """Decode MLX power-of-two affine packing independently on the CPU."""
    q = np.asarray(q, dtype=np.uint32)
    unpacked = ((q[..., None] >> (np.arange(32 // bits, dtype=np.uint32) * bits)) & ((1 << bits) - 1))
    unpacked = unpacked.reshape(q.shape[0], -1).astype(np.float64)
    scales = np.repeat(np.asarray(scales, dtype=np.float64), group_size, axis=-1)
    biases = np.repeat(np.asarray(biases, dtype=np.float64), group_size, axis=-1)
    if unpacked.shape != scales.shape or unpacked.shape != biases.shape:
        raise ValueError("packing/group mismatch")
    return unpacked * scales + biases


def run_pair(mx, np, name, baseline, candidate, oracle, blocks, limit=0.02, metadata=None):
    outputs = {}
    first_ms = {}
    for arm, fn in (("baseline", baseline), ("candidate", candidate)):
        mx.synchronize()
        start = time.perf_counter_ns()
        out = fn()
        mx.eval(out)
        first_ms[arm] = (time.perf_counter_ns() - start) / 1e6
        outputs[arm] = np.array(out).astype(np.float64)
    checks = {arm: error_metrics(np, oracle, out) for arm, out in outputs.items()}
    checks["candidate_vs_baseline"] = error_metrics(np, outputs["baseline"], outputs["candidate"])
    if any(v["relative_l2"] > limit for v in checks.values()):
        raise ValueError(f"{name}: numerical gate failed: {checks}")
    for _ in range(3):
        mx.eval(baseline())
        mx.eval(candidate())
    samples = []
    for block in range(blocks):
        for arm, fn in (("baseline", baseline), ("candidate", candidate), ("candidate", candidate), ("baseline", baseline)):
            start = time.perf_counter_ns()
            out = fn()
            mx.eval(out)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            # Check every measured output; reading is outside the timed span.
            check = error_metrics(np, oracle, np.array(out))
            if check["relative_l2"] > limit:
                raise ValueError(f"{name}: measured output failed")
            samples.append({"block": block, "arm": arm, "ms": elapsed, "relative_l2": check["relative_l2"]})
    return {"name": name, "metadata": metadata or {}, "first_call_ms_not_cold_process": first_ms,
            "checks": checks, "relative_l2_limit": limit, "samples": samples, **summarize(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.blocks <= 32:
        parser.error("blocks must be 2..32")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; use a fresh path")
    import numpy as np
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rng = np.random.default_rng(args.seed)
    report = {"schema": 1, "date": time.strftime("%Y-%m-%d"), "seed": args.seed,
              "os": platform.mac_ver()[0], "mlx": importlib.metadata.version("mlx"),
              "numpy": importlib.metadata.version("numpy"), "device": mx.device_info(),
              "source_sha256": source_hash, "production_qualified": False,
              "scope": "synthetic resident warm host latency including graph construction, dispatch and mx.eval; CPU conversion/checks excluded; no counters, model E2E or engine ranking",
              "arms_share_prepared_allocations": True, "runs": []}
    binary = Path(mx.__file__)
    report["core_binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    try:
        # Same FP32 algebra; compilation may fuse and change rounding.
        for rows in (1, 128, 1024):
            xnp = rng.normal(size=(rows, 1024)).astype(np.float32)
            wnp = rng.uniform(0.8, 1.2, size=(1024,)).astype(np.float32)
            x, w = mx.array(xnp), mx.array(wnp)
            mx.eval(x, w)
            def norm(a, scale):
                y = a * mx.rsqrt(mx.mean(a * a, axis=-1, keepdims=True) + 1e-5)
                return mx.tanh(y * scale) + a * 0.125
            compiled = mx.compile(norm)
            a64 = xnp.astype(np.float64)
            oracle = np.tanh(a64 / np.sqrt(np.mean(a64 * a64, axis=-1, keepdims=True) + 1e-5) * wnp) + a64 * 0.125
            report["runs"].append(run_pair(mx, np, f"compile-rms-chain-M{rows}", lambda: norm(x, w), lambda: compiled(x, w),
                oracle, args.blocks, limit=1e-5, metadata={"shape": [rows, 1024], "dtype": "float32", "baseline": "eager lazy expression", "candidate": "mx.compile same expression"}))
        # Packed and predecoded use the SAME encoded weights, not original FP16.
        wn = rng.normal(0, 1 / 32, size=(1024, 1024)).astype(np.float16)
        weight = mx.array(wn)
        mx.eval(weight)
        for bits in (4, 8):
            q, scales, biases = mx.quantize(weight, group_size=64, bits=bits)
            decoded = mx.dequantize(q, scales, biases, group_size=64, bits=bits)
            mx.eval(q, scales, biases, decoded)
            canonical = affine_oracle(np, np.array(q), np.array(scales), np.array(biases), bits)
            stored_bytes = int(q.nbytes + scales.nbytes + biases.nbytes)
            for rows in (1, 16, 256, 1024):
                xn = rng.normal(size=(rows, 1024)).astype(np.float16)
                x = mx.array(xn)
                mx.eval(x)
                oracle = np.einsum("mk,nk->mn", xn.astype(np.float64), canonical, optimize=False)
                report["runs"].append(run_pair(mx, np, f"affine{bits}-packed-vs-resident-M{rows}",
                    lambda: mx.quantized_matmul(x, q, scales, biases, transpose=True, group_size=64, bits=bits),
                    lambda: x @ decoded.T, oracle, args.blocks,
                    metadata={"M": rows, "N": 1024, "K": 1024, "bits": bits, "group_size": 64,
                              "packed_bytes": stored_bytes, "extra_resident_decoded_bytes": int(decoded.nbytes),
                              "baseline": "packed quantized_matmul", "candidate": "predecoded FP16 matmul; decode excluded", 
                              "unquantized_model_quality_tested": False}))
        for qlen, klen, dim in ((1, 4096, 64), (512, 512, 64), (192, 257, 80)):
            arrays = [rng.normal(size=(1, 4, n, dim)).astype(np.float16) for n in (qlen, klen, klen)]
            q, k, v = [mx.array(a) for a in arrays]
            mx.eval(q, k, v)
            a, b, c = [a.astype(np.float64) for a in arrays]
            score = np.einsum("bhqd,bhkd->bhqk", a, b, optimize=False) * dim**-0.5
            prob = np.exp(score - np.max(score, axis=-1, keepdims=True))
            prob /= np.sum(prob, axis=-1, keepdims=True)
            oracle = np.einsum("bhqk,bhkd->bhqd", prob, c, optimize=False)
            def materialized():
                scores = (q * dim**-0.5) @ mx.swapaxes(k, -1, -2)
                return mx.softmax(scores.astype(mx.float32), axis=-1).astype(mx.float16) @ v
            report["runs"].append(run_pair(mx, np, f"sdpa-Q{qlen}-K{klen}-D{dim}", materialized,
                lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=dim**-0.5), oracle, args.blocks,
                limit=0.005, metadata={"shape_q": list(q.shape), "shape_k": list(k.shape), "dtype": "float16", "mask": "none", "gqa": 1,
                                      "baseline": "materialized QK/FP32 softmax/PV", "candidate": "mx.fast.sdpa; actual kernel not captured"}))
        if source_hash != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("source changed during run")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]), "production_qualified": False}))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
