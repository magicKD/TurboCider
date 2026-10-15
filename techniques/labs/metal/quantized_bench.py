#!/usr/bin/env python3
"""Bounded same-packed-weight GPU GEMM/FFN experiments; never product routing.

One candidate per fresh directory, explicit optional real fixture. Numerical
equivalence of implementations and quality of the quantized model are separate
gates. Dense decoded residency is charged as memory, not called free speed.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time

from affine_gemm import GUARD, actual_strides, factory, geometry
from real_bf16_source import load as load_original_bf16


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            value.update(data)
    return value.hexdigest()


def local(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("fixture asset is missing or escapes its root")
    return path


def compare(a, b, np):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("shape or finite gate failed")
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    ex, ey = float(np.sum(x * x)), float(np.sum(y * y))
    error = float(np.sum((x - y) ** 2))
    return {"relative_l2": math.sqrt(error / ex) if ex else (0. if error == 0 else None),
            "cosine": float(np.sum(x * y) / math.sqrt(ex * ey)) if ex and ey else (1. if error == 0 else 0.),
            "max_abs": float(np.max(np.abs(x - y))), "finite": True}


def decode_numpy(q, scale, bias, bits, group, np):
    """Independent exact affine arithmetic in FP64, then one FP16 storage cast."""
    if q.dtype != np.uint32 or scale.dtype != np.float16 or bias.dtype != np.float16 or bits not in (4, 8):
        raise ValueError("Q4/Q8 uint32 storage and FP16 metadata required")
    n, words = q.shape
    factor = 32 // bits
    k = words * factor
    if k % group or scale.shape != (n, k // group) or bias.shape != scale.shape:
        raise ValueError("packed row pitch or metadata group stride differs")
    codes = np.empty((n, k), dtype=np.uint8)
    for lane in range(factor):
        codes[:, lane::factor] = ((q >> (lane * bits)) & ((1 << bits) - 1)).astype(np.uint8)
    value = codes.reshape(n, k // group, group).astype(np.float64)
    value *= scale.astype(np.float64)[:, :, None]
    value += bias.astype(np.float64)[:, :, None]
    result = value.reshape(n, k).astype(np.float16)
    if not np.isfinite(result).all():
        raise ValueError("decoded FP16 storage overflow")
    return result


def witness(x, weights, scope, np):
    rows = sorted({0, x.shape[0] // 3, 2 * x.shape[0] // 3, x.shape[0] - 1})
    n = weights[-1].shape[0] if scope == "ffn" else weights[0].shape[0]
    cols = sorted({0, n // 3, 2 * n // 3, n - 1})
    if scope == "gemm":
        reference = np.einsum("mk,nk->mn", x[rows].astype(np.float64), weights[0][cols].astype(np.float64), optimize=False)
    else:
        gu = np.einsum("mk,nk->mn", x[rows].astype(np.float64), weights[0].astype(np.float64), optimize=False)
        gate, up = np.split(gu, 2, axis=1)
        value = np.exp(-np.abs(gate))
        sigmoid = np.where(gate >= 0, 1 / (1 + value), value / (1 + value))
        reference = np.einsum("mk,nk->mn", (gate * sigmoid) * up, weights[1][cols].astype(np.float64), optimize=False)
    return rows, cols, reference


def paired(samples, baseline, expected_repeats=8):
    warm = [sample for sample in samples if not sample["warmup"]]
    if (len(warm) != expected_repeats or {sample["trial"] for sample in warm} != set(range(expected_repeats))
            or not all(sample.get("same_recipe_numerical_pass") is True for sample in samples)):
        raise ValueError("incomplete or unqualified trials")
    pairs = {}
    for sample in warm:
        wall = sample["wall_seconds"]
        if type(wall) not in (int,float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("finite positive measured wall required")
        arms = pairs.setdefault(sample["pair"], {})
        if sample["variant"] in arms:
            raise ValueError("duplicate paired arm")
        arms[sample["variant"]] = sample
    ratios = []
    for arms in pairs.values():
        if set(arms) != {baseline, "candidate"} or arms[baseline]["point"] != arms["candidate"]["point"]:
            raise ValueError("incomplete or mismatched pair")
        ratios.append(arms[baseline]["wall_seconds"] / arms["candidate"]["wall_seconds"])
    rng = random.Random(37)
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000))
    return {"source_kind": "derived", "pairs": len(ratios), "paired_ratios": ratios,
            "median_paired_speedup": statistics.median(ratios), "exploratory_bootstrap_95_percentile": [boot[124], boot[4874]],
            "method": "paired ratio median, seed37, 5000 bootstrap resamples; exploratory small n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--point", action="append")
    parser.add_argument("--scope", choices=("gemm", "ffn"), default="ffn")
    parser.add_argument("--rows", type=int, default=33)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--outputs", type=int, default=41, help="synthetic GEMM N only")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--group", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--candidate", choices=("decoded-cache", "decode-call", "sg32", "sg64", "sg64-coalesced", "sg64-skew", "packed"), default="decoded-cache")
    parser.add_argument("--baseline", choices=("packed", "fp16", "bf16"), default="packed")
    parser.add_argument("--source-checkpoint", type=Path, help="explicit original BF16 weights for the bf16 baseline")
    parser.add_argument("--capture-root", type=Path, help="explicit completed original BF16 capture for the bf16 baseline")
    parser.add_argument("--input-layout", choices=("contiguous", "stride2"), default="contiguous")
    parser.add_argument("--repeats", type=int, choices=(4, 8, 12, 16), default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin" or args.candidate == args.baseline:
        parser.error("requires actual macOS GPU and distinct arms")
    if args.fixture and args.scope != "ffn":
        parser.error("the real fixture contract is a complete FFN, not a sliced GEMM")
    if args.baseline == "bf16" and not (args.fixture and args.source_checkpoint and args.capture_root):
        parser.error("original BF16 baseline needs real fixture, explicit checkpoint and completed capture root")
    if args.point and len(set(args.point)) != len(args.point):
        parser.error("point IDs must be unique")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), Path(__file__).with_name("affine_gemm.py"), Path(__file__).with_name("affine_gemm.metal"),
                    Path(__file__).with_name("real_bf16_source.py")]
    (output / "source").mkdir()
    snapshots = []
    for source in source_files:
        shutil.copy2(source, output / "source" / source.name)
        snapshots.append({"path": "source/" + source.name, "sha256": sha(source)})
    report = {"schema": "tc-quantized-gpu-experiment-v1", "status": "starting", "phase": "setup",
              "recorded_at_singapore": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + 8 * 3600)) + "+08:00",
              "synthetic": args.fixture is None, "production_qualified": False, "source_snapshots": snapshots,
              "scope": args.scope, "baseline": args.baseline, "candidate": args.candidate,
              "bits": args.bits, "group_size": args.group, "input_layout": args.input_layout,
              "same_encoded_weights": args.baseline == "packed", "samples": [], "errors": [],
              "repeats": args.repeats, "warmup_trials": 4, "order": "ABBA",
              "gates": {"full_relative_l2_max": .02, "independent_sample_relative_l2_max": .02,
                        "original_bf16_relative_l2_max": .03, "original_bf16_cosine_min": .999},
              "limits": ["FP16-source affine quantization, not GGML Q4_0/Q8_0 or default BF16 weights.",
                         "Same encoded storage is algebraically equivalent, not a claim of bitwise decode/accumulation.",
                         "Resident source per process; cache is not reusable across source/LoRA/layout changes.",
                         "Both arms share prepared candidate allocations; isolated per-arm memory peak not measured.",
                         "Custom kernels include output poison/guard initialization in their measured boundary.",
                         "No original-model quality or request performance qualification from component timing."]}
    receipt = output / "report.json"

    def save():
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    mx = None
    try:
        import numpy as np
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        free = shutil.disk_usage(output).free
        report["disk_admission"] = {"available_bytes": free, "minimum_bytes": 512 * 1024 * 1024}
        if free < 512 * 1024 * 1024:
            raise ValueError("insufficient disk headroom; no other data deleted")
        info = mx.device_info()
        report["environment"] = {"macos": platform.mac_ver()[0], "python": platform.python_version(),
                                  "mlx": version("mlx"), "numpy": np.__version__,
                                  "device": {key: info.get(key) for key in ("device_name", "architecture", "memory_size", "max_recommended_working_set_size")},
                                  "library_assets": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
                                                     for path in [Path(mx.__file__), *sorted((Path(mx.__file__).parent / "lib").glob("*"))]
                                                     if path.is_file() and path.suffix in (".so", ".dylib", ".metallib")]}
        if args.fixture:
            root = args.fixture.resolve().parent
            fixture = json.loads(args.fixture.read_text())
            if fixture.get("schema") != "tc-real-ffn-fixture-v1" or fixture.get("loras") != []:
                raise ValueError("real base FFN fixture required")
            for name, expected in fixture["files"].items():
                if sha(local(root, name)) != expected:
                    raise ValueError("fixture content changed: " + name)
            weights = [np.load(local(root, fixture[name]), allow_pickle=False) for name in ("gate_up", "down")]
            m, h, f = fixture["shape_mhf"]
            points = [{"id": point["id"], "x": np.load(local(root, point["input"]), allow_pickle=False),
                       "original": np.load(local(root, point["original_reference"]), allow_pickle=False)}
                      for point in fixture["points"] if args.point is None or point["id"] in args.point]
            if not points or (args.point and set(args.point) != {point["id"] for point in points}):
                raise ValueError("missing requested real point")
            report["fixture_sha256"] = sha(args.fixture)
        else:
            m, h, f = args.rows, args.hidden, args.width
            rng = np.random.default_rng(17)
            shapes = [(2 * f, h), (h, f)] if args.scope == "ffn" else [(args.outputs, h)]
            weights = [rng.normal(0, .2 / math.sqrt(shape[1]), shape).astype(np.float16) for shape in shapes]
            points = [{"id": "synthetic-17", "x": rng.normal(0, .3, (m, h)).astype(np.float16), "original": None}]
        for matrix in weights:
            geometry(m, matrix.shape[0], matrix.shape[1], args.bits, args.group, 32)
            if matrix.dtype != np.float16 or not np.isfinite(matrix).all():
                raise ValueError("finite FP16 weight source required")
        if sum(matrix.nbytes for matrix in weights) > 512 * 1024 * 1024:
            raise ValueError("raw weight admission exceeded")
        if args.scope == "ffn" and (weights[0].shape != (2 * f, h) or weights[1].shape != (h, f)):
            raise ValueError("unmodified paired gate/up and matching complete down required")
        for point in points:
            if point["x"].shape != (m, h) or point["x"].dtype != np.float16 or not np.isfinite(point["x"]).all():
                raise ValueError("complete finite FP16 activation required")
        original_bf16_weights, gpu_bf16 = None, None
        if args.baseline == "bf16":
            original_bf16_weights = load_original_bf16(args.source_checkpoint, args.capture_root, fixture, points, np)
            gpu_bf16 = [mx.array(value, dtype=mx.bfloat16) for value in original_bf16_weights]
            mx.eval(gpu_bf16)
            report["original_source"] = {"selected_gate_up_tensor_sha256": fixture["original_gate_up_tensor_sha256"],
                                         "selected_down_tensor_sha256": fixture["original_down_tensor_sha256"],
                                         "capture_receipt_sha256": fixture["capture_receipt_sha256"],
                                         "input_output_abi": "original [1,M,H] BF16 input/output; candidate input cast and output restore included"}
        report["shape"] = {"m": m, "h": h, "f": f if args.scope == "ffn" else None, "projections_out_in": [list(w.shape) for w in weights]}
        gpu_original = [mx.array(matrix) for matrix in weights]
        mx.eval(gpu_original)
        started = time.perf_counter()
        packed = [mx.quantize(value, group_size=args.group, bits=args.bits, mode="affine") for value in gpu_original]
        mx.eval([value for triplet in packed for value in triplet])
        report["quantize_prepare_seconds"] = time.perf_counter() - started
        numpy_packed = [[np.array(value) for value in triplet] for triplet in packed]
        decoded = [decode_numpy(*values, args.bits, args.group, np) for values in numpy_packed]
        report["packed_identities"] = [{"parts": [{"dtype": str(value.dtype), "shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                                                    for value in values]} for values in numpy_packed]
        started = time.perf_counter()
        gpu_decoded = [mx.dequantize(*triplet, group_size=args.group, bits=args.bits, mode="affine") for triplet in packed]
        mx.eval(gpu_decoded)
        report["decode_prepare_seconds"] = time.perf_counter() - started
        report["dequantize_vs_cpu_storage"] = [compare(reference, np.array(value), np) for reference, value in zip(decoded, gpu_decoded)]
        packed_bytes = sum(value.nbytes for triplet in packed for value in triplet)
        decoded_bytes = sum(value.nbytes for value in gpu_decoded)
        report["logical_memory_bytes"] = {"source_kind": "derived", "packed_and_metadata": packed_bytes,
                                           "decoded_fp16": decoded_bytes, "packed_plus_decoded_cache": packed_bytes + decoded_bytes,
                                           "original_fp16": sum(value.nbytes for value in gpu_original),
                                           "original_bf16": None if gpu_bf16 is None else sum(value.nbytes for value in gpu_bf16), "hard_bound": False}
        if args.candidate.startswith("sg"):
            report["custom_geometry"] = [geometry(m, matrix.shape[0], matrix.shape[1], args.bits, args.group,
                                                   32 if args.candidate == "sg32" else 64, 8 if args.candidate == "sg64-skew" else 0)
                                          for matrix in weights]

        def dense(x, matrices):
            if args.scope == "gemm":
                return x @ matrices[0].T
            gate, up = mx.split(x @ matrices[0].T, 2, axis=-1)
            return ((gate * mx.sigmoid(gate)) * up) @ matrices[1].T

        report["points"] = []
        for point in points:
            point["witnesses"] = {}
            point["references"] = {}
            references = [("quantized", decoded), ("fp16", weights)]
            if original_bf16_weights is not None:
                references.append(("bf16", original_bf16_weights))
            for label, matrices in references:
                source_x = point["x_bf16_source"] if label == "bf16" else point["x"]
                ref_weights = [mx.array(value, dtype=mx.float32) for value in matrices]
                reference = dense(mx.array(source_x, dtype=mx.float32), ref_weights)
                mx.eval(reference)
                point["references"][label] = np.array(reference)
                rows, cols, cpu = witness(source_x, matrices, args.scope, np)
                point["witnesses"][label] = (rows, cols, cpu)
                metric = compare(cpu, point["references"][label][np.ix_(rows, cols)], np)
                if metric["relative_l2"] is None or metric["relative_l2"] > 1e-4:
                    raise ValueError("FP64 witness rejected reference")
                report["points"].append({"id": point["id"], "reference_kind": label, "rows": rows, "columns": cols,
                                         "reference_vs_fp64": metric, "input_sha256": hashlib.sha256(source_x.tobytes()).hexdigest(),
                                         "reference_input_encoding": "exact BF16 values represented in FP32" if label == "bf16" else "FP16"})
                del ref_weights, reference
        q_arguments = [value for triplet in packed for value in triplet]
        custom = factory(mx, args.bits, args.group, 32 if args.candidate == "sg32" else 64,
                         coalesced=args.candidate in ("sg64-coalesced", "sg64-skew"),
                         pad=8 if args.candidate == "sg64-skew" else 0) if args.candidate.startswith("sg") else None

        def implementation(method):
            def apply(x, *parameters):
                guards = []
                if method == "bf16":
                    return (dense(x, gpu_bf16),)
                if args.baseline == "bf16":
                    x = x.reshape(m, h).astype(mx.float16)

                def restore(value):
                    return value.reshape(1, m, h).astype(mx.bfloat16) if args.baseline == "bf16" else value

                if method == "fp16":
                    return (restore(dense(x, gpu_original)),)
                if method == "decoded-cache":
                    return (restore(dense(x, gpu_decoded)),)
                if method == "decode-call":
                    matrices = [mx.dequantize(*parameters[i:i + 3], group_size=args.group, bits=args.bits, mode="affine")
                                for i in range(0, len(parameters), 3)]
                    return (restore(dense(x, matrices)), *matrices)

                def projection(value, slot):
                    q, scale, bias = parameters[slot * 3:slot * 3 + 3]
                    if method == "packed":
                        return mx.quantized_matmul(value, q, scale, bias, transpose=True, group_size=args.group, bits=args.bits, mode="affine")
                    result, guard = custom(value, q, scale, bias)
                    guards.append(guard)
                    return result

                output = projection(x, 0)
                if args.scope == "ffn":
                    gate, up = mx.split(output, 2, axis=-1)
                    output = projection((gate * mx.sigmoid(gate)) * up, 1)
                return (restore(output), *guards)
            return mx.compile(apply)

        baseline_fn, candidate_fn = implementation(args.baseline), implementation(args.candidate)
        if args.baseline != "fp16":
            gpu_original = None
        if args.candidate != "decoded-cache":
            gpu_decoded = None
        del numpy_packed, decoded, weights, original_bf16_weights
        report["boundary"] = ("ready original BF16 [1,M,H] input -> complete BF16 [1,M,H] output; candidate FP16 input conversion and BF16 output restore included"
                              if args.baseline == "bf16" else "ready resident MLX FP16 input -> complete GEMM/full FFN output")
        report["completion"] = "mx.eval all outputs; no CPU readback or metrics cast in timed span"
        report["decode_call_contract"] = "packed arrays are dynamic compiled inputs; decoded matrices returned/materialized, no intermediate host fence; physical dispatch count unprofiled"
        report["cache_contract"] = "decode setup outside warm span, charged separately; immutable one-source packed identity; additional residency, no automatic cross-layer cache"
        report["phase"] = "measure"
        save()
        model_quality = []
        for trial in range(-4, args.repeats):
            candidate = (trial + 4) % 4 in (1, 2)
            point = points[((trial + 4) // 2) % len(points)]
            method = args.candidate if candidate else args.baseline
            source_x = point["x_bf16_source"] if args.baseline == "bf16" else point["x"]
            dtype = mx.bfloat16 if args.baseline == "bf16" else mx.float16
            if args.input_layout == "stride2":
                padded = np.zeros((m, h * 2), dtype=source_x.dtype)
                padded[:, ::2] = source_x
                device_x = mx.as_strided(mx.array(padded, dtype=dtype), shape=(m, h), strides=(h * 2, 2))
            else:
                device_x = mx.array(source_x, dtype=dtype)
            mx.eval(device_x)
            if trial == -4:
                report["actual_input_strides_elements"] = actual_strides(mx, device_x)
            if args.baseline == "bf16":
                device_x = device_x.reshape(1, m, h)
            started = time.perf_counter()
            values = (candidate_fn if candidate else baseline_fn)(device_x, *q_arguments)
            mx.eval(values)
            wall = time.perf_counter() - started
            if args.baseline == "bf16":
                if values[0].shape != (1, m, h) or values[0].dtype != mx.bfloat16:
                    raise ValueError("original BF16 consumer ABI not restored")
                actual = np.array(values[0].astype(mx.float32)).reshape(m, h)
            else:
                actual = np.array(values[0])
            if method.startswith("sg") and not all(bool(mx.all(value == GUARD).item()) for value in values[1:]):
                raise ValueError("custom output tail guard failed")
            label = method if method in ("fp16", "bf16") else "quantized"
            metric = compare(point["references"][label], actual, np)
            rows, cols, cpu = point["witnesses"][label]
            independent = compare(cpu, actual[np.ix_(rows, cols)], np)
            original = None if point["original"] is None else compare(point["original"], actual, np)
            numeric = metric["relative_l2"] is not None and metric["relative_l2"] <= .02 and independent["relative_l2"] is not None and independent["relative_l2"] <= .02
            quality = None if original is None else original["relative_l2"] is not None and original["relative_l2"] <= .03 and original["cosine"] >= .999
            model_quality.append(quality)
            report["samples"].append({"trial": trial, "warmup": trial < 0, "pair": trial // 2, "point": point["id"],
                                      "variant": "candidate" if candidate else args.baseline, "method": method,
                                      "wall_seconds": wall, "wall_source_kind": "measured", "same_recipe_numerical_pass": numeric,
                                      "vs_same_recipe_fp32": metric, "vs_independent_fp64": independent,
                                      "vs_original_bf16_replica": original, "original_component_budget_pass": quality,
                                      "guard": "pass" if method.startswith("sg") else "library-output-finite-check"})
            save()
            print(f'trial={trial} method={method} wall={wall:.6f}s numeric={numeric} original_budget={quality}', file=sys.stderr, flush=True)
            if not numeric:
                raise ValueError("same-recipe numerical gate failed; withhold timing ratios")
        statistics_value = paired(report["samples"], args.baseline, args.repeats)
        report["statistics"] = statistics_value
        report["statistics_scope"] = "same-encoded-weight implementation comparison only" if args.baseline == "packed" else "approximate quantized candidate vs original " + args.baseline + " component"
        report["original_component_budget_pass"] = all(value is True for value in model_quality) if args.fixture else None
        if args.baseline in ("fp16", "bf16") and not report["original_component_budget_pass"]:
            report.pop("statistics")
            report["statistics_withheld"] = "original-source component quality budget not passed"
        report["warm_median_seconds"] = {name: statistics.median(sample["wall_seconds"] for sample in report["samples"] if not sample["warmup"] and sample["variant"] == name)
                                          for name in (args.baseline, "candidate")}
        report["status"] = "same_recipe_numeric_pass"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"phase": report["phase"], "type": type(error).__name__, "message": str(error)})
    finally:
        if mx is not None:
            mx.synchronize()
            report["mlx_allocator_peak_bytes"] = {"value": mx.get_peak_memory(), "source_kind": "api-reported", "scope": "whole process including references, not isolated per-arm or system peak"}
        if any(sha(path) != saved["sha256"] for path, saved in zip(source_files, snapshots)):
            report["status"] = "failed"
            report["errors"].append({"phase": "source-identity", "type": "ValueError", "message": "source changed during run"})
            report.pop("statistics", None)
        report["process_peak_rss_bytes"] = {"value": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "source_kind": "measured", "scope": "macOS process, references/setup included"}
        report["phase"] = "terminal"
        save()
    print(json.dumps({key: value for key, value in report.items() if key not in ("samples", "points")}, indent=2))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
