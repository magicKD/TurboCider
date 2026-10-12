#!/usr/bin/env python3
"""Public Core ML W8A8 candidate screen; raw failures remain evidence.

One loaded model per arm; fixed graph A/B/A weight swaps. Compute-plan intent
and physical arithmetic are reported separately. No product route is changed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import struct
import time

import numpy as np
from convrot_w8a8_math import (fp16_ulp_distance, integer_dot, normalize,
                                normalized_output, quantize_rows, restore_normalized, rotate)

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "runtime_w8a8_export", ROOT / "tools/coreml/export_runtime_w8a8_probe.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def convrot_slice(path, prefix, outputs, hidden):
    """Bounded offline slice reader, NOT a verified product SourceLease."""
    if any(type(value) is not int or not 1 <= value <= 32768 for value in (outputs, hidden)):
        raise ValueError("invalid projection slice geometry")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate checkpoint header key")
            result[key] = value
        return result

    with Path(path).open("rb") as handle:
        before = os.fstat(handle.fileno())
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError("checkpoint header truncated")
        header_bytes = struct.unpack("<Q", size_bytes)[0]
        if not 0 < header_bytes <= 16 << 20 or header_bytes > before.st_size - 8:
            raise ValueError("invalid checkpoint header")
        header = json.loads(handle.read(header_bytes), object_pairs_hook=unique_keys)
        ranges = []

        def read(name, dtype, item, shape, count):
            record = header[name]
            if record["dtype"] != dtype or record["shape"] != shape:
                raise ValueError("ConvRot tensor geometry/dtype mismatch")
            begin, end = record["data_offsets"]
            if (type(begin) is not int or type(end) is not int or begin < 0
                    or end - begin != math.prod(shape) * item
                    or 8 + header_bytes + end > before.st_size or count * item > end - begin):
                raise ValueError("ConvRot tensor range invalid")
            if any(begin < other_end and other_begin < end for other_begin, other_end in ranges):
                raise ValueError("ConvRot payload ranges overlap")
            ranges.append((begin, end))
            handle.seek(8 + header_bytes + begin)
            raw = handle.read(count * item)
            if len(raw) != count * item:
                raise ValueError("checkpoint truncated")
            return raw

        shape = header[prefix + ".weight"]["shape"]
        if (len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape)
                or shape[1] != hidden or shape[0] < outputs):
            raise ValueError("projection scope mismatch")
        raw = read(prefix + ".weight", "I8", 1, shape, outputs * hidden)
        scale = read(prefix + ".weight_scale", "F32", 4, [shape[0], 1], outputs)
        after = os.fstat(handle.fileno())
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size,
                                 stat.st_mtime_ns, stat.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError("source changed during slice read")
    scales = np.frombuffer(scale, dtype="<f4").copy()
    if not np.all(np.isfinite(scales)):
        raise ValueError("nonfinite source scale")
    provenance = {"kind": "real_convrot_projection_slice", "tensor": prefix + ".weight",
                  "slice_rows": outputs, "weight_sha256": hashlib.sha256(raw).hexdigest(),
                  "scale_sha256": hashlib.sha256(scale).hexdigest(),
                  "checkpoint_content_sha256": None}
    return np.frombuffer(raw, dtype=np.int8).reshape(outputs, hidden).copy(), scales, provenance


def inspect_plan(path, units):
    from coremltools.models.compute_plan import MLComputePlan

    try:
        plan = MLComputePlan.load_from_path(str(path), compute_units=units)
        result = []

        def visit(block):
            for op in block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                result.append({"op": op.operator_name, "outputs": [value.name for value in op.outputs],
                               "preferred": type(usage.preferred_compute_device).__name__ if usage else "unknown"})
                for nested in op.blocks:
                    visit(nested)

        for function in plan.model_structure.program.functions.values():
            visit(function.block)
        return {"status": "available", "scope": "anticipated placement, not a runtime trace",
                "operations": result}
    except Exception as error:
        return {"status": "unknown", "error": str(error)}


def classify_screen(entry):
    """Fail closed: numerical/reuse proof cannot be replaced by a fast time."""
    if entry.get("status") == "failed":
        return "inconclusive_export_or_execution"
    if not entry.get("normalized_numeric_gate"):
        return "numeric_gate_failed"
    if not entry.get("graph_files_unchanged") or not entry.get("weight_a_repeat_exact"):
        return "unsupported_on_tuple_graph_or_repeat"
    if entry.get("arm") != "frozen_qdq" and not entry.get("weight_b_changes_output"):
        return "unsupported_on_tuple_dynamic_weights"
    matrix = [op for op in entry.get("compute_plan", {}).get("operations", [])
              if "matmul" in op["op"]]
    if matrix and all(op["preferred"] == "MLCPUComputeDevice" for op in matrix):
        return "unsupported_on_tuple_cpu_plan"
    return "inconclusive_arithmetic_not_observed"


def screen_exit_code(report):
    """Exit zero means a usable SCREEN receipt, never hardware qualification."""
    arms = report.get("arms", [])
    if (len(arms) != 3 or {arm.get("arm") for arm in arms} != {"fp16", "frozen_qdq", "runtime_qdq"}
            or any(arm.get("status") != "completed" for arm in arms)):
        return 2
    if any(len(arm.get("samples", [])) < 30 for arm in arms):
        return 2
    if any(not arm.get("normalized_numeric_gate") or not arm.get("graph_files_unchanged")
           or not arm.get("weight_a_repeat_exact")
           or (arm.get("arm") != "frozen_qdq" and not arm.get("weight_b_changes_output")) for arm in arms):
        return 1
    return 0


def main():
    import coremltools as ct

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--tile-k", type=int)
    parser.add_argument("--tile-n", type=int)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--max-scratch-bytes", type=int, default=1 << 30)
    parser.add_argument("--convrot-checkpoint", type=Path)
    parser.add_argument("--convrot-prefix", default="layers.0.feed_forward.w1")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    if not 30 <= args.iterations <= 300:
        parser.error("iterations must be 30..300")
    specs = [EXPORT.geometry(args.rows, args.hidden, args.width, arm, args.tile_k, args.tile_n)
             for arm in ("fp16", "frozen_qdq", "runtime_qdq")]
    # Input/oracle allowance, NOT the compiler/Core ML framework envelope.
    scratch = args.width * args.hidden * 23 + args.rows * args.hidden * 11 + args.rows * args.width * 24
    if scratch > args.max_scratch_bytes:
        parser.error("oracle/input scratch exceeds configured bound")
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(42)
    source_identity = {"kind": "synthetic_i8_fixture"}
    if args.convrot_checkpoint:
        wa, source_scales, source_identity = convrot_slice(
            args.convrot_checkpoint, args.convrot_prefix, args.width, args.hidden)
        activation = rng.normal(0, .5, size=(args.rows, args.hidden)).astype(np.float32)
        qx, source_activation_scales = quantize_rows(rotate(activation))
        source_identity["activation_source"] = "synthetic Gaussian + Comfy H256 + dynamic row A8"
    else:
        qx = rng.integers(-127, 128, size=(args.rows, args.hidden), dtype=np.int8)
        wa = rng.integers(-128, 128, size=(args.width, args.hidden), dtype=np.int8)
    wb = rng.integers(-128, 128, size=wa.shape, dtype=np.int8)
    if not args.convrot_checkpoint:
        wa[0, 0] = -128
        qx[0, 0] = -127
    wb[0, 0] = 127
    xn = normalize(qx)
    wn = [normalize(wa), normalize(wb)]
    integer = [integer_dot(qx, w, fast=True) for w in (wa, wb)]
    oracle = [normalized_output(dot) for dot in integer]
    sx = source_activation_scales if args.convrot_checkpoint else np.geomspace(.001, .05, args.rows).astype(np.float32)
    sw = source_scales if args.convrot_checkpoint else np.geomspace(.0001, .01, args.width).astype(np.float32)
    units = ct.ComputeUnit.CPU_ONLY if args.cpu_only else ct.ComputeUnit.CPU_AND_NE
    bias = [np.linspace(-.01, .01, args.width, dtype=np.float32),
            np.linspace(.02, -.02, args.width, dtype=np.float32)]
    activation_scales = [sx, sx * np.float32(1.25)]
    weight_scales = [sw, sw * np.linspace(.5, 1.5, args.width, dtype=np.float32)]
    restored = [restore_normalized(value, activation_scales[i], weight_scales[i], bias[i])
                for i, value in enumerate(oracle)]
    report = {
        "schema": "tc-runtime-w8a8-proof-receipt-v2",
        "scope": "single projection, public Core ML candidate; not model qualification",
        "geometry": specs[0], "compute_policy": units.name, "coremltools": ct.__version__,
        "managed_scratch_upper": scratch, "source": source_identity,
        "activation_sha256": hashlib.sha256(qx.tobytes()).hexdigest(),
        "weight_a_sha256": hashlib.sha256(wa.tobytes()).hexdigest(),
        "weight_b_sha256": hashlib.sha256(wb.tobytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                          [Path(__file__).resolve(), ROOT / "tools/coreml/export_runtime_w8a8_probe.py",
                           ROOT / "tools/validation/convrot_w8a8_math.py"]},
        "observed_placement": "unknown", "hardware_arithmetic": "unknown",
        "production_qualified": False, "compiler_framework_envelope": "unknown",
        "scale_bias_scope": "external FP32 restoration, different nonuniform scales/bias for A/B; not graph inputs",
        "input_preparation_scope": "codes + A8 and normalization prepared outside prediction timing",
        "sample_order_scope": "each arm separately, A/B/A within arm; not paired cross-arm performance qualification",
        "scale_bias_sha256": [{name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in
                              [("sx", activation_scales[i]), ("sw", weight_scales[i]), ("bias", bias[i])]}
                             for i in range(2)], "arms": []}

    def save():
        (args.output / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for spec in specs:
        arm = spec["arm"]
        entry = {"arm": arm, "status": "running", "samples": []}
        report["arms"].append(entry)
        save()
        if arm == "runtime_qdq" and report["arms"][1]["status"] != "completed":
            entry.update(status="not_run", feasibility="inconclusive_frozen_control_prerequisite")
            save()
            continue
        try:
            path = args.output / arm
            start = time.perf_counter()
            manifest = EXPORT.export(path, spec, wa if arm == "frozen_qdq" else None)
            entry["export_compile_seconds"] = time.perf_counter() - start
            entry["manifest_sha256"] = digest(path / "manifest.json")
            entry["program_ops"] = manifest["program_ops_after_convert"]
            start = time.perf_counter()
            model = ct.models.CompiledMLModel(str(path / "graph.mlmodelc"), compute_units=units)
            entry["load_seconds"] = time.perf_counter() - start
            entry["model_load_count"] = 1
            entry["export_compile_count"] = 1
            entry["runtime_compile_count"] = "unknown"
            first_a = None
            max_ulp = 0
            max_abs = restored_rel = 0.
            repeat = True
            changed = False
            # At least ten A/B/A cycles. Frozen control cannot switch weights.
            for i in range(args.iterations + 5):
                which = 0 if arm == "frozen_qdq" or i % 3 != 1 else 1
                start = time.perf_counter()
                data = {"x": xn}
                if arm != "frozen_qdq":
                    data["w"] = wn[which]
                binding = time.perf_counter() - start
                start = time.perf_counter()
                output = np.asarray(model.predict(data)["y"], dtype=np.float16)
                predict = time.perf_counter() - start
                if output.shape != oracle[which].shape or not np.all(np.isfinite(output)):
                    raise ArithmeticError("invalid output geometry/nonfinite")
                ulp = int(fp16_ulp_distance(output, oracle[which]).max())
                absolute = float(np.max(np.abs(output.astype(np.float32) - oracle[which].astype(np.float32))))
                start = time.perf_counter()
                current = restore_normalized(output, activation_scales[which], weight_scales[which], bias[which])
                restore = time.perf_counter() - start
                rel = float(np.linalg.norm((current - restored[which]).astype(np.float64))
                            / max(np.linalg.norm(restored[which].astype(np.float64)), 1e-12))
                max_ulp = max(max_ulp, ulp)
                max_abs = max(max_abs, absolute)
                restored_rel = max(restored_rel, rel)
                if which == 0:
                    if first_a is None:
                        first_a = output.copy()
                    else:
                        repeat = repeat and np.array_equal(first_a, output)
                elif first_a is not None:
                    changed = changed or not np.array_equal(first_a, output)
                if i >= 5:
                    entry["samples"].append({
                        "weights": "A" if which == 0 else "B", "binding_seconds": binding,
                        "predict_seconds": predict, "restore_seconds": restore,
                        "binding_predict_restore_seconds": binding + predict + restore,
                        "max_ulp": ulp, "max_abs": absolute, "restored_rel_l2": rel})
                else:
                    entry.setdefault("warmup_predict_seconds", []).append(predict)
            entry.update(
                max_ulp=max_ulp, max_abs=max_abs, restored_rel_l2=restored_rel,
                weight_a_repeat_exact=repeat,
                weight_b_changes_output=changed if arm != "frozen_qdq" else None,
                normalized_numeric_gate=max_ulp <= 2,
                median_predict_seconds=statistics.median(x["predict_seconds"] for x in entry["samples"]),
                median_binding_predict_restore_seconds=statistics.median(
                    x["binding_predict_restore_seconds"] for x in entry["samples"]))
            entry["graph_files_unchanged"] = all(digest(path / name) == value for name, value in manifest["files"].items())
            entry["compute_plan"] = inspect_plan(path / "graph.mlmodelc", units)
            entry["feasibility"] = classify_screen(entry)
            entry["status"] = "completed"
            print(arm, entry["feasibility"], "ULP", max_ulp, "median", entry["median_predict_seconds"], flush=True)
        except Exception as error:
            entry.update(status="failed", error=str(error), feasibility="inconclusive_export_or_execution")
            print(arm, "FAILED", str(error), flush=True)
        save()
    report["status"] = "completed"
    report["screen_exit_code"] = screen_exit_code(report)
    save()
    return report["screen_exit_code"]


if __name__ == "__main__":
    with open("/tmp/turbocider-z-image-gpu-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(main())
