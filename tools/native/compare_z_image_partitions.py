#!/usr/bin/env python3
"""Fixed 32-branch Z-Image prefix smoke against original BF16 GPU weights.

Synthetic hidden states only. No full-image quality, ANE residency, memory or
performance qualification. Writes the plan before loading/predicting models.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import os
import struct

import mlx.core as mx
import numpy as np
from benchmark_coreml_ffn_bridge import CoreMLFFN
from analyze_z_image_encoder_streaming import compare_arrays


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    checkpoint, manifest_path, library = (p.resolve() for p in (args.checkpoint, args.manifest, args.library))
    manifest = json.loads(manifest_path.read_text())
    shape = manifest["shape"]
    expected = {"K": 3840, "N": 3840, "mlp_width": 10240, "ane_mlp_start": 0,
                "ane_mlp_end": 5120, "buckets": [1088], "activation_scale": 8, "output_scale": 32}
    if any(shape.get(k) != v for k, v in expected.items()) or set(manifest["artifacts"]) != {str(i) for i in range(32)}:
        raise ValueError("requires the fixed b1088/w5120 32-branch candidate")
    if manifest["source"].get("loras"):
        raise ValueError("this oracle uses unmodified original checkpoint weights")
    if any(Path(entry["int8_pc"]).suffix != ".mlmodelc" for entry in manifest["artifacts"].values()):
        raise ValueError("compiled artifacts required")
    limits = {"relative_l2_max": .025, "cosine_min": .999, "relative_max_abs_max": .05}
    plan = {"scope": __doc__, "checkpoint": str(checkpoint), "manifest": str(manifest_path),
            "manifest_sha256": digest(manifest_path), "library_sha256": digest(library),
            "driver_sha256": digest(Path(__file__)), "shape": expected,
            "seed": 42, "input_scale": .25, "warmups": 0, "predictions_per_branch": 1,
            "rows": {"noise_refiner": 1024, "main": 1088}, "thresholds": limits,
            "reference": "Original BF16 weights and prefix MLP on MLX GPU; shared BF16-rounded input transported as FP16",
            "candidate": "Native Core ML bridge CPU+NE configuration; FP16 output cast to BF16 before restoring output scale",
            "failure_policy": "Keep every fixed numeric result; no retries or threshold changes. Runtime errors stop the run."}
    write(root / "plan.json", plan)
    report = {"scope": plan["scope"], "cases": [], "completed": False}
    write(root / "results.json", report)
    bridge = None
    try:
        mx.set_default_device(mx.gpu)
        initial = np.random.default_rng(42).standard_normal((1, 1088, 3840), dtype=np.float32) * .25
        rounded = mx.array(initial, dtype=mx.bfloat16)
        transported = np.asarray(rounded.astype(mx.float32)).astype(np.float16)
        x_full = mx.array(transported, dtype=mx.bfloat16)
        if not np.array_equal(np.asarray(x_full.astype(mx.float32)), transported.astype(np.float32)):
            raise ValueError("input transport is not exact for the shared oracle input")
        np.save(root / "input.npy", transported)
        with checkpoint.open("rb") as stream:
            snapshot = identity(os.fstat(stream.fileno()))
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_size <= 16 << 20:
                raise ValueError("invalid checkpoint header size")
            header = json.loads(stream.read(header_size))
            payload = 8 + header_size
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as memory:
                def weight(name, projection):
                    meta = header[name]
                    geometry = [3840, 10240] if projection == 2 else [10240, 3840]
                    if meta["dtype"] != "BF16" or meta["shape"] != geometry:
                        raise ValueError("oracle requires original BF16 FFN geometry")
                    lo, hi = meta["data_offsets"]
                    if lo < 0 or hi - lo != 3840 * 10240 * 2 or payload + hi > len(memory):
                        raise ValueError("invalid BF16 source range")
                    words = np.ndarray(geometry, dtype="<u2", buffer=memory, offset=payload + lo)
                    sliced = words[:, :5120] if projection == 2 else words[:5120]
                    decoded = (sliced.astype(np.uint32) << 16).view(np.float32)
                    result = mx.array(decoded, dtype=mx.bfloat16)
                    mx.eval(result)
                    return result

                bridge = CoreMLFFN(library, manifest_path, checkpoint, 1088, 0)
                report["create_seconds_diagnostic_only"] = bridge.create_seconds
                for block in range(32):
                    prefix = f"noise_refiner.{block}" if block < 2 else f"layers.{block - 2}"
                    rows = 1024 if block < 2 else 1088
                    x = x_full[:, :rows]
                    gate, down, up = (weight(f"{prefix}.feed_forward.w{i}.weight", i) for i in (1, 2, 3))
                    projected = x @ gate.T
                    reference = ((projected * mx.sigmoid(projected)) * (x @ up.T)) @ down.T
                    mx.eval(reference)
                    expected_output = np.asarray(reference.astype(mx.float32))
                    raw, seconds = bridge.predict(block, np.ascontiguousarray(transported[:, :rows]))
                    restored = mx.array(raw, dtype=mx.bfloat16) * mx.array(32, dtype=mx.bfloat16)
                    mx.eval(restored)
                    actual = np.asarray(restored.astype(mx.float32))
                    np.save(root / f"{block:02d}-reference.npy", expected_output)
                    np.save(root / f"{block:02d}-candidate.npy", actual)
                    metric = compare_arrays(expected_output, actual)
                    peak = float(np.max(np.abs(expected_output)))
                    relative_max = metric["max_abs"] / peak if metric["valid"] and peak else None
                    passed = bool(metric["valid"] and metric["relative_l2"] is not None and relative_max is not None
                                  and metric["relative_l2"] <= limits["relative_l2_max"]
                                  and metric["cosine"] >= limits["cosine_min"]
                                  and relative_max <= limits["relative_max_abs_max"])
                    report["cases"].append({"block": block, "prefix": prefix, "rows": rows,
                        "metrics": metric, "relative_max_abs": relative_max, "passed": passed,
                        "prediction_seconds_diagnostic_only": seconds,
                        "reference_sha256": digest(root / f"{block:02d}-reference.npy"),
                        "candidate_sha256": digest(root / f"{block:02d}-candidate.npy")})
                    write(root / "results.json", report)
                    print(block, passed, metric.get("relative_l2"), relative_max, flush=True)
                    del gate, down, up, projected, reference, restored, expected_output, actual
                    mx.clear_cache()
                if identity(os.fstat(stream.fileno())) != snapshot or identity(checkpoint.stat()) != snapshot:
                    raise ValueError("checkpoint changed during oracle run")
        metrics = bridge.metrics()
        report["native_metrics"] = metrics
        if metrics["runtime_calls_session_total"] != 32 or metrics["runtime_failures_session_total"] != 0 or not metrics["checkpoint_sha256_verified"]:
            raise ValueError("native source verification or complete prediction count failed")
        report["completed"] = True
        report["all_within_exploratory_thresholds"] = all(c["passed"] for c in report["cases"])
        write(root / "results.json", report)
    except BaseException as error:
        report["error"] = str(error)
        write(root / "results.json", report)
        raise
    finally:
        if bridge is not None:
            bridge.close()


if __name__ == "__main__":
    main()
