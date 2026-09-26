#!/usr/bin/env python3
"""Compare CPU_ONLY and CPU_AND_NE on one compiled Z-Image FFN partition.

This is a same-artifact, same-input, warmed ABBA latency check. A faster
CPU_AND_NE path supports, but does not prove, physical ANE placement or
concurrent GPU execution. End-to-end image quality must use the runtime's
CPU_AND_NE result rather than treating a CPU_ONLY output as an oracle.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--image-rows", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 2 or args.warmups < 1 or args.block < 0:
        parser.error("rounds >= 2, warmups >= 1 and nonnegative block required")

    import coremltools as ct
    import numpy as np

    manifest = json.loads(args.manifest.read_text())
    identity = manifest.get("export_identity", {})
    shape = manifest.get("shape", {})
    rows = shape.get("buckets")
    if (identity.get("activation_precision") != "int8" or
            shape.get("K") != 3840 or rows not in ([1056], [1024]) or
            (rows == [1024] and identity.get("image_only_token_rows") != 1024) or
            (rows == [1056] and identity.get("image_only_token_rows"))):
        parser.error("expected a fixed 1056-row or marked image-only 1024-row W8A8 manifest")
    artifact = manifest["artifacts"][str(args.block)]["int8_pc"]
    compiled = (args.manifest.parent / artifact).resolve()
    if compiled.suffix != ".mlmodelc" or not compiled.is_dir():
        parser.error("the selected manifest must reference a compiled .mlmodelc")
    raw = args.capture.read_bytes()
    capture = np.load(args.capture, allow_pickle=False)
    if capture.dtype != np.float16 or capture.shape not in ((rows[0], 3840),
                                                              (1056, 3840)):
        parser.error("capture must be a matching [rows, 3840] FP16 NPY tensor")
    if (not 0 < args.image_rows <= rows[0] or
            (rows == [1056] and args.image_rows == 1056) or
            (rows == [1024] and args.image_rows != 1024)):
        parser.error("image-rows must be 1024 for image-only or leave conditioning rows")
    value = np.ascontiguousarray(capture[:rows[0]].T[None, :, None, :])
    keys = ("CPU_ONLY", "CPU_AND_NE")
    models = {
        "CPU_ONLY": ct.models.CompiledMLModel(
            str(compiled), compute_units=ct.ComputeUnit.CPU_ONLY),
        "CPU_AND_NE": ct.models.CompiledMLModel(
            str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE),
    }
    for _ in range(args.warmups):
        for key in keys:
            models[key].predict({"x": value})
    samples = {key: [] for key in keys}
    outputs = {}
    for _ in range(args.rounds):
        for key in (keys[0], keys[1], keys[1], keys[0]):
            start = time.perf_counter()
            outputs[key] = models[key].predict({"x": value})["y"]
            samples[key].append((time.perf_counter() - start) * 1000)
    cpu = outputs[keys[0]].astype(np.float32)
    ne = outputs[keys[1]].astype(np.float32)
    if cpu.shape != ne.shape or cpu.shape != value.shape:
        raise ValueError("Core ML output shape differs between compute units")
    def relative_l2(left, right):
        return float(np.linalg.norm(left - right) /
                     max(float(np.linalg.norm(left)), 1e-12))
    report = {
        "scope": "same compiled model, held-out captured input, warmed ABBA; not a device trace",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "compiled": str(compiled),
        "capture": str(args.capture.resolve()),
        "capture_sha256": hashlib.sha256(raw).hexdigest(),
        "block": args.block,
        "model_rows": rows[0],
        "image_rows": args.image_rows,
        "rounds": args.rounds,
        "warmups": args.warmups,
        "platform": {"macos": platform.mac_ver()[0], "machine": platform.machine()},
        "medians_milliseconds": {key: statistics.median(v) for key, v in samples.items()},
        "minima_milliseconds": {key: min(v) for key, v in samples.items()},
        "cpu_only_over_cpu_and_ne": statistics.median(samples[keys[0]]) /
                                    statistics.median(samples[keys[1]]),
        "outputs": {
            "both_finite": bool(np.isfinite(cpu).all() and np.isfinite(ne).all()),
            "image_relative_l2": relative_l2(
                cpu[..., :args.image_rows], ne[..., :args.image_rows]),
            "caption_relative_l2": (relative_l2(
                cpu[..., args.image_rows:], ne[..., args.image_rows:])
                if args.image_rows < rows[0] else None),
            "maximum_absolute_difference": float(np.max(np.abs(cpu - ne))),
        },
        "physical_ane_residency": "unverified",
        "physical_gpu_ane_overlap": "unverified",
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error(f"refusing to overwrite existing report: {args.output}")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
