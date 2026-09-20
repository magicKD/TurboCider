#!/usr/bin/env python3
"""One frozen failed-branch FP16 diagnostic; does not amend INT8 smoke results."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import coremltools as ct
import mlx.core as mx
import numpy as np
from compare_z_image_partitions import digest, write
from analyze_z_image_encoder_streaming import compare_arrays


def native_tree_digest(root):
    # Native artifact_cache.mm encoding on the arm64 little-endian host.
    if root.is_symlink():
        raise ValueError("symlink artifact root")
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in artifact")
        if path.is_file():
            files.append(path)
    if not files:
        raise ValueError("empty artifact")
    h = hashlib.sha256()
    for path in sorted(files):
        name = path.relative_to(root).as_posix().encode()
        size = path.stat().st_size
        h.update(struct.pack("<Q", len(name))); h.update(name); h.update(struct.pack("<Q", size))
        read = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                read += len(block); h.update(block)
        if read != size:
            raise ValueError("artifact changed while hashing")
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--compile-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    compiled = json.loads(args.compile_result.read_text())
    source = Path(plan["output"]) / "manifest.json"
    manifest = json.loads(source.read_text())
    shape = manifest["shape"]
    if (manifest["export_identity"]["variant"] != "fp16" or manifest["source"]["blocks"] != [29]
            or manifest["source"]["checkpoint_sha256"] != plan["source_checkpoint_sha256"]
            or shape["buckets"] != [plan["bucket"]] or shape["ane_mlp_end"] != 5120
            or shape["ane_mlp_start"] != 0 or shape["output_scale"] != 32 or shape["activation_scale"] != 8):
        raise ValueError("artifact differs from frozen precision diagnostic")
    package = source.parent / manifest["artifacts"]["29"]["int8_pc"]
    if native_tree_digest(package) != compiled["identity"]["source_sha256"]:
        raise ValueError("compile receipt does not bind this source package")
    for key in ("input", "reference"):
        if digest(Path(plan[key])) != plan[key + "_sha256"]:
            raise ValueError("frozen input/reference changed")
    model_path = Path(compiled["artifact"])
    before = native_tree_digest(model_path)
    inputs = np.load(plan["input"], allow_pickle=False)
    reference = np.load(plan["reference"], allow_pickle=False)
    if inputs.dtype != np.float16 or inputs.shape != (1, 1088, 3840) or reference.shape != inputs.shape:
        raise ValueError("unexpected oracle shape/type")
    model = ct.models.CompiledMLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    raw = model.predict({"x": np.ascontiguousarray(inputs.transpose(0, 2, 1)[:, :, None, :])})["y"]
    raw = np.ascontiguousarray(raw[:, :, 0, :].transpose(0, 2, 1))
    mx.set_default_device(mx.gpu)
    restored = mx.array(raw, dtype=mx.bfloat16) * mx.array(32, dtype=mx.bfloat16)
    mx.eval(restored)
    candidate = np.asarray(restored.astype(mx.float32))
    np.save(args.output / "candidate.npy", candidate)
    metric = compare_arrays(reference, candidate)
    peak = float(np.max(np.abs(reference)))
    relative_max = metric["max_abs"] / peak if metric["valid"] and peak else None
    limits = plan["thresholds"]
    passed = bool(metric["valid"] and metric["relative_l2"] is not None and relative_max is not None
        and metric["relative_l2"] <= limits["relative_l2_max"] and metric["cosine"] >= limits["cosine_min"]
        and relative_max <= limits["relative_max_abs_max"])
    after = native_tree_digest(model_path)
    if before != after:
        raise ValueError("compiled artifact changed during prediction")
    report = {"scope": __doc__, "plan_sha256": digest(args.plan), "driver_sha256": digest(Path(__file__)),
              "compile_result": compiled, "compiled_tree_sha256": before, "block": 29,
              "predictions": 1, "warmups": 0, "compute_units": "CPU_AND_NE", "observed_ane_residency": "unknown",
              "metrics": metric, "relative_max_abs": relative_max, "passed": passed,
              "candidate_sha256": digest(args.output / "candidate.npy"),
              "qualification": "NONE: one branch and synthetic input; original 32-branch INT8 result unchanged"}
    write(args.output / "results.json", report)
    print(json.dumps({k: report[k] for k in ("block", "metrics", "relative_max_abs", "passed")}, indent=2))


if __name__ == "__main__":
    main()
