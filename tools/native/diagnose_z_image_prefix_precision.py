#!/usr/bin/env python3
"""Offline arithmetic diagnostics for failed Z prefix smoke cases.

Does not rerun Core ML, change thresholds, or replace the original BF16 oracle.
The FP16 GPU graph is a counterfactual, not a Core ML artifact validation.
"""
import argparse
import json
import mmap
from pathlib import Path
import os
import struct

import mlx.core as mx
import numpy as np
from compare_z_image_partitions import digest, identity, write
from analyze_z_image_encoder_streaming import compare_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.smoke_root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    results = json.loads((root / "results.json").read_text())
    if not results["completed"] or len(results["cases"]) != 32:
        raise ValueError("complete original 32-branch smoke required")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [case for case in results["cases"] if not case["passed"]]
    write(args.output / "plan.json", {"scope": __doc__, "cases": [c["block"] for c in cases],
        "original_plan_sha256": digest(root / "plan.json"), "original_results_sha256": digest(root / "results.json"),
        "driver_sha256": digest(Path(__file__)), "input_sha256": digest(root / "input.npy")})
    mx.set_default_device(mx.gpu)
    transported = np.load(root / "input.npy", allow_pickle=False)
    checkpoint = Path(plan["checkpoint"])
    rows = []
    with checkpoint.open("rb") as stream:
        snapshot = identity(os.fstat(stream.fileno()))
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length <= 16 << 20:
            raise ValueError("invalid header")
        header = json.loads(stream.read(length))
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as memory:
            for case in cases:
                block = case["block"]
                for kind in ("reference", "candidate"):
                    if digest(root / f"{block:02d}-{kind}.npy") != case[f"{kind}_sha256"]:
                        raise ValueError("original output changed")
                expected = np.load(root / f"{block:02d}-reference.npy", allow_pickle=False)
                candidate = np.load(root / f"{block:02d}-candidate.npy", allow_pickle=False)
                matrices = []
                for projection in (1, 2, 3):
                    meta = header[f"{case['prefix']}.feed_forward.w{projection}.weight"]
                    geometry = [3840, 10240] if projection == 2 else [10240, 3840]
                    lo, hi = meta["data_offsets"]
                    if meta["dtype"] != "BF16" or meta["shape"] != geometry or lo < 0 or hi - lo != 3840 * 10240 * 2 or 8 + length + hi > len(memory):
                        raise ValueError("unexpected source tensor")
                    words = np.ndarray(geometry, dtype="<u2", buffer=memory, offset=8 + length + lo)
                    selected = words[:, :5120] if projection == 2 else words[:5120]
                    decoded = (selected.astype(np.uint32) << 16).view(np.float32)
                    matrices.append(mx.array(decoded))
                    del selected, words
                mx.eval(matrices)
                def ordinary(dtype):
                    gate, down, up = (w.astype(dtype) for w in matrices)
                    x = mx.array(transported[:, :case["rows"]], dtype=dtype)
                    g = x @ gate.T
                    result = ((g * mx.sigmoid(g)) * (x @ up.T)) @ down.T
                    mx.eval(result)
                    return np.asarray(result.astype(mx.float32))
                repeated_bf16 = ordinary(mx.bfloat16)
                if not np.array_equal(repeated_bf16, expected):
                    raise ValueError("BF16 replay differs; diagnostic input/source is not the original oracle")
                f32 = ordinary(mx.float32)
                gate, down, up = (w.astype(mx.float16) for w in matrices)
                # Match the exporter's weight/input/output scaling order, but
                # MLX matmul is not evidence of Core ML convolution behavior.
                scaled_down = (matrices[1] * (8 * 8 / 32)).astype(mx.float16)
                x = mx.array(transported[:, :case["rows"]], dtype=mx.float16)
                g = x @ gate.T
                product = (g * mx.sigmoid(g) * mx.array(1 / 8, dtype=mx.float16)) * ((x @ up.T) * mx.array(1 / 8, dtype=mx.float16))
                raw = product @ scaled_down.T
                restored = raw.astype(mx.bfloat16) * mx.array(32, dtype=mx.bfloat16)
                mx.eval(restored)
                f16_graph = np.asarray(restored.astype(mx.float32))
                np.save(args.output / f"{block:02d}-fp32.npy", f32)
                np.save(args.output / f"{block:02d}-fp16-gpu-graph.npy", f16_graph)
                rows.append({"block": block, "original_bf16_replay_identical": True,
                    "bf16_vs_fp32": compare_arrays(f32, expected),
                    "candidate_vs_fp32": compare_arrays(f32, candidate),
                    "fp16_gpu_graph_vs_bf16": compare_arrays(expected, f16_graph),
                    "candidate_vs_fp16_gpu_graph": compare_arrays(f16_graph, candidate)})
            if identity(os.fstat(stream.fileno())) != snapshot or identity(checkpoint.stat()) != snapshot:
                raise ValueError("checkpoint changed")
    write(args.output / "results.json", {"scope": __doc__, "cases": rows,
        "primary_smoke_outcome_unchanged": results["all_within_exploratory_thresholds"]})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
