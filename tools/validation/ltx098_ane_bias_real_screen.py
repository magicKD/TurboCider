#!/usr/bin/env python3
"""Screen a bias-inclusive Core ML FFN on captured real LTX activations.

CPU_AND_NE can use CPU; physical ANE placement is unknown. Python/Core ML
prediction timings are not a GPU/ANE-overlap or end-to-end performance claim.
"""
import argparse
import json
from pathlib import Path
import time

from ltx098_comfy_reference import MODEL_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-row", type=int, default=3072)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    checkpoint, manifest, activation = (p.resolve(strict=True) for p in (args.checkpoint, args.manifest, args.activation))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the pinned source")
    spec = json.loads(manifest.read_text())
    if (spec.get("backend"), spec.get("biases"), spec.get("rows"), spec.get("hidden"), spec.get("width")) != (
            "ltx098_experimental_bias_fp16", True, 256, 2048, 8192):
        raise ValueError("expected experimental bias-inclusive 256x2048x8192 graph")
    compiled = manifest.parent / "graph.mlmodelc"
    for relative, digest in spec["files"].items():
        if relative.startswith("graph.mlmodelc/") and sha256(manifest.parent / relative) != digest:
            raise ValueError(f"compiled graph hash mismatch: {relative}")
    import coremltools as ct
    import mlx.core as mx
    import numpy as np

    captured = mx.load(str(activation))["input"]
    if captured.shape != (1, 4096, 2048) or args.start_row < 0 or args.start_row + 256 > 4096:
        raise ValueError("expected a valid 256-row slice of captured [1,4096,2048] FFN input")
    tensors = mx.load(str(checkpoint))
    prefix = "model.diffusion_model.transformer_blocks.0.ff.net."
    wu, bu = tensors[prefix + "0.proj.weight"], tensors[prefix + "0.proj.bias"]
    wd, bd = tensors[prefix + "2.weight"], tensors[prefix + "2.bias"]
    x = captured[0, args.start_row:args.start_row + 256, :]
    mx.eval(x, wu, bu, wd, bd)
    if wu.shape != (8192, 2048) or wd.shape != (2048, 8192) or bu.shape != (8192,) or bd.shape != (2048,):
        raise ValueError("checkpoint FFN weight/bias geometry mismatch")

    @mx.compile
    def gelu(value):
        f = value.astype(mx.float32)
        return (.5 * f * (1 + mx.tanh(.7978845608028654 * (f + .044715 * f * f * f)))).astype(value.dtype)

    def gpu():
        up = mx.matmul(x, mx.transpose(wu)) + bu
        return mx.matmul(gelu(up), mx.transpose(wd)) + bd

    gpu_times = []
    for index in range(args.repeats + 2):
        started = time.perf_counter()
        candidate = gpu()
        mx.eval(candidate)
        if index >= 2:
            gpu_times.append(time.perf_counter() - started)
    reference = np.array(candidate.astype(mx.float32))
    prepared = {"x": np.array(x.astype(mx.float16)), "wu": np.array(wu.astype(mx.float16)),
                "wd": np.array(wd.astype(mx.float16)), "bu": np.array(bu.astype(mx.float16)),
                "bd": np.array(bd.astype(mx.float16))}
    output.mkdir(parents=True)
    report = {"schema": "ltx098-ane-bias-real-screen-v1", "status": "running", "checkpoint_sha256": MODEL_SHA256,
              "manifest_sha256": sha256(manifest), "activation_sha256": sha256(activation), "start_row": args.start_row,
              "scope": "One real Comfy block-0 FFN input slice and real weights plus both biases; GPU BF16 vs Core ML FP16; CPU_AND_NE placement unknown; Python prediction includes interface overhead, no GPU overlap or full-model claim",
              "gpu_seconds": gpu_times, "modes": []}
    try:
        for name, units in (("CPU_ONLY", ct.ComputeUnit.CPU_ONLY), ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)):
            started = time.perf_counter()
            model = ct.models.CompiledMLModel(str(compiled), compute_units=units)
            load_seconds = time.perf_counter() - started
            checks = []
            # Nonzero biases and a changed signed weight pattern check both
            # bias inputs, orientation and runtime-weight rebinding.
            for scale in (.125, -.25):
                rr = np.arange(256, dtype=np.int32)[:, None]
                cc = np.arange(2048, dtype=np.int32)[None, :]
                sx = (((rr * 3 + cc * 7) % 17 - 8) / 8).astype(np.float16)
                swu, swd = np.zeros((8192, 2048), np.float16), np.zeros((2048, 8192), np.float16)
                swu[np.arange(8192), np.arange(8192) % 2048] = np.float16(scale)
                swd[np.arange(2048), np.arange(2048)] = np.float16(scale)
                sbu = np.full((8192,), .03125, np.float16)
                sbd = np.full((2048,), -.015625, np.float16)
                observed = np.asarray(model.predict({"x": sx, "wu": swu, "wd": swd, "bu": sbu, "bd": sbd})["y"], dtype=np.float32)
                z = sx.astype(np.float32) * scale + .03125
                expected = (.5 * z * (1 + np.tanh(.7978845608028654 * (z + .044715 * z ** 3))) * scale - .015625)
                error = np.abs(observed - expected)
                passed = bool(np.isfinite(observed).all() and np.all(error <= .0002 + .02 * np.abs(expected)))
                checks.append({"scale": scale, "passed": passed, "max_abs": float(error.max())})
            if not all(check["passed"] for check in checks):
                report["modes"].append({"compute_units": name, "status": "self_test_failed", "self_test": checks})
                continue
            times, observed = [], None
            for index in range(args.repeats + 2):
                started = time.perf_counter()
                observed = np.asarray(model.predict(prepared)["y"], dtype=np.float32)
                if index >= 2:
                    times.append(time.perf_counter() - started)
            if observed.shape != reference.shape or not np.isfinite(observed).all():
                raise ValueError(f"{name} returned invalid real-input output")
            delta = observed - reference
            report["modes"].append({"compute_units": name, "status": "complete", "physical_ane_placement": "unknown",
                "self_test": checks, "load_seconds": load_seconds, "prediction_seconds": times,
                "relative_l2_vs_gpu": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30)),
                "max_abs_vs_gpu": float(np.abs(delta).max()),
                "cosine_vs_gpu": float(np.sum(reference * observed) / max(np.linalg.norm(reference) * np.linalg.norm(observed), 1e-30))})
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(report["modes"][-1]), flush=True)
        report["status"] = "complete" if all(mode["status"] == "complete" for mode in report["modes"]) else "self_test_failed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
