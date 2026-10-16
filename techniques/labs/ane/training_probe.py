#!/usr/bin/env python3
"""Tiny manual-backprop experiment: one runtime-weight matmul, host FP32 SGD.

--cpu is a mathematical control, NOT ANE execution. --device uses public Core
ML inference graphs for forward/dX/dW; it is NOT MLUpdateTask or a whole LLM.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu", action="store_true")
    mode.add_argument("--device", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--size", type=int, choices=(32, 128, 256), default=32)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("use a new output directory")
    if args.device and platform.system() != "Darwin":
        parser.error("Core ML execution requires macOS")
    with redirect_stdout(sys.stderr):
        import numpy as np
    args.output_dir.mkdir(parents=True)
    root = args.output_dir
    (root / "source.py").write_bytes(Path(__file__).read_bytes())
    report = {"schema": "ane-training-probe-v1", "mode": "public-coreml" if args.device else "cpu-reference",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "seed": args.seed, "shape": [args.size, args.size], "status": "running", "calls": [], "steps": [],
              "relative_l2_gate": .015, "physical_ane_placement": "unverified",
              "optimizer": "host-fp32-SGD", "loss_scale": 64, "compile_count": 0,
              "environment": {"macos": platform.mac_ver()[0], "numpy": np.__version__}, "errors": []}
    def save():
        (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        rng = np.random.default_rng(args.seed)
        n = args.size
        x = rng.normal(0, .4, (n, n)).astype(np.float16)
        teacher = rng.normal(0, .2, (n, n)).astype(np.float16)
        target = np.einsum("ij,jk->ik", x.astype(np.float64), teacher.astype(np.float64), optimize=False)
        weight = np.zeros((n, n), np.float32)
        if args.device:
            with redirect_stdout(sys.stderr):
                import coremltools as ct
                from coremltools.converters.mil import Builder as mb
                from coremltools.converters.mil.mil import types
                @mb.program(input_specs=[mb.TensorSpec(shape=(n, n), dtype=types.fp16),
                                         mb.TensorSpec(shape=(n, n), dtype=types.fp16)],
                            opset_version=ct.target.macOS15)
                def graph(a, b):
                    return mb.matmul(x=a, y=b, name="y")
                model = ct.convert(graph, convert_to="mlprogram", skip_model_load=True,
                                   minimum_deployment_target=ct.target.macOS15)
                model.save(str(root / "matmul.mlpackage"))
                started = time.perf_counter()
                ct.models.utils.compile_model(str(root / "matmul.mlpackage"),
                                              destination_path=str(root / "matmul.mlmodelc"))
                report["compile_seconds"] = time.perf_counter() - started
                report["compile_count"] = 1
                loaded = ct.models.CompiledMLModel(str(root / "matmul.mlmodelc"), compute_units=ct.ComputeUnit.CPU_AND_NE)
                from compute_plan import inspect
                report["compute_plan"] = inspect(root / "matmul.mlmodelc", ct.ComputeUnit.CPU_AND_NE)
                report["environment"]["coremltools"] = ct.__version__
            def execute(a, b):
                return np.array(loaded.predict({"a": a, "b": b})["y"], copy=True)
        else:
            def execute(a, b):
                return np.einsum("ij,jk->ik", a.astype(np.float64), b.astype(np.float64), optimize=False).astype(np.float16)
        def checked(kind, a, b):
            aa, bb = np.ascontiguousarray(a, dtype=np.float16), np.ascontiguousarray(b, dtype=np.float16)
            ref = np.einsum("ij,jk->ik", aa.astype(np.float64), bb.astype(np.float64), optimize=False)
            started = time.perf_counter()
            value = execute(aa, bb)
            elapsed = time.perf_counter() - started
            if value.shape != ref.shape or not np.isfinite(value).all():
                raise ValueError("shape or nonfinite gradient")
            diff = np.linalg.norm(value.astype(np.float64) - ref)
            norm = np.linalg.norm(ref)
            error = float(diff / norm) if norm else float(diff)
            report["calls"].append({"kind": kind, "host_seconds": elapsed, "relative_l2": error,
                "input_sha256": hashlib.sha256(aa.tobytes() + bb.tobytes()).hexdigest(),
                "output_sha256": hashlib.sha256(value.tobytes()).hexdigest()})
            if error > report["relative_l2_gate"]:
                raise ValueError("gradient numeric gate rejected")
            return value.astype(np.float32)
        initial = float(.5 * np.sum(target * target) / n)
        for step in range(12):
            started = time.perf_counter()
            y = checked("forward", x, weight)
            residual = y.astype(np.float64) - target
            loss = float(.5 * np.sum(residual * residual) / n)
            dy = (residual / n * 64).astype(np.float16)
            checked("dX", dy, weight.T)
            dw = checked("dW", x.T, dy) / 64
            weight = weight - np.float32(.6) * dw
            if not np.isfinite(weight).all():
                raise ValueError("invalid master update")
            report["steps"].append({"step": step, "loss": loss,
                "host_step_seconds": time.perf_counter() - started,
                "master_sha256": hashlib.sha256(weight.tobytes()).hexdigest()})
            save()
        # Fresh forward validates that final updated runtime weights take effect.
        final = checked("final-forward", x, weight)
        final_loss = float(.5 * np.sum((final.astype(np.float64) - target) ** 2) / n)
        report.update(initial_loss=initial, final_loss=final_loss,
            loss_reduction=initial / final_loss, unique_master_generations=len({s["master_sha256"] for s in report["steps"]}),
            underflow_control={"unscaled_fp16": float(np.float16(1e-9)),
                               "scaled_then_fp32_unscaled": float(np.float16(1e-9 * 65536)) / 65536})
        if final_loss >= initial * .8 or report["unique_master_generations"] != 12:
            raise ValueError("loss/update generation gate rejected")
        report["status"] = "component-pass"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append(type(error).__name__ + ": " + str(error))
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ("compute_plan", "calls", "steps")}, indent=2))
    return int(report["status"] != "component-pass")


if __name__ == "__main__":
    raise SystemExit(main())
