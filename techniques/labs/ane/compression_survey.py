#!/usr/bin/env python3
"""Explicit-device, bounded public Core ML compression experiment (synthetic).

No weights/downloads/private APIs/system changes. Fixed numerical gates are not
model-quality qualification. All outputs, including first/warmup, are checked.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


GATES = {"fp16": .015, "w8": .06, "lut4": .35, "w8a8": .08, "mlx-fp16": .015}


def metrics(reference, actual, np):
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ValueError("wrong shape or nonfinite output")
    a, b = actual.astype(np.float64).ravel(), reference.ravel()
    norm = float(np.linalg.norm(b))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("invalid synthetic reference")
    return {"relative_l2": float(np.linalg.norm(a - b) / norm),
            "max_abs": float(np.max(np.abs(a - b)))}


def lloyd_palette(weight):
    """Original deterministic scalar Lloyd iteration, 16 entries; no sklearn."""
    import numpy as np
    values = weight.ravel().astype(np.float64)
    centers = np.quantile(values, (np.arange(16) + .5) / 16)
    for _ in range(24):
        indices = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        centers = np.array([np.mean(values[indices == i]) if np.any(indices == i)
                            else centers[i] for i in range(16)])
        centers.sort()
    lut = centers.astype(weight.dtype)
    indices = np.argmin(np.abs(values[:, None] - lut.astype(np.float64)[None, :]), axis=1)
    return lut, indices.astype(np.uint8)


def reference(x, weights, np):
    value = x.reshape(weights[0].shape[0], -1).astype(np.float64)
    for weight in weights:
        # Independent unfused FP64 expression; never uses Core ML/MLX outputs.
        value = np.maximum(np.einsum("oc,cr->or", weight.astype(np.float64), value,
                                     optimize=False), 0)
    return value.reshape(x.shape)


def base_program(ct, np, weights, rows):
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    channels = weights[0].shape[0]
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, channels, 1, rows), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def graph(x):
        for i, weight in enumerate(weights):
            x = mb.conv(x=x, weight=weight[:, :, None, None], name="conv_" + str(i))
            x = mb.relu(x=x, name="relu_" + str(i))
        return mb.identity(x=x, name="y")
    return ct.convert(graph, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                      compute_precision=ct.precision.FLOAT16, skip_model_load=True)


def size(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def operator_counts(model):
    return dict(Counter(op.op_type for op in model._mil_program.functions["main"].operations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="store_true", help="explicit opt-in; required")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--units", choices=("cpu-ne", "cpu"), default="cpu-ne")
    args = parser.parse_args()
    if not args.device or platform.system() != "Darwin":
        parser.error("explicit --device on macOS is required")
    if (args.output_dir.exists() or args.output_dir.is_symlink()
            or not 16 <= args.rows <= 512 or not 32 <= args.channels <= 512
            or not 1 <= args.layers <= 8 or args.rounds not in (10, 20)):
        parser.error("new output directory, rows16..512,C32..512,L1..8,rounds10/20 required")
    args.output_dir.mkdir(parents=True)
    output = args.output_dir
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    report = {"schema": "ane-compression-survey-v1", "status": "running",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shape": [1, args.channels, 1, args.rows], "layers": args.layers, "seed": args.seed,
        "rounds": args.rounds, "units": args.units, "gates_relative_l2": GATES,
        "scope": "synthetic host FP16 input through full graph to independent owned host output",
        "physical_ane_placement": "unverified", "integer_mac_observed": False,
        "environment": {"python": platform.python_version(), "macos": platform.mac_ver()[0],
            "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
            "soc": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()},
        "arms": {}, "samples": [], "errors": []}
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        with redirect_stdout(sys.stderr):
            import numpy as np
            import coremltools as ct
            import mlx.core as mx
        report["environment"].update(coremltools=ct.__version__, numpy=np.__version__)
        units = ct.ComputeUnit.CPU_AND_NE if args.units == "cpu-ne" else ct.ComputeUnit.CPU_ONLY
        rng = np.random.default_rng(args.seed)
        weights = [(rng.normal(0, 1 / math.sqrt(args.channels), (args.channels, args.channels))).astype(np.float16)
                   for _ in range(args.layers)]
        report["weight_sha256"] = hashlib.sha256(b"".join(w.tobytes() for w in weights)).hexdigest()
        samples = [np.random.default_rng(args.seed + 1000 + i).normal(0, .35, report["shape"]).astype(np.float16)
                   for i in range(args.rounds + 3)]
        oracle = [reference(x, weights, np) for x in samples]
        report["input_sha256"] = [hashlib.sha256(x.tobytes()).hexdigest() for x in samples]
        with redirect_stdout(sys.stderr):
            started = time.perf_counter()
            base = base_program(ct, np, weights, args.rows)
            report["base_export_seconds"] = time.perf_counter() - started
        loaded, functions = {}, {}
        # Acquisition/calibration/model loading is outside the warm benchmark.
        for name in ("fp16", "w8", "lut4", "w8a8"):
            arm = report["arms"][name] = {"status": "preparing", "gate": GATES[name]}
            try:
                started = time.perf_counter()
                with redirect_stdout(sys.stderr):
                    opt = ct.optimize.coreml
                    if name == "fp16":
                        model = base
                    elif name == "lut4":
                        config = opt.OptimizationConfig(global_config=opt.OpPalettizerConfig(
                            mode="custom", lut_function=lloyd_palette, weight_threshold=0))
                        model = opt.palettize_weights(base, config)
                    else:
                        config = opt.OptimizationConfig(global_config=opt.OpLinearQuantizerConfig(
                            mode="linear_symmetric", dtype="int8", weight_threshold=0))
                        target = base
                        if name == "w8a8":
                            calibration = [{"x": np.random.default_rng(args.seed + 100 + j).normal(
                                0, .45, report["shape"]).astype(np.float16)} for j in range(4)]
                            target = opt.linear_quantize_activations(base, config, calibration)
                        model = opt.linear_quantize_weights(target, config)
                arm["optimization_seconds"] = time.perf_counter() - started
                arm["mil_operator_counts"] = operator_counts(model)
                package, compiled = output / (name + ".mlpackage"), output / (name + ".mlmodelc")
                model.save(str(package))
                arm["package_bytes"] = size(package)
                started = time.perf_counter()
                with redirect_stdout(sys.stderr):
                    ct.models.utils.compile_model(str(package), destination_path=str(compiled))
                arm["compile_seconds"] = time.perf_counter() - started
                started = time.perf_counter()
                loaded[name] = ct.models.CompiledMLModel(str(compiled), compute_units=units)
                arm["load_seconds"] = time.perf_counter() - started
                from compute_plan import inspect
                try:
                    arm["compute_plan"] = inspect(compiled, units)
                except Exception as error:
                    arm["compute_plan_error"] = type(error).__name__ + ": " + str(error)
                def predict(x, model=loaded[name]):
                    return np.array(model.predict({"x": x})["y"], copy=True)
                functions[name] = predict
                arm["status"] = "ready"
            except Exception as error:
                arm.update(status="setup-failed", error=type(error).__name__ + ": " + str(error))
            save()
        mx_weights = [mx.array(w) for w in weights]
        mx.eval(*mx_weights)
        @mx.compile
        def gpu_graph(x):
            for w in mx_weights:
                x = mx.maximum(mx.matmul(w, x), 0)
            return x
        def gpu_predict(x):
            result = gpu_graph(mx.array(x.reshape(args.channels, args.rows)))
            mx.eval(result)
            return np.array(result, copy=True).reshape(x.shape)
        functions["mlx-fp16"] = gpu_predict
        report["arms"]["mlx-fp16"] = {"status": "ready", "gate": GATES["mlx-fp16"],
            "boundary": "host->MLX GPU graph->owned host; not prepared-GPU-only"}
        names = list(functions)
        for trial in range(-3, args.rounds):
            index = trial + 3
            order = names[index % len(names):] + names[:index % len(names)]
            if index % 2:
                order.reverse()
            for name in order:
                arm = report["arms"][name]
                if arm["status"] not in ("ready", "numerical-pass"):
                    continue
                started = time.perf_counter()
                actual = functions[name](samples[index])
                elapsed = time.perf_counter() - started
                numerical = metrics(oracle[index], actual, np)
                row = {"trial": trial, "arm": name, "order": order, "host_seconds": elapsed,
                       "input_sha256": report["input_sha256"][index], "metrics": numerical,
                       "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                       "phase": "warmup" if trial < 0 else "timed"}
                report["samples"].append(row)
                if "first_prediction_seconds" not in arm:
                    arm["first_prediction_seconds"] = elapsed
                if numerical["relative_l2"] > GATES[name]:
                    arm.update(status="numerical-rejected", rejected_trial=trial)
                else:
                    arm["status"] = "numerical-pass"
            save()
        for name, arm in report["arms"].items():
            rows = [r for r in report["samples"] if r["arm"] == name and r["phase"] == "timed"]
            if arm["status"] == "numerical-pass" and len(rows) == args.rounds:
                arm.update(status="component-pass", median_host_seconds=statistics.median(
                    r["host_seconds"] for r in rows), timed_calls=len(rows),
                    maximum_relative_l2=max(r["metrics"]["relative_l2"] for r in report["samples"] if r["arm"] == name))
        report["status"] = ("finished" if all(a["status"] == "component-pass"
            for a in report["arms"].values()) else "partial")
        mx.synchronize()
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append(type(error).__name__ + ": " + str(error))
    save()
    print(json.dumps({"status": report["status"], "arms": {n: {k: v for k, v in a.items()
        if k not in ("compute_plan", "mil_operator_counts")} for n, a in report["arms"].items()}}, indent=2))
    return int(report["status"] != "finished")


if __name__ == "__main__":
    raise SystemExit(main())
