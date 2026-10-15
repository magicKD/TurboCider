#!/usr/bin/env python3
"""Tiny PUBLIC Core ML FFN lifecycle and optional MLX/channel fork-join lab.

No original models, private API, product library, download or global env edits.
Frozen FP16 works with coremltools 8.3; runtime W8 uses the repo's exact public
macOS26/CoreML9 exporter, and explicitly reports unsupported on older tooling.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import importlib.util
import hashlib
import math
from pathlib import Path
import platform
import sys
import tempfile
import time

LABS = Path(__file__).resolve().parents[1]
ROOT = LABS.parents[1]
sys.path.insert(0, str(LABS))
from common import emit, file_sha, report


def load_exporter(name):
    source = ROOT / "tools/coreml" / name
    spec = importlib.util.spec_from_file_location("tc_lab_" + source.stem, source)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value, source


def numeric_metrics(reference, actual, np):
    if reference.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("output shape/nonfinite failure")
    a, b = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    denominator = np.linalg.norm(a)
    return {"relative_l2": float(np.linalg.norm(a - b) / denominator) if denominator else None,
            "max_abs": float(np.max(np.abs(a - b))), "finite": True}


def frozen_program(ct, np, wg, wu, wd, rows):
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    hidden = wg.shape[1]
    @mb.program(input_specs=[mb.TensorSpec(shape=(rows, hidden), dtype=types.fp16)], opset_version=ct.target.macOS15)
    def program(x):
        gate = mb.matmul(x=x, y=wg, transpose_y=True)
        up = mb.matmul(x=x, y=wu, transpose_y=True)
        activation = mb.real_div(x=gate, y=mb.add(x=np.float16(1),
            y=mb.exp(x=mb.mul(x=gate, y=np.float16(-1)))))
        h = mb.mul(x=activation, y=up)
        return mb.identity(x=mb.matmul(x=h, y=wd, transpose_y=True), name="y")
    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("frozen-fp16", "runtime-fp16", "runtime-w8a8"), default="frozen-fp16")
    parser.add_argument("--compute-units", choices=("cpu", "cpu-ne", "cpu-gpu", "all"), default="cpu-ne")
    parser.add_argument("--compare-gpu", action="store_true", help="frozen FP16 prefix || actual MLX GPU suffix")
    parser.add_argument("--rows", type=int, default=33)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--ane-channels", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--coreml-overlay", type=Path, help="optional isolated coremltools install directory; no existing environment changed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error("report already exists")
    if platform.system() != "Darwin":
        parser.error("actual public Core ML requires macOS")
    if not (1 <= args.rows <= 128 and 2 <= args.hidden <= 512 and 2 <= args.width <= 1024 and
            4 <= args.repeats <= 32 and args.repeats % 4 == 0):
        parser.error("bounded lab requires rows<=128,H<=512,F<=1024,repeats4..32 in ABBA multiples")
    if args.compare_gpu and (args.kind != "frozen-fp16" or not 0 < args.ane_channels < args.width):
        parser.error("GPU comparison needs frozen FP16 and a proper interior channel split")
    if args.kind == "runtime-w8a8":
        if (args.rows, args.hidden, args.width) != (33, 128, 512):
            parser.error("native shared W8 integration fixture uses M33/H128/F512; adapt its native source for another test geometry")
        from shared_w8 import run
        return run(overlay=args.coreml_overlay, output=args.output)
    if args.coreml_overlay:
        overlay = args.coreml_overlay.resolve()
        if not (overlay / "coremltools/__init__.py").is_file():
            parser.error("overlay must contain an explicitly installed coremltools package")
        sys.path.insert(0, str(overlay))
    with redirect_stdout(sys.stderr):
        import numpy as np
        import coremltools as ct
    units = {"cpu": ct.ComputeUnit.CPU_ONLY, "cpu-ne": ct.ComputeUnit.CPU_AND_NE,
             "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU, "all": ct.ComputeUnit.ALL}[args.compute_units]
    rng = np.random.default_rng(args.seed)
    m, h, f = args.rows, args.hidden, args.width
    wg, wu, wd = [(rng.normal(0, .04, shape)).astype(np.float16) for shape in ((f, h), (f, h), (h, f))]
    weight_digest = hashlib.sha256(b"".join(value.tobytes() for value in (wg, wu, wd))).hexdigest()
    prefix = args.ane_channels if args.compare_gpu else f
    exporter, source = None, Path(__file__)
    conversion_start = time.perf_counter()
    with redirect_stdout(sys.stderr):
        if args.kind == "frozen-fp16":
            program = frozen_program(ct, np, wg[:prefix], wu[:prefix], wd[:, :prefix], m)
        else:
            exporter, source = load_exporter("export_runtime_ane.py")
            # Match frozen's exp-SiLU/full-K three projections. Otherwise a
            # different reduction/lowering would confound a storage-ABI lesson.
            program = exporter.make_program(exporter.geometry("swiglu", m, h, f, max(h, f), f))
        model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                           compute_precision=ct.precision.FLOAT16, skip_model_load=True)
    conversion_seconds = time.perf_counter() - conversion_start
    with tempfile.TemporaryDirectory(prefix="tc-technique-coreml-") as directory:
        folder = Path(directory)
        package, compiled = folder / "graph.mlpackage", folder / "graph.mlmodelc"
        model.save(str(package))
        started = time.perf_counter()
        with redirect_stdout(sys.stderr):
            ct.models.utils.compile_model(str(package), destination_path=str(compiled))
        compile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        loaded = ct.models.CompiledMLModel(str(compiled), compute_units=units)
        load_seconds = time.perf_counter() - started
        mx, full_gpu, suffix_gpu = None, None, None
        if args.compare_gpu:
            import mlx.core as mx
            gw, uw, dw = [mx.array(value) for value in (wg, wu, wd)]
            mx.eval(gw, uw, dw)
            def ffn(x, first, count):
                gate = mx.matmul(x, gw[first:first + count].T)
                up = mx.matmul(x, uw[first:first + count].T)
                hidden = (gate * mx.sigmoid(gate)) * up
                return mx.matmul(hidden, dw[:, first:first + count].T)
            full_gpu = mx.compile(lambda x: ffn(x, 0, f))
            suffix_gpu = mx.compile(lambda x: ffn(x, prefix, f - prefix))
        calls, observations, first_prediction_seconds = 0, [], None
        max_relative, max_absolute = 0., 0.
        # One worker is persistent across trials. Its prediction owns all arrays
        # until result(); context teardown joins even when the GPU branch raises.
        with ThreadPoolExecutor(max_workers=1) as worker:
            try:
                for trial in range(-4, args.repeats):
                    candidate = (trial + 4) % 4 in (1, 2)
                    generation = (trial + 4) // 2
                    pair_rng = np.random.default_rng(args.seed + generation)
                    x = pair_rng.normal(0, .3, (m, h)).astype(np.float16)
                    # Independent FP64 einsum oracle, not a BLAS/device self-control.
                    # Some NumPy/Accelerate builds report stale FP status flags
                    # around matmul after foreign framework calls; do not merely
                    # suppress warnings or weaken finite checks to hide that.
                    xx, gg, uu, dd = [value.astype(np.float64) for value in (x, wg, wu, wd)]
                    gate = np.einsum("mk,nk->mn", xx, gg, optimize=False)
                    up = np.einsum("mk,nk->mn", xx, uu, optimize=False)
                    hidden_reference = (gate / (1 + np.exp(-gate))) * up
                    reference = np.einsum("mk,nk->mn", hidden_reference, dd, optimize=False)
                    staging_start = time.perf_counter()
                    features = {"x": x}
                    if args.kind == "runtime-fp16": features.update(wg=wg, wu=wu, wd=wd)
                    staging_seconds = time.perf_counter() - staging_start if mx is None or candidate else 0.
                    if mx is not None:
                        device_x = mx.array(x); mx.eval(device_x)  # common producer, outside target boundary
                    start = time.perf_counter()
                    prediction_seconds, gpu_wait_seconds = None, None
                    def predict():
                        nonlocal calls, first_prediction_seconds
                        now = time.perf_counter()
                        value = loaded.predict(features)
                        span = time.perf_counter() - now
                        calls += 1
                        if first_prediction_seconds is None: first_prediction_seconds = span
                        return np.array(value["y"], copy=True), span
                    if mx is not None and not candidate:
                        result = full_gpu(device_x)
                        mx.eval(result)
                        actual = np.array(result)
                    elif mx is not None:
                        future = worker.submit(predict)
                        head = suffix_gpu(device_x)
                        mx.async_eval(head)
                        prefix_output, prediction_seconds = future.result()
                        now = time.perf_counter()
                        # Deliberately includes Python/host-to-MLX restore and final join.
                        result = head + mx.array(prefix_output)
                        mx.eval(result)
                        gpu_wait_seconds = time.perf_counter() - now
                        actual = np.array(result)
                    else:
                        actual, prediction_seconds = predict()
                    wall = time.perf_counter() - start
                    metric = numeric_metrics(reference, actual, np)
                    max_relative = max(max_relative, metric["relative_l2"] or 0)
                    max_absolute = max(max_absolute, metric["max_abs"])
                    if metric["relative_l2"] is not None and metric["relative_l2"] > .02:
                        raise ValueError("toy FFN component numerical gate failed; no timing qualification")
                    if trial >= 0:
                        observations.append({"pair": trial // 2, "input_generation": generation,
                            "variant": ("hybrid" if candidate else "gpu") if mx is not None else "coreml",
                            "wall_seconds": wall, "host_staging_seconds": staging_seconds,
                            "request_boundary_seconds": wall + staging_seconds,
                            "prediction_host_seconds": prediction_seconds, "post_prediction_restore_join_host_seconds": gpu_wait_seconds,
                            "metrics": metric})
            finally:
                if mx is not None: mx.synchronize()
        # Model and worker stop borrowing compiled files before TemporaryDirectory cleanup.
        del loaded, model
    value = report("coreml-ffn", {"kind": args.kind, "shape": [m, h, f], "prefix_channels": prefix,
                   "seed": args.seed, "repeats": args.repeats, "weight_dtype": "fp16", "input_dtype": "fp16",
                   "reference_dtype": "fp64_einsum", "synthetic_weight_sha256": weight_digest, "macos": platform.mac_ver()[0],
                   "compute_units_policy": args.compute_units, "coremltools": ct.__version__,
                   "coreml_math_recipe": "fp16-exp-silu-full-k-three-projections",
                   "conversion_seconds": conversion_seconds, "compile_seconds": compile_seconds,
                   "load_seconds": load_seconds, "first_prediction_seconds": first_prediction_seconds,
                   "actual_public_prediction_calls_including_warmup": calls, "warmup_trials": 4,
                   "max_component_relative_l2": max_relative, "maximum_abs_error": max_absolute,
                   "samples": observations, "exporter_sha256": file_sha(source), "runner_sha256": file_sha(Path(__file__)),
                   "limits": "Public compute-unit policy is not device placement. Python features/restores are not zero-copy. Tiny synthetic FFN, base only, no E2E/model-quality claim."},
                   scope="actual public Core ML operator lifecycle; optional actual MLX GPU channel branch", backend="public_coreml_python+mlx_gpu" if args.compare_gpu else "public_coreml_python")
    emit(value, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
