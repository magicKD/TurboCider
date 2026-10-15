#!/usr/bin/env python3
"""Explicit pinned-ANEForge CPU contracts or small synchronous e5rt experiments.

Requires a caller-prepared, isolated source copy and explicitly built shim.
No downloads, install, auto-build, upstream pytest, profiling or unsafe async.
Use run_experiment.py to bound the owned process. All outputs stay in fresh dirs.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
from search_contracts import paired_result
sys.path.insert(0, str(ROOT / "labs/ane"))
from precision_oracle import reference as full_reference

FILES = ("__init__.py", "graph.py", "_compile.py", "_optimize.py", "_rewrite.py",
         "_runtime.py", "_circuit.py", "_targets.py", "_lib/ane_e5rt_dispatch.mm", "_lib/build.sh")
BOUNDARY = "host-fp16-array-to-new-host-fp32-array"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(reference, value, np):
    if value.shape != reference.shape or not np.isfinite(value).all():
        return {"finite": False, "relative_l2": None, "max_abs": None}
    error = value.astype(np.float64) - reference
    energy = float(np.sum(reference * reference))
    squared = float(np.sum(error * error))
    return {"finite": True, "relative_l2": (squared / energy)**.5 if energy else (0. if squared == 0 else None),
            "max_abs": float(np.max(np.abs(error)))}


def cpu_contracts(af, np, output):
    optimize = importlib.import_module("aneforge._optimize")
    compiler = importlib.import_module("aneforge._compile")
    rewrite = importlib.import_module("aneforge._rewrite")
    rng = np.random.default_rng(17)
    weights = [rng.normal(0, .1, (128, 64)).astype(np.float16) for _ in range(2)]
    graphs = [af.input((32, 128), max_abs=1.) @ w for w in weights]
    keys = [optimize._graph_key(g, [(32, 128)]) for g in graphs]
    programs = [compiler._lower_fused_to_dir(g) for g in graphs]
    content = [{"mil_sha256": sha(path / "model.mil"), "weights_sha256": sha(path / "weights.bin"),
                "upstream_content_key": path.name} for path in programs]
    cfg = {"int8": False, "decomp": [], "lossy": False}
    optimize._save_cache({keys[0]: {"config": cfg, "us": 123., "mocked_not_measured": True}})
    with patch.object(optimize, "build_variant", return_value="mock-only") as build, \
            patch.object(optimize, "measure", side_effect=AssertionError("cache hit must not measure")) as measured:
        result = optimize.tune(graphs[1], inputs=[np.zeros((32, 128), np.float16)])
        cache_hit = {"mocked_build_result": result, "build_calls": build.call_count,
                     "measurement_calls": measured.call_count, "weights_changed": True}
    with patch.object(optimize, "_fp32_reference", return_value=np.zeros((32, 64), np.float64)), \
            patch.object(optimize, "_measure_with_ref", return_value=(10., .2, np.zeros((32, 64), np.float32))), \
            patch.object(optimize, "build_variant", return_value="mock-only"):
        _, precision = optimize.tune_precision(graphs[0], target_error=.001,
                                               inputs=[np.zeros((32, 128), np.float16)])
    q, k, v = [af.input((1, 2, 8, 16)) for _ in range(3)]
    masked = af.sdpa(q, k, v, is_causal=True)
    unmasked = af.sdpa(q, k, v)
    return {"graph_keys": keys, "same_structure_key": keys[0] == keys[1], "emitted_programs": content,
            "different_weight_blob_and_content_key": content[0]["weights_sha256"] != content[1]["weights_sha256"]
                and content[0]["upstream_content_key"] != content[1]["upstream_content_key"],
            "cache_hit": cache_hit, "mocked_precision_unmet_budget": {
                "chosen_error": precision["chosen"]["relerr"], "reason": precision["reason"],
                "model_returned": True, "actual_device_executed": False},
            "route_guard": {"causal_candidates": optimize._route_ids(masked),
                            "unmasked_candidates": optimize._route_ids(unmasked),
                            "direct_helper_removes_sdpa": all(t.op != "sdpa" for t in
                                compiler._topo(rewrite.sdpa_to_decomposed(masked)))},
            "dylib_loaded": importlib.import_module("aneforge._runtime")._lib._lib is not None}


def device(af, np, args, output, report, save):
    import mlx.core as mx
    from importlib.metadata import version

    mx.set_default_device(mx.gpu)
    m, h, f = args.shape
    rng = np.random.default_rng(args.seed)
    wg, wu = [rng.normal(0, .04, (h, f)).astype(np.float16) for _ in range(2)]
    wd = rng.normal(0, .04, (f, h)).astype(np.float16)
    x = af.input((m, h), max_abs=1.)
    gate = x if args.case == "activation" else x @ wg
    graph = gate if args.case == "matmul" else gate.silu()
    if args.case == "ffn":
        graph = (gate.silu() * (x @ wu)) @ wd
    if args.case == "reduction":
        graph = x.sum([-1])
    candidate_graph = af.reduce_sum_to_matmul(graph) if args.candidate == "reduce-matmul" else graph
    if args.candidate == "exp-silu":
        activated = gate / ((gate * -1.).exp().adds(1.))
        candidate_graph = (activated * (x @ wu)) @ wd if args.case == "ffn" else activated
    report.update(shape_mhf=args.shape, seed=args.seed, case=args.case, candidate=args.candidate,
                  gpu_output_conversion=args.gpu_output_conversion,
                  gpu_compute=args.gpu_compute,
                  source_weight_sha256=[hashlib.sha256(w.tobytes()).hexdigest() for w in (wg, wu, wd)],
                  independent_oracle="full FP64 NumPy einsum(optimize=False), source inputs before FP16 intermediates",
                  independent_oracle_sha256=sha(ROOT / "labs/ane/precision_oracle.py"),
                  library_versions={"numpy": np.__version__, "mlx": version("mlx")},
                  gpu_identity={key: mx.device_info().get(key) for key in ("device_name", "architecture", "memory_size")},
                  cold_cache_claim=False, native_compute_device_mask=4, physical_ane_placement="not independently observed",
                  samples=[], setup={})
    models = {}
    try:
        for arm, value, int8 in (("ane-baseline", graph, False),
                                 ("ane-candidate", candidate_graph, args.candidate == "int8")):
            now = time.perf_counter_ns()
            models[arm] = af.compile(value, int8=int8, opt=0, build_dir=output / arm)
            report["setup"][arm] = {"emit_compile_load_ns": time.perf_counter_ns() - now,
                "n_graph_ops": models[arm].n_ops,
                "mil_sha256": sha(output / arm / "model.mil"),
                "weights_sha256": sha(output / arm / "weights.bin")}
            save()
        gpu_setup = time.perf_counter_ns()
        gpu_dtype = mx.float16 if args.gpu_compute == "fp16" else mx.float32
        weights_gpu = [mx.array(w, dtype=gpu_dtype) for w in (wg, wu, wd)]
        mx.eval(weights_gpu)
        def function(value):
            if args.case == "reduction":
                return mx.sum(value, axis=-1, keepdims=True)
            gate = value if args.case == "activation" else value @ weights_gpu[0]
            if args.case == "matmul":
                return gate
            if args.case in ("activation", "projection-silu"):
                return gate * mx.sigmoid(gate)
            up = value @ weights_gpu[1]
            return ((gate * mx.sigmoid(gate)) * up) @ weights_gpu[2]
        compiled = mx.compile((lambda value: function(value).astype(mx.float32))
                              if args.gpu_output_conversion == "compiled" else function)
        report["setup"]["gpu"] = {"weight_prepare_and_compile_wrapper_ns": time.perf_counter_ns() - gpu_setup,
                                     "lazy_kernel_compilation_in_first_call": True}
        def gpu(value):
            result = compiled(mx.array(value, dtype=gpu_dtype))
            return np.array(result).astype(np.float32)
        calls = {**models, "gpu": gpu}
        arms = ("gpu", "ane-baseline", "ane-candidate")
        orders = (arms, arms[::-1], (arms[1], arms[2], arms[0]),
                  (arms[0], arms[2], arms[1]), (arms[2], arms[0], arms[1]), (arms[1], arms[0], arms[2]))
        for phase, trials in (("first", 1), ("warmup", 2), ("measured", 6)):
            for trial in range(trials):
                value = rng.uniform(-.5, .5, (m, h)).astype(np.float16)
                if args.case == "reduction":
                    value[:, ::2] = np.float16(100.)
                    value[:, 1::2] = np.float16(-99.9375)
                    value[:, -1] += np.float16(trial / 8.)
                reference = full_reference(args.case, value, (wg, wu, wd), np)
                input_sha = hashlib.sha256(value.tobytes()).hexdigest()
                for arm in orders[trial % len(orders)]:
                    now = time.perf_counter_ns()
                    actual = calls[arm](value)
                    wall = time.perf_counter_ns() - now
                    report["samples"].append({"phase": phase, "trial": trial, "arm": arm, "wall_ns": wall,
                        "boundary": BOUNDARY, "input_sha256": input_sha, "output_dtype": str(actual.dtype),
                        "metrics": metrics(reference, actual, np)})
                    save()
        report["comparisons"] = [paired_result(report["samples"], a, b, .01) for a, b in
                                   (("ane-baseline", "ane-candidate"), ("gpu", "ane-baseline"), ("gpu", "ane-candidate"))]
        # Prepared GPU producer is a different boundary, explicitly not ratioed to host-fed ANE.
        report["resident_gpu_diagnostic"] = {"formal_performance_score": False, "samples": [],
            "boundary": "prepared GPU input to GPU output ready; validation/readback outside timing"}
        for trial in range(8):
            value = rng.uniform(-.5, .5, (m, h)).astype(np.float16)
            reference = full_reference(args.case, value, (wg, wu, wd), np)
            resident = mx.array(value, dtype=gpu_dtype)
            mx.eval(resident)
            now = time.perf_counter_ns()
            result = compiled(resident)
            mx.eval(result)
            wall = time.perf_counter_ns() - now
            report["resident_gpu_diagnostic"]["samples"].append({"warmup": trial < 2, "wall_ns": wall,
                "metrics": metrics(reference, np.array(result).astype(np.float32), np)})
        if args.case == "activation":
            report["activation_scalar_witness"] = []
            for point in (-.5, -.25, 0., .25, .5, 1.):
                value = np.full((m, h), point, dtype=np.float16)
                expected = point / (1 + np.exp(-point))
                row = {"input": point, "fp64_expected": float(expected)}
                for arm in arms:
                    actual = calls[arm](value)
                    row[arm] = {"minimum": float(actual.min()), "maximum": float(actual.max()),
                                "max_abs_error": float(np.max(np.abs(actual.astype(np.float64) - expected)))}
                report["activation_scalar_witness"].append(row)
        if args.case != "reduction":
            model = models["ane-baseline"]
            value = rng.uniform(-.5, .5, (m, h)).astype(np.float16)
            first = model(value)
            saved = first.copy()
            borrowed = model.output_view()
            snapshot = borrowed.copy()
            model(-value)
            changed = not np.array_equal(borrowed, snapshot)
            np.copyto(model.input_view(), value)
            borrowed.fill(np.nan)
            model.execute()
            report["alive_view_control"] = {"view_updated_by_next_call": changed,
                "model_result_copy_stable": np.array_equal(first, saved),
                "poison_fully_overwritten": bool(np.isfinite(borrowed).all()),
                "view_matches_model_first": np.array_equal(borrowed.astype(np.float32), first),
                "read_after_release_performed": False,
                "timing_used_for_formal_ratio": False}
            del borrowed, snapshot
        report["status"] = "measured"
    finally:
        # Synchronous calls only. Never keep/dereference borrowed views across release.
        for model in models.values():
            model.release()
        mx.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="isolated copy containing aneforge/")
    parser.add_argument("--mode", choices=("cpu", "device"), default="cpu")
    parser.add_argument("--case", choices=("matmul", "ffn", "reduction", "activation", "projection-silu"), default="matmul")
    parser.add_argument("--candidate", choices=("identity", "int8", "reduce-matmul", "exp-silu"), default="identity")
    parser.add_argument("--shape", type=int, nargs=3, default=(32, 128, 256), metavar=("M", "H", "F"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gpu-output-conversion", choices=("host", "compiled"), default="host")
    parser.add_argument("--gpu-compute", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(not 1 <= size <= cap or size % 32 for size, cap in zip(args.shape, (128, 512, 1024))):
        parser.error("aligned bounded M<=128/H<=512/F<=1024 required")
    if (args.case == "reduction") != (args.candidate == "reduce-matmul"):
        parser.error("reduction requires reduce-matmul and vice versa")
    if args.candidate == "exp-silu" and args.case not in ("ffn", "activation", "projection-silu"):
        parser.error("exp-silu requires an activation-bearing case")
    source = args.source_root.resolve()
    package = source / "aneforge"
    if not source.is_relative_to((ROOT.parent / "outputs").resolve()) or not package.is_dir():
        parser.error("use an explicit isolated source copy beneath repository outputs/")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "tc-aneforge-research-v1", "status": "starting", "mode": args.mode,
              "production_qualified": False, "runner_sha256": sha(Path(__file__)), "errors": [],
              "macos": platform.mac_ver()[0], "python": platform.python_version(),
              "source_sha256": {name: sha(package / name) for name in FILES},
              "source_contract_helper_sha256": sha(ROOT / "labs/search_contracts.py")}
    (output / "source").mkdir()
    for path in (Path(__file__), ROOT / "labs/search_contracts.py", ROOT / "labs/ane/precision_oracle.py"):
        shutil.copy2(path, output / "source" / path.name)
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save()
    try:
        if shutil.disk_usage(output).free < 512 * 1024**2:
            raise ValueError("512 MiB disk headroom required; nothing deleted")
        # Only this explicit child process is changed; HOME and security flags are untouched.
        for name in list(os.environ):
            if name.startswith(("ANEFORGE_", "ANE_", "DYLD_")):
                del os.environ[name]
        os.environ.update(ANEFORGE_CACHE_DIR=str(output / "cache"), ANEFORGE_CACHE=str(output / "build-cache"),
                          ANEFORGE_NO_AUTOBUILD="1", ANEFORGE_COMPILE_BREAKER_STRICT="1")
        dylib = package / "_lib/libane_e5rt_dispatch.dylib"
        if args.mode == "device":
            if platform.system() != "Darwin" or not dylib.is_file():
                raise ValueError("macOS and an explicitly prebuilt isolated shim required")
            os.environ["ANEFORGE_DYLIB"] = str(dylib)
            report["shim_sha256"] = sha(dylib)
        sys.path.insert(0, str(source))
        import numpy as np
        import aneforge as af
        if Path(af.__file__).resolve().parent != package:
            raise ValueError("unexpected upstream import path")
        if args.mode == "cpu":
            report["contracts"] = cpu_contracts(af, np, output)
            report["status"] = "cpu-contracts-observed"
        else:
            device(af, np, args, output, report, save)
        report["source_inputs_unchanged"] = report["source_sha256"] == {name: sha(package / name) for name in FILES}
        if not report["source_inputs_unchanged"]:
            raise ValueError("upstream source changed during probe")
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    save()
    print(json.dumps({"status": report["status"], "errors": report["errors"]}))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
