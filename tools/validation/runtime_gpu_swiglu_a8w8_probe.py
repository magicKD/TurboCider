"""Bounded, checkpoint-free portable GPU SwiGLU A8W8 comparison.

Run serially:
  .venv/bin/python -B tools/validation/runtime_gpu_swiglu_a8w8_probe.py \
    --size tiny --output outputs/runtime-gpu-swiglu-tiny

Only tiny (32/64/96) and medium (128/512/768) are accepted. First requests and
warm totals include CPU->MLX entry, three weight FWHT/normalization/quantization,
original-basis GPU LoRA, full forward, synchronization and NumPy readback. The
integer path uses scalar int32 Metal ALUs, not native matrix acceleration.
The floating dequant path has its own rounding oracle. No Core ML is used.
"""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools/metal"), str(ROOT / "tools/validation")]
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_owned
from runtime_ane_swiglu_int8_probe import errors, fixtures, unquantized_oracle, write_report
from runtime_swiglu_a8w8 import GpuCandidate, VARIANTS, geometry, integer_dot_reference, reference


def sample(candidate, case, variant):
    import numpy as np
    mx = candidate.mx
    mx.synchronize()
    started = time.perf_counter()
    entered = candidate.enter(case)
    x, weights, adapters = entered
    mx.eval(x, *weights, *[value for a, b, _ in adapters for value in (a, b)])
    mx.synchronize()
    entered_at = time.perf_counter()
    prepared = candidate.prepare_request(entered, variant)
    values, dg, du, (a, b, _) = prepared
    mx.eval(*[value for pair in values for value in pair], dg, du, a, b)
    mx.synchronize()
    prepared_at = time.perf_counter()
    output = candidate.forward(prepared, variant)
    mx.eval(output["h"], output["y"])
    mx.synchronize()
    forwarded_at = time.perf_counter()
    actual = {name: np.array(value) for name, value in output.items()}
    mx.synchronize()
    completed_at = time.perf_counter()
    return actual, {"entry_ms": (entered_at - started) * 1000,
                    "prepare_ms": (prepared_at - entered_at) * 1000,
                    "forward_ms": (forwarded_at - prepared_at) * 1000,
                    "readback_ms": (completed_at - forwarded_at) * 1000,
                    "total_ms": (completed_at - started) * 1000}


def arithmetic_smoke(candidate):
    import numpy as np
    mx = candidate.mx
    a = np.array([[127, -127, 0, 1, -1], [-2, 3, -4, 5, -6]], np.int8)
    b = np.array([[-127, 127, 1, -1, 1], [6, -5, 4, -3, 2], [0, 0, 0, 0, 0]], np.int8)
    rng = np.random.default_rng(213)
    matrices = [("signed_tail", a, b),
                ("random_tail", rng.integers(-128, 128, (3, 17), dtype=np.int8),
                 rng.integers(-128, 128, (5, 17), dtype=np.int8)),
                ("negative_extreme", np.full((1, 768), -128, np.int8), np.full((2, 768), -128, np.int8)),
                ("positive_extreme", np.full((1, 768), 127, np.int8), np.full((2, 768), 127, np.int8))]
    receipts = []
    for name, x, w in matrices:
        result = candidate.integer_dot(mx.array(x), mx.array(w))
        mx.eval(result)
        mx.synchronize()
        actual, expected = np.array(result), integer_dot_reference(x, w)
        exact = actual.dtype == np.int32 and np.array_equal(actual, expected)
        receipts.append({"case": name, "input_shapes": [list(x.shape), list(w.shape)],
                         "numpy_dtype": str(actual.dtype), "int64_oracle_exact": exact})
        if not exact:
            raise ValueError("GPU signed INT8/int32 dot failed its exact independent int64 oracle")
    group = candidate.config["group"]
    basis = np.eye(group, dtype=np.float32)
    actual = candidate.rotate(mx.array(basis))
    mx.eval(actual)
    mx.synchronize()
    from runtime_ane_swiglu_int8_candidate import hadamard
    np.testing.assert_allclose(np.array(actual), hadamard(basis, group), atol=1e-7, rtol=1e-6)
    return {"signed_int8_int32_dot_exact": True, "integer_cases": receipts, "FWHT_basis_order_checked": True,
            "native_matrix_acceleration_proven": False}


def worker(args):
    import numpy as np
    config = geometry(args.size)
    report = json.loads((args.output / "report.json").read_text())
    candidate = None
    try:
        started = time.perf_counter()
        candidate = GpuCandidate(config)
        mx = candidate.mx
        report["gpu_initialization_ms"] = (time.perf_counter() - started) * 1000
        report["mlx_version"] = importlib.metadata.version("mlx")
        report["mlx_python_extension_sha256"] = sha256_file(Path(mx.__file__))
        report["device_info"] = mx.device_info()
        cases = fixtures(config["rows"], config["hidden"], config["width"], config["rank"])
        observed = {}
        reports = report["variants"] = {}
        for variant in VARIANTS:
            observed[variant] = {}
            row = reports[variant] = {"arithmetic": config["variant_arithmetic"][variant], "cases": []}
            for case in cases:
                actual, timing = sample(candidate, case, variant)
                expected = reference(case, config, variant)
                exact = unquantized_oracle(case)
                result = {"case": case[0], "timing": timing, "vs_own_cpu_oracle": {}, "vs_unquantized_math": {}}
                row["cases"].append(result)
                if case[0] == "A_lora":
                    row["first_complete_request"] = timing
                for name in ("h", "y"):
                    if actual[name].dtype != np.float16:
                        raise ValueError("MLX candidate did not retain declared FP16 output")
                    result["vs_own_cpu_oracle"][name] = errors(actual[name], expected[name])
                    result["vs_unquantized_math"][name] = errors(actual[name], exact[name])
                    if result["vs_own_cpu_oracle"][name]["relative_l2"] > .03:
                        raise ValueError(f"{variant}/{case[0]}/{name} differs from its own CPU rounding contract")
                    if case[0] == "zero" and np.any(actual[name] != 0):
                        raise ValueError("zero input/adapter produced nonzero output")
                observed[variant][case[0]] = actual
                write_report(args.output, report)
            row["A_after_B_exact"] = all(np.array_equal(observed[variant]["A_lora"][name],
                                                         observed[variant]["A_lora_again"][name]) for name in ("h", "y"))
            if not row["A_after_B_exact"]:
                raise ValueError("runtime input/weight/adapter rebinding failed")
            row["warm_samples"] = []
        # Exact integer/basis smoke is outside measured requests; running it
        # here avoids hiding the first full request's kernel specialization.
        report["arithmetic_smoke"] = arithmetic_smoke(candidate)
        for _ in range(args.warm_rounds):
            for variant in (*VARIANTS, *reversed(VARIANTS)):
                actual, timing = sample(candidate, cases[0], variant)
                for name in ("h", "y"):
                    if not np.array_equal(actual[name], observed[variant]["A_lora"][name]):
                        raise ValueError(f"{variant}/{name} changed during identical warmed requests")
                reports[variant]["warm_samples"].append(timing)
        for variant in VARIANTS:
            row = reports[variant]
            row["warm_median_ms"] = {key: statistics.median(value[key] for value in row["warm_samples"])
                                     for key in row["warm_samples"][0]}
        report["quantized_vs_fp16_control"] = [{"variant": variant, "case": case[0],
            **{name: errors(observed[variant][case[0]][name], observed["fp16_control"][case[0]][name]) for name in ("h", "y")}}
            for variant in VARIANTS[1:] for case in cases]
        report["memory"] = {"active_bytes": mx.get_active_memory(), "cache_bytes": mx.get_cache_memory(),
                            "peak_bytes": mx.get_peak_memory(), "scope": "MLX allocator, not system footprint"}
        report["worker_success"] = True
        write_report(args.output, report)
        return 0
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["worker_success"] = False
        write_report(args.output, report)
        raise
    finally:
        if candidate is not None:
            candidate.mx.synchronize()
        candidate = None
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--size", choices=("tiny", "medium"), default="tiny")
    parser.add_argument("--warm-rounds", type=int, choices=(1, 2), default=1,
                        help="2 or 4 warm samples per variant; no unbounded loop")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    config = geometry(args.size)
    sources = [Path(__file__), ROOT / "tools/metal/runtime_swiglu_a8w8.py",
               ROOT / "tools/coreml/runtime_ane_swiglu_int8_candidate.py",
               ROOT / "tools/validation/runtime_ane_swiglu_int8_probe.py"]
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}
    report = {"scope": "portable full-SwiGLU synthetic GPU controls; no checkpoint or Core ML",
              "platform": platform.platform(), "config": config, "source_sha256": hashes,
              "timing_scope": "CPU->MLX entry + all three weight FWHT/quant preparation + GPU original-basis LoRA + forward + eval/synchronize + NumPy readback",
              "comparison_scope": "three GPU variants only; not a paired comparison with the earlier Core ML FP16-input candidate",
              "gpu_input_contract": "FP32 source matrices/row scales/LoRA factors and gate-up deltas; corrected h and final y FP16; Core ML scales/factors/deltas had FP16 input boundaries",
              "first_scope": "first complete request per variant in this process; includes specialization, may share MLX/driver caches; initialization separately recorded",
              "native_int8_matrix_acceleration": False, "production_speedup_verified": False,
              "artifact_scope": "no model export; Metal/MLX system driver caches outside this temporary directory are not controlled",
              "success": False}
    write_report(output, report)
    started = time.monotonic()
    fixture = tempfile.TemporaryDirectory(prefix="tc-gpu-swiglu-")
    scratch = Path(fixture.name).resolve()
    environment = benchmark_environment()
    environment["TMPDIR"] = str(scratch) + "/"
    report["temporary_directory"] = str(scratch)
    command = [sys.executable, "-B", Path(__file__).resolve(), "--worker", "--output", output,
               "--size", args.size, "--warm-rounds", args.warm_rounds]
    report["command"] = [str(value) for value in command]
    write_report(output, report)
    try:
        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
            result = run_owned(report["command"], cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, timeout=90)
        report = json.loads((output / "report.json").read_text())
        report["worker_exit_code"] = result.returncode
        report["success"] = result.returncode == 0 and report.get("worker_success") is True
    except BaseException as error:
        report = json.loads((output / "report.json").read_text())
        report["error"] = f"{type(error).__name__}: {error}"
        report["success"] = False
        if getattr(error, "__notes__", None):
            report["cleanup_notes"] = error.__notes__
    finally:
        try:
            fixture.cleanup()
            report["owned_temporary_directory_removed"] = not scratch.exists()
        except BaseException as error:
            report["owned_temporary_directory_removed"] = False
            report["cleanup_error"] = str(error)
        try:
            report["source_sha256_after"] = {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}
            report["sources_unchanged"] = report["source_sha256_after"] == hashes
        except BaseException as error:
            report["sources_unchanged"] = False
            report["source_hash_error"] = str(error)
        report["success"] = report["success"] and report["sources_unchanged"] and report["owned_temporary_directory_removed"]
        report["total_seconds"] = time.monotonic() - started
        write_report(output, report)
    print("Evidence: " + str(output / "report.json"), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
