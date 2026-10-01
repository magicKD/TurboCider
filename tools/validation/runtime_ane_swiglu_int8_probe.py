"""One isolated synthetic full-SwiGLU FP16/QDQ comparison (no checkpoint).

Example, run serially with other Core ML/Metal work idle:
  .venv/bin/python -B tools/validation/runtime_ane_swiglu_int8_probe.py \
    --output outputs/runtime-ane-swiglu-int8-tiny

All weights and low-rank factors remain dynamic inputs. Timings include CPU
gate/up LoRA preparation, rotation/normalization and full prediction, including
FP32 nonlinear/restore/down-LoRA work and h output. CPU_AND_NE and compute-plan
preference are not physical ANE/INT8 traces. Owned graph artifacts are removed
on success, failure, timeout and interruption; logs/reports/MIL text remain.
The installed predict bridge promotes input dict arrays to NumPy FP32 and may
return FP32 copies of FP16 MLMultiArray outputs. Both conversions are timed;
the declared model interface and exact output FP16 roundtrip are verified.
"""
import argparse
import collections
import gc
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_owned

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/coreml"))


def write_report(output, report):
    temporary = output / ".report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output / "report.json")


def errors(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite or incorrectly shaped candidate output")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {"relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-12)),
            "max_absolute": float(np.max(np.abs(difference)))}


def verify_prediction_boundary(actual, row, spec, phase):
    """Require an FP16 model/MIL contract and lossless Python bridge promotion.

    Apple's Python proxy can promote an MLMultiArray Float16 to a NumPy Float32
    array. A wider NumPy dtype alone does not prove a wider model interface.
    Accept it only when the declared interfaces agree and every finite value
    is exactly representable in FP16. Leave actual arrays untouched for metrics.
    """
    import numpy as np
    if set(actual) != set(spec["outputs"]):
        raise ValueError("prediction must expose complete output and original hidden")
    record = {"phase": phase, "outputs": {}}
    row.setdefault("prediction_boundaries", []).append(record)
    for name, shape in spec["outputs"].items():
        value = np.asarray(actual[name])
        description = row["model_description_outputs"].get(name, {})
        mil_dtype = row["mil_interface_outputs"].get(name)
        receipt = record["outputs"][name] = {
            "model_description_dtype": description.get("dtype"), "mil_output_dtype": mil_dtype,
            "numpy_dtype": str(value.dtype), "numpy_shape": list(value.shape),
            "finite": bool(np.isfinite(value).all()), "fp16_roundtrip_exact": False,
            "python_bridge_promoted": value.dtype == np.float32}
        if (description.get("feature_type") != "multiArrayType" or description.get("dtype") != "FLOAT16"
                or description.get("shape") != shape or mil_dtype != "FLOAT16"):
            raise ValueError(f"{name} model description/MIL did not retain the FP16 output interface")
        if list(value.shape) != shape or value.dtype not in (np.dtype(np.float16), np.dtype(np.float32)) or not receipt["finite"]:
            raise ValueError(f"{name} has invalid Python output shape/dtype/finite values")
        with np.errstate(over="ignore", invalid="ignore"):
            restored = value.astype(np.float16).astype(value.dtype)
        receipt["fp16_roundtrip_exact"] = bool(np.array_equal(value, restored) and
                                                np.array_equal(np.signbit(value), np.signbit(restored)))
        if not receipt["fp16_roundtrip_exact"]:
            raise ValueError(f"{name} Python output is not an exact FP16 bridge promotion")


def fixtures(rows, hidden, width, rank):
    import numpy as np
    rng = np.random.default_rng(179)
    x = rng.normal(0, .6, (rows, hidden)).astype(np.float32)
    weights = [rng.normal(0, .15, shape).astype(np.float32)
               for shape in ((width, hidden), (width, hidden), (hidden, width))]
    adapters = [(rng.normal(0, .12, (rank, n)).astype(np.float32),
                 rng.normal(0, .20, (m, rank)).astype(np.float32), scale)
                for n, m, scale in ((hidden, width, .9), (hidden, width, -.6), (width, hidden, .7))]
    zero = [(np.zeros_like(a), np.zeros_like(b), scale) for a, b, scale in adapters]
    # A -> changed runtime weights/factors -> A proves an export did not bind
    # one preparation's model data. Base/zero use the same complete graph.
    return [("A_lora", x, weights, adapters),
            ("B_lora", x * .7, [weights[0] * -.8, weights[1] * 1.2, weights[2] * .5],
             [(a * .9, b * -1.1, -scale) for a, b, scale in adapters]),
            ("A_lora_again", x, weights, adapters),
            ("base", x, weights, zero), ("zero", np.zeros_like(x), weights, zero)]


def prepare_case(case, spec):
    from runtime_ane_swiglu_int8_candidate import lora_delta, prepare_inputs
    _, x, weights, adapters = case
    return prepare_inputs(x, *weights, spec, dg=lora_delta(x, *adapters[0]),
                          du=lora_delta(x, *adapters[1]), down_a=adapters[2][0],
                          down_b=adapters[2][1], down_scale=adapters[2][2])


def unquantized_oracle(case):
    import numpy as np
    _, x, (wg, wu, wd), adapters = case
    def lowrank(value, factors):
        a, b, scale = factors
        low = np.einsum("ik,rk->ir", value, a.astype(np.float64), optimize=False)
        return np.einsum("ir,or->io", low, b.astype(np.float64), optimize=False) * scale
    x = x.astype(np.float64)
    gate = np.einsum("ik,jk->ij", x, wg.astype(np.float64), optimize=False) + lowrank(x, adapters[0])
    up = np.einsum("ik,jk->ij", x, wu.astype(np.float64), optimize=False) + lowrank(x, adapters[1])
    h = gate / (1 + np.exp(-gate)) * up
    return {"h": h, "y": np.einsum("ik,jk->ij", h, wd.astype(np.float64), optimize=False) + lowrank(h, adapters[2])}


def worker(args):
    import coremltools as ct
    import numpy as np
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.proto import FeatureTypes_pb2, MIL_pb2
    from google.protobuf.text_format import MessageToString
    from qwen21_ane_placement import inspect_plan
    from runtime_ane_swiglu_int8_candidate import export, make_program, reference, specification

    report = json.loads((args.output / "report.json").read_text())
    report["coremltools"] = ct.__version__
    report["python_bridge_contract"] = {
        "official_output_bridge_source": "https://github.com/apple/coremltools/blob/main/coremlpython/CoreMLPythonUtils.mm",
        "function": "Utils::convertArrayValueToPython",
        "installed_compiled_model_python": str(Path(ct.__file__).parent / "models/_compiled_model.py"),
        "installed_proxy_sha256": sha256_file(Path(ct.__file__).parent / "libcoremlpython.so"),
        "verification": "model description and MIL FLOAT16; NumPy FLOAT16 or exactly reversible FLOAT32 promotion",
        "timing_includes": "predict input FP16-to-FP32 preprocessing and output MLMultiArray-to-NumPy bridge"}
    report["models"] = {}
    cases = fixtures(args.rows, args.hidden, args.width, args.rank)
    models, specs, observed = {}, {}, {}
    try:
        for precision in ("fp16", "qdq_int8"):
            spec = specification(args.rows, args.hidden, args.width, precision, args.group, args.rank)
            specs[precision] = spec
            path = args.scratch / precision
            row = report["models"][precision] = {"input_bytes": spec["input_bytes"], "cases": []}
            started = time.perf_counter()
            export(path, spec, program_factory=make_program, compute_precision=ct.precision.FLOAT32)
            row["export_compile_seconds"] = time.perf_counter() - started
            proto = ct.models.MLModel(str(path / "graph.mlpackage"), skip_model_load=True).get_spec()
            (args.output / f"{precision}-converted-mil.txt").write_text(MessageToString(proto.mlProgram))
            function = proto.mlProgram.functions["main"]
            operations = list(function.block_specializations.values())[0].operations
            counts = collections.Counter(op.type for op in operations)
            row["converted_op_counts"] = dict(counts)
            row["converted_outputs"] = [{"operator": op.type, "outputs": [
                {"name": value.name, "dtype": MIL_pb2.DataType.Name(value.type.tensorType.dataType)}
                for value in op.outputs]} for op in operations if op.type != "const"]
            row["model_description_outputs"] = {
                feature.name: {"feature_type": feature.type.WhichOneof("Type"),
                               "dtype": FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name(feature.type.multiArrayType.dataType),
                               "shape": list(feature.type.multiArrayType.shape)}
                for feature in proto.description.output}
            row["model_description_inputs"] = {
                feature.name: {"feature_type": feature.type.WhichOneof("Type"),
                               "dtype": FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name(feature.type.multiArrayType.dataType),
                               "shape": list(feature.type.multiArrayType.shape)}
                for feature in proto.description.input}
            outputs = {value.name: MIL_pb2.DataType.Name(value.type.tensorType.dataType)
                       for op in operations for value in op.outputs}
            row["mil_interface_outputs"] = {name: outputs.get(name) for name in spec["outputs"]}
            if counts["matmul"] != 5 or counts["quantize"] != (5 if precision == "qdq_int8" else 0):
                raise ValueError("converted graph lost a base or down-LoRA projection/QDQ boundary")
            if counts["dequantize"] != counts["quantize"] or {v.name for v in function.inputs} != set(spec["inputs"]):
                raise ValueError("converted runtime input/QDQ contract changed")
            plan = MLComputePlan.load_from_path(str(path / "graph.mlmodelc"), compute_units=ct.ComputeUnit.CPU_AND_NE)
            row["planned_devices"] = inspect_plan(plan)
            del plan
            started = time.perf_counter()
            models[precision] = ct.models.CompiledMLModel(str(path / "graph.mlmodelc"), compute_units=ct.ComputeUnit.CPU_AND_NE)
            row["load_seconds"] = time.perf_counter() - started
            prepared = prepare_case(cases[0], spec)
            row["first_prediction_input_bridge"] = {
                "numpy_dtypes_before_predict": {name: str(value.dtype) for name, value in prepared.items()}}
            started = time.perf_counter()
            actual = models[precision].predict(prepared)
            row["first_prediction_ms"] = (time.perf_counter() - started) * 1000
            row["first_prediction_input_bridge"]["numpy_dtypes_after_predict"] = {
                name: str(value.dtype) for name, value in prepared.items()}
            verify_prediction_boundary(actual, row, spec, "first_prediction")
            observed[precision] = {}
            for case in cases:
                prepared = prepare_case(case, spec)
                cpu = reference(prepared, spec)
                actual = models[precision].predict(prepared)
                verify_prediction_boundary(actual, row, spec, case[0])
                oracle = unquantized_oracle(case)
                result = {"case": case[0], "vs_cpu_graph_reference": {}, "vs_unquantized_math": {}}
                for name in ("y", "h"):
                    result["vs_cpu_graph_reference"][name] = errors(actual[name], cpu[name])
                    result["vs_unquantized_math"][name] = errors(actual[name], oracle[name])
                    if result["vs_cpu_graph_reference"][name]["relative_l2"] > .025:
                        raise ValueError(f"{precision}/{case[0]}/{name} differs from CPU graph contract")
                    if case[0] == "zero" and np.any(actual[name] != 0):
                        raise ValueError("zero input/adapter produced nonzero output")
                observed[precision][case[0]] = actual
                row["cases"].append(result)
                write_report(args.output, report)
            row["A_after_B_exact"] = all(np.array_equal(observed[precision]["A_lora"][name],
                                                         observed[precision]["A_lora_again"][name]) for name in ("y", "h"))
            if not row["A_after_B_exact"]:
                raise ValueError("runtime input rebinding did not restore A after B")
        samples = {precision: [] for precision in models}
        # One small ABBA sequence; diagnostic samples, not a stability claim.
        for precision in ("fp16", "qdq_int8", "qdq_int8", "fp16"):
            started = time.perf_counter()
            prepared = prepare_case(cases[0], specs[precision])
            prepared_at = time.perf_counter()
            actual = models[precision].predict(prepared)
            predicted_at = time.perf_counter()
            verify_prediction_boundary(actual, report["models"][precision], specs[precision],
                                       f"warm_sample_{len(samples[precision]) + 1}")
            samples[precision].append({"prepare_ms": (prepared_at - started) * 1000,
                                       "prediction_ms": (predicted_at - prepared_at) * 1000,
                                       "total_ms": (predicted_at - started) * 1000})
            for name in ("h", "y"):
                errors(actual[name], observed[precision]["A_lora"][name])
        for precision in models:
            report["models"][precision]["samples"] = samples[precision]
            report["models"][precision]["median_ms"] = {
                name: statistics.median(sample[name] for sample in samples[precision])
                for name in ("prepare_ms", "prediction_ms", "total_ms")}
        report["qdq_vs_fp16"] = [{"case": case[0], **{name: errors(observed["qdq_int8"][case[0]][name],
                                                                         observed["fp16"][case[0]][name])
                                                          for name in ("h", "y")}} for case in cases]
        report["diagnostic_latency_ratio_fp16_over_qdq"] = (
            report["models"]["fp16"]["median_ms"]["total_ms"] / report["models"]["qdq_int8"]["median_ms"]["total_ms"])
        report["worker_success"] = True
        write_report(args.output, report)
        return 0
    except BaseException as error:
        report["worker_success"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        write_report(args.output, report)
        raise
    finally:
        models.clear()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scratch", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    from runtime_ane_swiglu_int8_candidate import specification
    spec = specification(args.rows, args.hidden, args.width, "fp16", args.group, args.rank)
    if args.worker:
        if args.scratch is None:
            parser.error("worker requires its owned scratch directory")
        return worker(args)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    sources = [Path(__file__), ROOT / "tools/coreml/runtime_ane_swiglu_int8_candidate.py",
               ROOT / "tools/coreml/runtime_ane_int8_candidate.py", ROOT / "tools/coreml/export_runtime_ane.py"]
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}
    report = {"scope": "one synthetic geometry; dynamic three-projection SwiGLU and runtime LoRA; no model",
              "platform": platform.platform(), "shape": [args.rows, args.hidden, args.width],
              "hadamard_group": args.group, "lora_rank": args.rank, "input_bytes_per_graph": spec["input_bytes"],
              "timing_scope": "CPU gate/up low-rank + all input rotation/staging + full graph prediction; 2 warmed samples per variant",
              "placement_scope": "CPU_AND_NE policy/compute-plan preferences; no hardware residency trace",
              "fp32_bridge_scope": "nonlinearity, hidden FWHT/scale, restore, down low-rank; original FP16 h output included",
              "hardware_int8_verified": False, "production_speedup_verified": False,
              "source_sha256": source_hashes, "success": False}
    write_report(output, report)
    started = time.monotonic()
    fixture = tempfile.TemporaryDirectory(prefix="tc-swiglu-int8-")
    scratch = Path(fixture.name).resolve()
    private_tmp = scratch / "private-tmp"
    private_tmp.mkdir(mode=0o700)
    environment = benchmark_environment()
    environment["TMPDIR"] = str(private_tmp) + "/"
    command = [sys.executable, "-B", Path(__file__).resolve(), "--worker", "--output", output,
               "--scratch", scratch, "--rows", args.rows, "--hidden", args.hidden,
               "--width", args.width, "--group", args.group, "--rank", args.rank]
    report["temporary_fixture"] = str(scratch)
    report["command"] = [str(value) for value in command]
    write_report(output, report)
    try:
        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
            result = run_owned(report["command"], cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, timeout=240)
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
            report["explicit_artifacts_removed"] = not scratch.exists()
        except BaseException as error:
            report["explicit_artifacts_removed"] = False
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
            report["success"] = False
        try:
            report["source_sha256_after"] = {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}
            report["sources_unchanged"] = report["source_sha256_after"] == source_hashes
        except BaseException as error:
            report["sources_unchanged"] = False
            report["source_hash_error"] = f"{type(error).__name__}: {error}"
        report["success"] = report["success"] and report["sources_unchanged"] and report["explicit_artifacts_removed"]
        report["total_seconds"] = time.monotonic() - started
        write_report(output, report)
    print("Evidence: " + str(output / "report.json"), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
