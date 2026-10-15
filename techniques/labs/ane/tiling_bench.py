#!/usr/bin/env python3
"""Public Core ML tiling vs compiled fused MLX, including the GPU consumer.

Runs ONE candidate per fresh output directory. Generic real FP16 fixture input
is optional; default inputs are deterministic synthetic data. All variants use
the same source arrays and independent sampled FP64 witnesses. Never downloads,
clears system caches, imports product code or changes production routing.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time

from compute_plan import inspect
from tiling_graphs import geometry, program


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def local(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing or escaping fixture asset")
    return path


def metrics(reference, candidate, np):
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("shape or nonfinite output failure")
    a, b = reference.astype(np.float64).ravel(), candidate.astype(np.float64).ravel()
    ea, eb = float(np.sum(a * a)), float(np.sum(b * b))
    error = float(np.sum((a - b) ** 2))
    return {"relative_l2": math.sqrt(error / ea) if ea else (0.0 if error == 0 else None),
            "cosine": float(np.sum(a * b) / math.sqrt(ea * eb)) if ea and eb else (1.0 if error == 0 else 0.0),
            "max_abs": float(np.max(np.abs(a - b))), "finite": True}


def witness(x, gu, down, np):
    rows = sorted({0, x.shape[0] // 3, 2 * x.shape[0] // 3, x.shape[0] - 1})
    cols = sorted({0, down.shape[0] // 3, 2 * down.shape[0] // 3, down.shape[0] - 1})
    projected = np.einsum("mk,nk->mn", x[rows].astype(np.float64), gu.astype(np.float64), optimize=False)
    gate, up = np.split(projected, 2, axis=1)
    # Stable SiLU; all intermediates are independent FP64 CPU arithmetic.
    sigmoid = np.exp(-np.abs(gate))
    sigmoid = np.where(gate >= 0, 1 / (1 + sigmoid), sigmoid / (1 + sigmoid))
    hidden = (gate * sigmoid) * up
    output = np.einsum("mk,nk->mn", hidden, down[cols].astype(np.float64), optimize=False)
    return rows, cols, output


def timed_system(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    return {"returncode": result.returncode, "stdout": result.stdout.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, help="explicit fixture.json from any conforming local exporter")
    parser.add_argument("--point", action="append", help="explicit point IDs, repeatable")
    parser.add_argument("--rows", type=int, default=33)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--tile-k", type=int, default=0)
    parser.add_argument("--tile-n", type=int, default=0)
    parser.add_argument("--runtime-weights", action="store_true")
    parser.add_argument("--layout", choices=("rows", "channels"), default="rows")
    parser.add_argument("--silu", choices=("sigmoid", "exp"), default="sigmoid")
    parser.add_argument("--compute-units", choices=("cpu-ne", "cpu"), default="cpu-ne")
    parser.add_argument("--route", choices=("coreml", "serial", "hybrid"), default="coreml")
    parser.add_argument("--baseline", choices=("gpu", "serial"), default="gpu",
                        help="serial baseline is only a matched overlap diagnostic for --route hybrid")
    parser.add_argument("--host-timeline", action="store_true", help="diagnostic host events only; no GPU/ANE timestamps")
    parser.add_argument("--ane-channels", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin" or args.repeats not in (4, 8, 12, 16, 24, 32):
        parser.error("requires macOS and 4..32 repeats in ABBA groups")
    if args.point and len(set(args.point)) != len(args.point):
        parser.error("point IDs must be unique")
    if args.baseline == "serial" and args.route != "hybrid":
        parser.error("serial baseline requires a matched hybrid candidate")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot_directory = output / "source"
    snapshot_directory.mkdir()
    source_snapshots = []
    for name in ("tiling_bench.py", "tiling_graphs.py", "compute_plan.py"):
        source = Path(__file__).with_name(name)
        destination = snapshot_directory / name
        shutil.copy2(source, destination)
        source_snapshots.append({"path": "source/" + name, "sha256": digest(destination)})
    report = {"schema": "tc-coreml-tiling-experiment-v1", "status": "starting", "synthetic": args.fixture is None,
              "production_qualified": False, "physical_ane_placement": "unverified", "phase": "setup",
              "runner_sha256": digest(Path(__file__)), "graphs_sha256": digest(Path(__file__).with_name("tiling_graphs.py")),
              "source_snapshots": source_snapshots, "samples": [], "points": [], "errors": [],
              "gates": {"fp16_source_relative_l2_max": .02, "original_reference_relative_l2_max": .03,
                        "original_reference_cosine_min": .999, "independent_witness_relative_l2_max": .02}}
    receipt = output / "report.json"

    def save():
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    mx = None
    try:
        with redirect_stdout(sys.stderr):
            import numpy as np
            import mlx.core as mx
            import coremltools as ct
        report["environment"] = {"chip": timed_system(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
                                 "ram_bytes": timed_system(["/usr/sbin/sysctl", "-n", "hw.memsize"]),
                                 "macos": platform.mac_ver()[0], "os_build": timed_system(["sw_vers", "-buildVersion"]),
                                 "sdk": timed_system(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
                                 "coremltools": ct.__version__, "numpy": np.__version__,
                                 "mlx": version("mlx"),
                                 "thermal_before": timed_system(["pmset", "-g", "therm"])}
        if args.fixture:
            root = args.fixture.resolve().parent
            fixture = json.loads(args.fixture.read_text())
            if fixture.get("schema") != "tc-real-ffn-fixture-v1" or fixture.get("loras") != []:
                raise ValueError("unsupported fixture schema or adapter contract")
            for name, expected in fixture["files"].items():
                if digest(local(root, name)) != expected:
                    raise ValueError("fixture asset changed: " + name)
            gu, down = [np.load(local(root, fixture[key]), allow_pickle=False) for key in ("gate_up", "down")]
            points = []
            for point in fixture["points"]:
                if args.point is None or point["id"] in args.point:
                    points.append({"id": point["id"], "input": np.load(local(root, point["input"]), allow_pickle=False),
                                   "original_reference": np.load(local(root, point["original_reference"]), allow_pickle=False)})
            if not points or (args.point and set(args.point) != {point["id"] for point in points}):
                raise ValueError("missing requested real fixture points")
            m, h, f = fixture["shape_mhf"]
            report["fixture_sha256"] = digest(args.fixture)
            report["original_reference_scope"] = fixture["original_reference_scope"]
        else:
            m, h, f = args.rows, args.hidden, args.width
            geometry(m, h, f, args.tile_k, args.tile_n)
            rng = np.random.default_rng(17)
            gu, down = [rng.normal(0, .2 / math.sqrt(size[1]), size).astype(np.float16)
                        for size in ((2 * f, h), (h, f))]
            points = [{"id": "synthetic-17", "input": rng.normal(0, .3, (m, h)).astype(np.float16),
                       "original_reference": None}]
        if gu.shape != (2 * f, h) or down.shape != (h, f) or any(value.dtype != np.float16 or not np.isfinite(value).all() for value in (gu, down)):
            raise ValueError("finite FP16 paired full weights required")
        for point in points:
            if point["input"].shape != (m, h) or point["input"].dtype != np.float16 or not np.isfinite(point["input"]).all():
                raise ValueError("complete finite FP16 input required; provenance must separately establish real rows")
        prefix = f if args.route == "coreml" else args.ane_channels
        if not 0 < prefix <= f or (args.route != "coreml" and prefix == f):
            raise ValueError("serial/hybrid require a proper interior channel split")
        spec = geometry(m, h, prefix, args.tile_k, args.tile_n)
        geometry(m, h, f, 0, 0)
        # Conversion can hold several weight-file copies. This is a conservative
        # admission policy, not a measured bound on the compiler's workspace.
        free = shutil.disk_usage(output).free
        required = 512 * 1024 * 1024 + (0 if args.runtime_weights else 4 * 3 * h * prefix * 2)
        report["disk_admission"] = {"available_bytes": free, "required_bytes": required,
                                    "source_kind": "derived", "hard_bound": False}
        if free < required:
            raise ValueError("insufficient disk headroom for bounded compiler workspace; no unrelated data removed")
        report.update(shape_mhf=[m, h, f], coreml_geometry=spec, layout=args.layout, route=args.route,
                      baseline=args.baseline,
                      diagnostic_host_timeline=args.host_timeline,
                      prefix_channels=prefix, runtime_weights=args.runtime_weights, recorded_order="ABBA",
                      silu_lowering=args.silu,
                      compute_units_policy="CPU_AND_NE" if args.compute_units == "cpu-ne" else "CPU_ONLY",
                      cache_state="fresh process; system/driver caches not reset",
                      boundary="ready resident MLX FP16 input and resident weights -> actual output restored/joined and mx.eval complete",
                      excluded="conversion/load, initial weight preparation, oracle and output readback; no streaming-weight claim")
        report["weight_sha256"] = hashlib.sha256(gu.tobytes() + down.tobytes()).hexdigest()
        gpu_gu, gpu_down = mx.array(gu), mx.array(down)
        mx.eval(gpu_gu, gpu_down)

        def ffn(x, wgu, wd):
            gate, up = mx.split(x @ wgu.T, 2, axis=-1)
            return ((gate * mx.sigmoid(gate)) * up) @ wd.T

        full_gpu = mx.compile(lambda x: ffn(x, gpu_gu, gpu_down))
        ref_gu, ref_down = gpu_gu.astype(mx.float32), gpu_down.astype(mx.float32)
        mx.eval(ref_gu, ref_down)
        fp32_reference = mx.compile(lambda x: ffn(x.astype(mx.float32), ref_gu, ref_down))
        for point in points:
            x = point["input"]
            reference = fp32_reference(mx.array(x))
            mx.eval(reference)
            point["reference"] = np.array(reference)
            point["witness_rows"], point["witness_cols"], point["witness"] = witness(x, gu, down, np)
            metric = metrics(point["witness"], point["reference"][np.ix_(point["witness_rows"], point["witness_cols"])], np)
            if metric["relative_l2"] is None or metric["relative_l2"] > 1e-4:
                raise ValueError("independent CPU witness rejected FP32 reference")
            report["points"].append({"id": point["id"], "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                                     "independent_fp64_witness_rows": point["witness_rows"],
                                     "independent_fp64_witness_cols": point["witness_cols"], "fp32_reference_vs_witness": metric})
        del ref_gu, ref_down, fp32_reference
        prefix_gu = np.ascontiguousarray(np.concatenate((gu[:prefix], gu[f:f + prefix]), axis=0))
        prefix_down = np.ascontiguousarray(down[:, :prefix])
        if args.route != "coreml":
            suffix_gu = mx.array(np.ascontiguousarray(np.concatenate((gu[prefix:f], gu[f + prefix:]), axis=0)))
            suffix_down = mx.array(np.ascontiguousarray(down[:, prefix:]))
            mx.eval(suffix_gu, suffix_down)
            suffix_gpu = mx.compile(lambda x: ffn(x, suffix_gu, suffix_down))
        report["phase"] = "convert"
        save()
        started = time.perf_counter()
        with redirect_stdout(sys.stderr):
            graph = program(ct, np, prefix_gu, prefix_down, m, args.tile_k, args.tile_n, args.runtime_weights, args.layout, args.silu)
            model = ct.convert(graph, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                               compute_precision=ct.precision.FLOAT16, skip_model_load=True)
        report["conversion_host_seconds"] = time.perf_counter() - started
        package, compiled = output / "graph.mlpackage", output / "graph.mlmodelc"
        model.save(str(package))
        report["phase"] = "compile"
        save()
        started = time.perf_counter()
        with redirect_stdout(sys.stderr):
            ct.models.utils.compile_model(str(package), destination_path=str(compiled))
        report["compile_host_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        units = ct.ComputeUnit.CPU_AND_NE if args.compute_units == "cpu-ne" else ct.ComputeUnit.CPU_ONLY
        loaded = ct.models.CompiledMLModel(str(compiled), compute_units=units)
        report["load_host_seconds"] = time.perf_counter() - started
        try:
            plan = inspect(compiled, units)
            (output / "compute-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
            report["compute_plan"] = {"status": "predictions_available", "operations": plan["operation_count"],
                                      "preferred_device_counts": {name: sum(str(op["preferred_device"]) == name for op in plan["operations"])
                                                                  for name in sorted({str(op["preferred_device"]) for op in plan["operations"]})}}
        except Exception as error:
            report["compute_plan"] = {"status": "unavailable", "error": str(error)}
        report["plan_loaded_before_first_prediction"] = True
        report["phase"] = "predict"
        save()
        runtime_weights = {"weight_gu": prefix_gu, "weight_down": prefix_down}
        if args.layout == "channels":
            runtime_weights = {key: value[None, None] for key, value in runtime_weights.items()}
        with ThreadPoolExecutor(max_workers=1) as worker:
            for trial in range(-4, args.repeats):
                candidate = (trial + 4) % 4 in (1, 2)
                point = points[((trial + 4) // 2) % len(points)]
                device_x = mx.array(point["input"])
                mx.eval(device_x)
                start = time.perf_counter()
                events = []

                def event(name, stamp):
                    if not args.host_timeline:
                        return
                    events.append({"name": name, "start_ns": round((stamp - start) * 1e9),
                                   "duration_ns": round((time.perf_counter() - stamp) * 1e9),
                                   "source_kind": "measured", "clock_domain": "this-process-perf-counter-host-only"})

                prediction, staging, restore = None, 0., 0.
                if candidate or args.baseline == "serial":
                    host_x = np.array(device_x)
                    if args.layout == "channels":
                        host_x = np.ascontiguousarray(host_x.T)[None, None]
                    features = {"x": host_x, **(runtime_weights if args.runtime_weights else {})}
                    staging = time.perf_counter() - start
                    event("host-input-staging", start)

                    def predict():
                        stamp = time.perf_counter()
                        result = loaded.predict(features)
                        actual = np.array(result["y"], copy=True)
                        event("coreml-predict-host-call-and-output-copy", stamp)
                        return actual, time.perf_counter() - stamp

                    selected_route = args.route if candidate else "serial"
                    if selected_route == "hybrid":
                        future = worker.submit(predict)
                        stamp = time.perf_counter()
                        tail = suffix_gpu(device_x)
                        mx.async_eval(tail)
                        event("mlx-suffix-host-submit", stamp)
                        host_y, prediction = future.result()
                    else:
                        host_y, prediction = predict()
                        tail = suffix_gpu(device_x) if selected_route == "serial" else None
                    stamp = time.perf_counter()
                    if args.layout == "channels":
                        host_y = np.ascontiguousarray(host_y.reshape(h, m).T)
                    head = mx.array(host_y)
                    actual = head if tail is None else head + tail
                    mx.eval(actual)
                    restore = time.perf_counter() - stamp
                    event("restore-join-host-until-output-ready", stamp)
                else:
                    actual = full_gpu(device_x)
                    mx.eval(actual)
                    event("full-ffn-host-until-output-ready", start)
                wall = time.perf_counter() - start
                actual = np.array(actual)
                metric = metrics(point["reference"], actual, np)
                sampled = metrics(point["witness"], actual[np.ix_(point["witness_rows"], point["witness_cols"])], np)
                original = None if point["original_reference"] is None else metrics(point["original_reference"], actual, np)
                passed = (metric["relative_l2"] is not None and metric["relative_l2"] <= .02
                          and sampled["relative_l2"] is not None and sampled["relative_l2"] <= .02
                          and (original is None or (original["relative_l2"] is not None
                               and original["relative_l2"] <= .03 and original["cosine"] >= .999)))
                sample = {"trial": trial, "warmup": trial < 0, "pair": trial // 2, "point": point["id"],
                          "trial_start_host_seconds": start,
                          "variant": "candidate" if candidate else args.baseline, "wall_seconds": wall,
                          "host_events": sorted(events, key=lambda item: item["start_ns"]),
                          "input_staging_host_seconds": staging, "prediction_host_seconds": prediction,
                          "post_prediction_restore_join_host_seconds": restore,
                          "vs_fp32_same_fp16_source": metric, "vs_independent_fp64_witness": sampled,
                          "vs_original_bf16_replica": original, "component_numeric_pass": passed}
                report["samples"].append(sample)
                save()
                print(f'trial={trial} arm={sample["variant"]} wall={wall:.6f}s numeric={passed}', file=sys.stderr, flush=True)
                if not passed:
                    raise ValueError("fixed numerical gate failed; timing samples retained but speedup withheld")
        values = {name: [sample["wall_seconds"] for sample in report["samples"]
                         if not sample["warmup"] and sample["variant"] == name] for name in (args.baseline, "candidate")}
        report["warm_median_seconds"] = {name: statistics.median(samples) for name, samples in values.items()}
        report["warm_paired_speedups"] = [next(sample["wall_seconds"] for sample in report["samples"] if not sample["warmup"] and sample["pair"] == pair and sample["variant"] == args.baseline) /
                                          next(sample["wall_seconds"] for sample in report["samples"] if not sample["warmup"] and sample["pair"] == pair and sample["variant"] == "candidate") for pair in range(args.repeats // 2)]
        report["median_paired_speedup"] = statistics.median(report["warm_paired_speedups"])
        report["status"] = "component_numeric_pass"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"phase": report["phase"], "type": type(error).__name__, "message": str(error)})
    finally:
        if mx is not None:
            mx.synchronize()
        if "environment" in report:
            report["environment"]["thermal_after"] = timed_system(["pmset", "-g", "therm"])
        if any(digest(Path(__file__).with_name(Path(item["path"]).name)) != item["sha256"] for item in source_snapshots):
            report["status"] = "failed"
            report["errors"].append({"phase": "source-identity", "type": "ValueError", "message": "source changed during experiment"})
            for key in ("warm_median_seconds", "warm_paired_speedups", "median_paired_speedup"):
                report.pop(key, None)
        report["process_peak_rss_bytes"] = {"value": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                            "source_kind": "measured", "scope": "macOS process high-water including compile/oracles, not request peak or hard bound"}
        report["phase"] = "terminal"
        save()
    print(json.dumps({key: value for key, value in report.items() if key not in ("samples", "points")}, indent=2))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
