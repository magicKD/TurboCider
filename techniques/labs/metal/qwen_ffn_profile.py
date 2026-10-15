#!/usr/bin/env python3
"""Explicit real BF16 Qwen FFN host profiling; no device timestamps invented.

Uses existing source/capture identity readers. Original compiled output-ready
samples and deliberately staged, fully-drained host spans are separate records.
The latter changes execution boundaries and cannot be used as formal speedup.
No capture bundle, weights, arrays, viewer, downloads or product edits emitted.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import shutil
import time

from quantized_bench import compare, local, witness
from real_bf16_source import load as load_original_bf16


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def validate_profile_events(events):
    if [event.get("name") for event in events] != ["gate-up-projection", "silu-times-up", "down-projection"]:
        raise ValueError("complete ordered FFN diagnostic phases required")
    end = 0
    for event in events:
        if (event.get("source_kind") != "measured" or event.get("clock_domain") != "this-process-perf-counter-host-only"
                or type(event.get("start_ns")) is not int or type(event.get("duration_ns")) is not int
                or event["start_ns"] < end or event["duration_ns"] <= 0):
            raise ValueError("nonmonotonic or non-host diagnostic event")
        end = event["start_ns"] + event["duration_ns"]
    return True


def numeric_pass(replica, fp64_sample):
    return all(metric.get("finite") is True and type(metric.get("relative_l2")) in (int,float)
               and math.isfinite(metric["relative_l2"]) and 0 <= metric["relative_l2"] <= .03
               for metric in (replica, fp64_sample))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--point", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin" or not 1 <= len(args.point) <= 3 or len(set(args.point)) != len(args.point):
        parser.error("macOS and 1..3 distinct explicit real point IDs required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parent
    names = ("qwen_ffn_profile.py", "quantized_bench.py", "real_bf16_source.py", "affine_gemm.py", "affine_gemm.metal")
    (output / "source").mkdir()
    snapshots = []
    for name in names:
        shutil.copy2(source_root / name, output / "source" / name)
        snapshots.append({"path": "source/" + name, "sha256": digest(source_root / name)})
    report = {"schema": "tc-qwen-bf16-host-profile-v1", "status": "starting", "synthetic": False,
              "production_qualified": False, "request_quality_qualified": False, "physical_device_timestamps": None,
              "native_counter_calibration_applied": False, "loras": [], "source_snapshots": snapshots,
              "fixture_sha256": digest(args.fixture), "formal_samples": [], "diagnostic_samples": [], "errors": [],
              "gates": {"replica_relative_l2_max": .03, "fp64_sample_relative_l2_max": .03},
              "recorded_at_singapore": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + 8 * 3600)) + "+08:00"}

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    mx = None
    before = args.source_checkpoint.stat()
    save()
    try:
        if shutil.disk_usage(output).free < 512 * 1024 * 1024:
            raise ValueError("insufficient disk headroom; no other data removed")
        import numpy as np
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        info = mx.device_info()
        report["environment"] = {"macos": platform.mac_ver()[0], "python": platform.python_version(),
            "mlx": version("mlx"), "numpy": np.__version__,
            "device": {key: info.get(key) for key in ("device_name", "architecture", "memory_size")},
            "library_assets": [{"name": path.name, "sha256": digest(path)} for path in
                [Path(mx.__file__), *sorted((Path(mx.__file__).parent / "lib").glob("*"))]
                if path.is_file() and path.suffix in (".so", ".dylib", ".metallib")]}
        fixture = json.loads(args.fixture.read_text())
        if fixture.get("schema") != "tc-real-ffn-fixture-v1" or fixture.get("loras") != []:
            raise ValueError("original base FFN fixture required")
        m, h, f = fixture["shape_mhf"]
        if not (1 <= m <= 2048 and 1 <= h <= 4096 and 1 <= f <= 12288):
            raise ValueError("bounded full original FFN shape required")
        root = args.fixture.resolve().parent
        for name, expected in fixture["files"].items():
            if digest(local(root, name)) != expected:
                raise ValueError("fixture asset identity differs")
        points = [{"id": point["id"], "x": np.load(local(root, point["input"]), allow_pickle=False),
                   "replica": np.load(local(root, point["original_reference"]), allow_pickle=False)}
                  for point in fixture["points"] if point["id"] in args.point]
        if {point["id"] for point in points} != set(args.point):
            raise ValueError("missing requested original point")
        source = load_original_bf16(args.source_checkpoint, args.capture_root, fixture, points, np)
        report.update(shape_mhf=[m,h,f], source_layer=fixture["source_layer"], source_dtype="bfloat16",
                      consumer_abi=[1,m,h], capture_receipt_sha256=fixture["capture_receipt_sha256"],
                      original_tensor_sha256={key: fixture[key] for key in
                          ("original_gate_up_tensor_sha256", "original_down_tensor_sha256")})
        matrices = [mx.array(value, dtype=mx.bfloat16) for value in source]
        mx.eval(matrices)

        def dense(x):
            gate, up = mx.split(x @ matrices[0].T, 2, axis=-1)
            return ((gate * mx.sigmoid(gate)) * up) @ matrices[1].T

        compiled = mx.compile(dense)
        for point in points:
            x = mx.array(point["x_bf16_source"].reshape(1,m,h), dtype=mx.bfloat16)
            mx.eval(x)
            rows, columns, fp64 = witness(point["x_bf16_source"], source, "ffn", np)

            def metrics(value):
                actual = np.array(value.astype(mx.float32)).reshape(m,h)
                if list(value.shape) != [1,m,h] or value.dtype != mx.bfloat16:
                    raise ValueError("original BF16 consumer ABI changed")
                replica = compare(point["replica"], actual, np)
                sampled = compare(fp64, actual[np.ix_(rows, columns)], np)
                return {"replica": replica, "fp64_sample": sampled,
                        "component_numeric_pass": numeric_pass(replica, sampled)}

            for trial in range(-2,4):
                started = time.perf_counter_ns()
                result = compiled(x)
                mx.eval(result)
                duration = time.perf_counter_ns() - started
                report["formal_samples"].append({"point": point["id"], "trial": trial, "warmup": trial < 0,
                    "duration_ns": duration, "source_kind": "measured", "formal_performance_score": True,
                    "scope": "same original compiled BF16 graph, host-until-output-ready; not device kernel time", **metrics(result)})
                save()
            for trial in range(2):
                mx.synchronize()
                origin = time.perf_counter_ns()
                events = []

                def phase(name, function):
                    start = time.perf_counter_ns()
                    value = function()
                    mx.eval(value)
                    mx.synchronize()
                    end = time.perf_counter_ns()
                    events.append({"name": name, "start_ns": start-origin, "duration_ns": end-start,
                        "source_kind": "measured", "clock_domain": "this-process-perf-counter-host-only",
                        "completion": "mx.eval plus mx.synchronize; changes program boundaries"})
                    return value

                gu = phase("gate-up-projection", lambda: x @ matrices[0].T)

                def epilogue():
                    gate, up = mx.split(gu,2,axis=-1)
                    return (gate * mx.sigmoid(gate)) * up

                hidden = phase("silu-times-up", epilogue)
                result = phase("down-projection", lambda: hidden @ matrices[1].T)
                elapsed = time.perf_counter_ns() - origin
                validate_profile_events(events)
                report["diagnostic_samples"].append({"point": point["id"], "trial": trial,
                    "formal_performance_score": False, "scope": "deliberately staged full FFN; CPU host spans only",
                    "host_events": events, "host_boundary_ns": elapsed, **metrics(result)})
                save()
        after = args.source_checkpoint.stat()
        report["checkpoint_stat_unchanged"] = all(getattr(before,key)==getattr(after,key) for key in
                                                  ("st_size","st_mtime_ns","st_ctime_ns","st_ino","st_dev"))
        report["source_inputs_unchanged"] = all(digest(source_root / Path(item["path"]).name)==item["sha256"] for item in snapshots)
        if not report["checkpoint_stat_unchanged"] or not report["source_inputs_unchanged"]:
            raise ValueError("checkpoint or source changed during profile")
        report["status"] = "component-numeric-pass" if all(row["component_numeric_pass"] for row in
            report["formal_samples"]+report["diagnostic_samples"]) else "component-numeric-failed"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        if mx is not None:
            try:
                mx.synchronize()
            except Exception as error:
                report["status"] = "failed"
                report["errors"].append({"type": type(error).__name__, "message": str(error)})
        save()
    print(json.dumps({"status": report["status"], "points": args.point, "production_qualified": False}))
    return int(report["status"] != "component-numeric-pass")


if __name__ == "__main__":
    raise SystemExit(main())
