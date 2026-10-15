#!/usr/bin/env python3
"""Explicit real-Qwen full BF16 GPU vs private ANE prefix serial/hybrid replay.

The deliberately copy-based bridge charges GPU->host FP16, Model feed/read/cast,
restore and GPU join. This is not shared-IOSurface or zero-copy qualification.
Only an explicitly prepared pinned upstream copy and existing local sources.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs/metal"))
sys.path.insert(0, str(ROOT / "labs"))
sys.path.insert(0, str(ROOT / "labs/ane"))
from real_bf16_source import load as load_original
from silu_domain import channel_slices
from qwen_witness import selected as witness

UPSTREAM_FILES = ("__init__.py", "graph.py", "_compile.py", "_runtime.py", "_rewrite.py", "_blob.py", "_targets.py", "_circuit.py",
                  "_lib/ane_e5rt_dispatch.mm")
BOUNDARY = "prepared original BF16 GPU input to BF16 GPU output with explicit MLX drain; conversion/feed/restore/join included"


class NumericalRejection(ValueError):
    pass


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            value.update(data)
    return value.hexdigest()


def local(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("explicit bounded local asset required")
    return path


def metric(reference, value, np):
    if reference.shape != value.shape or not np.isfinite(value).all():
        return {"finite": False, "relative_l2": None, "cosine": None, "max_abs": None}
    a, b = reference.astype(np.float64).ravel(), value.astype(np.float64).ravel()
    energy, other = float(np.sum(a * a)), float(np.sum(b * b))
    delta = a - b
    squared = float(np.sum(delta * delta))
    return {"finite": True, "relative_l2": (squared / energy)**.5 if energy else (0. if squared == 0 else None),
            "cosine": float(np.sum(a * b) / math.sqrt(energy * other)) if energy and other else (1. if squared == 0 else 0.),
            "max_abs": float(np.max(np.abs(delta)))}


def passed(value):
    error, cosine = value.get("relative_l2"), value.get("cosine")
    return (value.get("finite") is True and type(error) in (int, float) and math.isfinite(error) and 0 <= error <= .03
            and type(cosine) in (int, float) and math.isfinite(cosine) and cosine >= .999)


def ratios(samples, baseline, candidate, repeats=6):
    rows = [row for row in samples if row["phase"] == "measured"]
    groups = {}
    for row in rows:
        groups.setdefault(row["trial"], {})[row["arm"]] = row
    if set(groups) != set(range(repeats)) or any(set(g) != {"gpu", "serial", "hybrid"} for g in groups.values()):
        raise ValueError("complete balanced measured groups required")
    values = [group[baseline]["wall_ns"] / group[candidate]["wall_ns"] for group in groups.values()]
    rng = random.Random(73)
    boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(2000))
    return {"baseline": baseline, "candidate": candidate, "paired_ratios": values,
            "median_paired_host_ratio": statistics.median(values), "bootstrap_95pct": [boot[49], boot[1949]],
            "production_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--point", required=True)
    parser.add_argument("--share", type=int, choices=(256, 512, 1024, 2048), default=512)
    parser.add_argument("--repeats", type=int, choices=(6, 24), default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    source = args.source_root.resolve(); package = source / "aneforge"
    own_sources = [Path(__file__), ROOT / "labs/silu_domain.py", ROOT / "labs/metal/real_bf16_source.py",
                   ROOT / "labs/ane/qwen_witness.py", ROOT / "labs/ane/precision_oracle.py"]
    report = {"schema": "tc-aneforge-qwen-ffn-v1", "status": "starting", "production_qualified": False,
              "point": args.point, "share": args.share, "boundary": BOUNDARY, "samples": [], "comparisons": None,
              "physical_ane_placement": "not independently observed", "zero_copy": False,
              "gates": {"relative_l2_max": .03, "cosine_min": .999}, "errors": [],
              "source_snapshots": [], "upstream_sha256": {name: sha(package / name) for name in UPSTREAM_FILES},
              "fixture_sha256": sha(args.fixture), "checkpoint_stat_before": {}, "setup": {}}
    report["measured_groups"] = args.repeats
    report["pre_arm_validation_drain"] = True
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save()
    mx, net, pool = None, None, None
    library_files = []
    try:
        if (platform.system() != "Darwin" or not source.is_relative_to((ROOT.parent / "outputs").resolve())
                or shutil.disk_usage(output).free < 512 * 1024**2):
            raise ValueError("macOS, explicit ignored upstream copy and 512 MiB disk headroom required")
        dylib = package / "_lib/libane_e5rt_dispatch.dylib"
        if not dylib.is_file():
            raise ValueError("explicit previously built shim required; no auto-build")
        for key in list(os.environ):
            if key.startswith(("ANEFORGE_", "ANE_", "DYLD_", "TURBOCIDER_", "MPSGRAPH_")):
                del os.environ[key]
        os.environ.update(ANEFORGE_DYLIB=str(dylib), ANEFORGE_NO_AUTOBUILD="1", ANEFORGE_CACHE_DIR=str(output / "cache"),
                          ANEFORGE_CACHE=str(output / "build-cache"), ANEFORGE_COMPILE_BREAKER_STRICT="1")
        sys.path.insert(0, str(source))
        import aneforge as af
        import mlx.core as mx
        import numpy as np
        if Path(af.__file__).resolve().parent != package:
            raise ValueError("unexpected upstream import")
        mx.set_default_device(mx.gpu)
        (output / "source").mkdir()
        for path in own_sources:
            shutil.copy2(path, output / "source" / path.name)
            report["source_snapshots"].append({"path": "source/" + path.name, "sha256": sha(path)})
        report.update(shim_sha256=sha(dylib), macos=platform.mac_ver()[0], python=platform.python_version(),
                      numpy=np.__version__, mlx=version("mlx"),
                      device={key: mx.device_info().get(key) for key in ("device_name", "architecture", "memory_size")})
        library_files = [Path(mx.__file__), *sorted((Path(mx.__file__).parent / "lib").glob("*"))]
        library_files = [path for path in library_files if path.is_file() and path.suffix in (".so", ".dylib", ".metallib")]
        report["mlx_library_assets"] = [{"name": path.name, "sha256": sha(path)} for path in library_files]
        before = args.checkpoint.stat()
        report["checkpoint_stat_before"] = {key: getattr(before, key) for key in ("st_size", "st_mtime_ns", "st_ctime_ns")}
        fixture = json.loads(args.fixture.read_text()); root = args.fixture.resolve().parent
        if fixture.get("schema") != "tc-real-ffn-fixture-v1" or fixture.get("loras") != [] or fixture.get("shape_mhf") != [1024, 4096, 12288]:
            raise ValueError("complete original Qwen base-layer fixture required")
        for name, expected in fixture["files"].items():
            if sha(local(root, name)) != expected:
                raise ValueError("fixture content differs")
        points = [point for point in fixture["points"] if point["id"] == args.point]
        if len(points) != 1:
            raise ValueError("unique explicit real point required")
        info = points[0]
        point = {"id": args.point, "x": np.load(local(root, info["input"]), allow_pickle=False, mmap_mode="r")}
        gu, down = load_original(args.checkpoint, args.capture_root, fixture, [point], np)
        original = np.load(local(root, info["original_reference"]), allow_pickle=False, mmap_mode="r").reshape(1024, 4096)
        x = point["x_bf16_source"]
        m, h, f = fixture["shape_mhf"]; share = args.share
        slicing = channel_slices(f, share)
        prefix_gu = np.concatenate((gu[:share], gu[f:f + share]), axis=0)
        prefix_down = np.ascontiguousarray(down[:, :share])
        native_gu, native_down = prefix_gu.astype(np.float16), prefix_down.astype(np.float16)
        if not np.isfinite(native_gu).all() or not np.isfinite(native_down).all():
            raise ValueError("native source conversion overflow")
        report.update(shape_mhf=[m,h,f], channel_slices=slicing, source_layer=fixture["source_layer"], loras=[],
            original_weight_sha256={key: fixture[key] for key in ("original_gate_up_tensor_sha256", "original_down_tensor_sha256")},
            original_input_tensor_sha256=info["original_input_tensor_sha256"], original_capture_file_sha256=info["capture_file_sha256"],
            capture_receipt_sha256=fixture["capture_receipt_sha256"],
            native_weight_sha256=[hashlib.sha256(w.tobytes()).hexdigest() for w in (native_gu,native_down)],
            logical_private_weight_bytes=native_gu.nbytes + native_down.nbytes,
            logical_per_call_gpu_host_input_bytes=m*h*2, logical_per_call_host_gpu_partial_bytes=m*h*4)
        rows, cols, cpu, _ = witness(x, gu, down, np)
        prows, pcols, pcpu, ranges = witness(x, prefix_gu, prefix_down, np)
        report["independent_fp64_witness"] = {"rows": rows, "cols": cols, "prefix_rows": prows, "prefix_cols": pcols,
            "arithmetic": "independent full-K/F NumPy FP64 einsum and stable exp(-abs); selected rows/columns only", **ranges}
        native_x = af.input((m,h), max_abs=float(np.max(np.abs(point["x"]))))
        gate, up = native_x @ native_gu[:share].T, native_x @ native_gu[share:].T
        small = (gate.abs() * -1.).exp()
        selected = af.select(gate.greater_equal(gate * 0.), gate, gate * small)
        graph = ((selected / small.adds(1.)) * up) @ native_down.T
        now = time.perf_counter_ns(); net = af.compile(graph, opt=0, build_dir=output / "ane-model")
        report["setup"]["native_emit_compile_load_ns"] = time.perf_counter_ns() - now
        report["native_requested_device_mask"] = net._prog._device_mask
        if report["native_requested_device_mask"] != 4:
            raise ValueError("unexpected private device-mask policy")
        report["private_mil_sha256"] = sha(output / "ane-model/model.mil")
        report["private_weights_blob_sha256"] = sha(output / "ane-model/weights.bin")
        save()
        now = time.perf_counter_ns()
        wgu, wd = mx.array(gu, dtype=mx.bfloat16), mx.array(down, dtype=mx.bfloat16)
        wpg, wpd = mx.array(prefix_gu, dtype=mx.bfloat16), mx.array(prefix_down, dtype=mx.bfloat16)
        wrg = mx.contiguous(mx.concatenate((wgu[share:f], wgu[f+share:]), axis=0))
        wrd = mx.contiguous(wd[:,share:])
        mx.eval(wgu,wd,wpg,wpd,wrg,wrd); mx.synchronize()
        def formula(value, wg, wdown):
            g,u = mx.split(value @ wg.T, 2, axis=-1)
            return ((g * mx.sigmoid(g)) * u) @ wdown.T
        full = mx.compile(lambda value: formula(value,wgu,wd))
        suffix = mx.compile(lambda value: formula(value,wrg,wrd))
        prefix = mx.compile(lambda value: formula(value,wpg,wpd))
        reference_input = mx.array(x.reshape(1,m,h),dtype=mx.bfloat16)
        pref = prefix(reference_input); mx.eval(pref); mx.synchronize()
        prefix_reference = np.array(pref.astype(mx.float32)).reshape(m,h)
        gpu_gate, _ = mx.split(reference_input @ wpg.T, 2, axis=-1)
        bounds = [mx.min(gpu_gate), mx.max(gpu_gate), mx.sum(mx.abs(gpu_gate) > 11)]
        mx.eval(bounds); mx.synchronize()
        report["full_prefix_gpu_gate_domain"] = {"arithmetic": "original BF16 GPU projection, all actual rows and selected channels",
            "minimum": float(bounds[0].item()), "maximum": float(bounds[1].item()),
            "abs_over_11": int(bounds[2].item()), "elements": m * share,
            "not_a_proof_of_private_fp16_intermediate_values": True}
        report["setup"]["gpu_weights_and_prefix_reference_ns"] = time.perf_counter_ns() - now
        report["logical_gpu_weight_bytes"] = sum(w.nbytes for w in (wgu,wd,wpg,wpd,wrg,wrd))
        pool = ThreadPoolExecutor(max_workers=1)
        arms = ("gpu", "serial", "hybrid")
        orders = (arms, arms[::-1], (arms[1],arms[2],arms[0]),(arms[0],arms[2],arms[1]),(arms[2],arms[0],arms[1]),(arms[1],arms[0],arms[2]))
        for phase, count in (("first",1),("warmup",2),("measured",args.repeats)):
            for trial in range(count):
                # Same complete actual values, fresh resource identity, common ready producer outside timing.
                gpu_x = mx.array(x.reshape(1,m,h),dtype=mx.bfloat16); mx.eval(gpu_x); mx.synchronize()
                for arm in orders[trial % 6]:
                    # Validation casts/readback belong to the preceding diagnostic,
                    # not to the next arm's timed production work.
                    mx.synchronize()
                    events = []
                    started = time.perf_counter_ns()
                    def event(name, begin, end):
                        events.append({"name":name,"start_ns":begin-started,"duration_ns":end-begin,"clock_domain":"same-process-perf-counter-host"})
                    def private_call(values):
                        begin = time.perf_counter_ns(); result = net(values)
                        event("private-model-feed-execute-read-fp32",begin,time.perf_counter_ns())
                        return result
                    partial = None
                    if arm == "gpu":
                        result = full(gpu_x); mx.eval(result); mx.synchronize()
                    else:
                        begin=time.perf_counter_ns()
                        half = np.array(gpu_x.astype(mx.float16)).reshape(m,h)
                        event("gpu-fp16-cast-host-readback",begin,time.perf_counter_ns())
                        if arm == "hybrid":
                            future = pool.submit(private_call, half)
                        else:
                            partial = private_call(half)
                        begin=time.perf_counter_ns(); tail = suffix(gpu_x); mx.eval(tail); mx.synchronize()
                        event("gpu-suffix-output-ready",begin,time.perf_counter_ns())
                        if arm == "hybrid":
                            partial = future.result()
                        begin=time.perf_counter_ns()
                        result = (tail.astype(mx.float32) + mx.array(partial.reshape(1,m,h),dtype=mx.float32)).astype(mx.bfloat16)
                        mx.eval(result); mx.synchronize()
                        event("partial-gpu-upload-fp32-join-bf16-drain",begin,time.perf_counter_ns())
                    wall=time.perf_counter_ns()-started
                    actual=np.array(result.astype(mx.float32)).reshape(m,h)
                    metrics={"full_original":metric(original,actual,np), "independent_full_sample":metric(cpu,actual[np.ix_(rows,cols)],np)}
                    if partial is not None:
                        metrics.update(prefix_original=metric(prefix_reference,partial,np),
                                       independent_prefix_sample=metric(pcpu,partial[np.ix_(prows,pcols)],np))
                    qualified=all(passed(value) for value in metrics.values())
                    if arm == "gpu":
                        qualified = qualified and metrics["full_original"]["relative_l2"] == 0. and metrics["full_original"]["max_abs"] == 0.
                    report["samples"].append({"phase":phase,"trial":trial,"arm":arm,"wall_ns":wall,"point":args.point,
                        "output_dtype":"bfloat16","metrics":metrics,"numeric_pass":qualified,
                        "host_events":sorted(events,key=lambda item:item["start_ns"]),"physical_overlap_qualified":False})
                    save()
                    if not qualified:
                        raise NumericalRejection("fixed original/prefix/independent gate rejected this arm")
        report["comparisons"]=[ratios(report["samples"],a,b,args.repeats) for a,b in (("gpu","serial"),("gpu","hybrid"),("serial","hybrid"))]
        report["source_inputs_unchanged"]=(report["upstream_sha256"]=={name:sha(package/name) for name in UPSTREAM_FILES}
            and all(sha(path)==snapshot["sha256"] for path,snapshot in zip(own_sources,report["source_snapshots"])))
        report["mlx_library_assets_unchanged"] = report["mlx_library_assets"] == [{"name": path.name, "sha256": sha(path)} for path in library_files]
        report["checkpoint_stat_unchanged"]=all(getattr(args.checkpoint.stat(),key)==value for key,value in report["checkpoint_stat_before"].items())
        if not report["source_inputs_unchanged"] or not report["checkpoint_stat_unchanged"] or not report["mlx_library_assets_unchanged"]:
            raise ValueError("source generation changed")
        report["status"]="measured"; save()
    except NumericalRejection as error:
        report["status"]="numerical-rejected"; report["errors"].append({"type":type(error).__name__,"message":str(error)}); save()
    except (Exception,KeyboardInterrupt) as error:
        report["status"]="failed"; report["comparisons"]=None
        report["errors"].append({"type":type(error).__name__,"message":str(error)}); save()
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        if mx is not None:
            mx.synchronize()
        if net is not None:
            net.release()
        report["source_inputs_unchanged"]=(report["upstream_sha256"]=={name:sha(package/name) for name in UPSTREAM_FILES}
            and all(sha(path)==snapshot["sha256"] for path,snapshot in zip(own_sources,report["source_snapshots"])))
        if report["checkpoint_stat_before"]:
            report["checkpoint_stat_unchanged"]=all(getattr(args.checkpoint.stat(),key)==value for key,value in report["checkpoint_stat_before"].items())
        if "mlx_library_assets" in report:
            report["mlx_library_assets_unchanged"] = report["mlx_library_assets"] == [{"name": path.name, "sha256": sha(path)} for path in library_files]
        if (not report["source_inputs_unchanged"] or report.get("checkpoint_stat_unchanged") is False
                or report.get("mlx_library_assets_unchanged") is False):
            report["status"]="failed"; report["comparisons"]=None
            report["errors"].append({"type":"SourceChanged","message":"post-cleanup source generation check failed"})
        save()
    print(json.dumps({"status":report["status"],"errors":report["errors"]}))
    return 3 if report["status"]=="numerical-rejected" else int(report["status"]!="measured")


if __name__=="__main__":
    raise SystemExit(main())
