#!/usr/bin/env python3
"""Real base512 block0 Private QKV screen. Not a complete image benchmark."""
import argparse
from collections import Counter
import ctypes as C
import json
import math
import os
import shutil
from pathlib import Path
import statistics
import sys

import analyze
from collect import ROOT, Observer, model_snapshot, sha256_file, write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

SOURCES = tuple("tools/profiling/qwen_base/" + name for name in
                ("qkv_collect.py", "qkv_probe.mm", "qkv_ane.hpp", "collect.py", "analyze.py", "observer.mm")) + ("tools/native/build_qwen_base_qkv_probe.sh",)
ARMS = ("gpu-original", "gpu-staged", "private-prestaged", "private-restaged")


def require_output_space(free_bytes):
    # Full independent controls currently occupy208MiB plus metadata/cache.
    if free_bytes < (2 << 30) + (256 << 20):
        raise ValueError("QKV capture needs256MiB output allowance plus2GiB free reserve; no model/cache deletion is automatic")


def validate(report, repeats):
    if (report.get("status") != "complete" or report.get("complete_request") is not False or report.get("image_export") is not False or
            report.get("chunk") not in (256, 512) or report.get("repeats") != repeats or report.get("warmups") != 2 or
            report.get("fresh_inputs_weights_outputs_verified") is not True or report.get("borrowed_values_unchanged") is not True or
            report.get("predictions") != 5 + (repeats + 2) * 2 or report.get("weight_stages") != repeats + 5):
        raise ValueError("complete base component and actual source/predict restoration evidence required")
    request = report.get("actual_source_request") or {}
    if ((request.get("model"), request.get("width"), request.get("height"), request.get("steps"), request.get("frames")) !=
            ("qwen-image-2.1", 512, 512, 40, 1) or request.get("audio") is not False or request.get("lora_count") != 0 or
            request.get("execution") != "gpu" or request.get("residency") != "resident" or request.get("allow_approximation") is not False or
            not ((request.get("operation") == "image.generate" and request.get("reference_count") == 0) or
                 (request.get("operation") == "image.edit" and request.get("reference_count") == 1 and request.get("reference_size") == 1024))):
        raise ValueError("actual full-base40 ordinary GPU source request required")
    samples = report.get("samples", [])
    if len(samples) != 4 * (repeats + 2):
        raise ValueError("all rotated warm/measured complete-block samples required")
    counts = Counter()
    for row in samples:
        if row.get("arm") not in ARMS or type(row.get("seconds")) not in (int, float) or not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("actual finite complete-block times required")
        if row.get("warmup") is not (row["rep"] < 0):
            raise ValueError("warmup cannot enter the denominator")
        counts[row["arm"], row["rep"]] += 1
        for field in ("weight_stage_seconds", "activation_seconds", "submit_seconds", "join_seconds"):
            if type(row.get(field)) not in (int, float) or not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError("finite phase spans required")
        if row["arm"] == "private-restaged" and row["weight_stage_seconds"] <= 0:
            raise ValueError("honest restaging arm cannot omit weight work")
    if set(counts) != {(a, r) for a in ARMS for r in range(-2, repeats)} or any(v != 1 for v in counts.values()):
        raise ValueError("one sample per arm/rep required")
    result = {a: dict(seconds=[s["seconds"] for s in samples if s["arm"] == a and not s["warmup"]]) for a in ARMS}
    for arm in ARMS:
        result[arm]["median_seconds"] = statistics.median(result[arm]["seconds"])
    base = result["gpu-original"]["median_seconds"]
    for arm in ARMS:
        result[arm]["reduction_vs_original_percent"] = 100 * (1 - result[arm]["median_seconds"] / base)
    return result


def observer_contract(value, predictions):
    rows = analyze.observer_evidence(value)
    model_ids = {r["model"] for r in rows if r["name"] == "ane.evaluate_api" and r["request"] >= 1}
    if len(model_ids) != 1:
        raise ValueError("one fixed runtime QKV model must serve changed weights/inputs")
    workload = [r for r in rows if r["request"] >= 1]
    if sum(r["name"] == "ane.evaluate_api" for r in workload) != predictions:
        raise ValueError("actual QKV evaluate submissions differ from successful native predictions")
    if any(r["name"] in ("ane.compile_api", "ane.load_api") for r in rows if r["request"] >= 1):
        raise ValueError("hot weight/input changes compiled or loaded a model")
    return dict(runtime_model_ids=sorted(model_ids), hot_no_compile=True, hot_no_load=True,
                successful_evaluate_calls=predictions, scope="actual Private selectors; no ANE hardware timing or physical INT8/overlap proof")


def environment():
    result = {k: v for k, v in benchmark_environment().items() if not k.startswith("TURBOCIDER_") and k != "MLX_DISABLE_COMPILE"}
    result.update(TURBOCIDER_ANE_BACKEND="private", TURBOCIDER_ALLOW_PRIVATE_ANE="1")
    return result


def child(args):
    observer = Observer(args.observer) if args.mode == "timeline" else None
    if observer:
        observer.lib.tc_qwen_profile_scope(-1)  # source encoder/prefill is not the -2 blit calibration
    C.CDLL(str(args.native_library), mode=C.RTLD_GLOBAL)
    lib = C.CDLL(str(args.library))
    scope_type = C.CFUNCTYPE(None, C.c_int)
    scope = scope_type(lambda i: observer.lib.tc_qwen_profile_scope(i)) if observer else scope_type()
    fn = lib.tc_qwen_base_qkv_probe
    fn.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p, C.c_int, C.c_int, C.c_int, scope_type]
    fn.restype = C.c_int
    try:
        code = fn(str(args.model).encode(), str(args.reference).encode() if args.case == "edit" else b"-",
                  str(args.output).encode(), args.chunk, int(args.layout == "tiled"), args.repeats, scope)
        if code:
            raise RuntimeError(f"bounded base QKV probe returned{code}; inspect stderr/report")
    finally:
        if observer:
            write_json(args.output / "observer.json", observer.take("tc_qwen_profile_finish"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, default=ROOT / "build/qwen-base-qkv-capture-v1-20261018/libturbocider.dylib")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/qwen21/teapot-blue-edit-native-512.png")
    parser.add_argument("--observer", type=Path, default=ROOT / "build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib")
    parser.add_argument("--case", choices=("generate", "edit"), default="generate")
    parser.add_argument("--chunk", type=int, choices=(256, 512), default=512)
    parser.add_argument("--layout", choices=("tiled", "single"), default="tiled")
    parser.add_argument("--mode", choices=("stock", "timeline"), default="stock")
    parser.add_argument("--repeats", type=int, choices=(4, 8, 10, 20), default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    for key in ("library", "native_library", "model", "reference", "observer"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    if args.child:
        child(args)
        return
    manifest = json.loads((args.native_library.parent / "runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        if any(sha256_file(ROOT / e["path"]) != e["sha256"] for e in manifest["inputs"]["sources"]):
            raise ValueError("frozen native source identity changed")
    native_sources()
    existing_parent = next(p for p in args.output.parents if p.exists())
    require_output_space(shutil.disk_usage(existing_parent).free)
    sources = {name: sha256_file(ROOT / name) for name in SOURCES}
    assets = {name: sha256_file(getattr(args, name)) for name in ("library", "native_library", "reference", "observer")}
    identity = model_snapshot(args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    trial = args.output / "trial"
    trial.mkdir()
    summary = dict(schema="tc-base-private-qkv-screen-v1", status="running", qualification_passed=False,
                   case=args.case, chunk=args.chunk, layout=args.layout, mode=args.mode, repeats=args.repeats,
                   source_sha256=sources, assets=assets, runtime_build_id=manifest["runtime_build_id"], model_snapshot=identity,
                   scope="real base512 layer0 decode component; ordinary full-block GPU control; no full request/image/FFN-ANE combination claim")
    write_json(args.output / "source.json", {name: (ROOT / name).read_text() for name in SOURCES})
    write_json(args.output / "summary.json", summary)
    load = LoadObservation(args.output / "load.jsonl")
    try:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--library", str(args.library),
                   "--native-library", str(args.native_library), "--model", str(args.model), "--reference", str(args.reference),
                   "--observer", str(args.observer), "--case", args.case, "--chunk", str(args.chunk), "--layout", args.layout,
                   "--mode", args.mode, "--repeats", str(args.repeats), "--output", str(trial)]
        with (args.output / "stdout.log").open("x") as stdout, (args.output / "stderr.log").open("x") as stderr:
            summary["memory"] = run_sampled(command, repo=ROOT, output=args.output, stem="trial", env=environment(), stdout=stdout,
                                            stderr=stderr, timeout=900, interval_ms=100, max_gap_ms=500, observer=load)
        report = json.loads((trial / "report.json").read_text())
        if (report["loaded_runtime_id"] != manifest["runtime_build_id"] or report["cpp_sha256"] != sources["tools/profiling/qwen_base/qkv_probe.mm"] or
                report["header_sha256"] != sources["tools/profiling/qwen_base/qkv_ane.hpp"] or report["chunk"] != args.chunk or report["tiled"] != (args.layout == "tiled")):
            raise ValueError("compiled source/loaded runtime/geometry differs")
        summary["analysis"] = validate(report, args.repeats)
        if args.mode == "timeline":
            value = json.loads((trial / "observer.json").read_text())
            if value["source_sha256"] != sources["tools/profiling/qwen_base/observer.mm"]:
                raise ValueError("observer compiled source differs")
            summary["observer"] = observer_contract(value, report["predictions"])
        summary["arrays_sha256"] = sha256_file(trial / "arrays.safetensors")
        native_sources()
        summary["status"] = "complete"
    except Exception as error:
        summary.update(status="failed", error=str(error))
    try:
        summary["load"] = load.verify()
    except ValueError as error:
        summary["load_error"] = str(error)
    if (model_snapshot(args.model) != identity or any(sha256_file(ROOT / name) != digest for name, digest in sources.items()) or
            any(sha256_file(getattr(args, name)) != digest for name, digest in assets.items())):
        summary.update(status="failed", error="bound source or assets changed")
    write_json(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("model_snapshot", "source_sha256")}, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
