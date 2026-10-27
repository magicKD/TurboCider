#!/usr/bin/env python3
"""Qwen-only full-request cache/FFN composition experiment; no downloads.

One native session per route, off/warmup then off/candidates/off for each
workload. Diagnostic cache parameters deliberately match the public GPU
presets without changing their GPU-only admission. Cold and text-cache state
are explicit. CPU image metrics are observations, not semantic qualification.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from flux9_bf16_framework_screen import write_json
from qwen21_dit_cache_report import image_metrics
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
PRESETS = {"conservative": (.15, 1), "balanced": (.25, 2), "fast": (.25, 4)}
CASES = {
    "generate": "A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.",
    "edit": "Change the teapot to bright red. Preserve its shape, handle, spout, table and background.",
}
DEFAULT_MANIFEST = ROOT / "outputs/benchmark-best-configs/qwen21-frozen-w6144-r1024-cache-20261011/manifest-99f1b3499b89b96f5d04a23cb91c3f3496083e1d66b09c6dd2e4500d0d6ca509.json"


def sequence(modes):
    if not modes or len(set(modes)) != len(modes) or any(m not in PRESETS for m in modes):
        raise ValueError("unique known candidate modes required")
    return ("off", "off", *modes, "off")


def make_request(args, case, mode, index):
    r = dict(model="qwen-image-2.1", operation="image.generate" if case == "generate" else "image.edit",
        prompt=CASES[case], width=512, height=512, steps=40, seed=args.seed, audio=False,
        residency="resident", allow_approximation=True,
        execution="gpu_ane" if args.child == "frozen" else "gpu",
        output=str(args.output / f"{index}.png"))
    if case == "edit":
        r["inputs"] = [dict(kind="image", role="reference", path=str(args.reference))]
    if args.child == "frozen":
        r.update(qwen21_w8a8=True, ane_manifest=str(args.manifest))
    if mode == "off" or getattr(args, "cache_api", "diagnostic") == "preset":
        # Explicit Off overrides diagnostics. Named-preset experiments keep
        # the selected mode in the request instead of inheriting an env flag.
        r["qwen21_dit_cache"] = mode
    return r


def validate_row(row, route, prior_calls):
    m, r = row["metrics"], row["request"]
    wall = row["request_wall_including_export"]
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("positive finite complete request wall required")
    if ((m["model"], m["operation"], m["width"], m["height"], m["steps"], m["seed"], m["actual_denoise_steps"]) !=
            (r["model"], r["operation"], 512, 512, 40, r["seed"], 40)):
        raise ValueError("actual workload/steps differ")
    if m["prompt_cache_hit"] is not (row["position"] != 0):
        raise ValueError("conditioning cache lifecycle differs")
    cache = m.get("qwen21_dbcache")
    saved = 0
    if row["mode"] == "off":
        if cache:
            raise ValueError("explicit Off performed temporal reuse")
    else:
        threshold, maximum = PRESETS[row["mode"]]
        expected_mode = row["mode"] if row.get("cache_api") == "preset" else "diagnostic"
        if (not cache or cache.get("mode") != expected_mode or cache.get("front_blocks") != 8 or
                cache.get("back_blocks") != 0 or cache.get("warmup_steps") != 8 or
                abs(cache.get("threshold", 0) - threshold) > 1e-6 or cache.get("max_consecutive") != maximum or
                type(cache.get("cached_steps")) is not int or not 0 < cache["cached_steps"] <= 31 or
                cache.get("saved_middle_blocks") != cache["cached_steps"] * 24):
            raise ValueError("actual cache policy/coverage differs")
        saved = cache["saved_middle_blocks"]
    h = m.get("hybrid")
    if route == "gpu":
        if h:
            raise ValueError("GPU control unexpectedly performed hybrid work")
        return prior_calls
    if (not h or h.get("runtime_failed") is not False or type(h.get("runtime_failures_session_total")) is not int or
            h.get("runtime_failures_session_total") != 0 or
            h.get("checkpoint_sha256_verified") is not True or h.get("runtime_weight") is not None or
            h.get("ane_mlp_range") != [0, 6144] or h.get("compute_units") != "cpuAndNeuralEngine"):
        raise ValueError("successful source-bound frozen FFN required, not fallback")
    calls = h.get("runtime_calls_session_total")
    if type(calls) is not int or calls - prior_calls != 39 * 32 - saved:
        raise ValueError("actual hybrid prediction delta differs from executed blocks")
    return calls


def child(args):
    args.output.mkdir(parents=True, exist_ok=True)
    lib = C.CDLL(str(args.library))
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_free.argtypes = [C.c_void_p]

    def consume(p):
        value = C.string_at(p).decode() if p.value else ""
        if p.value:
            lib.tc_string_free(p)
        return value

    # Disable between-request prefix snapshots on EVERY route so the study
    # isolates temporal reuse and cannot borrow a warm-prefix advantage.
    os.environ["TURBOCIDER_QWEN21_PREFIX_SNAPSHOT"] = "0"
    if args.cache_api == "diagnostic":
        os.environ["TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC"] = "1"
    else:
        for key in tuple(os.environ):
            if key.startswith("TURBOCIDER_QWEN21_DBCACHE_"):
                os.environ.pop(key)
    if args.child == "frozen" and "edit" in args.cases.split(","):
        # Preserve original1024 reference geometry. This existing explicit
        # diagnostic is required independently of temporal cache admission.
        os.environ["TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC"] = "1"
    engine, error = C.c_void_p(), C.c_void_p()
    started = time.perf_counter()
    if lib.tc_engine_create_model(b"qwen-image-2.1", str(args.model).encode(), C.byref(engine), C.byref(error)):
        raise RuntimeError(consume(error))
    report = dict(status="running", route=args.child, constructor_seconds=time.perf_counter() - started,
        library_sha256=sha256_file(args.library), reference_sha256=sha256_file(args.reference), runs=[])
    prior_calls = 0
    try:
        for case in args.cases.split(","):
            for position, mode in enumerate(sequence(args.modes.split(","))):
                threshold, maximum = PRESETS.get(mode, (.15, 1))
                if args.cache_api == "diagnostic":
                    os.environ.update(TURBOCIDER_QWEN21_DBCACHE_THRESHOLD=str(threshold),
                                      TURBOCIDER_QWEN21_DBCACHE_MAX_CONSECUTIVE=str(maximum))
                index = len(report["runs"])
                request = make_request(args, case, mode, index)
                write_json(args.output / f"request-{index}.json", request)
                result, error = C.c_void_p(), C.c_void_p()
                started = time.perf_counter()
                status = lib.tc_engine_generate(engine, json.dumps(request).encode(), None, None, C.byref(result), C.byref(error))
                wall = time.perf_counter() - started
                raw, failure = consume(result), consume(error)
                if status:
                    raise RuntimeError(failure)
                row = dict(index=index, case=case, position=position, mode=mode, cache_api=args.cache_api, warmup=position == 0,
                    request=request, request_wall_including_export=wall, metrics=json.loads(raw))
                # Preserve evidence before validation, including a failed gate.
                report["runs"].append(row)
                write_json(args.output / "report.json", report)
                prior_calls = validate_row(row, args.child, prior_calls)
                print(json.dumps(dict(route=args.child, case=case, mode=mode, warmup=row["warmup"],
                    seconds=wall, cached_steps=row["metrics"].get("qwen21_dbcache", {}).get("cached_steps", 0))), flush=True)
        report["status"] = "complete"
        write_json(args.output / "report.json", report)
    finally:
        lib.tc_engine_free(engine)


def compare(report):
    import numpy as np
    from PIL import Image
    rows = []
    for case in dict.fromkeys(r["case"] for r in report["runs"]):
        measured = [r for r in report["runs"] if r["case"] == case and not r["warmup"]]
        controls = [r for r in measured if r["mode"] == "off"]
        if len(controls) != 2:
            raise ValueError("both bracketing controls required")
        def pixels(row):
            with Image.open(row["request"]["output"]) as image:
                image.load()
                if image.size != (512, 512):
                    raise ValueError("complete 512px PNG required")
                return np.asarray(image.convert("RGBA"))
        a, b = [pixels(r) for r in controls]
        if not np.array_equal(a, b):
            raise ValueError("bracketing Off output changed; persistent-state contamination possible")
        baseline = statistics.mean(r["request_wall_including_export"] for r in controls)
        for row in measured:
            candidate = pixels(row)
            rows.append(dict(case=case, mode=row["mode"], index=row["index"],
                wall_seconds=row["request_wall_including_export"], baseline_seconds=baseline,
                reduction_percent=100 * (1 - row["request_wall_including_export"] / baseline),
                rgb=image_metrics(a[:, :, :3], candidate[:, :, :3]),
                alpha_max_abs_255=int(np.abs(a[:, :, 3].astype(int) - candidate[:, :, 3]).max()),
                rgba_sha256=hashlib.sha256(candidate.tobytes()).hexdigest()))
    return rows


def parent(args):
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema="tc-qwen-cache-hybrid-screen-v1", status="running", qualification_passed=False,
        scope=("512px/40step base BF16 generate/edit, original1024 reference preprocessing; " +
               ("named request preset" if args.cache_api == "preset" else "diagnostic preset-equivalent cache") +
               "; warm conditioning, no prefix snapshot; not physical ANE trace or broad quality qualification"),
        library_sha256=sha256_file(args.library), reference_sha256=sha256_file(args.reference),
        manifest_sha256=sha256_file(args.manifest), script_sha256=sha256_file(Path(__file__)),
        cache_api=args.cache_api, trials=[])
    write_json(args.output / "screen-source.json", dict(source=Path(__file__).read_text(),
                                                       sha256=summary["script_sha256"]))
    write_json(args.output / "summary.json", summary)
    for route in args.routes.split(","):
        output = args.output / route
        output.mkdir()
        observer = LoadObservation(output / "load.jsonl")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", route,
            "--library", str(args.library), "--model", str(args.model), "--manifest", str(args.manifest),
            "--reference", str(args.reference), "--output", str(output), "--seed", str(args.seed),
            "--modes", args.modes, "--cases", args.cases, "--cache-api", args.cache_api]
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            memory = run_sampled(command, repo=ROOT, output=output, stem="trial", env=benchmark_environment(),
                stdout=stdout, stderr=stderr, timeout=2400, interval_ms=100, max_gap_ms=500, observer=observer)
        load, load_error = None, None
        try:
            load = observer.verify()
        except ValueError as error:
            load_error = str(error)
        report = json.loads((output / "report.json").read_text())
        if report["status"] != "complete":
            raise ValueError("incomplete native report")
        summary["trials"].append(dict(route=route, memory=memory, load=load, load_error=load_error,
                                      observations=compare(report)))
        write_json(args.output / "summary.json", summary)
    summary["status"] = "complete"
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def comparison_cases(reports):
    cases = tuple(dict.fromkeys(row["case"] for row in reports["gpu"]["runs"]))
    if (not cases or any(case not in CASES for case in cases) or
            set(cases) != {row["case"] for row in reports["frozen"]["runs"]}):
        raise ValueError("both routes must cover the same nonempty known cases")
    return cases


def analyze(output):
    """Joint approximation error against full GPU, not just frozen controls."""
    import numpy as np
    from PIL import Image
    reports = {route: json.loads((output / route / "report.json").read_text()) for route in ("gpu", "frozen")}
    if any(r["status"] != "complete" for r in reports.values()):
        raise ValueError("both complete route reports required")
    def pixels(row):
        with Image.open(row["request"]["output"]) as im:
            im.load()
            if im.size != (512, 512):
                raise ValueError("complete 512px PNG required")
            return np.asarray(im.convert("RGBA"))
    observations = []
    for case in comparison_cases(reports):
        reference = next(r for r in reports["gpu"]["runs"] if r["case"] == case and not r["warmup"] and r["mode"] == "off")
        a = pixels(reference)
        baseline = statistics.mean(r["request_wall_including_export"] for r in reports["gpu"]["runs"]
                                  if r["case"] == case and not r["warmup"] and r["mode"] == "off")
        for route, report in reports.items():
            for mode in dict.fromkeys(r["mode"] for r in report["runs"]):
                rows = [r for r in report["runs"] if r["case"] == case and not r["warmup"] and r["mode"] == mode]
                if not rows:
                    continue
                if any({k: v for k, v in r["request"].items() if k not in
                        ("output", "execution", "qwen21_w8a8", "ane_manifest", "qwen21_dit_cache")} !=
                       {k: v for k, v in reference["request"].items() if k not in
                        ("output", "execution", "qwen21_w8a8", "ane_manifest", "qwen21_dit_cache")} for r in rows):
                    raise ValueError("joint quality comparison changed workload/source request")
                b = pixels(rows[0])
                seconds = statistics.median(r["request_wall_including_export"] for r in rows)
                observations.append(dict(case=case, route=route, mode=mode, samples=len(rows), seconds=seconds,
                    reduction_vs_gpu_off_percent=100 * (1 - seconds / baseline),
                    rgb_vs_gpu_off=image_metrics(a[:, :, :3], b[:, :, :3]),
                    alpha_max_abs_vs_gpu_off=int(np.abs(a[:, :, 3].astype(int) - b[:, :, 3]).max())))
    result = dict(schema="tc-qwen-cache-hybrid-joint-quality-v1", qualification_passed=False,
                  scope="same task/seed/source library; joint approximate output vs full GPU; no semantic quality guarantee",
                  observations=observations)
    write_json(output / "joint-quality.json", result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--reference", type=Path, default=ROOT / "results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--routes", default="frozen,gpu")
    p.add_argument("--cases", default="generate,edit")
    p.add_argument("--modes", default="conservative,balanced,fast")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--child", choices=("gpu", "frozen"))
    p.add_argument("--cache-api", choices=("diagnostic", "preset"), default="diagnostic")
    p.add_argument("--analyze-only", action="store_true", help="CPU-only joint quality vs complete GPU route")
    args = p.parse_args()
    sequence(args.modes.split(","))
    if any(r not in ("gpu", "frozen") for r in args.routes.split(",")) or any(c not in CASES for c in args.cases.split(",")):
        p.error("known routes and cases required")
    active_routes = (args.child,) if args.child else args.routes.split(",")
    if (not args.analyze_only and args.cache_api == "preset" and "frozen" in active_routes and
            any(mode != "conservative" for mode in args.modes.split(","))):
        p.error("frozen named-preset composition supports conservative only; choose --modes conservative")
    for key in ("library", "model", "manifest", "reference"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    if args.analyze_only:
        analyze(args.output)
    else:
        (child if args.child else parent)(args)


if __name__ == "__main__":
    main()
