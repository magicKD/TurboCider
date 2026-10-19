#!/usr/bin/env python3
"""Base-only Qwen512/40 GPU vs Private runtime W8 staging pilot; no LoRA."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

from flux9_bf16_framework_screen import write_json
from image_framework_native import generate_requests
from qwen_encoder_residency_screen import validate_weight_code_cache
from runtime_ane_common import benchmark_environment, sha256_file, validate_results, validate_fixed_async, validate_fp32_channel_join
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = dict(generate="A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.",
               edit="Change the teapot to bright red. Preserve its shape, handle, spout, table and background.")
SOURCES = ("tools/validation/qwen_base_runtime_screen.py", "tools/validation/image_framework_native.py",
           "tools/validation/qwen_encoder_residency_screen.py", "tools/validation/runtime_ane_common.py")


def environment(route):
    if route == "gpu":
        return {}
    if route not in ("runtime-off", "runtime-surface"):
        raise ValueError("base GPU or explicit runtime staging route required")
    return dict(TURBOCIDER_ANE_BACKEND="private", TURBOCIDER_ALLOW_PRIVATE_ANE="1",
                TURBOCIDER_PRIVATE_ANE_GPU_IO="1", TURBOCIDER_PRIVATE_ANE_DATA_PATH="w8a8",
                TURBOCIDER_PRIVATE_ANE_CHANNELS="4096", TURBOCIDER_PRIVATE_ANE_SCALE_CACHE="1",
                TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE="1", TURBOCIDER_PRIVATE_ANE_PREFETCH="0",
                TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE="1", TURBOCIDER_PRIVATE_ANE_ROWS="descriptor",
                TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1", TURBOCIDER_RUNTIME_ANE_CHUNKS="1",
                TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN="1",
                TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES=str(2 << 30 if route == "runtime-surface" else 0),
                TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_MODE="surface" if route == "runtime-surface" else "copy")


def requests(args):
    values = []
    for i, seed in enumerate((73, 74, 73)):
        request = dict(model="qwen-image-2.1", operation="image." + args.case, prompt=PROMPTS[args.case],
                       width=512, height=512, steps=40, frames=1, audio=False, seed=seed,
                       residency="resident", allow_approximation=True, qwen21_dit_cache="off",
                       execution="gpu" if args.route == "gpu" else "gpu_ane", output=str(args.output / f"{i}.png"))
        if args.route != "gpu":
            request.update(hybrid_mlp_mode="runtime", ane_manifest=str(args.manifest))
        if args.case == "edit":
            request.update(qwen21_reference_size=1024, inputs=[dict(kind="image", role="reference", path=str(args.reference))])
        values.append(request)
    return values


def validate(report, route, case):
    rows = report.get("runs", [])
    if len(rows) != 3:
        raise ValueError("one first and both full hot requests required")
    for i, row in enumerate(rows):
        m, wall = row["metrics"], row["request_wall_including_export"]
        if (type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0 or
                (m["model"], m["operation"], m["width"], m["height"], m["steps"], m["actual_denoise_steps"], m["seed"]) !=
                ("qwen-image-2.1", "image." + case, 512, 512, 40, 40, (73, 74, 73)[i]) or
                m.get("lora_applied_projections", 0) != 0 or m.get("qwen21_dbcache") or m["prompt_cache_hit"] is not (i > 0)):
            raise ValueError("complete base-only forty-step request/conditioning boundary differs")
        if any(word in m["acceleration_selection"].lower() for word in ("viggle", "student")):
            raise ValueError("base pilot must not inherit adapter/student recipes")
        hybrid = m.get("hybrid")
        if route == "gpu":
            if hybrid:
                raise ValueError("complete GPU control unexpectedly used hybrid work")
        else:
            runtime = (hybrid or {}).get("runtime_weight") or {}
            if (runtime.get("executor_backend") != "private_ane" or runtime.get("data_path") != "w8a8_hadamard"
                    or runtime.get("ane_channels") != 4096 or runtime.get("gpu_channels") != 8192):
                raise ValueError("actual dynamic Private W8 partition differs")
            for name in ("fallback_blocks_session_total", "overflow_retries_session_total"):
                if runtime.get(name) != 0:
                    raise ValueError("runtime fallback/retry cannot be counted as this fixed pilot")
    metrics = [row["metrics"] for row in rows]
    validate_results(metrics, "gpu" if route == "gpu" else "runtime", 3, model_id="qwen-image-2.1", expect_lora=False,
                     runtime_backend="private" if route != "gpu" else "public", expect_device_io=route != "gpu",
                     expected_data_path="w8a8_hadamard" if route != "gpu" else None)
    if route != "gpu":
        validate_fixed_async(metrics, True)
        validate_fp32_channel_join(metrics, True)
        counts = [row["hybrid"]["runtime_weight"]["hybrid_blocks_session_total"] for row in metrics]
        if counts != [1280, 2560, 3840]:
            raise ValueError("all forty-by-thirty-two actual FFNs must execute the fixed runtime partition")
        validate_weight_code_cache(metrics, 2 << 30 if route == "runtime-surface" else 0,
                                   "surface" if route == "runtime-surface" else "copy")


def child(args):
    os.environ.update(environment(args.route))
    report = generate_requests(args.library, "qwen-image-2.1", args.model, requests(args), args.output, args.route)
    validate(report, args.route, args.case)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("gpu", "runtime-off", "runtime-surface"), required=True)
    parser.add_argument("--case", choices=("generate", "edit"), default="generate")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/qwen21/teapot-blue-edit-native-512.png")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    for name in ("library", "model", "reference", "manifest"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    if args.child:
        child(args)
        return
    manifest = json.loads((args.library.parent / "runtime-build/runtime-build-manifest.json").read_text())
    for entry in manifest["inputs"]["sources"]:
        if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            parser.error("current native source differs from linked library")
    args.output.mkdir(parents=True, exist_ok=False)
    trial = args.output / "trial"
    trial.mkdir()
    sources = {name: sha256_file(ROOT / name) for name in SOURCES}
    summary = dict(schema="tc-qwen-base-runtime-screen-v1", status="running", qualification_passed=False,
                   route=args.route, case=args.case, source_sha256=sources,
                   assets={name: sha256_file(getattr(args, name)) for name in ("library", "reference", "manifest")},
                   runtime_build_id=manifest["runtime_build_id"], environment=environment(args.route),
                   scope="base/no LoRA, actual40 updates, no temporal/prefix cache; first separate, two hot requests; not formal performance qualification")
    write_json(args.output / "source.json", {name: (ROOT / name).read_text() for name in SOURCES})
    write_json(args.output / "summary.json", summary)
    observer = LoadObservation(args.output / "load.jsonl")
    try:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--library", str(args.library),
                   "--output", str(trial), "--route", args.route, "--case", args.case, "--model", str(args.model),
                   "--reference", str(args.reference), "--manifest", str(args.manifest)]
        with (args.output / "stdout.log").open("x") as stdout, (args.output / "stderr.log").open("x") as stderr:
            summary["memory"] = run_sampled(command, repo=ROOT, output=args.output, stem="trial", env=benchmark_environment(),
                                            stdout=stdout, stderr=stderr, timeout=1800, interval_ms=100, max_gap_ms=500, observer=observer)
        report = json.loads((trial / "report.json").read_text())
        validate(report, args.route, args.case)
        summary["analysis"] = dict(first_seconds=report["runs"][0]["request_wall_including_export"],
                                   full_hot_seconds=[row["request_wall_including_export"] for row in report["runs"][1:]],
                                   hot_median_seconds=statistics.median(row["request_wall_including_export"] for row in report["runs"][1:]))
        summary["status"] = "complete"
    except Exception as error:
        summary.update(status="failed", error=str(error))
    try:
        summary["load"] = observer.verify()
    except ValueError as error:
        summary["load_error"] = str(error)
    if any(sha256_file(ROOT / name) != digest for name, digest in sources.items()):
        summary.update(status="failed", error="bound source changed")
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
