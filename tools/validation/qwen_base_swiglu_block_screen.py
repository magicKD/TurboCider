#!/usr/bin/env python3
"""Original-base/paired-B/SwiGLU full-block research, not whole-image speed."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess

from flux9_bf16_framework_screen import write_json
from qwen_student_prefix_screen import adapter_pin
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("gpu-original", "strong-control", "base-paired-b32", "base-paired-b64")
V1_ARMS = (*ARMS[:3], "base-paired-b16")
PRIMARY, CONTROL = ARMS[3], ARMS[1]
WARMUPS, SAMPLES = 3, 16
SOURCES = ("tools/validation/qwen_base_swiglu_block_screen.py",
           "tools/native/qwen21_base_swiglu_probe.mm",
           "tools/native/qwen21_base_swiglu_prepare.hpp",
           "tools/native/build_qwen21_base_swiglu_probe.sh")


def quality(value):
    if (value.get("bits_exact") is not True or type(value.get("changed")) is not int or value["changed"] != 0
            or value.get("elements") != 4194304 or value.get("relative_l2") != 0 or value.get("max_abs") != 0
            or type(value.get("cosine")) not in (int, float) or not math.isfinite(value["cosine"])
            or value["cosine"] < .999999999):
        raise ValueError("complete typed block including signed-zero bits must be exact")


def analyze(native, order, legacy=False):
    arms = V1_ARMS if legacy else ARMS
    primary = arms[2] if legacy else PRIMARY
    if (native.get("status") != "measured" or native.get("image_export") is not False
            or native.get("order") != order or native.get("runtime_lora_applied_terms") != 227
            or native.get("extra_owned_weight_bytes") != 0 or native.get("completed_full_dit_steps") != 1
            or native.get("stopped_step") != 1 or native.get("stopped_layer") != 0
            or native.get("fresh_arguments_verified") is not True or native.get("borrowed_values_unchanged") is not True
            or native.get("all_outputs_lazy_before_eval") is not True
            or native.get("timing_readback_policy") != "first-fresh-restored-and-final-only-v1"
            or len(native.get("actual_layer_source_sha256", {})) != 22
            or not isinstance(native.get("ordinary_block_sha256"), str) or len(native["ordinary_block_sha256"]) != 64):
        raise ValueError("bounded full-block work/source/lifecycle receipt differs")
    first, final = native.get("first_replays", []), native.get("final_quality", [])
    if [row["arm"] for row in first] != list(arms[1:]) or [row["arm"] for row in final] != list(arms):
        raise ValueError("all first and final complete outputs required")
    for row in first + final:
        quality(row["quality"])
    for row in first:
        if type(row.get("seconds")) not in (int, float) or not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("finite first-use wall required")
    expected = [(iteration, visit, arms[((3 - visit if order == "reverse" else visit) + iteration + 4) % 4])
                for iteration in range(-WARMUPS, SAMPLES) for visit in range(4)]
    rows = native.get("samples", [])
    if len(rows) != len(expected):
        raise ValueError("all balanced warm and measured calls required")
    values = {arm: [] for arm in arms}
    for row, key in zip(rows, expected):
        if (row["iteration"], row["visit"], row["arm"]) != key:
            raise ValueError("frozen visit rotation/order differs")
        wall = row["seconds"]
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("finite complete block wall required")
        if row["iteration"] >= 0:
            values[row["arm"]].append(wall)
    medians = {arm: statistics.median(samples) for arm, samples in values.items()}
    reduction = 100 * (1 - medians[primary] / medians[CONTROL])
    return dict(median_seconds=medians, primary=primary, strong_control=CONTROL,
                primary_reduction_percent=reduction, primary_local_signal=reduction >= 1.,
                measured_calls=4 * SAMPLES, warm_calls=4 * WARMUPS,
                first_fresh_restored_final_bitwise_exact=True,
                scope="one real step1/layer0; no per-timed-sample readback, whole-image latency or multiplied historical gains")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    parser.add_argument("--lora", type=Path, default=ROOT / "models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("binary", "library", "model", "lora"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.reference:
        args.reference = args.reference.resolve(strict=True)
    pin = adapter_pin(args.lora)
    if pin["rank"] != "r128" or sha256_file(args.lora) != pin["sha256"]:
        parser.error("original released r128 required")
    manifest = json.loads((args.library.parent / "runtime-build/runtime-build-manifest.json").read_text())
    for entry in manifest["inputs"]["sources"]:
        if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            parser.error("native sources differ from linked library")
    hashes = {name: sha256_file(ROOT / name) for name in SOURCES}
    assets = {name: sha256_file(getattr(args, name)) for name in ("binary", "library", "lora")}
    if args.reference:
        assets["reference"] = sha256_file(args.reference)
    descriptor = json.loads(subprocess.check_output([str(args.binary), "--describe"], cwd=ROOT, env=benchmark_environment(), text=True))
    if (descriptor.get("recipe") != "original-base-paired-b-swiglu-v2-b64" or descriptor.get("arms") != list(ARMS)
            or descriptor.get("primary") != PRIMARY or descriptor.get("strong_control") != CONTROL
            or descriptor.get("warmups") != WARMUPS or descriptor.get("samples_per_arm") != SAMPLES
            or descriptor.get("cpp_sha256") != hashes[SOURCES[1]] or descriptor.get("header_sha256") != hashes[SOURCES[2]]
            or descriptor.get("loaded_runtime_id") != manifest["runtime_build_id"]):
        parser.error("binary/source/runtime/workload descriptor differs")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema="tc-qwen-base-swiglu-block-screen-v1", status="running", qualification_passed=False,
                   order=args.order, edit=bool(args.reference), descriptor=descriptor, source_sha256=hashes, asset_sha256=assets)
    write_json(args.output / "source.json", {name: (ROOT / name).read_text() for name in SOURCES})
    write_json(args.output / "summary.json", summary)
    observer = LoadObservation(args.output / "load.jsonl")
    try:
        command = [str(args.binary), str(args.model), str(args.lora), str(args.output / "trial"),
                   str(args.reference) if args.reference else "-", args.order]
        with (args.output / "stdout.log").open("x") as stdout, (args.output / "stderr.log").open("x") as stderr:
            summary["memory"] = run_sampled(command, repo=ROOT, output=args.output, stem="trial", env=benchmark_environment(),
                                            stdout=stdout, stderr=stderr, timeout=1200, interval_ms=100, max_gap_ms=500, observer=observer)
        native = json.loads((args.output / "trial/native.json").read_text())
        if native["loaded_runtime_id"] != descriptor["loaded_runtime_id"]:
            raise ValueError("actual runtime differs")
        summary["analysis"] = analyze(native, args.order)
        summary["status"] = "complete"
    except Exception as error:
        summary.update(status="failed", error=str(error))
    try:
        summary["load"] = observer.verify()
    except ValueError as error:
        summary["load_error"] = str(error)
    if (any(sha256_file(ROOT / name) != digest for name, digest in hashes.items())
            or any(sha256_file(getattr(args, name)) != digest for name, digest in assets.items())):
        summary.update(status="failed", error="bound source/asset changed")
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
