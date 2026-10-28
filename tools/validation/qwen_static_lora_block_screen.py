#!/usr/bin/env python3
"""Bounded ordinary r128/joint block: static Private prefix vs complete GPU.

Not an image/request benchmark. Never changes a default or promotes a backend.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

from flux9_bf16_framework_screen import write_json
from qwen_student_prefix_screen import adapter_pin
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("gpu-original", "gpu-staged", "private-serial", "private-hybrid")


def analyze(native):
    if (native.get("status") != "measured" or native.get("runtime_lora_applied_terms") != 227
            or native.get("successful_private_calls") != 30 or native.get("image_export") is not False
            or native.get("borrowed_values_unchanged") is not True):
        raise ValueError("complete bounded ordinary-block evidence required")
    samples = native["samples"]
    if len(samples) != 60:
        raise ValueError("all first/warm/measured arms required")
    for i in range(-3, 12):
        group = [row for row in samples if row["iteration"] == i]
        if [row["arm"] for row in group] != [NAMES[(visit+i+4) % 4] for visit in range(4)]:
            raise ValueError("balanced arm order differs")
        for row in group:
            q = row["quality"]
            numeric = (row["seconds"], q["relative_l2"], q["cosine"], q["max_abs"])
            if (any(type(value) not in (int, float) or not math.isfinite(value) for value in numeric)
                    or row["seconds"] <= 0 or not 0 <= q["relative_l2"] <= .001 or not .999 <= q["cosine"] <= 1.000001
                    or q["max_abs"] < 0 or type(q["elements"]) is not int or q["elements"] != 1024*4096
                    or type(q["changed"]) is not int or not 0 <= q["changed"] <= q["elements"]):
                raise ValueError("whole-block fixed numerical gate differs")
            if row["arm"].startswith("gpu-") and q["changed"] != 0:
                raise ValueError("strong GPU replay is not exact")
            if row["arm"] != "gpu-original":
                f = row["ffn_quality"]
                if (f["elements"] != 1024*4096 or any(type(f[key]) not in (int, float) or not math.isfinite(f[key])
                        for key in ("relative_l2", "cosine", "max_abs")) or not 0 <= f["relative_l2"] <= .03
                        or not .999 <= f["cosine"] <= 1.000001 or f["max_abs"] < 0
                        or (row["arm"] == "gpu-staged" and f["changed"] != 0)):
                    raise ValueError("complete FFN first/warm/measured numerical gate differs")
    values = {name: [row["seconds"] for row in samples if row["arm"] == name and row["iteration"] >= 0]
              for name in NAMES}
    medians = {name: statistics.median(value) for name, value in values.items()}
    return dict(whole_block_seconds=values, medians=medians,
                reduction_vs_original_gpu_percent=100*(1-medians["private-hybrid"]/medians["gpu-original"]),
                scope="whole layer0 block only; original GPU is denominator; no request extrapolation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    parser.add_argument("--lora", type=Path, default=ROOT / "models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    parser.add_argument("--share", type=int, choices=(1024, 2048), default=2048)
    parser.add_argument("--shared-ranks", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ("binary", "library", "model", "lora", "reference"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve(strict=True))
    pin = adapter_pin(args.lora)
    if pin["rank"] != "r128" or sha256_file(args.lora) != pin["sha256"]:
        parser.error("original pinned r128 required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "tools/native/qwen21_static_lora_block_probe.mm",
               ROOT / "tools/native/qwen21_static_lora_mil.hpp", ROOT / "tools/native/build_qwen21_static_lora_block_probe.sh"]
    identity = {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}
    write_json(output / "source.json", {str(path.relative_to(ROOT)): path.read_text() for path in sources})
    summary = dict(schema="tc-qwen-static-lora-screen-v1", status="running", qualification_passed=False,
                   source_sha256=identity, binary_sha256=sha256_file(args.binary),
                   library_sha256=sha256_file(args.library), adapter_sha256=pin["sha256"],
                   prefix_channels=args.share, case="edit" if args.reference else "generate",
                   input_rank_recipe="shared" if args.shared_ranks else "separate",
                   gates=dict(minimum_whole_block_reduction_percent=3., maximum_block_rel_l2=.001,
                              maximum_ffn_rel_l2=.03, minimum_cosine=.999),
                   scope="one ordinary r128/joint step1 layer0; static base weights, full online LoRA; not complete image/request or physical overlap")
    write_json(output / "summary.json", summary)
    command = [str(args.binary), str(args.model), str(args.lora), str(output / "trial"), str(args.share),
               summary["input_rank_recipe"], str(args.reference) if args.reference else "-"]
    if args.reference:
        summary["reference_sha256"] = sha256_file(args.reference)
    observer = LoadObservation(output / "load.jsonl")
    try:
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            summary["memory"] = run_sampled(command, repo=ROOT, output=output, stem="trial", env=benchmark_environment(),
                stdout=stdout, stderr=stderr, timeout=1200, interval_ms=100, max_gap_ms=500, observer=observer)
        native = json.loads((output / "trial/native.json").read_text())
        summary["analysis"] = analyze(native)
        summary["status"] = "complete"
    except Exception as error:
        summary.update(status="failed", error=str(error))
        native_file = output / "trial/native.json"
        if native_file.is_file():
            native = json.loads(native_file.read_text())
            summary["native_status"] = native.get("status")
            summary["native_error"] = native.get("error")
    try:
        summary["load"] = observer.verify()
    except ValueError as error:
        summary["load_error"] = str(error)
    for path in sources:
        if sha256_file(path) != identity[str(path.relative_to(ROOT))]:
            summary.update(status="failed", error="source changed during measurement")
    if sha256_file(args.binary) != summary["binary_sha256"] or sha256_file(args.library) != summary["library_sha256"]:
        summary.update(status="failed", error="native artifact changed during measurement")
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
