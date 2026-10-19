#!/usr/bin/env python3
"""Qwen conservative residual cache × immutable reference-prefix snapshots.

One serial session; exact seeds have independent snapshot-off controls. Also
measure the established snapshot-enabled GPU/cache-off route. All output and
memory/load evidence is retained. No models, adapters or references fetched.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from flux9_bf16_framework_screen import write_json
from image_framework_native import generate_requests
from qwen21_dit_cache_report import image_metrics
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
PROMPT = "Change the teapot to bright red. Preserve its shape, handle, spout, table and background."
DEFAULT_MANIFEST = ROOT / "outputs/benchmark-best-configs/qwen21-frozen-w6144-r1024-cache-20261011/manifest-99f1b3499b89b96f5d04a23cb91c3f3496083e1d66b09c6dd2e4500d0d6ca509.json"


def points(seed):
    # (label, mode, explicit snapshot flag, seed, expected prefix hit)
    return [
        ("warmup", "conservative", 0, seed, False),
        ("control-a", "conservative", 0, seed, False),
        ("populate", "conservative", 1, seed, False),
        ("hit-a", "conservative", 1, seed, True),
        ("hit-b", "conservative", 1, seed + 1, True),
        ("control-b", "conservative", 0, seed + 1, False),
        ("control-a-repeat", "conservative", 0, seed, False),
        ("gpu-off-populate", "off", 1, seed, False),
        ("gpu-off-hit", "off", 1, seed, True),
    ]


def requests(args):
    values = []
    for index, (_, mode, _, seed, _) in enumerate(points(args.seed)):
        request = dict(model="qwen-image-2.1", operation="image.edit", prompt=args.prompt,
            width=512, height=512, steps=40, seed=seed, audio=False, execution="gpu",
            residency="resident", allow_approximation=True, qwen21_dit_cache=mode,
            inputs=[dict(kind="image", role="reference", path=str(args.reference))],
            output=str(args.output / f"{index}.png"))
        if getattr(args, "route", "gpu") == "frozen" and mode != "off":
            request.update(execution="gpu_ane", qwen21_w8a8=True, ane_manifest=str(args.manifest))
        values.append(request)
    return values


def validate(report, seed, route="gpu"):
    schedule = points(seed)
    if len(report["runs"]) != len(schedule):
        raise ValueError("all scheduled requests required")
    prior_calls = 0
    for i, (row, point) in enumerate(zip(report["runs"], schedule)):
        label, mode, flag, expected_seed, expected_hit = point
        m = row["metrics"]
        wall = row["request_wall_including_export"]
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("finite positive complete wall required")
        if ((m["model"], m["operation"], m["width"], m["height"], m["steps"], m["actual_denoise_steps"], m["seed"]) !=
                ("qwen-image-2.1", "image.edit", 512, 512, 40, 40, expected_seed)):
            raise ValueError("actual edit/geometry/steps/seed differ")
        if m["prompt_cache_hit"] is not (i != 0):
            raise ValueError("conditioning lifecycle differs")
        selection = m["acceleration_selection"]
        hit = "edit prefix KV snapshot hit" in selection
        if hit != expected_hit or (flag == 1 and "edit prefix KV snapshot " not in selection):
            raise ValueError(f"actual prefix hit/miss differs at {label}: {selection}")
        cache = m.get("qwen21_dbcache")
        if mode == "off":
            if cache:
                raise ValueError("off retained temporal cache state")
        elif (not cache or cache.get("mode") != mode or cache.get("front_blocks") != 8 or
                cache.get("back_blocks") != 0 or cache.get("warmup_steps") != 8 or
                cache.get("max_consecutive") != 1 or abs(cache.get("threshold", 0) - .15) > 1e-6 or
                type(cache.get("cached_steps")) is not int or not 0 < cache["cached_steps"] <= 31 or
                cache.get("saved_middle_blocks") != 24 * cache["cached_steps"]):
            raise ValueError("actual conservative cache policy/coverage differs")
        h = m.get("hybrid")
        if route == "frozen" and mode != "off":
            if (not h or h.get("checkpoint_sha256_verified") is not True or h.get("runtime_failed") is not False or
                    type(h.get("runtime_failures_session_total")) is not int or h["runtime_failures_session_total"] != 0 or
                    h.get("runtime_weight") is not None or h.get("ane_mlp_range") != [0, 6144] or
                    h.get("compute_units") != "cpuAndNeuralEngine"):
                raise ValueError("actual successful source-bound frozen work required")
            calls = h.get("runtime_calls_session_total")
            expected = 39 * 32 - cache["saved_middle_blocks"]
            if type(calls) is not int or calls - prior_calls != expected:
                raise ValueError("frozen calls differ; snapshot step zero must remain full GPU")
            prior_calls = calls
        elif h:
            raise ValueError("pure GPU control unexpectedly retained hybrid work")


def child(args):
    values = requests(args)
    schedule = points(args.seed)
    if args.route == "frozen":
        os.environ["TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC"] = "1"
    def configure(index, request):
        os.environ["TURBOCIDER_QWEN21_PREFIX_SNAPSHOT"] = str(schedule[index][2])
        print(json.dumps(dict(starting=schedule[index][0], index=index, seed=request["seed"])), flush=True)
    report = generate_requests(args.library, "qwen-image-2.1", args.model, values, args.output,
                               args.route + "-prefix-cache", before_request=configure)
    validate(report, args.seed, args.route)
    report["status"] = "complete"
    write_json(args.output / "report.json", report)


def analyze(output, report):
    import numpy as np
    from PIL import Image
    def pixels(index):
        with Image.open(output / f"{index}.png") as im:
            im.load()
            if im.size != (512, 512):
                raise ValueError("complete512 image required")
            return np.asarray(im.convert("RGBA"))
    def seconds(index):
        return report["runs"][index]["request_wall_including_export"]
    def compare(a, b):
        x, y = pixels(a), pixels(b)
        return dict(rgb=image_metrics(x[:, :, :3], y[:, :, :3]),
                    alpha_max_abs=int(np.abs(x[:, :, 3].astype(int) - y[:, :, 3]).max()),
                    rgba_equal=bool(np.array_equal(x, y)),
                    candidate_rgba_sha256=hashlib.sha256(y.tobytes()).hexdigest())
    stability = compare(1, 6)
    if not stability["rgba_equal"]:
        raise ValueError("snapshot on/off contaminated repeated conservative control")
    observations = []
    for label, reference, candidate, controls in (("same-seed", 1, 3, [1, 6]), ("new-seed", 5, 4, [5])):
        baseline = statistics.mean(seconds(i) for i in controls)
        observations.append(dict(label=label, baseline_seconds=baseline, candidate_seconds=seconds(candidate),
            reduction_percent=100 * (1 - seconds(candidate) / baseline), quality=compare(reference, candidate)))
    return dict(observations=observations, control_restoration=stability,
        vs_established_gpu_snapshot_off_cache=dict(baseline_seconds=seconds(8),
            candidate_seconds=seconds(3), reduction_percent=100 * (1 - seconds(3) / seconds(8)),
            quality=compare(8, 3)))


def evaluate_observation_gates(analysis, gates):
    observations = analysis["observations"]
    return dict(
        speed_passed=bool(observations) and all(
            r["reduction_percent"] >= gates["minimum_speed_reduction_percent"] for r in observations),
        incremental_quality_passed=bool(observations) and all(
            r["quality"]["rgb"]["ssim"] >= gates["minimum_incremental_rgb_ssim"] and
            r["quality"]["rgb"]["mae_255"] <= gates["maximum_incremental_mae_255"] and
            r["quality"]["alpha_max_abs"] <= gates["maximum_incremental_alpha_abs"] for r in observations),
        control_restoration_passed=analysis["control_restoration"]["rgba_equal"] is True)


def parent(args):
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema="tc-qwen-prefix-cache-screen-v1", status="running", qualification_passed=False,
        scope=("base BF16 512/40 edit; standard1024 reference; conservative snapshot on/off; same+new seeds; conditioning warm except first; " +
               ("frozen W8A8/6144-channel FFN; final cache-off controls pure GPU" if args.route == "frozen" else "GPU only") +
               "; no physical trace or broad quality qualification"), route=args.route,
        library_sha256=sha256_file(args.library), reference_sha256=sha256_file(args.reference),
        manifest_sha256=sha256_file(args.manifest) if args.route == "frozen" else None,
        script_sha256=sha256_file(Path(__file__)),
        observation_gates=dict(minimum_speed_reduction_percent=5., minimum_incremental_rgb_ssim=.98,
                               maximum_incremental_mae_255=3., maximum_incremental_alpha_abs=4),
        schedule=[dict(label=p[0], mode=p[1], snapshot=p[2], seed=p[3], expected_hit=p[4])
                  for p in points(args.seed)])
    write_json(args.output / "source.json", dict(source=Path(__file__).read_text()))
    write_json(args.output / "summary.json", summary)
    observer = LoadObservation(args.output / "load.jsonl")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--library", str(args.library),
        "--model", str(args.model), "--reference", str(args.reference), "--output", str(args.output),
        "--seed", str(args.seed), "--prompt", args.prompt, "--route", args.route, "--manifest", str(args.manifest)]
    with (args.output / "stdout.log").open("x") as stdout, (args.output / "stderr.log").open("x") as stderr:
        memory = run_sampled(command, repo=ROOT, output=args.output, stem="trial", env=benchmark_environment(),
            stdout=stdout, stderr=stderr, timeout=2400, interval_ms=100, max_gap_ms=500, observer=observer)
    load, load_error = None, None
    try:
        load = observer.verify()
    except ValueError as error:
        load_error = str(error)
    report = json.loads((args.output / "report.json").read_text())
    validate(report, args.seed, args.route)
    analysis = analyze(args.output, report)
    summary.update(status="complete", memory=memory, load=load, load_error=load_error,
                   analysis=analysis, observation_gate_results=evaluate_observation_gates(
                       analysis, summary["observation_gates"]))
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--reference", type=Path, default=ROOT / "results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--route", choices=("gpu", "frozen"), default="gpu")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    for key in ("library", "model", "reference"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.manifest = args.manifest.resolve(strict=args.route == "frozen")
    args.output = args.output.resolve()
    (child if args.child else parent)(args)


if __name__ == "__main__":
    main()
