#!/usr/bin/env python3
"""RETIRED graph-retention experiment: archived research binary only.

The pilot failed its net-benefit gate. Current native libraries reject the
retired flag; use this runner only to reproduce the pinned archived binary.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

from flux9_bf16_framework_screen import write_json
from image_framework_native import generate_requests
from qwen21_dit_cache_report import image_metrics
from qwen_student_prefix_screen import ADAPTER, PROMPTS, adapter_pin
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
ARCHIVED_LIBRARY_SHA256 = "16c5c6ed2fa617c809441564bf268757f1ffc2b1994a1210ea4b1f2ec7d61a7d"
GRAPH = re.compile(r"student graph cache (hit|miss|declined); block wrappers built=(\d+); block C\+\+ traces=(\d+); request values cleared before VAE")


def require_archived_library(library):
    if sha256_file(library) != ARCHIVED_LIBRARY_SHA256:
        raise ValueError("graph retention is retired: this runner requires the pinned archived research binary; "
                         "see docs/status/qwen512-student-graph-negative-2026-10-18.md")


def points(seed):
    # label, case, prompt kind, graph enabled, seed, conditioning hit, prefix hit
    return [
        ("warmup", "generate", "a", 0, seed, False, False),
        ("control-a", "generate", "a", 0, seed, True, False),
        ("graph-prime", "generate", "a", 1, seed, True, False),
        ("graph-hit", "generate", "a", 1, seed, True, False),
        ("graph-new-seed", "generate", "a", 1, seed + 1, True, False),
        ("graph-fresh-same-shape", "generate", "b", 1, seed, False, False),
        ("graph-new-geometry", "generate", "c", 1, seed, False, False),
        ("geometry-control", "generate", "c", 0, seed, True, False),
        ("fresh-control", "generate", "b", 0, seed, False, False),
        ("control-rewarm", "generate", "a", 0, seed, False, False),
        ("control-a-repeat", "generate", "a", 0, seed, True, False),
        ("control-new-seed", "generate", "a", 0, seed + 1, True, False),
        ("edit-populate", "edit", "a", 0, seed, False, False),
        ("edit-control-a", "edit", "a", 0, seed, True, True),
        ("edit-graph-prime", "edit", "a", 1, seed, True, True),
        ("edit-graph-hit", "edit", "a", 1, seed, True, True),
        ("edit-graph-new-seed", "edit", "a", 1, seed + 1, True, True),
        ("edit-control-new-seed", "edit", "a", 0, seed + 1, True, True),
        ("edit-control-a-repeat", "edit", "a", 0, seed, True, True),
    ]


def prompt(case, kind):
    value = PROMPTS[case]
    if kind == "b":
        return value.replace("blue", "red")
    return value + " A small gold star is clearly visible on its side." if kind == "c" else value


def requests(args):
    values = []
    for i, (_, case, kind, _, seed, _, _) in enumerate(points(args.seed)):
        r = dict(model="qwen-image-2.1", operation="image." + case, prompt=prompt(case, kind),
            width=512, height=512, steps=6, seed=seed, frames=1, audio=False,
            execution="gpu", residency="resident", allow_approximation=True, qwen21_dit_cache="off",
            lora_strategy="inference_time", loras=[dict(path=str(args.lora), strength=1., role="transformer")],
            output=str(args.output / f"{i}.png"))
        if case == "edit":
            r.update(qwen21_reference_size=args.reference_size,
                     inputs=[dict(kind="image", role="reference", path=str(args.reference))])
        values.append(r)
    return values


def validate(report, seed):
    if len(report["runs"]) != len(points(seed)):
        raise ValueError("all graph/control requests required")
    observations = []
    for i, (row, point) in enumerate(zip(report["runs"], points(seed))):
        label, case, kind, enabled, expected_seed, condition_hit, prefix_hit = point
        m = row["metrics"]
        if (m["model"], m["operation"], m["steps"], m["actual_denoise_steps"], m["width"], m["height"], m["seed"]) != (
                "qwen-image-2.1", "image." + case, 6, 6, 512, 512, expected_seed):
            raise ValueError("actual six-step workload differs")
        wall = row["request_wall_including_export"]
        if type(wall) not in (float, int) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("finite positive full request wall required")
        if (m.get("hybrid") or m.get("encoder_hybrid") or m.get("student_ffn_reuse") or m.get("qwen21_dbcache") or
                m.get("lora_applied_projections") != 227 or m.get("lora_strategy") != "inference_time"):
            raise ValueError("full GPU student without extra approximation required")
        if m["prompt_cache_hit"] is not condition_hit:
            raise ValueError("conditioning lifecycle differs at " + label)
        selected = m["acceleration_selection"]
        if "joint BF16 LoRA A/B operands" not in selected:
            raise ValueError("joint arithmetic must remain fixed")
        if ("edit prefix KV snapshot hit" in selected) != prefix_hit:
            raise ValueError("prefix-value reuse differs at " + label)
        graph = GRAPH.search(selected)
        if not enabled:
            if graph:
                raise ValueError("disabled graph policy remained active")
            continue
        if not graph:
            raise ValueError("native graph observation missing at " + label)
        status, builds, traces = graph[1], int(graph[2]), int(graph[3])
        expected = "miss" if i in (2, 6, 14) else "hit"
        if status != expected:
            raise ValueError("unexpected graph geometry/admission at " + label)
        if status == "hit" and (builds != 0 or traces != 0):
            raise ValueError("reused graph unexpectedly rebuilt/retraced")
        if status == "miss" and (builds <= 0 or traces <= 0):
            raise ValueError("miss did not observe actual construction/trace")
        observations.append(dict(index=i, label=label, status=status, builds=builds, cpp_traces=traces))
    return observations


def child(args):
    sequence = points(args.seed)
    os.environ["TURBOCIDER_QWEN21_LORA_BF16_AB"] = "1"
    def before(index, request):
        label, case, _, enabled, _, _, _ = sequence[index]
        os.environ["TURBOCIDER_QWEN21_RETAIN_STUDENT_GRAPH"] = str(enabled)
        os.environ["TURBOCIDER_QWEN21_PREFIX_SNAPSHOT"] = "1" if case == "edit" else "0"
        if case == "edit" and args.reference_size == 512 and adapter_pin(args.lora)["rank"] == "r256":
            os.environ["TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC"] = "1"
        else:
            os.environ.pop("TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC", None)
        print(json.dumps(dict(index=index, label=label, retain_graph=enabled)), flush=True)
    report = generate_requests(args.library, "qwen-image-2.1", args.model, requests(args), args.output,
                               "student-graph", before_request=before)
    report["graph_observations"] = validate(report, args.seed)
    report["status"] = "complete"
    write_json(args.output / "report.json", report)


def analyze(output, report):
    import numpy as np
    from PIL import Image
    def pixels(i):
        with Image.open(output / f"{i}.png") as im:
            im.load()
            if im.size != (512, 512):
                raise ValueError("complete512 PNG required")
            return np.asarray(im.convert("RGBA"))
    def comparison(a, b):
        x, y = pixels(a), pixels(b)
        return dict(rgba_equal=bool(np.array_equal(x, y)), rgb=image_metrics(x[:, :, :3], y[:, :, :3]),
                    alpha_max_abs=int(np.abs(x[:, :, 3].astype(int) - y[:, :, 3]).max()))
    for a, b in ((1, 10), (13, 18)):
        if not comparison(a, b)["rgba_equal"]:
            raise ValueError("return-to-control changed output")
    wall = lambda i: report["runs"][i]["request_wall_including_export"]
    rows = []
    for label, control, candidate, controls in (("generation-repeat", 1, 3, [1, 10]),
            ("generation-new-seed", 11, 4, [11]), ("generation-fresh-same-shape", 8, 5, [8]),
            ("edit-repeat-strong-prefix", 13, 15, [13, 18]), ("edit-new-seed-strong-prefix", 17, 16, [17])):
        seconds = statistics.mean(wall(i) for i in controls)
        rows.append(dict(label=label, baseline_seconds=seconds, candidate_seconds=wall(candidate),
            reduction_percent=100 * (1 - wall(candidate) / seconds), quality=comparison(control, candidate)))
    return dict(observations=rows, geometry_change_quality=comparison(7, 6), controls_restored=True)


def parent(args):
    identity = adapter_pin(args.lora)
    if sha256_file(args.lora) != identity["sha256"]:
        raise ValueError("pinned original adapter required")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema="tc-qwen-student-graph-v1", status="running", qualification_passed=False,
        scope="same pinned six-step GPU/joint source; graph-only retention; request values reset; edit controls retain same immutable prefix; C++ construction/trace counters not GPU shader counters",
        library_sha256=sha256_file(args.library), adapter_sha256=identity["sha256"],
        reference_sha256=sha256_file(args.reference), script_sha256=sha256_file(Path(__file__)),
        gates=dict(minimum_reduction_percent=3., require_matched_rgba_equal=True,
                   require_graph_hit_zero_cpp_traces=True))
    write_json(args.output / "source.json", dict(source=Path(__file__).read_text()))
    write_json(args.output / "summary.json", summary)
    observer = LoadObservation(args.output / "load.jsonl")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--library", str(args.library),
        "--model", str(args.model), "--lora", str(args.lora), "--reference", str(args.reference),
        "--reference-size", str(args.reference_size), "--seed", str(args.seed), "--output", str(args.output)]
    with (args.output / "stdout.log").open("x") as stdout, (args.output / "stderr.log").open("x") as stderr:
        memory = run_sampled(command, repo=ROOT, output=args.output, stem="trial", env=benchmark_environment(),
            stdout=stdout, stderr=stderr, timeout=1800, interval_ms=100, max_gap_ms=500, observer=observer)
    load, load_error = None, None
    try:
        load = observer.verify()
    except ValueError as error:
        load_error = str(error)
    report = json.loads((args.output / "report.json").read_text())
    traces = validate(report, args.seed)
    summary.update(status="complete", memory=memory, load=load, load_error=load_error,
                   graph_observations=traces, analysis=analyze(args.output, report))
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora", type=Path, default=ADAPTER)
    p.add_argument("--reference", type=Path, default=ROOT / "results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--reference-size", type=int, choices=(512, 1024), default=512)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    for key in ("library", "model", "lora", "reference"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    require_archived_library(args.library)
    args.output = args.output.resolve()
    (child if args.child else parent)(args)


if __name__ == "__main__":
    main()
