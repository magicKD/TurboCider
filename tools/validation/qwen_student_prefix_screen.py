#!/usr/bin/env python3
"""Pinned six-step r128/r256 GPU students: precision × prefix replay.

No comparison to base40 latency or image equivalence. No new weights, temporal
FFN reuse or ANE. Every candidate keeps its own precision-matched controls.
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
ADAPTER = ROOT / "models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
ADAPTER_SHA256 = "2a0148f5c73abbed5f97da5ea356e439318aadb281d01fce4af39cdf43728803"
ADAPTER_PINS = {
    ADAPTER.name: dict(rank="r256", sha256=ADAPTER_SHA256),
    "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors": dict(
        rank="r128", sha256="bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b"),
}
PROMPTS = dict(generate="A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.",
              edit="Change the teapot to bright red. Preserve its shape, handle, spout, table and background.")


def adapter_pin(path):
    try:
        return ADAPTER_PINS[Path(path).name]
    except KeyError as error:
        raise ValueError("published pinned six-step adapter filename required") from error


def points(seed):
    # case, label, snapshot, seed, expected hit, expected conditioning hit
    return [
        ("generate", "warmup", 0, seed, False, False),
        ("generate", "control-a", 0, seed, False, True),
        ("generate", "new-seed", 0, seed + 1, False, True),
        ("generate", "control-a-repeat", 0, seed, False, True),
        ("edit", "warmup", 0, seed, False, False),
        ("edit", "control-a", 0, seed, False, True),
        ("edit", "populate", 1, seed, False, True),
        ("edit", "hit-a", 1, seed, True, True),
        ("edit", "hit-b", 1, seed + 1, True, True),
        ("edit", "control-b", 0, seed + 1, False, True),
        ("edit", "control-a-repeat", 0, seed, False, True),
    ]


def schedule(seed, precision):
    if precision != "switch":
        return points(seed)
    return [("edit", label, 1, seed, hit, i != 0) for i, (label, hit) in enumerate((
        ("original-populate", False), ("original-hit", True),
        ("joint-invalidate", False), ("joint-hit", True),
        ("original-restore-miss", False), ("original-restore-hit", True)))]


def actual_precision(precision, index):
    return "joint" if precision == "joint" or (precision == "switch" and index in (2, 3)) else "original"


def requests(args):
    values = []
    for i, (case, _, _, seed, _, _) in enumerate(schedule(args.seed, getattr(args, "child", None) or "original")):
        r = dict(model="qwen-image-2.1", operation="image." + case, prompt=PROMPTS[case],
            width=512, height=512, steps=6, seed=seed, frames=1, audio=False,
            execution="gpu", residency="resident", allow_approximation=True, qwen21_dit_cache="off",
            lora_strategy="inference_time", loras=[dict(path=str(args.lora), strength=1., role="transformer")],
            output=str(args.output / f"{i}.png"))
        if case == "edit":
            r.update(qwen21_reference_size=args.reference_size,
                     inputs=[dict(kind="image", role="reference", path=str(args.reference))])
        values.append(r)
    return values


def validate(report, seed, precision):
    expected = schedule(seed, precision)
    if len(report["runs"]) != len(expected):
        raise ValueError("complete student request sequence required")
    for i, (row, point) in enumerate(zip(report["runs"], expected)):
        case, label, snapshot, expected_seed, hit, conditioning_hit = point
        m = row["metrics"]
        wall = row["request_wall_including_export"]
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("finite complete request wall required")
        if ((m["model"], m["operation"], m["width"], m["height"], m["steps"], m["actual_denoise_steps"], m["seed"]) !=
                ("qwen-image-2.1", "image." + case, 512, 512, 6, 6, expected_seed)):
            raise ValueError("actual student task differs")
        if (m.get("hybrid") or m.get("encoder_hybrid") or m.get("student_ffn_reuse") or m.get("qwen21_dbcache") or
                m.get("lora_strategy") != "inference_time" or m.get("lora_applied_projections") != 227):
            raise ValueError("full pinned GPU adapter required; no ANE/temporal reuse")
        if m["prompt_cache_hit"] is not conditioning_hit:
            raise ValueError("conditioning lifecycle differs")
        selection = m["acceleration_selection"]
        if ("joint BF16 LoRA A/B operands" in selection) != (actual_precision(precision, i) == "joint"):
            raise ValueError("actual LoRA arithmetic recipe differs")
        if ("edit prefix KV snapshot hit" in selection) != hit:
            raise ValueError(f"actual prefix hit differs at {case}/{label}")
        if snapshot and "edit prefix KV snapshot " not in selection:
            raise ValueError("explicit student prefix route was not admitted")


def child(args):
    sequence = schedule(args.seed, args.child)
    def before(index, request):
        case, label, snapshot, _, _, _ = sequence[index]
        precision = actual_precision(args.child, index)
        os.environ["TURBOCIDER_QWEN21_LORA_BF16_AB"] = "1" if precision == "joint" else "0"
        os.environ["TURBOCIDER_QWEN21_PREFIX_SNAPSHOT"] = str(snapshot)
        if case == "edit" and args.reference_size == 512 and adapter_pin(args.lora)["rank"] == "r256":
            os.environ["TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC"] = "1"
        else:
            os.environ.pop("TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC", None)
        print(json.dumps(dict(index=index, case=case, label=label, precision=precision)), flush=True)
    report = generate_requests(args.library, "qwen-image-2.1", args.model, requests(args), args.output,
                               args.child, before_request=before)
    validate(report, args.seed, args.child)
    report.update(status="complete", library_sha256=sha256_file(args.library))
    write_json(args.output / "report.json", report)


def analyze(output, report, precision="original"):
    import numpy as np
    from PIL import Image
    def pixels(i):
        with Image.open(output / f"{i}.png") as im:
            im.load()
            if im.size != (512, 512):
                raise ValueError("complete512 PNG required")
            return np.asarray(im.convert("RGBA"))
    def compare(a, b):
        x, y = pixels(a), pixels(b)
        return dict(rgba_equal=bool(np.array_equal(x, y)), rgb=image_metrics(x[:, :, :3], y[:, :, :3]),
                    alpha_max_abs=int(np.abs(x[:, :, 3].astype(int) - y[:, :, 3]).max()),
                    candidate_rgba_sha256=hashlib.sha256(y.tobytes()).hexdigest())
    if precision == "switch":
        miss, hit = compare(0, 4), compare(1, 5)
        if not miss["rgba_equal"] or not hit["rgba_equal"]:
            raise ValueError("precision switch contaminated restored original prefix/output")
        return dict(original_miss_restored=miss, original_hit_restored=hit,
                    joint_snapshot_quality=compare(2, 3))
    for a, b in ((1, 3), (5, 10)):
        if not compare(a, b)["rgba_equal"]:
            raise ValueError("return-to-control changed student output")
    wall = lambda i: report["runs"][i]["request_wall_including_export"]
    observations = []
    for label, reference, candidate, controls in (("same-seed", 5, 7, [5, 10]), ("new-seed", 9, 8, [9])):
        baseline = statistics.mean(wall(i) for i in controls)
        observations.append(dict(label=label, baseline_seconds=baseline, candidate_seconds=wall(candidate),
            reduction_percent=100 * (1 - wall(candidate) / baseline), quality=compare(reference, candidate)))
    return dict(generation_seconds=[wall(1), wall(2), wall(3)], edit_observations=observations,
                generate_control_restored=True, edit_control_restored=True)


def parent(args):
    pin = adapter_pin(args.lora)
    digest = sha256_file(args.lora)
    if digest != pin["sha256"]:
        raise ValueError("local adapter does not match its pinned original")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema="tc-qwen-student-prefix-v1", status="running", qualification_passed=False,
        scope="pinned " + pin["rank"] + " six-step student only; original/joint arithmetic separate; no teacher40 denominator or image equivalence; no ANE or temporal cache",
        library_sha256=sha256_file(args.library), adapter_sha256=digest,
        adapter_rank=pin["rank"], adapter_filename=args.lora.name, adapter_bytes=args.lora.stat().st_size,
        reference_sha256=sha256_file(args.reference), reference_size=args.reference_size,
        script_sha256=sha256_file(Path(__file__)), trials=[],
        observation_gates=dict(minimum_prefix_reduction_percent=5., minimum_incremental_ssim=.99,
                              maximum_incremental_mae_255=2., maximum_alpha_abs=2))
    write_json(args.output / "source.json", dict(source=Path(__file__).read_text()))
    write_json(args.output / "summary.json", summary)
    for precision in args.precisions.split(","):
        output = args.output / precision
        output.mkdir()
        observer = LoadObservation(output / "load.jsonl")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", precision,
            "--library", str(args.library), "--model", str(args.model), "--lora", str(args.lora),
            "--reference", str(args.reference), "--reference-size", str(args.reference_size),
            "--output", str(output), "--seed", str(args.seed)]
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            memory = run_sampled(command, repo=ROOT, output=output, stem="trial", env=benchmark_environment(),
                stdout=stdout, stderr=stderr, timeout=1800, interval_ms=100, max_gap_ms=500, observer=observer)
        load, load_error = None, None
        try:
            load = observer.verify()
        except ValueError as error:
            load_error = str(error)
        report = json.loads((output / "report.json").read_text())
        validate(report, args.seed, precision)
        summary["trials"].append(dict(precision=precision, memory=memory, load=load, load_error=load_error,
                                      analysis=analyze(output, report, precision)))
        write_json(args.output / "summary.json", summary)
    summary["status"] = "complete"
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def cross_precision(output):
    """CPU-only original-vs-joint audit; never a teacher40 comparison."""
    import numpy as np
    from PIL import Image
    summary = json.loads((output / "summary.json").read_text())
    if summary["status"] != "complete" or summary["adapter_sha256"] not in {p["sha256"] for p in ADAPTER_PINS.values()}:
        raise ValueError("completed pinned-student cohort required")
    reports = {p: json.loads((output / p / "report.json").read_text()) for p in ("original", "joint")}
    rows = []
    for i in (1, 2, 5, 9):
        requests_pair = [json.loads((output / p / f"request-{i}.json").read_text()) for p in reports]
        if {k: v for k, v in requests_pair[0].items() if k != "output"} != {
                k: v for k, v in requests_pair[1].items() if k != "output"}:
            raise ValueError("precision comparison changed the request workload")
        pixels = []
        for p in reports:
            with Image.open(output / p / f"{i}.png") as im:
                im.load()
                if im.size != (512, 512):
                    raise ValueError("complete student512 image required")
                pixels.append(np.asarray(im.convert("RGBA")))
        a, b = pixels
        r = requests_pair[0]
        rows.append(dict(index=i, operation=r["operation"], seed=r["seed"],
            original_seconds=reports["original"]["runs"][i]["request_wall_including_export"],
            joint_seconds=reports["joint"]["runs"][i]["request_wall_including_export"],
            rgb=image_metrics(a[:, :, :3], b[:, :, :3]),
            alpha_max_abs=int(np.abs(a[:, :, 3].astype(int) - b[:, :, 3]).max())))
    result = dict(schema="tc-qwen-student-precision-audit-v1", qualification_passed=False,
        scope="same pinned six-step adapter/inputs/seed/library; no prefix on compared requests; no teacher40 fidelity claim",
        adapter_sha256=summary["adapter_sha256"],
        source_reports_sha256={p: sha256_file(output / p / "report.json") for p in reports}, rows=rows)
    write_json(output / "precision-comparison.json", result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora", type=Path, default=ADAPTER)
    p.add_argument("--reference", type=Path, default=ROOT / "results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--reference-size", type=int, choices=(512, 1024), default=512)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--precisions", default="original,joint")
    p.add_argument("--child", choices=("original", "joint", "switch"))
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--analyze-only", action="store_true", help="CPU-only original/joint precision audit")
    args = p.parse_args()
    if any(value not in ("original", "joint", "switch") for value in args.precisions.split(",")):
        p.error("known precision families required")
    for key in ("library", "model", "lora", "reference"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    if args.analyze_only:
        cross_precision(args.output)
    else:
        (child if args.child else parent)(args)


if __name__ == "__main__":
    main()
