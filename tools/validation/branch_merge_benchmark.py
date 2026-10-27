#!/usr/bin/env python3
"""Serial same-source before/after merge screen; no model work on import."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from flux9_bf16_framework_screen import PROMPTS, check_generations, generation, write_json
from image_framework_native import generate_requests
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled
import qwen21_mflux_framework_screen as qwen
import z_mflux_framework_screen as zimage

ROOT = Path(__file__).resolve().parents[2]
QWEN_MODEL = ROOT / "models/Comfy-Org-Qwen-Image-2.1"
QWEN_MANIFEST = ROOT / "outputs/benchmark-best-configs/qwen21-frozen-w6144-r1024-cache-20261011/manifest-99f1b3499b89b96f5d04a23cb91c3f3496083e1d66b09c6dd2e4500d0d6ca509.json"
ORDER = (("before", "native-gpu"), ("after", "native-gpu"),
         ("after", "native-frozen"), ("before", "native-frozen"),
         ("before", "native-frozen"), ("after", "native-frozen"),
         ("after", "native-gpu"), ("before", "native-gpu"))


def child(args):
    model = QWEN_MODEL if args.model == "qwen" else zimage.COMMON
    model_id = "qwen-image-2.1" if args.model == "qwen" else "z-image-turbo"
    if args.child == "cache":
        values = []
        prompts = (PROMPTS[1], "A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.")
        for prompt in prompts:
            for mode in ("off", "off", "conservative", "balanced", "fast", "off"):
                values.append(dict(model=model_id, operation="image.generate", prompt=prompt,
                    width=512, height=512, steps=40, seed=29, audio=False, execution="gpu",
                    residency="resident", allow_approximation=True, qwen21_dit_cache=mode,
                    output=str(args.output / f"{len(values)}.png")))
    else:
        values = (qwen.requests(args.output, 40, 3, args.child, QWEN_MANIFEST) if args.model == "qwen"
                  else zimage.requests(args.output, args.child, 8, 3))
    report = generate_requests(args.library, model_id, model, values, args.output, args.child)
    if args.child != "cache":
        (qwen.validate_runs if args.model == "qwen" else zimage.validate_runs)(
            report, args.child, 40 if args.model == "qwen" else 8, 3)
        if args.model == "z" and args.child == "native-frozen":
            for index, row in enumerate(report["runs"]):
                hybrid = row["metrics"]["hybrid"]
                if (hybrid.get("runtime_calls_session_total") != (index + 1) * 8 * 32 or
                        hybrid.get("checkpoint_sha256_verified") is not True):
                    raise ValueError("source-verified complete frozen Z work required")
    else:
        for index, row in enumerate(report["runs"]):
            metrics = row["metrics"]
            if (metrics["actual_denoise_steps"] != 40 or metrics.get("hybrid") or
                    metrics["prompt_cache_hit"] is not (index % 6 != 0)):
                raise ValueError("cache study requires actual complete GPU steps and warm conditioning")
            receipt = metrics.get("qwen21_dbcache", {})
            if values[index]["qwen21_dit_cache"] != "off" and receipt.get("cached_steps", 0) <= 0:
                raise ValueError("candidate performed no residual reuse")


def png_digests(output, count):
    from PIL import Image
    values = []
    for index in range(count):
        path = output / f"{index}.png"
        with Image.open(path) as image:
            image.load()
            if image.size != (512, 512):
                raise ValueError("complete 512px PNG required")
            rgba = hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()
        values.append(dict(file_sha256=sha256_file(path), rgba_sha256=rgba))
    return values


def aggregate(trials):
    result = {}
    for model in sorted({trial["model"] for trial in trials}):
        for route in ("native-gpu", "native-frozen"):
            groups = {}
            for branch in ("before", "after"):
                matches = [trial for trial in trials if (trial["model"], trial["branch"], trial["route"]) == (model, branch, route)]
                if not matches:
                    continue
                samples = [row["request_wall_including_export"] for trial in matches for row in trial["report"]["runs"][1:3]]
                groups[branch] = dict(fresh_warm_samples=samples, median_seconds=statistics.median(samples),
                    cold_seconds=[trial["report"]["runs"][0]["request_wall_including_export"] for trial in matches],
                    denoise_median_seconds=statistics.median(row["metrics"]["timings_seconds"]["denoise"]
                        for trial in matches for row in trial["report"]["runs"][1:3]),
                    process_tree_peak_bytes=[trial["memory"]["verified"]["tree_peak_phys_footprint_bytes"] for trial in matches],
                    swap_out_bytes=[trial["memory"]["verified"]["swap_out_bytes"] for trial in matches])
            if len(groups) == 2:
                old, new = groups["before"]["median_seconds"], groups["after"]["median_seconds"]
                groups.update(speedup=old / new, elapsed_reduction_percent=(old - new) / old * 100)
                pixels = [trial["pngs"] for trial in trials
                          if (trial["model"], trial["route"]) == (model, route)]
                groups["all_before_after_rgba_exact"] = all(
                    [item["rgba_sha256"] for item in trial] == [item["rgba_sha256"] for item in pixels[0]]
                    for trial in pixels)
            result[model + "/" + route] = groups
    return result


def parent(args):
    args.output.mkdir(parents=True, exist_ok=False)
    libraries = {name: getattr(args, name).resolve(strict=True) for name in ("before", "after")}
    evidence = ("docs/design/validation/qwen21-valid-framework-current-20261011.json",
                "docs/design/validation/z-valid-framework-current-20261011.json")
    sources = {ROOT / name for item in evidence for name in json.loads((ROOT / item).read_text())["model_sha256"]}
    sources.update((QWEN_MANIFEST, zimage.FROZEN, *libraries.values()))
    sources.update(path.parent / "runtime-build/runtime-build-manifest.json" for path in libraries.values())
    identities = {str(path): generation(path) for path in sources}
    summary = dict(schema="tc-branch-merge-screen-v1", status="running", trials=[],
        started_at=datetime.now().astimezone().isoformat(),
        host={"cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
              "physical_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
              "macos": subprocess.check_output(["sw_vers"], text=True).strip(),
              "power": subprocess.check_output(["pmset", "-g", "batt"], text=True).strip()},
        scope="M4 Max/64GiB; same BF16 sources, full PNG wall; 512px/40 Qwen or 8 Z steps; cold + two fresh prompts per process; per-route ABBA branch order; no repeat-prompt sample; optional W8A8 remains approximate",
        library_sha256={name: sha256_file(path) for name, path in libraries.items()},
        source_generations=identities, script_sha256=sha256_file(Path(__file__)),
        build_manifests={name: json.loads((path.parent / "runtime-build/runtime-build-manifest.json").read_text()) for name, path in libraries.items()},
        old_evidence=list(evidence), summary={})
    target = args.output / "summary.json"
    write_json(target, summary)
    try:
        for model in args.models.split(","):
            for branch, route in ORDER:
                check_generations(identities)
                index = len(summary["trials"])
                output = args.output / f"{index:02}-{model}-{branch}-{route}"
                output.mkdir()
                env = zimage.route_environment(route) if model == "z" else benchmark_environment()
                env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", route,
                           "--model", model, "--library", str(libraries[branch]), "--output", str(output)]
                print(json.dumps(dict(starting=index, model=model, branch=branch, route=route)), flush=True)
                with (output / "stdout.txt").open("x") as out, (output / "stderr.txt").open("x") as err:
                    memory = run_sampled(command, repo=ROOT, output=output, stem="run", env=env,
                        stdout=out, stderr=err, timeout=1800, interval_ms=100, max_gap_ms=500,
                        async_evidence=True, child_roles=(("python_helper", str(Path(sys.executable).resolve(strict=True))),))
                check_generations(identities)
                report = json.loads((output / "report.json").read_text())
                summary["trials"].append(dict(index=index, model=model, branch=branch, route=route,
                    report=report, memory=memory, pngs=png_digests(output, 3), directory=output.name))
                summary["summary"] = aggregate(summary["trials"])
                write_json(target, summary)
                print(json.dumps(dict(finished=index, seconds=[row["request_wall_including_export"] for row in report["runs"]])), flush=True)
        summary["status"] = "complete_diagnostic"
        summary["finished_at"] = datetime.now().astimezone().isoformat()
    except BaseException as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        write_json(target, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default="z,qwen")
    parser.add_argument("--child", choices=("native-gpu", "native-frozen", "cache"))
    parser.add_argument("--model", choices=("z", "qwen"))
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    args.output = args.output.absolute()
    if args.child:
        if not args.model or not args.library or (args.child == "cache" and args.model != "qwen"):
            parser.error("child requires a model/library; cache requires Qwen")
        args.output.mkdir(parents=True, exist_ok=True)
        child(args)
    else:
        if not args.before or not args.after or any(model not in ("z", "qwen") for model in args.models.split(",")):
            parser.error("both libraries and known models required")
        parent(args)


if __name__ == "__main__":
    main()
