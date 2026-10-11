#!/usr/bin/env python3
"""Complete LTX A/V candidate screen, with persistent versus clean-exec scope."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from flux9_bf16_framework_screen import generation, check_generations, write_json
from ltx_base_benchmark_screen import make_profile
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "gpu-quality-batch": dict(mode="quality", hybrid=False, single=False),
    "gpu-sol-batch": dict(mode="sol", hybrid=False, single=False),
    "gpu-text256-batch": dict(mode="fast_approx", hybrid=False, single=False),
    "int8-quality-batch": dict(mode="quality", hybrid=True, single=False),
    "int8-sol-batch": dict(mode="sol", hybrid=True, single=False),
    "int8-text256-batch": dict(mode="fast_approx", hybrid=True, single=False),
    "gpu-quality-exec": dict(mode="quality", hybrid=False, single=True),
    "int8-quality-exec": dict(mode="quality", hybrid=True, single=True),
}
PROMPT = "A cinematic red fox running through a snowy forest"


def make_request(name, output, profile=None):
    if name not in CASES:
        raise ValueError("known LTX full-media candidate required")
    case = CASES[name]
    result = dict(model="ltx-2.5-distilled", operation="video.generate", prompt=PROMPT,
        width=704, height=448, frames=97, fps=24, steps=11, seed=42, audio=True,
        ltx_backend="c_metal", ltx_fast_av=True, residency="component_staged",
        execution="gpu_ane" if case["hybrid"] else "gpu", allow_approximation=case["hybrid"] or case["mode"] != "quality",
        output=str(output))
    if case["mode"] != "quality":
        result.update(ltx_sol_stage1=False, ltx_sol_stage2=True, ltx_sol_tau=1.0,
                      ltx_sol_dense_edge_blocks=1, ltx_sol_dense_edge_steps=0)
    if case["mode"] == "fast_approx":
        result["ltx_stage2_text_rows"] = 256
    if case["hybrid"]:
        if not profile:
            raise ValueError("existing source-matched INT8 profile required")
        result["ane_manifest"] = str(profile)
    return result


def validate_rows(rows, name):
    if name not in CASES or len(rows) != 2:
        raise ValueError("two actual complete full-media requests required")
    case = CASES[name]
    for index, row in enumerate(rows):
        if (row.get("model"), row.get("operation"), row.get("width"), row.get("height"), row.get("frames"), row.get("audio")) != (
                "ltx-2.5-distilled", "video.generate", 704, 448, 97, True):
            raise ValueError("short clip/non-base geometry cannot serve as LTX performance")
        if row.get("lora_applied_projections", 0):
            raise ValueError("base-only LTX required")
        if row.get("conditioning_cache_hit") is not (index == 1):
            raise ValueError("actual per-arm empty/shared cache cold-hit lifecycle differs")
        if case["hybrid"] and (row.get("execution") != "gpu_ane" or (not case["single"] and not row.get("ane_profile"))):
            raise ValueError("actual configured INT8 artifact route missing")
        if not case["hybrid"] and row.get("execution") != "gpu":
            raise ValueError("GPU control cannot relabel hybrid")
        if case["single"] and row.get("video_vae_isolation") != "exec":
            raise ValueError("clean-exec candidate actually used fallback decoder")
        if not case["single"]:
            if row.get("ltx_sol_stage2", False) is not (case["mode"] != "quality") or row.get("ltx_stage2_text_rows", 0) != (
                    256 if case["mode"] == "fast_approx" else 0):
                raise ValueError("configured Sol/text-row recipe was not returned by native request")


def probe_media(path):
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], text=True))
    video = next(item for item in probe["streams"] if item["codec_type"] == "video")
    audio = next(item for item in probe["streams"] if item["codec_type"] == "audio")
    if (video["width"], video["height"], int(video["nb_frames"]), video["r_frame_rate"], audio["channels"], int(audio["sample_rate"])) != (
            704, 448, 97, "24/1", 2, 48000):
        raise ValueError("complete unresized 97-frame stereo output required")
    return dict(sha256=sha256_file(path), video=video, audio=audio)


def request_seconds(row, single):
    # The clean finalizer reports its own pre-finalizer+decode estimate; it
    # is not the same API boundary as a persistent native request_wall.
    key = "request_wall_estimate" if single else "request_wall"
    value = row["timings_seconds"][key]
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError("positive actual full-media timing required")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cli", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--order", default="gpu-quality-batch,int8-quality-batch,gpu-sol-batch,int8-sol-batch,gpu-text256-batch,int8-text256-batch")
    args = parser.parse_args()
    order = args.order.split(",")
    if args.output.exists() or any(name not in CASES for name in order):
        parser.error("fresh output and explicitly declared serial candidates required")
    cli, model = args.cli.resolve(strict=True), args.model.resolve(strict=True)
    args.output = args.output.absolute()
    args.output.mkdir(parents=True)
    profiles = {}
    if any(CASES[name]["hybrid"] for name in order):
        path, directory = make_profile("int8", args.output)
        profiles["int8"] = (path, directory)
    executables = [cli.parent / name for name in ("turbocider", "ltx-video-vae-decode", "ltx-video-finalizer",
        "ltx-audio-vae-decode", "ltx-vocoder-decode", "ltx-bwe-decode", "ltx-audio-mux", "ltx-gemma-encode")]
    if not all(path.is_file() for path in executables):
        raise ValueError("complete exact bundled helper set required")
    sources = [*model.rglob("*.safetensors"), model / "config.json", model / "gemma4-12b-ltx-v1/tokenizer.json",
               cli.parent / "libturbocider.dylib", *executables, Path(__file__).resolve()]
    for path, directory in profiles.values():
        sources += [path, *directory.rglob("manifest.json")]
    sources = sorted(set(sources))
    identities = {str(path): generation(path) for path in sources}
    # One initial window verification, not a hash before each inference.
    hashes = {str(path): sha256_file(path) for path in sources}
    summary = dict(schema="tc-ltx-complete-config-screen-v1", status="running", qualification_passed=False,
        order=order, trials=[], source_sha256=hashes,
        scope="original local ConvRot base; full704x448x97/24fps/11step/stereo; persistent cold/warm versus separate new-process cache-hit scopes, no LoRA")
    destination = args.output / "summary.json"
    write_json(destination, summary)
    try:
        for arm, name in enumerate(order):
            check_generations(identities)
            case = CASES[name]
            output = args.output / f"{arm}-{name}"
            output.mkdir()
            cache = output / "conditioning-cache"
            cache.mkdir()
            env = benchmark_environment()
            env["TURBOCIDER_LTX_CONDITIONING_CACHE_DIR"] = str(cache)
            paths = []
            for iteration in range(2):
                actual = make_request(name, output / f"{iteration}.mp4", profiles.get("int8", (None,))[0])
                path = output / f"request-{iteration}.json"
                write_json(path, actual)
                paths.append(str(path))
            commands = [[str(cli), "generate", str(model), path] for path in paths] if case["single"] else [[str(cli), "batch", str(model), *paths]]
            rows, memories, parent_walls = [], [], []
            print(json.dumps(dict(starting=arm, candidate=name)), flush=True)
            for iteration, command in enumerate(commands):
                with (output / f"stdout-{iteration}.jsonl").open("x") as out, (output / f"stderr-{iteration}.txt").open("x") as err:
                    start = time.perf_counter()
                    memory = run_sampled(command, repo=ROOT, output=output, stem=f"run-{iteration}", env=env,
                        stdout=out, stderr=err, timeout=1800, interval_ms=100, max_gap_ms=500, async_evidence=True,
                        child_roles=tuple(("ltx_helper_" + path.stem.replace("-", "_"), str(path)) for path in executables))
                    parent_walls.append(time.perf_counter() - start)
                memories.append(memory)
                rows += [json.loads(line) for line in (output / f"stdout-{iteration}.jsonl").read_text().splitlines()]
            validate_rows(rows, name)
            check_generations(identities)
            summary["trials"].append(dict(index=arm, name=name, configured=case, rows=rows, memory=memories,
                parent_wall_including_sampler_and_verification=parent_walls,
                hybrid_evidence_scope="configured execution/profile; clean finalizer forwards execution but not ane_profile or common image call counters; not physical ANE trace",
                native_timing_boundary="request_wall_estimate from clean finalizer" if case["single"] else "native request_wall",
                warm_scope="new process, conditioning cache hit, constructor/model load still counted" if case["single"] else "same process immediate warm/conditioning cache hit",
                media=[probe_media(output / f"{index}.mp4") for index in range(2)]))
            write_json(destination, summary)
        summary["routes"] = []
        for name in sorted(set(order)):
            trials = [trial for trial in summary["trials"] if trial["name"] == name]
            warm = [request_seconds(trial["rows"][1], trial["configured"]["single"]) for trial in trials]
            summary["routes"].append(dict(name=name, warm_seconds=warm, warm_median=statistics.median(warm),
                cold_seconds=[request_seconds(trial["rows"][0], trial["configured"]["single"]) for trial in trials],
                warm_scope=trials[0]["warm_scope"]))
        summary["status"] = "complete_diagnostic"
    except BaseException as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        write_json(destination, summary)


if __name__ == "__main__":
    main()
