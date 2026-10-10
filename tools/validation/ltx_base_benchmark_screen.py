#!/usr/bin/env python3
"""Current local LTX base GPU/INT8/FP16 full-media serial screen."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from flux9_bf16_framework_screen import generation, check_generations
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_profile(mode, output):
    profile = json.loads((ROOT / "profiles/ltx-ane-local-20260913.json").read_text())
    variant = "int8" if mode == "int8" else "fp16"
    profile["variant"] = "int8_pc" if mode == "int8" else "fp16"
    directory = ROOT / f"models/coreml/ltx_mlp_a6912_dual_{variant}_20260913"
    if not directory.is_dir():
        raise ValueError("matching local frozen LTX graph directory missing")
    profile["mlp_stage1"] = profile["mlp_stage2"] = str(directory)
    path = output / f"{mode}-profile.json"
    write_json(path, profile)
    return path, directory


def validate_rows(rows, mode):
    if len(rows) != 2:
        raise ValueError("cold and immediate same-prompt warm full-media requests required")
    for row in rows:
        if (row.get("model") != "ltx-2.5-distilled" or row.get("operation") != "video.generate" or
                row.get("width") != 704 or row.get("height") != 448 or row.get("frames") != 97 or
                row.get("steps", 11) != 11 or row.get("seed", 42) != 42 or row.get("audio") is not True):
            raise ValueError("wrong actual LTX base task")
        if row.get("lora_applied_projections", 0) != 0:
            raise ValueError("base-only LTX required")
        if mode != "gpu" and (row.get("execution") != "gpu_ane" or not row.get("ane_profile")):
            raise ValueError("LTX configured artifact route evidence missing; no invented common image-hybrid counters")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", default="gpu,int8,fp16")
    args = parser.parse_args()
    order = args.order.split(",")
    if args.output.exists() or not order or any(mode not in ("gpu", "int8", "fp16") for mode in order):
        parser.error("fresh output and explicit serial GPU/INT8/FP16 modes required")
    cli, model = args.cli.resolve(strict=True), args.model.resolve(strict=True)
    args.output = args.output.absolute()
    args.output.mkdir(parents=True)
    profiles = {mode: make_profile(mode, args.output) for mode in set(order) if mode != "gpu"}
    vae_helper = cli.parent / "ltx-video-vae-decode"
    if not vae_helper.is_file():
        raise ValueError("exact bundled LTX decoder executable missing")
    sources = [cli, cli.parent / "libturbocider.dylib", vae_helper, *model.rglob("*.safetensors"),
               model / "config.json", model / "gemma4-12b-ltx-v1/tokenizer.json"]
    for path, directory in profiles.values():
        sources += [path, *directory.rglob("manifest.json")]
    sources = sorted(set(sources))
    identities = {str(path): generation(path) for path in sources}
    hashes = {str(path): sha256_file(path) for path in sources}
    summary = dict(schema="tc-ltx-base-full-media-screen-v1", status="running", qualification_passed=False,
                   date="2026-10-11", order=order, source_sha256=hashes, trials=[],
                   scope="704x448x97/11steps/seed42/audio; component_staged C/Metal; independent process per route; cold/warm split; no LoRA")
    destination = args.output / "summary.json"
    write_json(destination, summary)
    try:
        for index, mode in enumerate(order):
            check_generations(identities)
            output = args.output / f"{index}-{mode}"
            output.mkdir()
            env = benchmark_environment()
            cache = output / "conditioning-cache"
            cache.mkdir()
            env["TURBOCIDER_LTX_CONDITIONING_CACHE_DIR"] = str(cache)
            paths = []
            for iteration in range(2):
                request = dict(model="ltx-2.5-distilled", operation="video.generate",
                    prompt="A cinematic red fox running through a snowy forest", width=704, height=448,
                    frames=97, fps=24, steps=11, seed=42, audio=True, ltx_backend="c_metal",
                    ltx_fast_av=True, residency="component_staged", execution="gpu" if mode == "gpu" else "gpu_ane",
                    allow_approximation=mode != "gpu", output=str(output / f"{iteration}.mp4"))
                if mode != "gpu":
                    request["ane_manifest"] = str(profiles[mode][0])
                path = output / f"request-{iteration}.json"
                write_json(path, request)
                paths.append(str(path))
            print(json.dumps(dict(starting=index, mode=mode)), flush=True)
            with (output / "stdout.jsonl").open("x") as out, (output / "stderr.txt").open("x") as err:
                memory = run_sampled([str(cli), "batch", str(model), *paths], repo=ROOT, output=output,
                    stem="run", env=env, stdout=out, stderr=err, timeout=1800, interval_ms=100, max_gap_ms=500,
                    child_roles=(("ltx_native_helper", str(cli)), ("ltx_video_decoder", str(vae_helper.resolve())),))
            rows = [json.loads(line) for line in (output / "stdout.jsonl").read_text().splitlines()]
            validate_rows(rows, mode)
            check_generations(identities)
            media = []
            for iteration in range(2):
                path = output / f"{iteration}.mp4"
                probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
                    "-show_format", "-of", "json", str(path)], text=True))
                video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
                audio = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
                if (video["width"], video["height"], int(video["nb_frames"])) != (704, 448, 97):
                    raise ValueError("full-sized complete LTX video output required")
                media.append(dict(sha256=sha256_file(path), video=video, audio=audio, format=probe["format"]))
            summary["trials"].append(dict(index=index, mode=mode, rows=rows, memory=memory, media=media))
            write_json(destination, summary)
        summary["status"] = "complete_diagnostic"
    except BaseException as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        write_json(destination, summary)


if __name__ == "__main__":
    main()
