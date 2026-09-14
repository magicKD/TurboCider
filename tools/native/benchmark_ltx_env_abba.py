#!/usr/bin/env python3
"""ABBA benchmark for an opt-in LTX environment-controlled candidate."""

from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from benchmark_ltx_resident import (
    bf16_metrics,
    consume,
    create_engine,
    load_library,
    peak_rss_bytes,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidate-env", required=True)
    parser.add_argument("--candidate-value", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio", action="store_true")
    parser.add_argument(
        "--prompt", default="A cinematic red fox running through a snowy forest"
    )
    parser.add_argument(
        "--residency",
        choices=("resident", "component_staged", "streamed"),
        default="resident",
    )
    parser.add_argument("--memory-budget-bytes", type=int, default=12 * (1 << 30))
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.width <= 0 or args.height <= 0 or args.frames <= 0:
        parser.error("geometry must be positive")
    if args.steps != 11:
        parser.error("the LTX distilled route requires --steps 11")
    if "=" in args.candidate_env or not args.candidate_env:
        parser.error("--candidate-env must be one environment variable name")
    if args.residency == "streamed" and args.memory_budget_bytes <= 0:
        parser.error("streamed residency requires a positive memory budget")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(rows: list[dict], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def sequence(rounds: int) -> list[str]:
    result: list[str] = []
    for index in range(rounds):
        result.extend(("baseline", "candidate", "candidate", "baseline")
                      if index % 2 == 0
                      else ("candidate", "baseline", "baseline", "candidate"))
    return result


def main() -> int:
    args = arguments()
    cache = Path(args.cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TURBOCIDER_LTX_CONDITIONING_CACHE_DIR"] = str(cache)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = Path(args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)

    library = load_library(Path(args.library))
    engine = create_engine(library, Path(args.model).resolve())
    original = os.environ.get(args.candidate_env)
    rows: list[dict] = []
    report_value = {
        "format": "turbocider-ltx-env-abba-v1",
        "workload": {
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "steps": args.steps,
            "seed": args.seed,
            "prompt": args.prompt,
            "execution": "gpu",
            "residency": args.residency,
            "memory_budget_bytes": (
                args.memory_budget_bytes if args.residency == "streamed" else 0
            ),
            "audio": args.audio,
        },
        "candidate": {
            "environment": args.candidate_env,
            "value": args.candidate_value,
        },
        "method": (
            "one retained native engine; baseline and candidate receive one warmup "
            "request each; measured requests use alternating ABBA/BAAB rounds; "
            "candidate environment is changed only at request graph-construction "
            "boundaries; all requests dump the final Stage-2 BF16 video latent"
        ),
        "warmup_sequence": ["baseline", "candidate"],
        "measured_sequence": sequence(args.rounds),
        "runs": rows,
    }

    def write_report() -> None:
        report.write_text(json.dumps(report_value, indent=2) + "\n")

    def run(variant: str, phase: str, index: int) -> dict:
        if variant == "candidate":
            os.environ[args.candidate_env] = args.candidate_value
        else:
            # The production C/Metal path now defaults FAST_AV on.  Explicitly
            # disable it here so this tool remains a strict historical baseline
            # versus candidate comparison.
            os.environ[args.candidate_env] = "0"
        stem = f"{phase}-{index:02d}-{variant}"
        destination = output / f"{stem}.mp4"
        dump = output / f"{stem}-tensors"
        request = {
            "schema_version": 1,
            "model": "ltx-2.5-distilled",
            "operation": "video.generate",
            "prompt": args.prompt,
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "fps": 24,
            "steps": args.steps,
            "seed": args.seed,
            "execution": "gpu",
            "residency": args.residency,
            "memory_budget_bytes": (
                args.memory_budget_bytes if args.residency == "streamed" else 0
            ),
            "audio": args.audio,
            "dump_tensors": str(dump),
            "output": str(destination),
        }
        value = c.c_void_p()
        error = c.c_void_p()
        started = time.perf_counter()
        status = library.tc_engine_generate(
            engine, json.dumps(request).encode(), None, None,
            c.byref(value), c.byref(error),
        )
        client_wall = time.perf_counter() - started
        parsed = consume(library, value)
        failure = consume(library, error)
        if status:
            raise RuntimeError(failure)
        result = json.loads(parsed)
        timings = result["timings_seconds"]
        latent = dump / "stage2_video.bf16"
        row = {
            "phase": phase,
            "index": index,
            "variant": variant,
            "client_wall_seconds": client_wall,
            "request_wall_seconds": float(timings["request_wall"]),
            "stage1_seconds": float(timings["stage1"]),
            "stage2_seconds": float(timings["stage2"]),
            "denoise_seconds": float(timings["stage1"]) + float(timings["stage2"]),
            "output": str(destination),
            "output_sha256": file_sha256(destination),
            "stage2_video": str(latent),
            "stage2_video_sha256": file_sha256(latent),
            "block_streaming": result.get("block_streaming"),
        }
        print(json.dumps({
            "phase": phase,
            "index": index,
            "variant": variant,
            "request_wall_seconds": row["request_wall_seconds"],
            "denoise_seconds": row["denoise_seconds"],
        }), flush=True)
        return row

    try:
        for index, variant in enumerate(report_value["warmup_sequence"]):
            report_value.setdefault("warmups", []).append(
                run(variant, "warmup", index)
            )
            write_report()
        for index, variant in enumerate(report_value["measured_sequence"]):
            rows.append(run(variant, "measured", index))
            write_report()
    finally:
        library.tc_engine_free(engine)
        if original is None:
            os.environ.pop(args.candidate_env, None)
        else:
            os.environ[args.candidate_env] = original

    baseline = [row for row in rows if row["variant"] == "baseline"]
    candidate = [row for row in rows if row["variant"] == "candidate"]
    reference_latent = Path(baseline[0]["stage2_video"])
    quality = []
    for row in rows:
        metrics = bf16_metrics(reference_latent, Path(row["stage2_video"]))
        metrics.update({"index": row["index"], "variant": row["variant"]})
        quality.append(metrics)
    baseline_denoise = median(baseline, "denoise_seconds")
    candidate_denoise = median(candidate, "denoise_seconds")
    baseline_wall = median(baseline, "request_wall_seconds")
    candidate_wall = median(candidate, "request_wall_seconds")
    report_value["summary"] = {
        "baseline_samples": len(baseline),
        "candidate_samples": len(candidate),
        "baseline_median_request_wall_seconds": baseline_wall,
        "candidate_median_request_wall_seconds": candidate_wall,
        "request_wall_speedup": baseline_wall / candidate_wall,
        "baseline_median_denoise_seconds": baseline_denoise,
        "candidate_median_denoise_seconds": candidate_denoise,
        "denoise_speedup": baseline_denoise / candidate_denoise,
        "candidate_denoise_change_percent": (
            (candidate_denoise / baseline_denoise - 1.0) * 100.0
        ),
        "all_latents_finite": all(item["finite"] for item in quality),
        "all_latents_byte_exact": all(item["byte_exact"] for item in quality),
        "all_media_byte_exact": len({row["output_sha256"] for row in rows}) == 1,
        "process_peak_rss_bytes": peak_rss_bytes(),
        "process_peak_rss_scope": (
            "benchmark worker self; spawned isolated video decoder excluded"
        ),
    }
    report_value["latent_quality"] = quality
    write_report()
    print(json.dumps(report_value["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
