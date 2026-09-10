#!/usr/bin/env python3
"""Paired LTX resident/streamed benchmark through retained native engines."""

from __future__ import annotations

import argparse
import ctypes as c
import json
import math
import os
import platform
import resource
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

from video_quality_gate import compare as compare_video
from video_quality_gate import passes as video_passes


def consume(library, pointer):
    if not pointer.value:
        return None
    value = c.string_at(pointer).decode()
    library.tc_string_free(pointer)
    return value


def bf16_metrics(reference: Path, candidate: Path) -> dict:
    reference_bytes = reference.read_bytes()
    candidate_bytes = candidate.read_bytes()
    size_equal = len(reference_bytes) == len(candidate_bytes)
    if not size_equal or not reference_bytes or len(reference_bytes) % 2:
        return {
            "size_equal": size_equal,
            "reference_bytes": len(reference_bytes),
            "candidate_bytes": len(candidate_bytes),
            "elements": 0,
            "byte_exact": False,
            "finite": False,
            "rel_l2": float("inf"),
            "cosine": 0.0,
            "max_abs": float("inf"),
        }
    difference2 = reference2 = candidate2 = dot = 0.0
    max_abs = 0.0
    finite = True
    elements = len(reference_bytes) // 2
    for (left,), (right,) in zip(
        struct.iter_unpack("<H", reference_bytes),
        struct.iter_unpack("<H", candidate_bytes),
    ):
        expected = struct.unpack("<f", struct.pack("<I", left << 16))[0]
        actual = struct.unpack("<f", struct.pack("<I", right << 16))[0]
        if not math.isfinite(expected) or not math.isfinite(actual):
            finite = False
            continue
        difference = actual - expected
        max_abs = max(max_abs, abs(difference))
        difference2 += difference * difference
        reference2 += expected * expected
        candidate2 += actual * actual
        dot += expected * actual
    return {
        "size_equal": True,
        "reference_bytes": len(reference_bytes),
        "candidate_bytes": len(candidate_bytes),
        "elements": elements,
        "byte_exact": reference_bytes == candidate_bytes,
        "finite": finite,
        "rel_l2": math.sqrt(difference2 / reference2) if reference2 else 0.0,
        "cosine": (
            dot / math.sqrt(reference2 * candidate2)
            if reference2 and candidate2 else 1.0
        ),
        "max_abs": max_abs,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt", default="A cinematic red fox running through a snowy forest"
    )
    parser.add_argument(
        "--residency",
        choices=("paired", "resident", "component_staged", "streamed"),
        default="paired",
    )
    parser.add_argument(
        "--execution", choices=("gpu", "gpu_ane"), default="gpu"
    )
    parser.add_argument("--ane-manifest")
    parser.add_argument("--memory-budget-bytes", type=int, default=12 * (1 << 30))
    parser.add_argument("--require-speedup", type=float, default=0.0)
    parser.add_argument("--require-quality", action="store_true")
    parser.add_argument("--require-memory-budget", action="store_true")
    parser.add_argument("--max-latent-rel-l2", type=float, default=1e-4)
    parser.add_argument("--min-latent-cosine", type=float, default=0.99999)
    parser.add_argument("--min-mean-correlation", type=float, default=0.99)
    parser.add_argument("--min-frame-correlation", type=float, default=0.95)
    parser.add_argument("--min-mean-cosine", type=float, default=0.995)
    parser.add_argument("--max-mean-mae-255", type=float, default=5.0)
    parser.add_argument("--max-motion-relative-error", type=float, default=0.15)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.width <= 0 or args.height <= 0 or args.frames <= 0:
        parser.error("geometry must be positive")
    if args.steps != 11:
        parser.error("the LTX distilled route requires --steps 11")
    if args.memory_budget_bytes < 0:
        parser.error("--memory-budget-bytes must be non-negative")
    if args.residency == "streamed" and not args.memory_budget_bytes:
        parser.error("streamed residency requires a non-zero memory budget")
    if args.execution == "gpu_ane" and not args.ane_manifest:
        parser.error("gpu_ane execution requires --ane-manifest")
    if args.execution == "gpu_ane" and args.residency in ("paired", "streamed"):
        parser.error("gpu_ane execution requires resident or component_staged")
    if args.ane_manifest and args.execution != "gpu_ane":
        parser.error("--ane-manifest requires --execution gpu_ane")
    if args.require_speedup < 0:
        parser.error("--require-speedup must be non-negative")
    if (args.require_speedup or args.require_quality or
            args.require_memory_budget) and args.residency != "paired":
        parser.error("quality, speed, and memory gates require --residency paired")
    if args.max_latent_rel_l2 < 0 or not 0.0 <= args.min_latent_cosine <= 1.0:
        parser.error("latent quality thresholds are out of range")
    return args


def load_library(path: Path):
    library = c.CDLL(str(path.resolve()))
    library.tc_string_free.argtypes = [c.c_void_p]
    library.tc_engine_create_model_candidate.argtypes = [
        c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)
    ]
    library.tc_engine_generate.argtypes = [
        c.c_void_p, c.c_char_p, c.c_void_p, c.c_void_p,
        c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)
    ]
    library.tc_engine_free.argtypes = [c.c_void_p]
    return library


def create_engine(library, model: Path):
    engine = c.c_void_p()
    error = c.c_void_p()
    status = library.tc_engine_create_model_candidate(
        b"ltx-2.5-distilled", str(model.resolve()).encode(),
        c.byref(engine), c.byref(error),
    )
    failure = consume(library, error)
    if status:
        raise RuntimeError(failure)
    return engine


def warm_median(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    warm = values[1:] if len(values) > 1 else values
    return statistics.median(warm)


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def run_mode(library, model: Path, output: Path, report: Path,
             report_value: dict, args: argparse.Namespace,
             residency: str) -> dict:
    directory = output / residency
    directory.mkdir(parents=True, exist_ok=True)
    engine = create_engine(library, model)
    rows = []
    try:
        for index in range(args.runs):
            destination = directory / f"request-{index}.mp4"
            dump = directory / f"request-{index}-tensors"
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
                "execution": args.execution,
                "residency": residency,
                "memory_budget_bytes": (
                    args.memory_budget_bytes if residency == "streamed" else 0
                ),
                "audio": False,
                "dump_tensors": str(dump.resolve()),
                "output": str(destination.resolve()),
            }
            if args.execution == "gpu_ane":
                request["allow_approximation"] = True
                request["ane_manifest"] = str(
                    Path(args.ane_manifest).resolve()
                )
            result = c.c_void_p()
            error = c.c_void_p()
            started = time.perf_counter()
            status = library.tc_engine_generate(
                engine, json.dumps(request).encode(), None, None,
                c.byref(result), c.byref(error),
            )
            client_wall = time.perf_counter() - started
            value = consume(library, result)
            failure = consume(library, error)
            if status:
                raise RuntimeError(failure)
            parsed = json.loads(value)
            timings = parsed["timings_seconds"]
            row = {
                "run": index,
                "client_wall_seconds": client_wall,
                "request_wall_seconds": float(timings["request_wall"]),
                "denoise_seconds": float(timings["stage1"]) + float(timings["stage2"]),
                "output": str(destination.resolve()),
                "stage2_video": str((dump / "stage2_video.bf16").resolve()),
                "output_bytes": destination.stat().st_size,
                "result": parsed,
            }
            rows.append(row)
            report_value["modes"][residency] = {"runs": rows}
            report.write_text(json.dumps(report_value, indent=2) + "\n")
            print(json.dumps({
                "residency": residency,
                "run": index,
                "request_wall_seconds": row["request_wall_seconds"],
                "denoise_seconds": row["denoise_seconds"],
                "block_streaming": parsed.get("block_streaming"),
            }), flush=True)
    finally:
        library.tc_engine_free(engine)
    return {
        "runs": rows,
        "first_request_seconds": rows[0]["request_wall_seconds"],
        "warm_median_seconds": warm_median(rows, "request_wall_seconds"),
        "warm_denoise_median_seconds": warm_median(rows, "denoise_seconds"),
        "process_peak_rss_bytes": peak_rss_bytes(),
        "process_peak_rss_scope": (
            "benchmark worker self; spawned isolated video decoder excluded"
        ),
    }


def compare_modes(args: argparse.Namespace, resident: dict, streamed: dict) -> dict:
    latent_pairs = []
    video_pairs = []
    for reference, candidate in zip(resident["runs"], streamed["runs"]):
        latent = bf16_metrics(
            Path(reference["stage2_video"]), Path(candidate["stage2_video"])
        )
        latent["run"] = reference["run"]
        latent_pairs.append(latent)
        video = compare_video(
            Path(reference["output"]), Path(candidate["output"]),
            ffmpeg=args.ffmpeg or "ffmpeg", ffprobe=args.ffprobe or "ffprobe",
        )
        video["run"] = reference["run"]
        video_pairs.append(video)
    latent_quality = all(
        item["size_equal"] and item["finite"] and
        float(item["rel_l2"]) <= args.max_latent_rel_l2 and
        float(item["cosine"]) >= args.min_latent_cosine
        for item in latent_pairs
    )
    video_quality = all(
        video_passes(
            item,
            min_mean_correlation=args.min_mean_correlation,
            min_frame_correlation=args.min_frame_correlation,
            min_mean_cosine=args.min_mean_cosine,
            max_mean_mae_255=args.max_mean_mae_255,
            max_motion_relative_error=args.max_motion_relative_error,
        )
        for item in video_pairs
    )
    streaming_info = streamed["runs"][-1]["result"].get("block_streaming", {})
    denoiser_working_set = int(streaming_info.get("estimated_working_set_bytes", 0))
    denoiser_budget = int(streaming_info.get("memory_budget_bytes", 0))
    streamed_peak_rss = int(streamed.get("process_peak_rss_bytes", 0))
    return {
        "first_request_speedup": (
            resident["first_request_seconds"] / streamed["first_request_seconds"]
        ),
        "warm_request_speedup": (
            resident["warm_median_seconds"] / streamed["warm_median_seconds"]
        ),
        "warm_denoise_speedup": (
            resident["warm_denoise_median_seconds"] /
            streamed["warm_denoise_median_seconds"]
        ),
        "latent_pairs": latent_pairs,
        "video_pairs": video_pairs,
        "quality_thresholds": {
            "maximum_latent_rel_l2": args.max_latent_rel_l2,
            "minimum_latent_cosine": args.min_latent_cosine,
            "minimum_mean_rgb_correlation": args.min_mean_correlation,
            "minimum_frame_rgb_correlation": args.min_frame_correlation,
            "minimum_mean_rgb_cosine": args.min_mean_cosine,
            "maximum_mean_rgb_mae_255": args.max_mean_mae_255,
            "maximum_motion_relative_error": args.max_motion_relative_error,
        },
        "latent_quality_passed": latent_quality,
        "video_quality_passed": video_quality,
        "quality_gate_passed": latent_quality and video_quality,
        "memory": {
            "budget_scope": streamed["runs"][-1]["result"]["plan"].get(
                "memory_budget_scope"
            ),
            "denoiser_budget_bytes": denoiser_budget,
            "estimated_denoiser_working_set_bytes": denoiser_working_set,
            "denoiser_budget_gate_passed": bool(
                denoiser_budget and denoiser_working_set and
                denoiser_working_set <= denoiser_budget
            ),
            "resident_process_peak_rss_bytes": int(
                resident.get("process_peak_rss_bytes", 0)
            ),
            "streamed_process_peak_rss_bytes": streamed_peak_rss,
            "process_peak_rss_scope": streamed.get("process_peak_rss_scope"),
            "streamed_process_peak_to_denoiser_budget_ratio": (
                streamed_peak_rss / denoiser_budget if denoiser_budget else None
            ),
        },
    }


def run_paired_workers(args: argparse.Namespace, output: Path,
                       report: Path, report_value: dict) -> None:
    script = Path(__file__).resolve()
    for residency in ("resident", "streamed"):
        mode_report = output / f"{residency}-report.json"
        command = [
            sys.executable, str(script),
            "--library", str(Path(args.library).resolve()),
            "--model", str(Path(args.model).resolve()),
            "--cache", str((Path(args.cache).resolve() / residency)),
            "--output", str(output),
            "--report", str(mode_report),
            "--runs", str(args.runs),
            "--width", str(args.width),
            "--height", str(args.height),
            "--frames", str(args.frames),
            "--steps", str(args.steps),
            "--seed", str(args.seed),
            "--prompt", args.prompt,
            "--residency", residency,
            "--memory-budget-bytes", str(args.memory_budget_bytes),
        ]
        completed = subprocess.run(command)
        if completed.returncode:
            raise RuntimeError(
                f"{residency} benchmark worker failed with status "
                f"{completed.returncode}"
            )
        mode_value = json.loads(mode_report.read_text())
        report_value["modes"][residency] = mode_value["modes"][residency]
        report.write_text(json.dumps(report_value, indent=2) + "\n")


def main() -> int:
    args = arguments()
    cache = Path(args.cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TURBOCIDER_LTX_CONDITIONING_CACHE_DIR"] = str(cache)
    output = Path(args.output).resolve()
    report = Path(args.report).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    modes = (
        ("resident", "streamed") if args.residency == "paired"
        else (args.residency,)
    )
    report_value = {
        "format": "turbocider-ltx-residency-c-abi-benchmark-v2",
        "workload": {
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "steps": args.steps,
            "seed": args.seed,
            "prompt": args.prompt,
            "execution": args.execution,
        },
        "memory_budget_bytes": args.memory_budget_bytes,
        "method": (
            "one clean worker process and retained engine per residency; first "
            "request excluded from warm medians; per-process peak RSS recorded; "
            "stage2 BF16 latent and decoded RGB compared by run index"
        ),
        "modes": {},
    }
    if args.residency == "paired":
        run_paired_workers(args, output, report, report_value)
    else:
        library = load_library(Path(args.library))
        model = Path(args.model).resolve()
        for residency in modes:
            report_value["modes"][residency] = run_mode(
                library, model, output, report, report_value, args, residency
            )
    if "resident" in report_value["modes"] and "streamed" in report_value["modes"]:
        report_value["comparison"] = compare_modes(
            args, report_value["modes"]["resident"],
            report_value["modes"]["streamed"],
        )
        report_value["comparison"]["required_warm_speedup"] = args.require_speedup
        report_value["comparison"]["speed_gate_passed"] = (
            not args.require_speedup or
            report_value["comparison"]["warm_request_speedup"] >=
            args.require_speedup
        )
    report.write_text(json.dumps(report_value, indent=2) + "\n")
    print(json.dumps(report_value.get("comparison", {}), indent=2), flush=True)
    comparison = report_value.get("comparison", {})
    if args.require_speedup and not comparison.get("speed_gate_passed", False):
        return 2
    if args.require_quality and not comparison.get("quality_gate_passed", False):
        return 3
    if (args.require_memory_budget and
            not comparison.get("memory", {}).get(
                "denoiser_budget_gate_passed", False)):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
