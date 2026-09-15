#!/usr/bin/env python3
"""Qualify H3 Qwen3-VL encoder SDPA and optional GPU/ANE prefill.

Each candidate runs in a separate resident process using identical deterministic
token IDs.  The first encoder pass is reported separately from the warm median,
and the final 50-layer hidden states are compared after timing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path


DEFAULT_TOKENS = (64, 128, 256, 512)
INVALID_METRIC = 1e30


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", type=Path,
        default=Path("build/native/h3-mlx-encoder-benchmark-probe"),
    )
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--encoder-ane-manifest", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-speedup", type=float, default=1.1)
    parser.add_argument(
        "--max-warm-cv", type=float, default=0.25,
        help="maximum warm coefficient of variation for both compared paths",
    )
    parser.add_argument(
        "--min-warm-samples", type=int, default=3,
        help="minimum warm samples required for a retention decision",
    )
    parser.add_argument(
        "--warmup-discard", type=int, default=1,
        help="post-first-pass settling samples excluded from warm statistics",
    )
    parser.add_argument("--max-relative-l2", type=float, default=0.025)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-relative-abs", type=float, default=0.05)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate inputs and print commands without running Metal",
    )
    args = parser.parse_args()
    if args.runs < 1 or args.runs > 50:
        parser.error("--runs must be in 1...50")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if (not args.tokens or any(value < 1 or value > 512 for value in args.tokens)
            or len(set(args.tokens)) != len(args.tokens)):
        parser.error("--tokens must be unique values in 1...512")
    if not math.isfinite(args.min_speedup) or args.min_speedup <= 0:
        parser.error("--min-speedup must be finite and positive")
    if not math.isfinite(args.max_warm_cv) or args.max_warm_cv < 0:
        parser.error("--max-warm-cv must be finite and nonnegative")
    if args.warmup_discard < 0 or args.warmup_discard >= args.runs:
        parser.error("--warmup-discard must be in 0...runs-1")
    if (args.min_warm_samples < 1 or args.min_warm_samples > 50 or
            args.runs - args.warmup_discard < args.min_warm_samples):
        parser.error("--min-warm-samples must fit after --warmup-discard")
    if not math.isfinite(args.max_relative_l2) or args.max_relative_l2 < 0:
        parser.error("--max-relative-l2 must be finite and nonnegative")
    if not math.isfinite(args.min_cosine) or not -1 <= args.min_cosine <= 1:
        parser.error("--min-cosine must be in [-1, 1]")
    if not math.isfinite(args.max_relative_abs) or args.max_relative_abs < 0:
        parser.error("--max-relative-abs must be finite and nonnegative")
    return args


def compiled_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    shape = manifest.get("shape")
    artifacts = manifest.get("artifacts")
    if manifest.get("schema_version") != 2 or not isinstance(shape, dict):
        raise ValueError("H3 encoder manifest must use schema_version 2")
    if (shape.get("K") != 5120 or shape.get("N") != 5120 or
            shape.get("mlp_width") != 25600 or
            shape.get("ane_mlp_start") != 0 or
            type(shape.get("ane_mlp_end")) is not int or
            not 0 < shape["ane_mlp_end"] < 25600):
        raise ValueError("H3 encoder manifest geometry is incompatible")
    buckets = shape.get("buckets")
    if (not isinstance(buckets, list) or not buckets or
            any(type(bucket) is not int or bucket < 1 or bucket > 8192
                for bucket in buckets) or buckets != sorted(set(buckets))):
        raise ValueError("H3 encoder manifest has invalid sequence buckets")
    if not isinstance(artifacts, dict):
        raise ValueError("H3 encoder manifest is missing artifacts")
    if not 1 <= len(artifacts) <= 50:
        raise ValueError("H3 encoder manifest must contain 1...50 prefix blocks")
    required = {str(index) for index in range(len(artifacts))}
    if set(artifacts) != required:
        raise ValueError("H3 encoder manifest blocks must be the contiguous 0-based prefix")
    for block in required:
        variants = artifacts[block]
        if not isinstance(variants, dict) or not variants:
            raise ValueError(f"H3 encoder block {block} has no artifact")
        for relative in variants.values():
            artifact = path.parent / relative
            if artifact.suffix != ".mlmodelc" or not artifact.is_dir():
                raise ValueError(
                    "--encoder-ane-manifest must contain compiled .mlmodelc artifacts"
                )
    return manifest


def command_for(args: argparse.Namespace, tokens: int, folder: Path,
                hybrid: bool) -> list[str]:
    command = [
        str(args.probe), str(args.text_encoder), str(args.tokenizer),
        str(tokens), str(args.runs), str(folder / "conditioning.safetensors"),
    ]
    if hybrid:
        command.append(str(args.encoder_ane_manifest))
    return command


def run_probe(command: list[str], folder: Path, timeout: int,
              fused: bool, warmup_discard: int) -> dict:
    folder.mkdir(parents=True)
    environment = os.environ.copy()
    for name in (
        "TURBOCIDER_H3_FUSED_SDPA", "TURBOCIDER_H3_DISABLE_FUSED_SDPA",
        "TURBOCIDER_H3_FUSED_SDPA_MIN_TOKENS",
        "TURBOCIDER_H3_FUSED_SDPA_VALIDATE",
        "TURBOCIDER_H3_QWEN3_HYBRID_VALIDATE",
    ):
        environment.pop(name, None)
    if fused:
        environment["TURBOCIDER_H3_FUSED_SDPA"] = "1"
        environment["TURBOCIDER_H3_FUSED_SDPA_MIN_TOKENS"] = "1"
    started = time.perf_counter()
    process = subprocess.run(
        ["/usr/bin/time", "-l", *command], text=True, capture_output=True,
        env=environment, timeout=timeout, check=False,
    )
    wall_seconds = time.perf_counter() - started
    (folder / "stdout.log").write_text(process.stdout)
    (folder / "stderr.log").write_text(process.stderr)
    peak = re.search(r"(\d+)\s+maximum resident set size", process.stderr)
    process_record = {
        "command": command,
        "returncode": process.returncode,
        "process_wall_seconds": wall_seconds,
        "process_peak_rss_bytes": int(peak.group(1)) if peak else None,
    }
    (folder / "process.json").write_text(
        json.dumps(process_record, indent=2, sort_keys=True) + "\n"
    )
    if process.returncode:
        raise RuntimeError(
            f"H3 encoder probe failed in {folder}:\n{process.stderr[-4000:]}"
        )
    metrics = json.loads(process.stdout)
    samples = metrics.get("samples", [])
    if (len(samples) != metrics.get("runs", 0) + 1 or not samples or
            not samples[0].get("warmup") or
            any(sample.get("warmup") for sample in samples[1:])):
        raise ValueError("probe did not return one first pass and N warm passes")
    all_warm = [sample["seconds"] for sample in samples[1:]]
    warm = all_warm[warmup_discard:]
    if not warm:
        raise ValueError("probe has no warm samples after settling discard")
    return {
        **process_record,
        "probe": metrics,
        "first_encoder_seconds": samples[0]["seconds"],
        "discarded_warm_seconds": all_warm[:warmup_discard],
        "warm_seconds": warm,
        "warm_median_seconds": statistics.median(warm),
        "warm_cv": statistics.pstdev(warm) / statistics.mean(warm),
    }


def tensor_quality(candidate_path: Path, reference_path: Path) -> dict:
    import mlx.core as mx

    candidate = mx.load(str(candidate_path))["hidden_states"].astype(mx.float32)
    reference = mx.load(str(reference_path))["hidden_states"].astype(mx.float32)
    if candidate.shape != reference.shape:
        return {
            "shape_match": False,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
            "relative_l2": INVALID_METRIC,
            "cosine": -1.0,
            "max_abs": INVALID_METRIC,
            "relative_max_abs": INVALID_METRIC,
        }
    delta = candidate - reference
    scale = mx.maximum(
        mx.maximum(mx.max(mx.abs(candidate)), mx.max(mx.abs(reference))),
        mx.array(1.0, dtype=mx.float32),
    )
    scaled_candidate = candidate / scale
    scaled_reference = reference / scale
    scaled_delta = delta / scale
    delta2 = mx.sum(scaled_delta * scaled_delta)
    candidate2 = mx.sum(scaled_candidate * scaled_candidate)
    reference2 = mx.sum(scaled_reference * scaled_reference)
    dot = mx.sum(scaled_candidate * scaled_reference)
    max_abs = mx.max(mx.abs(delta))
    reference_max = mx.max(mx.abs(reference))
    finite = mx.logical_and(
        mx.all(mx.isfinite(candidate)), mx.all(mx.isfinite(reference))
    )
    mx.eval(delta2, candidate2, reference2, dot, max_abs, reference_max, finite)
    if not bool(finite.item()):
        return {
            "shape_match": True, "all_finite": False,
            "relative_l2": INVALID_METRIC, "cosine": -1.0,
            "max_abs": INVALID_METRIC, "relative_max_abs": INVALID_METRIC,
        }
    delta_energy = float(delta2.item())
    candidate_energy = float(candidate2.item())
    reference_energy = float(reference2.item())
    absolute = float(max_abs.item())
    amplitude = float(reference_max.item())
    denominator = math.sqrt(candidate_energy * reference_energy)
    return {
        "shape_match": True,
        "all_finite": True,
        "relative_l2": math.sqrt(delta_energy / reference_energy)
        if reference_energy > 0 else (0.0 if delta_energy == 0 else INVALID_METRIC),
        "cosine": max(-1.0, min(1.0, float(dot.item()) / denominator
        if denominator > 0 else (1.0 if candidate_energy == reference_energy == 0
                                 else 0.0))),
        "max_abs": absolute,
        "relative_max_abs": absolute / amplitude
        if amplitude > 0 else (0.0 if absolute == 0 else INVALID_METRIC),
    }


def compact(result: dict) -> dict:
    samples = result["probe"]["samples"]
    record = {
        "load_seconds": result["probe"]["load_seconds"],
        "first_encoder_seconds": result["first_encoder_seconds"],
        "discarded_warm_seconds": result["discarded_warm_seconds"],
        "warm_seconds": result["warm_seconds"],
        "warm_median_seconds": result["warm_median_seconds"],
        "warm_cv": result["warm_cv"],
        "process_wall_seconds": result["process_wall_seconds"],
        "process_peak_rss_bytes": result["process_peak_rss_bytes"],
        "mlx_peak_bytes": max(sample["mlx_peak_bytes"] for sample in samples),
    }
    for key in (
        "ane_calls_session_total", "ane_seconds_session_total",
        "runtime_failures_session_total", "runtime_failed",
        "runtime_failure_block",
        "output_copy_bytes_session_total", "coreml_bucket",
        "coreml_block_count", "minimum_profitable_rows",
        "coreml_load_seconds",
        "prefill_actual_tokens", "prefill_compute_tokens",
        "prefill_selected_bucket", "prefill_padding_tokens",
        "prefill_fixed_shape", "minimum_profitable_rows",
        "prefill_plan_reason",
    ):
        if key in samples[-1]:
            record[key] = samples[-1][key]
    return record


def main() -> int:
    args = arguments()
    args.probe = args.probe.resolve()
    args.text_encoder = args.text_encoder.resolve()
    args.tokenizer = args.tokenizer.resolve()
    args.output = args.output.resolve()
    if not args.probe.is_file() or not os.access(args.probe, os.X_OK):
        raise ValueError("--probe must be an executable H3 encoder benchmark probe")
    if not args.text_encoder.is_dir() or not args.tokenizer.is_dir():
        raise ValueError("--text-encoder and --tokenizer must be directories")
    manifest = None
    if args.encoder_ane_manifest:
        args.encoder_ane_manifest = args.encoder_ane_manifest.resolve()
        if not args.encoder_ane_manifest.is_file():
            raise ValueError("--encoder-ane-manifest does not exist")
        manifest = compiled_manifest(args.encoder_ane_manifest)
        if max(args.tokens) > max(manifest["shape"]["buckets"]):
            raise ValueError("H3 encoder manifest has no bucket for this sweep")
    if args.output.exists():
        raise ValueError("--output must be a new path so prior evidence is preserved")

    variants = ["gpu_reference", "gpu_sdpa"]
    if manifest:
        variants.append("gpu_ane_sdpa")
    planned = []
    for index, tokens in enumerate(args.tokens):
        order = variants if index % 2 == 0 else list(reversed(variants))
        planned.append({
            "tokens": tokens,
            "order": order,
            "commands": {
                variant: command_for(
                    args, tokens, args.output / str(tokens) / variant,
                    variant == "gpu_ane_sdpa",
                )
                for variant in order
            },
        })
    if args.dry_run:
        print(json.dumps({
            "format": "turbocider-h3-encoder-sweep-plan-v1",
            "manifest": str(args.encoder_ane_manifest) if manifest else None,
            "planned": planned,
        }, indent=2, sort_keys=True))
        return 0

    args.output.mkdir(parents=True)
    report = {
        "format": "turbocider-h3-encoder-sweep-v1",
        "hardware": {
            "machine": platform.machine(),
            "chip": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=False,
            ).stdout.strip(),
        },
        "mlx_version": importlib.metadata.version("mlx"),
        "text_encoder": str(args.text_encoder),
        "tokenizer": str(args.tokenizer),
        "encoder_ane_manifest": str(args.encoder_ane_manifest) if manifest else None,
        "hybrid_prefix_blocks": len(manifest["artifacts"]) if manifest else 0,
        "method": (
            "separate candidate processes; deterministic token IDs; one first pass "
            "and N warm resident passes; post-first-pass settling samples are "
            "discarded from warm statistics; process order reverses by sequence length; "
            "final 50-layer hidden states compared after timing"
        ),
        "runs": args.runs,
        "gates": {
            "min_warm_speedup": args.min_speedup,
            "max_warm_cv": args.max_warm_cv,
            "min_warm_samples": args.min_warm_samples,
            "warmup_discard": args.warmup_discard,
            "max_relative_l2": args.max_relative_l2,
            "min_cosine": args.min_cosine,
            "max_relative_abs": args.max_relative_abs,
        },
        "cases": [],
    }
    report_path = args.output / "report.json"
    for item in planned:
        results = {}
        for variant in item["order"]:
            results[variant] = run_probe(
                item["commands"][variant],
                args.output / str(item["tokens"]) / variant,
                args.timeout,
                fused=variant != "gpu_reference",
                warmup_discard=args.warmup_discard,
            )
        candidates = {}
        reference_folder = args.output / str(item["tokens"]) / "gpu_reference"
        for variant in variants[1:]:
            quality = tensor_quality(
                args.output / str(item["tokens"]) / variant /
                    "conditioning.safetensors",
                reference_folder / "conditioning.safetensors",
            )
            quality_passed = (
                quality.get("shape_match", False) and
                quality.get("all_finite", False) and
                quality["relative_l2"] <= args.max_relative_l2 and
                quality["cosine"] >= args.min_cosine and
                quality["relative_max_abs"] <= args.max_relative_abs
            )
            # Gate the fused GPU candidate against the exact graph, but gate
            # GPU+ANE against the already-qualified fused GPU path.  Comparing
            # hybrid only with the slower exact graph can retain a topology
            # that fails to improve the best available GPU implementation.
            speed_reference = (
                "gpu_sdpa" if variant == "gpu_ane_sdpa"
                else "gpu_reference"
            )
            speedup_vs_gpu_reference = (
                results["gpu_reference"]["warm_median_seconds"] /
                results[variant]["warm_median_seconds"]
            )
            speedup = (
                results[speed_reference]["warm_median_seconds"] /
                results[variant]["warm_median_seconds"]
            )
            conservative_speedup = (
                min(results[speed_reference]["warm_seconds"]) /
                max(results[variant]["warm_seconds"])
            )
            stability_passed = bool(
                len(results[variant]["warm_seconds"]) >=
                    args.min_warm_samples and
                results[variant]["warm_cv"] <= args.max_warm_cv
            )
            reference_stability_passed = bool(
                len(results[speed_reference]["warm_seconds"]) >=
                    args.min_warm_samples and
                results[speed_reference]["warm_cv"] <= args.max_warm_cv
            )
            timing = compact(results[variant])
            hybrid_executed = variant != "gpu_ane_sdpa" or (
                timing.get("coreml_block_count", 0) > 0 and
                timing.get("ane_calls_session_total", 0) ==
                    timing.get("coreml_block_count", 0) * (args.runs + 1) and
                timing.get("runtime_failed") is not True and
                timing.get("prefill_fixed_shape") is True and
                timing.get("prefill_actual_tokens") == item["tokens"] and
                timing.get("prefill_selected_bucket") ==
                    timing.get("coreml_bucket") and
                timing.get("minimum_profitable_rows", 0) <= item["tokens"]
            )
            candidates[variant] = {
                "timing": timing,
                "warm_speedup": speedup,
                "conservative_warm_speedup": conservative_speedup,
                "warm_speedup_reference": speed_reference,
                "warm_speedup_vs_gpu_reference": speedup_vs_gpu_reference,
                "quality": quality,
                "quality_passed": quality_passed,
                "speed_passed": speedup >= args.min_speedup,
                "stability_passed": stability_passed,
                "reference_stability_passed": reference_stability_passed,
                "hybrid_executed": hybrid_executed,
                "retain": (
                    quality_passed and speedup >= args.min_speedup and
                    stability_passed and hybrid_executed
                ),
            }
        case = {
            "tokens": item["tokens"],
            "order": item["order"],
            "gpu_reference": compact(results["gpu_reference"]),
            "candidates": candidates,
        }
        report["cases"].append(case)
        temporary = report_path.with_name(report_path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(report_path)
        print(json.dumps(case, sort_keys=True), flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
