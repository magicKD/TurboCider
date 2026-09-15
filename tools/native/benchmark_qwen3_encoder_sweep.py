#!/usr/bin/env python3
"""Qualify Qwen3 encoder GPU/Core ML prefill over a sequence-length sweep.

Each backend runs in a separate process with one recorded first encoder pass
and multiple warm passes in the same resident process.  Core ML source
packages are rejected: setup and warm timing must use a compiled-cache manifest.
The final deterministic conditioning tensors are compared after timing so the
quality calculation does not contaminate the measured samples.
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
import sys
import time
from pathlib import Path


DEFAULT_TOKENS = (64, 128, 256, 512, 1024, 2048)
INVALID_METRIC = 1e30
MODE_GEOMETRY = {
    "z_image": {"hidden": 2560, "mlp_width": 9728, "required_blocks": 35},
    "flux_klein": {"hidden": 4096, "mlp_width": 12288, "required_blocks": 27},
}
OUTPUT_LAYERS = {
    "z_image": (34,),
    "flux_klein": (8, 17, 26),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", type=Path, default=Path("build/native/qwen3-quant-probe")
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mode", choices=("z_image", "flux_klein"), required=True)
    parser.add_argument("--encoder-ane-manifest", type=Path, required=True)
    parser.add_argument(
        "--mlx-reference",
        action="store_true",
        help=(
            "run the standard mlx-lm Qwen3 encoder adapter beside native GPU/"
            "hybrid (requires an MLX-capable Python environment)"
        ),
    )
    parser.add_argument(
        "--mlx-python",
        type=Path,
        default=Path(os.environ.get("TURBOCIDER_MLX_PYTHON", sys.executable)),
        help="Python executable used for --mlx-reference",
    )
    parser.add_argument(
        "--mlx-script",
        type=Path,
        default=Path(__file__).with_name("qwen3_mlx_encoder_probe.py"),
        help="reference adapter script used for --mlx-reference",
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-speedup", type=float, default=1.1)
    parser.add_argument(
        "--max-warm-cv", type=float, default=0.25,
        help="maximum warm coefficient of variation for both compared backends",
    )
    parser.add_argument(
        "--min-mlx-reference-speedup",
        type=float,
        default=1.0,
        help=(
            "minimum hybrid speedup over the standard mlx-lm adapter when "
            "--mlx-reference is enabled (default: 1.0)"
        ),
    )
    parser.add_argument("--max-relative-l2", type=float, default=0.025)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-relative-abs", type=float, default=0.05)
    parser.add_argument(
        "--force-fused-sdpa", action="store_true",
        help="force MLX fused SDPA for every swept sequence length",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate paths/manifest and print the planned commands without running Metal",
    )
    args = parser.parse_args()
    if args.runs < 1 or args.runs > 50:
        parser.error("--runs must be in 1...50")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if not args.tokens or any(value < 1 or value > 2048 for value in args.tokens):
        parser.error("--tokens values must be in 1...2048")
    if len(set(args.tokens)) != len(args.tokens):
        parser.error("--tokens values must be unique")
    if not math.isfinite(args.min_speedup) or args.min_speedup <= 0:
        parser.error("--min-speedup must be finite and positive")
    if not math.isfinite(args.max_warm_cv) or args.max_warm_cv < 0:
        parser.error("--max-warm-cv must be finite and nonnegative")
    if (not math.isfinite(args.min_mlx_reference_speedup) or
            args.min_mlx_reference_speedup <= 0):
        parser.error("--min-mlx-reference-speedup must be finite and positive")
    if not math.isfinite(args.max_relative_l2) or args.max_relative_l2 < 0:
        parser.error("--max-relative-l2 must be finite and nonnegative")
    if not math.isfinite(args.min_cosine) or not -1 <= args.min_cosine <= 1:
        parser.error("--min-cosine must be in [-1, 1]")
    if not math.isfinite(args.max_relative_abs) or args.max_relative_abs < 0:
        parser.error("--max-relative-abs must be finite and nonnegative")
    return args


def load_compiled_manifest(path: Path, mode: str) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("encoder manifest must use schema_version 2")
    shape = manifest.get("shape")
    artifacts = manifest.get("artifacts")
    if not isinstance(shape, dict) or not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("encoder manifest is missing shape/artifacts")
    buckets = shape.get("buckets")
    if (not isinstance(buckets, list) or not buckets or
            any(type(bucket) is not int or bucket < 1 or bucket > 8192
                for bucket in buckets) or
            buckets != sorted(set(buckets))):
        raise ValueError("encoder manifest has no sequence buckets")
    geometry = MODE_GEOMETRY[mode]
    if (shape.get("K") != geometry["hidden"] or
            shape.get("N") != geometry["hidden"] or
            shape.get("mlp_width") != geometry["mlp_width"] or
            shape.get("ane_mlp_start") != 0 or
            type(shape.get("ane_mlp_end")) is not int or
            not 0 < shape["ane_mlp_end"] <= geometry["mlp_width"]):
        raise ValueError(f"encoder manifest geometry does not match {mode}")
    block_indices = set()
    for block, variants in artifacts.items():
        if (not isinstance(block, str) or not block.isdigit() or
                str(int(block)) != block):
            raise ValueError(f"encoder manifest has invalid block key {block!r}")
        block_indices.add(int(block))
        if not isinstance(variants, dict) or not variants:
            raise ValueError(f"encoder block {block} has no artifact variant")
        for relative in variants.values():
            candidate = path.parent / relative
            if candidate.suffix != ".mlmodelc" or not candidate.is_dir():
                raise ValueError(
                    "--encoder-ane-manifest must point to a compiled-cache "
                    "manifest whose artifacts are .mlmodelc directories"
                )
    required = set(range(geometry["required_blocks"]))
    missing = sorted(required - block_indices)
    if missing:
        raise ValueError(
            f"encoder manifest needs blocks 0...{geometry['required_blocks'] - 1} "
            f"for {mode}; missing {missing[:8]}"
        )
    return manifest


def command_for(args: argparse.Namespace, tokens: int, folder: Path,
                hybrid: bool) -> list[str]:
    command = [
        str(args.probe), str(args.weights), str(tokens), str(args.runs),
        str(folder / "conditioning.safetensors"), args.mode,
    ]
    if hybrid:
        command.append(str(args.encoder_ane_manifest))
    return command


def mlx_command_for(args: argparse.Namespace, tokens: int, folder: Path) -> list[str]:
    return [
        str(args.mlx_python), str(args.mlx_script), str(args.weights),
        str(tokens), str(args.runs), str(folder / "conditioning.safetensors"),
        args.mode,
    ]


def run_probe(command: list[str], folder: Path, timeout: int,
              force_fused_sdpa: bool, force_hybrid_prefill: bool = False) -> dict:
    folder.mkdir(parents=True)
    environment = os.environ.copy()
    # Internal per-layer validation executes the exact MLP beside the candidate
    # and would invalidate timing. Final encoder output quality is checked below.
    environment.pop("TURBOCIDER_QWEN3_HYBRID_VALIDATE", None)
    for name in (
        "TURBOCIDER_QWEN3_FUSED_SDPA",
        "TURBOCIDER_QWEN3_DISABLE_FUSED_SDPA",
        "TURBOCIDER_QWEN3_FUSED_SDPA_MIN_TOKENS",
        "TURBOCIDER_QWEN3_ANE_MIN_TOKENS",
        "TURBOCIDER_QWEN3_DISABLE_MLP_TAIL_PADDING",
        "TURBOCIDER_QWEN3_MLP_MAX_PADDING_PERCENT",
    ):
        environment.pop(name, None)
    if force_fused_sdpa:
        environment["TURBOCIDER_QWEN3_FUSED_SDPA"] = "1"
        environment["TURBOCIDER_QWEN3_FUSED_SDPA_MIN_TOKENS"] = "1"
    if force_hybrid_prefill:
        # Qualification runs intentionally remove the production profitability
        # threshold so the report can show both executed and rejected buckets.
        environment["TURBOCIDER_QWEN3_ANE_MIN_TOKENS"] = "1"
        environment["TURBOCIDER_QWEN3_MLP_MAX_PADDING_PERCENT"] = "10000"
    started = time.perf_counter()
    process = subprocess.run(
        ["/usr/bin/time", "-l", *command], text=True, capture_output=True,
        env=environment, timeout=timeout, check=False,
    )
    wall_seconds = time.perf_counter() - started
    (folder / "stdout.log").write_text(process.stdout)
    (folder / "stderr.log").write_text(process.stderr)
    match = re.search(r"(\d+)\s+maximum resident set size", process.stderr)
    process_record = {
        "command": command,
        "returncode": process.returncode,
        "process_wall_seconds": wall_seconds,
        "process_peak_rss_bytes": int(match.group(1)) if match else None,
    }
    (folder / "process.json").write_text(
        json.dumps(process_record, indent=2, sort_keys=True) + "\n"
    )
    if process.returncode:
        raise RuntimeError(
            f"Qwen3 encoder probe failed in {folder}:\n{process.stderr[-4000:]}"
        )
    metrics = json.loads(process.stdout)
    warm = [sample["seconds"] for sample in metrics["samples"]
            if not sample["warmup"]]
    if len(warm) != len(metrics["samples"]) - 1 or not warm:
        raise ValueError("probe did not return one first pass followed by warm samples")
    return {
        **process_record,
        "probe": metrics,
        "first_encoder_seconds": metrics["samples"][0]["seconds"],
        "warm_seconds": warm,
        "warm_median_seconds": statistics.median(warm),
        "warm_cv": statistics.pstdev(warm) / statistics.mean(warm),
    }


def run_mlx_reference(command: list[str], folder: Path, timeout: int) -> dict:
    """Run the standard MLX adapter in a clean resident process."""
    folder.mkdir(parents=True)
    environment = os.environ.copy()
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    started = time.perf_counter()
    process = subprocess.run(
        ["/usr/bin/time", "-l", *command], text=True, capture_output=True,
        env=environment, timeout=timeout, check=False,
    )
    wall_seconds = time.perf_counter() - started
    (folder / "stdout.log").write_text(process.stdout)
    (folder / "stderr.log").write_text(process.stderr)
    match = re.search(r"(\d+)\s+maximum resident set size", process.stderr)
    process_record = {
        "command": command,
        "returncode": process.returncode,
        "process_wall_seconds": wall_seconds,
        "process_peak_rss_bytes": int(match.group(1)) if match else None,
    }
    (folder / "process.json").write_text(
        json.dumps(process_record, indent=2, sort_keys=True) + "\n"
    )
    if process.returncode:
        raise RuntimeError(
            f"MLX reference encoder probe failed in {folder}:\n"
            f"{process.stderr[-4000:]}"
        )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("MLX reference probe returned no JSON")
    metrics = json.loads(lines[-1])
    warm = [sample["seconds"] for sample in metrics["samples"]
            if not sample["warmup"]]
    if len(warm) != len(metrics["samples"]) - 1 or not warm:
        raise ValueError("MLX reference probe did not return warm samples")
    return {
        **process_record,
        "probe": metrics,
        "first_encoder_seconds": metrics["first_encoder_seconds"],
        "warm_seconds": warm,
        "warm_median_seconds": statistics.median(warm),
        "warm_cv": statistics.pstdev(warm) / statistics.mean(warm),
    }


def tensor_quality(candidate_path: Path, reference_path: Path) -> dict:
    import mlx.core as mx

    candidate = mx.load(str(candidate_path))["conditioning"].astype(mx.float32)
    reference = mx.load(str(reference_path))["conditioning"].astype(mx.float32)
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
    finite = mx.logical_and(mx.all(mx.isfinite(candidate)), mx.all(mx.isfinite(reference)))
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
        "relative_l2": (
            math.sqrt(delta_energy / reference_energy)
            if reference_energy > 0 else (
                0.0 if delta_energy == 0 else INVALID_METRIC
            )
        ),
        "cosine": max(-1.0, min(1.0, (
            float(dot.item()) / denominator
            if denominator > 0 else (1.0 if candidate_energy == reference_energy == 0 else 0.0)
        ))),
        "max_abs": absolute,
        "relative_max_abs": (
            absolute / amplitude if amplitude > 0 else (
                0.0 if absolute == 0 else INVALID_METRIC
            )
        ),
    }


def compact_backend(result: dict) -> dict:
    probe = result["probe"]
    last = probe["samples"][-1]
    compact = {
        "load_seconds": probe["load_seconds"],
        "first_encoder_seconds": result["first_encoder_seconds"],
        "warm_seconds": result["warm_seconds"],
        "warm_median_seconds": result["warm_median_seconds"],
        "warm_cv": result["warm_cv"],
        "process_wall_seconds": result["process_wall_seconds"],
        "process_peak_rss_bytes": result["process_peak_rss_bytes"],
        "mlx_peak_bytes": max(sample["mlx_peak_bytes"] for sample in probe["samples"]),
    }
    for key in (
        "coreml_bucket", "coreml_block_count", "coreml_load_seconds",
        "coreml_model_load_seconds",
        "coreml_interface_setup_seconds", "coreml_output_backing_setup_seconds",
        "qualified_flexible_backing",
        "ane_calls_session_total", "ane_seconds_session_total",
        "ane_first_runtime_calls_session_total",
        "ane_first_runtime_seconds_session_total",
        "ane_subsequent_runtime_calls_session_total",
        "ane_subsequent_runtime_seconds_session_total",
        "runtime_failures_session_total", "runtime_failed",
        "runtime_failure_block",
        "output_copy_bytes_session_total",
        "prefill_actual_tokens", "prefill_compute_tokens",
        "prefill_selected_bucket", "prefill_padding_tokens",
        "prefill_fixed_shape", "minimum_profitable_rows",
        "prefill_plan_reason",
    ):
        if key in last:
            compact[key] = last[key]
    return compact


def compact_mlx_reference(result: dict) -> dict:
    probe = result["probe"]
    return {
        "load_seconds": probe["load_seconds"],
        "first_encoder_seconds": result["first_encoder_seconds"],
        "warm_seconds": result["warm_seconds"],
        "warm_median_seconds": result["warm_median_seconds"],
        "warm_cv": result["warm_cv"],
        "process_wall_seconds": result["process_wall_seconds"],
        "process_peak_rss_bytes": result["process_peak_rss_bytes"],
        "mlx_peak_bytes": max(sample["mlx_peak_bytes"] for sample in probe["samples"]),
        "output_layers": probe["output_layers"],
        "weight_files": probe["weight_files"],
        "attention_compute_dtype": probe["attention_compute_dtype"],
        "residual_compute_dtype": probe["residual_compute_dtype"],
        "conditioning_dtype": probe["conditioning_dtype"],
        "eval_interval": probe["eval_interval"],
        "mlx_version": probe.get("mlx_version"),
        "mlx_lm_version": probe.get("mlx_lm_version"),
    }


def quality_gate_passed(quality: dict, args: argparse.Namespace) -> bool:
    return bool(
        quality.get("shape_match", False) and
        quality.get("all_finite", False) and
        quality["relative_l2"] <= args.max_relative_l2 and
        quality["cosine"] >= args.min_cosine and
        quality["relative_max_abs"] <= args.max_relative_abs
    )


def main() -> int:
    args = arguments()
    args.probe = args.probe.resolve()
    args.weights = args.weights.resolve()
    args.encoder_ane_manifest = args.encoder_ane_manifest.resolve()
    # Preserve virtual-environment launcher symlinks. Resolving them to the
    # base interpreter drops the venv's mlx-lm site-packages selection.
    args.mlx_python = Path(os.path.abspath(args.mlx_python))
    args.mlx_script = args.mlx_script.resolve()
    args.output = args.output.resolve()
    if not args.probe.is_file() or not os.access(args.probe, os.X_OK):
        raise ValueError("--probe must be an executable qwen3-quant-probe")
    if not args.weights.exists():
        raise ValueError("--weights does not exist")
    if not args.encoder_ane_manifest.is_file():
        raise ValueError("--encoder-ane-manifest does not exist")
    if args.mlx_reference:
        if not args.mlx_python.is_file() or not os.access(args.mlx_python, os.X_OK):
            raise ValueError("--mlx-python must be an executable Python interpreter")
        if not args.mlx_script.is_file():
            raise ValueError("--mlx-script does not exist")
    manifest = load_compiled_manifest(args.encoder_ane_manifest, args.mode)
    maximum_bucket = max(manifest["shape"]["buckets"])
    if max(args.tokens) > maximum_bucket:
        raise ValueError(
            f"sequence sweep needs {max(args.tokens)} rows but manifest supports {maximum_bucket}"
        )
    if args.output.exists():
        raise ValueError("--output must be a new path so prior evidence is preserved")

    planned = []
    for index, tokens in enumerate(args.tokens):
        if args.mlx_reference:
            orders = (
                ("gpu", "hybrid", "mlx_reference"),
                ("hybrid", "mlx_reference", "gpu"),
                ("mlx_reference", "gpu", "hybrid"),
            )
            order = orders[index % len(orders)]
        else:
            order = ("gpu", "hybrid") if index % 2 == 0 else ("hybrid", "gpu")
        commands = {}
        for variant in order:
            folder = args.output / str(tokens) / variant
            commands[variant] = (
                mlx_command_for(args, tokens, folder)
                if variant == "mlx_reference"
                else command_for(args, tokens, folder, variant == "hybrid")
            )
        planned.append({
            "tokens": tokens,
            "order": list(order),
            "commands": commands,
        })
    if args.dry_run:
        print(json.dumps({
            "format": "turbocider-qwen3-encoder-sweep-plan-v1",
            "manifest": str(args.encoder_ane_manifest),
            "force_fused_sdpa": args.force_fused_sdpa,
            "hybrid_prefill_policy": (
                "qualification hybrid process forces ANE eligibility and allows "
                "padding so the report records executed-vs-fallback planner cases"
            ),
            "buckets": manifest["shape"]["buckets"],
            "blocks": len(manifest["artifacts"]),
            "mlx_reference": args.mlx_reference,
            **({
                "mlx_python": str(args.mlx_python),
                "mlx_script": str(args.mlx_script),
                "min_mlx_reference_speedup": args.min_mlx_reference_speedup,
            } if args.mlx_reference else {}),
            "planned": planned,
        }, indent=2, sort_keys=True))
        return 0

    args.output.mkdir(parents=True)
    report = {
        "format": "turbocider-qwen3-encoder-sweep-v1",
        "hardware": {
            "machine": platform.machine(),
            "chip": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=False,
            ).stdout.strip(),
        },
        "mlx_version": importlib.metadata.version("mlx"),
        "mode": args.mode,
        "weights": str(args.weights),
        "encoder_ane_manifest": str(args.encoder_ane_manifest),
        "manifest_buckets": manifest["shape"]["buckets"],
        "manifest_blocks": len(manifest["artifacts"]),
        "mlx_reference": args.mlx_reference,
        **({
            "mlx_python": str(args.mlx_python),
            "mlx_script": str(args.mlx_script),
        } if args.mlx_reference else {}),
        "force_fused_sdpa": args.force_fused_sdpa,
        "hybrid_prefill_policy": (
            "qualification hybrid process sets ANE minimum tokens to 1 and "
            "padding limit to 10000%; production routing keeps conservative "
            "defaults and remains fail-closed"
        ),
        "method": (
            "separate backend processes; one first pass and N warm resident passes; "
            "backend order alternates by sequence length; final conditioning quality "
            "is computed after timing"
        ),
        "runs": args.runs,
        "gates": {
            "min_warm_speedup": args.min_speedup,
            "min_warm_samples": 3,
            "max_warm_cv": args.max_warm_cv,
            **({
                "min_mlx_reference_speedup": args.min_mlx_reference_speedup,
            } if args.mlx_reference else {}),
            "max_relative_l2": args.max_relative_l2,
            "min_cosine": args.min_cosine,
            "max_relative_abs": args.max_relative_abs,
        },
        "cases": [],
    }
    report_path = args.output / "report.json"
    for item in planned:
        tokens = item["tokens"]
        results = {}
        for variant in item["order"]:
            folder = args.output / str(tokens) / variant
            results[variant] = (
                run_mlx_reference(item["commands"][variant], folder, args.timeout)
                if variant == "mlx_reference"
                else run_probe(
                    item["commands"][variant], folder, args.timeout,
                    args.force_fused_sdpa,
                    force_hybrid_prefill=variant == "hybrid",
                )
            )
        quality = tensor_quality(
            args.output / str(tokens) / "hybrid" / "conditioning.safetensors",
            args.output / str(tokens) / "gpu" / "conditioning.safetensors",
        )
        quality_passed = quality_gate_passed(quality, args)
        speedup = (
            results["gpu"]["warm_median_seconds"] /
            results["hybrid"]["warm_median_seconds"]
        )
        speed_passed = speedup >= args.min_speedup
        stability_passed = bool(
            args.runs >= 3 and
            results["gpu"]["warm_cv"] <= args.max_warm_cv and
            results["hybrid"]["warm_cv"] <= args.max_warm_cv
        )
        hybrid_summary = compact_backend(results["hybrid"])
        hybrid_executed = (
            hybrid_summary.get("ane_calls_session_total", 0) > 0 and
            hybrid_summary.get("coreml_block_count") ==
                MODE_GEOMETRY[args.mode]["required_blocks"] and
            hybrid_summary.get("qualified_flexible_backing") is True and
            hybrid_summary.get("runtime_failed") is not True and
            hybrid_summary.get("prefill_fixed_shape") is True and
            hybrid_summary.get("prefill_actual_tokens") == tokens
        )
        case = {
            "tokens": tokens,
            "order": item["order"],
            "gpu": compact_backend(results["gpu"]),
            "hybrid": hybrid_summary,
            "hybrid_executed": hybrid_executed,
            "warm_speedup": speedup,
            "quality": quality,
            "quality_passed": quality_passed,
            "speed_passed": speed_passed,
            "stability_passed": stability_passed,
            "retain": (
                quality_passed and speed_passed and stability_passed and
                hybrid_executed
            ),
        }
        if args.mlx_reference:
            mlx_summary = compact_mlx_reference(results["mlx_reference"])
            mlx_quality = tensor_quality(
                args.output / str(tokens) / "mlx_reference" / "conditioning.safetensors",
                args.output / str(tokens) / "gpu" / "conditioning.safetensors",
            )
            hybrid_mlx_quality = tensor_quality(
                args.output / str(tokens) / "hybrid" / "conditioning.safetensors",
                args.output / str(tokens) / "mlx_reference" / "conditioning.safetensors",
            )
            mlx_speedup = (
                results["mlx_reference"]["warm_median_seconds"] /
                results["hybrid"]["warm_median_seconds"]
            )
            gpu_mlx_speedup = (
                results["mlx_reference"]["warm_median_seconds"] /
                results["gpu"]["warm_median_seconds"]
            )
            mlx_quality_passed = quality_gate_passed(mlx_quality, args)
            hybrid_mlx_quality_passed = quality_gate_passed(
                hybrid_mlx_quality, args
            )
            mlx_speed_passed = mlx_speedup >= args.min_mlx_reference_speedup
            mlx_stability_passed = bool(
                args.runs >= 3 and
                results["mlx_reference"]["warm_cv"] <= args.max_warm_cv
            )
            case.update({
                "mlx_reference": mlx_summary,
                "mlx_reference_quality_vs_gpu": mlx_quality,
                "mlx_reference_quality_vs_gpu_passed": mlx_quality_passed,
                "hybrid_quality_vs_mlx_reference": hybrid_mlx_quality,
                "hybrid_quality_vs_mlx_reference_passed": hybrid_mlx_quality_passed,
                "gpu_vs_mlx_reference_warm_speedup": gpu_mlx_speedup,
                "hybrid_vs_mlx_reference_warm_speedup": mlx_speedup,
                "mlx_reference_speed_passed": mlx_speed_passed,
                "mlx_reference_stability_passed": mlx_stability_passed,
            })
            case["retain"] = bool(
                case["retain"] and mlx_quality_passed and
                hybrid_mlx_quality_passed and mlx_speed_passed and
                mlx_stability_passed
            )
        report["cases"].append(case)
        temporary = report_path.with_name(report_path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(report_path)
        print(json.dumps(case, sort_keys=True), flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
