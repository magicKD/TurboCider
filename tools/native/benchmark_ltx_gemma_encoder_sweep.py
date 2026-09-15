#!/usr/bin/env python3
"""Benchmark complete LTX Gemma conditioning in a resident encoder process.

The staged, fused-MLP and fused-MLP-plus-GPU-tap variants use the same
checkpoint, tokenizer, prompt and Metal runtime.  Each process performs one
first encode followed by warm encodes, then writes the final raw
video/audio/mask conditioning tensors.  Quality is checked after timing so
the comparison does not perturb samples.

The default variants establish the GPU/fused-GPU baseline.  Supplying an ANE
bank adds an explicitly experimental integrated variant whose execution
telemetry, quality, stability, and speed all have to pass independently.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path


VIDEO_DIM = 4096
AUDIO_DIM = 2048
MAX_ROWS = 1024
INVALID_METRIC = 1e30
VARIANTS = ("staged", "fused", "fused_gpu_taps")
ALL_VARIANTS = VARIANTS + ("hybrid_ane",)
MIN_QUALIFYING_WARM_RUNS = 3
PROBE_FORMAT = "turbocider-native-gemma-raw-candidate-v1"
PROBE_MARKER = re.compile(
    r'\{\s*"format"\s*:\s*"' + re.escape(PROBE_FORMAT) + r'"'
)
COREML_E5_DIAGNOSTIC_START = (
    "Missing E5 bundle resource required for loading "
    "ExecutionStreamOperation. Must re-compile the E5. Resource path = "
)
COREML_E5_DIAGNOSTIC_END = " @ GetANEFModel"
MAX_COREML_STDOUT_DIAGNOSTICS = 16


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", type=Path, default=Path("build/native/ltx-gemma-encode"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--shader", type=Path, default=Path("build/native/ltx_shaders.metal"),
    )
    parser.add_argument(
        "--prompt", action="append", required=True,
        help="prompt to benchmark; repeat for multiple prompt/row cases",
    )
    parser.add_argument(
        "--label", action="append",
        help="optional label for each --prompt (defaults to prompt index)",
    )
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument(
        "--variants", nargs="+", choices=ALL_VARIANTS,
        default=list(VARIANTS),
        help=("GPU variants to run; staged is the reference and the other "
              "variants are compared against it"),
    )
    parser.add_argument(
        "--ane-manifest", type=Path,
        help="optional ltx-gemma-ane-mlp-v1 layer manifest or directory; "
             "adds the integrated hybrid_ane variant",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-speedup", type=float, default=1.1)
    parser.add_argument("--max-warm-cv", type=float, default=0.25)
    parser.add_argument("--max-relative-l2", type=float, default=0.025)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-relative-abs", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.prompt:
        parser.error("at least one --prompt is required")
    if args.label is not None and len(args.label) != len(args.prompt):
        parser.error("--label must be supplied once for each --prompt")
    if args.warm_runs < 1 or args.warm_runs > 50:
        parser.error("--warm-runs must be in 1...50")
    if "staged" not in args.variants:
        parser.error("--variants must include staged as the quality reference")
    if len(args.variants) < 2:
        parser.error("--variants must include at least one staged comparison")
    if len(set(args.variants)) != len(args.variants):
        parser.error("--variants values must be unique")
    if "hybrid_ane" in args.variants and args.ane_manifest is None:
        parser.error("hybrid_ane requires --ane-manifest")
    labels = args.label or [str(index) for index in range(len(args.prompt))]
    if (len(set(labels)) != len(labels) or
            any(not label or label in (".", "..") or Path(label).name != label
                for label in labels)):
        parser.error("--label values must be unique file names")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if not math.isfinite(args.min_speedup) or args.min_speedup <= 0:
        parser.error("--min-speedup must be finite and positive")
    if not math.isfinite(args.max_warm_cv) or args.max_warm_cv < 0:
        parser.error("--max-warm-cv must be finite and nonnegative")
    if not math.isfinite(args.max_relative_l2) or args.max_relative_l2 < 0:
        parser.error("--max-relative-l2 must be finite and nonnegative")
    if not math.isfinite(args.min_cosine) or not -1 <= args.min_cosine <= 1:
        parser.error("--min-cosine must be in [-1, 1]")
    if not math.isfinite(args.max_relative_abs) or args.max_relative_abs < 0:
        parser.error("--max-relative-abs must be finite and nonnegative")
    return args


def parse_peak_rss(stderr: str) -> int | None:
    match = re.search(r"(\d+)\s+maximum resident set size", stderr)
    return int(match.group(1)) if match else None


def parse_probe_stdout(stdout: str, folder: Path) -> tuple[dict, int]:
    encoded = stdout.strip()
    markers = list(PROBE_MARKER.finditer(encoded))
    if len(markers) != 1:
        raise ValueError(f"probe produced unexpected stdout in {folder}")
    payload_start = markers[0].start()
    prefix = encoded[:payload_start]
    diagnostic_count = 0
    while prefix:
        if (diagnostic_count >= MAX_COREML_STDOUT_DIAGNOSTICS or
                not prefix.startswith(COREML_E5_DIAGNOSTIC_START)):
            raise ValueError(f"probe produced unexpected stdout in {folder}")
        diagnostic_end = prefix.find(
            COREML_E5_DIAGNOSTIC_END, len(COREML_E5_DIAGNOSTIC_START),
        )
        if diagnostic_end < 0:
            raise ValueError(f"probe produced unexpected stdout in {folder}")
        resource_path = prefix[
            len(COREML_E5_DIAGNOSTIC_START):diagnostic_end
        ]
        if (not resource_path or "\n" in resource_path or
                "\r" in resource_path):
            raise ValueError(f"probe produced unexpected stdout in {folder}")
        prefix = prefix[
            diagnostic_end + len(COREML_E5_DIAGNOSTIC_END):
        ]
        diagnostic_count += 1
    try:
        payload = json.loads(encoded[payload_start:])
    except json.JSONDecodeError as failure:
        raise ValueError(
            f"probe produced invalid JSON stdout in {folder}"
        ) from failure
    return payload, diagnostic_count


def warm_summary(values: list[float]) -> dict:
    mean = statistics.fmean(values) if values else None
    deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "warm_median_seconds": statistics.median(values) if values else None,
        "warm_mean_seconds": mean,
        "warm_min_seconds": min(values) if values else None,
        "warm_max_seconds": max(values) if values else None,
        "warm_pstdev_seconds": deviation if values else None,
        "warm_cv": deviation / mean if mean else 0.0,
    }


def warm_qualification(
    reference: dict, candidate: dict, max_warm_cv: float,
) -> dict:
    return {
        "warm_samples_passed": (
            reference["warm_runs"] >= MIN_QUALIFYING_WARM_RUNS and
            candidate["warm_runs"] >= MIN_QUALIFYING_WARM_RUNS
        ),
        "stability_passed": (
            reference["warm_cv"] <= max_warm_cv and
            candidate["warm_cv"] <= max_warm_cv
        ),
    }


def command_for(args: argparse.Namespace, prompt: str, folder: Path,
                variant: str = "staged") -> list[str]:
    command = [
        str(args.probe), str(args.checkpoint), str(args.tokenizer),
        str(args.shader), prompt, str(folder), str(args.warm_runs),
    ]
    if variant == "hybrid_ane":
        command.append(str(args.ane_manifest))
    return command


def variant_order(variants: list[str], case_index: int) -> list[str]:
    offset = case_index % len(variants)
    return variants[offset:] + variants[:offset]


def normalize_variant(variant: str | bool) -> str:
    """Accept the old boolean API while exposing named GPU variants."""
    if isinstance(variant, bool):
        return "fused" if variant else "staged"
    if variant not in ALL_VARIANTS:
        raise ValueError(f"unknown Gemma encoder variant: {variant}")
    return variant


def ane_telemetry(payload: dict, expected_ane: bool, folder: Path) -> dict:
    integer_names = (
        "ane_layers_available", "ane_layers_attempted",
        "ane_layers_succeeded", "ane_layers_fallback",
        "ane_cache_hits", "ane_cache_misses",
        "ane_preload_models_session_total", "ane_preload_workers",
        "ane_selected_bucket",
        "ane_padding_rows", "ane_minimum_profitable_rows",
        "ane_execution_rows",
    )
    result = {}
    for name in integer_names:
        value = payload.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid {name} telemetry in {folder}")
        result[name] = value
    for name in ("ane_requested", "ane_used", "ane_output_backing_used"):
        value = payload.get(name, False)
        if not isinstance(value, bool):
            raise ValueError(f"invalid {name} telemetry in {folder}")
        result[name] = value
    total_seconds = payload.get("ane_total_seconds", 0.0)
    preload_seconds = payload.get("ane_preload_seconds_session_total", 0.0)
    for name, value in (("ane_total_seconds", total_seconds),
                        ("ane_preload_seconds_session_total", preload_seconds)):
        if (isinstance(value, bool) or
                not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid {name} telemetry in {folder}")
        result[name] = value
    plan_reason = payload.get("ane_plan_reason")
    if not isinstance(plan_reason, str) or not plan_reason:
        raise ValueError(f"invalid ane_plan_reason telemetry in {folder}")
    result["ane_plan_reason"] = plan_reason

    available = result["ane_layers_available"]
    attempted = result["ane_layers_attempted"]
    succeeded = result["ane_layers_succeeded"]
    fallback = result["ane_layers_fallback"]
    hits = result["ane_cache_hits"]
    misses = result["ane_cache_misses"]
    execution_rows = result["ane_execution_rows"]
    selected = result["ane_selected_bucket"]
    padding = result["ane_padding_rows"]
    minimum = result["ane_minimum_profitable_rows"]
    preloaded = result["ane_preload_models_session_total"]
    preload_workers = result["ane_preload_workers"]
    preload_valid = (
        (preloaded == preload_workers == 0 and preload_seconds == 0) or
        (preloaded > 0 and 1 <= preload_workers <= 16 and preload_seconds > 0)
    )
    if result["ane_requested"] is not expected_ane:
        raise ValueError(f"ANE request telemetry mismatch in {folder}")
    if expected_ane:
        rows = payload["rows"]
        valid = (
            available == attempted and
            succeeded + fallback == attempted and
            hits + misses == attempted and
            ((selected == 0 and padding == minimum == 0) or
             (selected >= rows and padding == selected - rows and
              1 <= minimum <= selected)) and
            (attempted == 0 or selected > 0) and
            result["ane_used"] is (succeeded > 0) and
            (succeeded == 0 or (
                result["ane_output_backing_used"] and
                execution_rows == selected * succeeded and
                total_seconds > 0
            )) and
            (succeeded > 0 or (
                not result["ane_output_backing_used"] and
                execution_rows == 0 and total_seconds == 0
            )) and preload_valid
        )
    else:
        valid = (
            not result["ane_used"] and
            not result["ane_output_backing_used"] and
            all(result[name] == 0 for name in integer_names) and
            total_seconds == 0 and preload_seconds == 0 and preload_valid and
            plan_reason == "no_ane_manifest"
        )
    if not valid:
        raise ValueError(f"inconsistent ANE execution telemetry in {folder}")
    return result


def resident_weight_telemetry(
    payload: dict, expected_resident: bool, folder: Path,
) -> dict:
    enabled = payload.get("resident_weights_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"invalid resident_weights_enabled telemetry in {folder}")
    result = {"resident_weights_enabled": enabled}
    for name in (
        "resident_weight_cache_hits", "resident_weight_cache_misses",
        "resident_weight_bytes",
    ):
        value = payload.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid {name} telemetry in {folder}")
        result[name] = value
    if enabled is not expected_resident:
        raise ValueError(f"resident-weight request telemetry mismatch in {folder}")
    hits = result["resident_weight_cache_hits"]
    misses = result["resident_weight_cache_misses"]
    resident_bytes = result["resident_weight_bytes"]
    if enabled:
        valid = hits > 0 and misses == 0 and resident_bytes > 0
    else:
        valid = hits == misses == resident_bytes == 0
    if not valid:
        raise ValueError(
            f"inconsistent resident-weight telemetry in {folder}"
        )
    return result


def run_case(
    command: list[str], folder: Path, timeout: int,
    variant: str | bool,
) -> dict:
    variant = normalize_variant(variant)
    folder.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("TURBOCIDER_LTX_GEMMA_FUSED_MLP", None)
    environment.pop("TURBOCIDER_LTX_GEMMA_GPU_TAPS", None)
    if variant in ("fused", "fused_gpu_taps", "hybrid_ane"):
        environment["TURBOCIDER_LTX_GEMMA_FUSED_MLP"] = "1"
    if variant in ("fused_gpu_taps", "hybrid_ane"):
        environment["TURBOCIDER_LTX_GEMMA_GPU_TAPS"] = "1"
    expected_resident = (
        environment.get("TURBOCIDER_LTX_GEMMA_RESIDENT_WEIGHTS") == "1"
    )
    process = subprocess.run(
        ["/usr/bin/time", "-l", *command],
        capture_output=True, text=True, check=False, timeout=timeout,
        env=environment,
    )
    (folder / "stdout.log").write_text(process.stdout)
    (folder / "stderr.log").write_text(process.stderr)
    record = {
        "variant": variant,
        "command": command,
        "returncode": process.returncode,
        "process_peak_rss_bytes": parse_peak_rss(process.stderr),
    }
    (folder / "process.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    if process.returncode:
        raise RuntimeError(
            f"Gemma encoder probe failed in {folder}:\n{process.stderr[-4000:]}"
        )
    payload, coreml_stdout_diagnostics = parse_probe_stdout(
        process.stdout, folder,
    )
    record["coreml_stdout_diagnostics"] = coreml_stdout_diagnostics
    (folder / "process.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    warm = payload.get("warm_seconds")
    expected_fused = variant in ("fused", "fused_gpu_taps", "hybrid_ane")
    expected_gpu_taps = variant in ("fused_gpu_taps", "hybrid_ane")
    expected_ane = variant == "hybrid_ane"
    if payload.get("format") != PROBE_FORMAT:
        raise ValueError(f"probe format mismatch in {folder}")
    create_seconds = payload.get("create_seconds")
    prepare_seconds = payload.get("prepare_seconds", 0.0)
    if (not isinstance(payload.get("rows"), int) or
            not 0 < payload["rows"] <= MAX_ROWS or
            payload.get("video_dim") != VIDEO_DIM or
            payload.get("audio_dim") != AUDIO_DIM or
            payload.get("fused_mlp") is not expected_fused or
            payload.get("gpu_taps") is not expected_gpu_taps or
            payload.get("runs") != args_runs_from_command(command) or
            not isinstance(warm, list) or
            payload.get("warm_runs") != len(warm) or
            not isinstance(create_seconds, (int, float)) or
            not math.isfinite(create_seconds) or create_seconds <= 0 or
            not isinstance(prepare_seconds, (int, float)) or
            not math.isfinite(prepare_seconds) or prepare_seconds < 0 or
            not isinstance(payload.get("first_seconds"), (int, float)) or
            not math.isfinite(payload["first_seconds"]) or
            payload["first_seconds"] <= 0 or
            not all(isinstance(value, (int, float)) and math.isfinite(value)
                    and value > 0 for value in warm)):
        raise ValueError(f"probe geometry/timing mismatch in {folder}")
    telemetry = ane_telemetry(payload, expected_ane, folder)
    resident_telemetry = resident_weight_telemetry(
        payload, expected_resident, folder,
    )
    artifacts = {
        "video": folder / "raw_video_context.bf16",
        "audio": folder / "raw_audio_context.bf16",
        "mask": folder / "raw_text_mask.bf16",
    }
    expected_bytes = {
        "video": payload["rows"] * VIDEO_DIM * 2,
        "audio": payload["rows"] * AUDIO_DIM * 2,
        "mask": payload["rows"] * 2,
    }
    for name, path in artifacts.items():
        if not path.is_file() or path.stat().st_size != expected_bytes[name]:
            raise ValueError(f"probe wrote invalid {name} conditioning in {folder}")
    return {
        "rows": payload["rows"],
        "video_dim": payload["video_dim"],
        "audio_dim": payload["audio_dim"],
        "fused_mlp": payload["fused_mlp"],
        "gpu_taps": payload["gpu_taps"],
        **telemetry,
        **resident_telemetry,
        "runs": payload["runs"],
        "warm_runs": payload["warm_runs"],
        "create_seconds": create_seconds,
        "prepare_seconds": prepare_seconds,
        "first_seconds": payload["first_seconds"],
        "warm_seconds": warm,
        **warm_summary(warm),
        "process_peak_rss_bytes": record["process_peak_rss_bytes"],
        "coreml_stdout_diagnostics": coreml_stdout_diagnostics,
        **artifacts,
    }


def args_runs_from_command(command: list[str]) -> int:
    try:
        index = -2 if len(command) == 8 else -1
        return int(command[index]) + 1
    except (ValueError, IndexError) as error:
        raise ValueError("probe command is missing warm-run count") from error


def bf16(path: Path):
    import numpy as np

    words = np.fromfile(path, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def tensor_quality(candidate_path: Path, reference_path: Path) -> dict:
    import numpy as np

    if not candidate_path.is_file() or not reference_path.is_file():
        return {
            "shape_match": False, "all_finite": False,
            "relative_l2": INVALID_METRIC, "cosine": -1.0,
            "max_abs": INVALID_METRIC, "relative_max_abs": INVALID_METRIC,
        }
    candidate = bf16(candidate_path).astype(np.float64)
    reference = bf16(reference_path).astype(np.float64)
    if candidate.shape != reference.shape:
        return {
            "shape_match": False, "candidate_elements": int(candidate.size),
            "reference_elements": int(reference.size),
            "relative_l2": INVALID_METRIC, "cosine": -1.0,
            "max_abs": INVALID_METRIC, "relative_max_abs": INVALID_METRIC,
        }
    finite = bool(np.isfinite(candidate).all() and np.isfinite(reference).all())
    if not finite:
        return {
            "shape_match": True, "all_finite": False,
            "relative_l2": INVALID_METRIC, "cosine": -1.0,
            "max_abs": INVALID_METRIC, "relative_max_abs": INVALID_METRIC,
        }
    delta = candidate - reference
    delta2 = float(np.dot(delta, delta))
    candidate2 = float(np.dot(candidate, candidate))
    reference2 = float(np.dot(reference, reference))
    denominator = math.sqrt(candidate2 * reference2)
    maximum = float(np.max(np.abs(delta), initial=0.0))
    amplitude = float(np.max(np.abs(reference), initial=0.0))
    return {
        "shape_match": True,
        "all_finite": True,
        "relative_l2": math.sqrt(delta2 / reference2)
        if reference2 > 0 else (0.0 if delta2 == 0 else INVALID_METRIC),
        "cosine": max(-1.0, min(
            1.0, float(np.dot(candidate, reference)) / denominator
            if denominator > 0 else (1.0 if candidate2 == reference2 == 0 else 0.0),
        )),
        "max_abs": maximum,
        "relative_max_abs": maximum / amplitude
        if amplitude > 0 else (0.0 if maximum == 0 else INVALID_METRIC),
    }


def conditioning_quality(candidate: dict, reference: dict) -> dict:
    if candidate["rows"] != reference["rows"]:
        return {
            name: {"shape_match": False, "all_finite": False,
                   "relative_l2": INVALID_METRIC, "cosine": -1.0,
                   "max_abs": INVALID_METRIC,
                   "relative_max_abs": INVALID_METRIC}
            for name in ("video", "audio", "mask")
        }
    return {
        name: tensor_quality(candidate[name], reference[name])
        for name in ("video", "audio", "mask")
    }


def quality_gate(qualities: dict[str, dict], args: argparse.Namespace) -> bool:
    return all(
        quality.get("shape_match", False) and
        quality.get("all_finite", False) and
        quality["relative_l2"] <= args.max_relative_l2 and
        quality["cosine"] >= args.min_cosine and
        quality["relative_max_abs"] <= args.max_relative_abs
        for quality in qualities.values()
    )


def compact(result: dict) -> dict:
    return {
        key: result[key] for key in (
            "rows", "video_dim", "audio_dim", "fused_mlp", "gpu_taps",
            "ane_requested", "ane_used", "ane_layers_available",
            "ane_layers_attempted", "ane_layers_succeeded",
            "ane_layers_fallback", "ane_cache_hits", "ane_cache_misses",
            "ane_selected_bucket", "ane_padding_rows",
            "ane_minimum_profitable_rows", "ane_plan_reason",
            "ane_execution_rows",
            "ane_total_seconds", "ane_preload_seconds_session_total",
            "ane_preload_models_session_total", "ane_preload_workers",
            "ane_output_backing_used",
            "resident_weights_enabled", "resident_weight_cache_hits",
            "resident_weight_cache_misses", "resident_weight_bytes",
            "runs", "warm_runs",
            "create_seconds", "prepare_seconds", "first_seconds",
            "warm_seconds",
            "warm_median_seconds", "warm_mean_seconds", "warm_min_seconds",
            "warm_max_seconds", "warm_pstdev_seconds", "warm_cv",
            "process_peak_rss_bytes", "coreml_stdout_diagnostics",
        )
    }


def main() -> int:
    args = arguments()
    args.probe = args.probe.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.tokenizer = args.tokenizer.resolve()
    args.shader = args.shader.resolve()
    args.output = args.output.resolve()
    labels = args.label or [str(index) for index in range(len(args.prompt))]
    variants = list(args.variants)
    if args.ane_manifest is not None and "hybrid_ane" not in variants:
        variants.append("hybrid_ane")
    planned = []
    for index, (label, prompt) in enumerate(zip(labels, args.prompt)):
        order = variant_order(variants, index)
        planned.append({
            "label": label,
            "prompt": prompt,
            "order": list(order),
            "commands": {
                variant: command_for(
                    args, prompt,
                    args.output / label / variant, variant,
                ) for variant in order
            },
        })
    if args.dry_run:
        print(json.dumps({
            "format": "turbocider-ltx-gemma-encoder-sweep-plan-v2",
            "warm_runs": args.warm_runs,
            "variants": variants,
            "min_qualifying_warm_runs": MIN_QUALIFYING_WARM_RUNS,
            "max_warm_cv": args.max_warm_cv,
            "quality": "full video/audio/mask conditioning compared after timing",
            "ane": "integrated_ltx_gemma_ane_mlp_with_per_layer_gpu_fallback"
                   if "hybrid_ane" in variants else
                   "not_claimed_single_layer_qualification_only",
            "planned": planned,
        }, indent=2, sort_keys=True))
        return 0

    for path, label in (
        (args.probe, "--probe"), (args.checkpoint, "--checkpoint"),
        (args.tokenizer, "--tokenizer"), (args.shader, "--shader"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if args.ane_manifest is not None:
        args.ane_manifest = args.ane_manifest.resolve()
        if not args.ane_manifest.exists():
            raise ValueError(f"--ane-manifest does not exist: {args.ane_manifest}")
    if not os.access(args.probe, os.X_OK):
        raise ValueError("--probe must be executable")
    if args.output.exists():
        raise ValueError("--output must be a new path so prior evidence is preserved")

    args.output.mkdir(parents=True)
    report = {
        "format": "turbocider-ltx-gemma-encoder-sweep-v2",
        "hardware": {
            "machine": platform.machine(),
            "chip": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=False,
            ).stdout.strip(),
        },
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "shader": str(args.shader),
        "warm_runs": args.warm_runs,
        "variants": variants,
        "method": (
            "separate staged/fused GPU processes plus a fused_gpu_taps "
            "process; optional integrated hybrid_ane process with fixed-shape "
            "per-layer ANE MLP and GPU fallback; one first encode followed by "
            "resident warm encodes; full raw video/audio/mask outputs compared "
            "after timing"
        ),
        "gates": {
            "min_warm_speedup": args.min_speedup,
            "min_qualifying_warm_runs": MIN_QUALIFYING_WARM_RUNS,
            "max_warm_cv": args.max_warm_cv,
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
            results[variant] = run_case(
                item["commands"][variant],
                args.output / item["label"] / variant,
                args.timeout, variant,
            )
        reference = results["staged"]
        quality_by_variant = {}
        speedups = {}
        for variant in variants:
            if variant == "staged":
                continue
            candidate = results[variant]
            qualities = conditioning_quality(candidate, reference)
            quality_by_variant[variant] = {
                "tensors": qualities,
                "passed": quality_gate(qualities, args),
            }
            speed_reference = (results.get("fused_gpu_taps", reference)
                               if variant == "hybrid_ane" else reference)
            speedups[variant] = (
                speed_reference["warm_median_seconds"] /
                candidate["warm_median_seconds"]
                if speed_reference["warm_median_seconds"] and
                candidate["warm_median_seconds"] else 0.0
            )
            quality_by_variant[variant]["speed_passed"] = (
                speedups[variant] >= args.min_speedup
            )
            quality_by_variant[variant].update(warm_qualification(
                speed_reference, candidate, args.max_warm_cv
            ))
            execution_passed = variant != "hybrid_ane" or (
                candidate["ane_requested"] and candidate["ane_used"] and
                candidate["ane_layers_succeeded"] > 0 and
                candidate["ane_layers_fallback"] == 0 and
                candidate["ane_output_backing_used"]
            )
            quality_by_variant[variant]["execution_passed"] = execution_passed
            quality_by_variant[variant]["retain"] = (
                quality_by_variant[variant]["passed"] and
                quality_by_variant[variant]["speed_passed"] and
                quality_by_variant[variant]["warm_samples_passed"] and
                quality_by_variant[variant]["stability_passed"] and
                execution_passed
            )
        legacy_variant = "fused" if "fused" in results else next(
            variant for variant in variants if variant != "staged"
        )
        legacy_quality = quality_by_variant[legacy_variant]
        gpu_taps_ab = None
        if "fused" in results and "fused_gpu_taps" in results:
            taps_quality = conditioning_quality(
                results["fused_gpu_taps"], results["fused"]
            )
            gpu_taps_ab = {
                "quality": taps_quality,
                "quality_passed": quality_gate(taps_quality, args),
                "first_speedup": (
                    results["fused"]["first_seconds"] /
                    results["fused_gpu_taps"]["first_seconds"]
                ),
                "warm_speedup": (
                    results["fused"]["warm_median_seconds"] /
                    results["fused_gpu_taps"]["warm_median_seconds"]
                ),
            }
        case = {
            "label": item["label"],
            "prompt": item["prompt"],
            "order": item["order"],
            "rows": reference["rows"],
            "variants": {variant: compact(results[variant])
                         for variant in variants},
            "staged": compact(reference),
            "quality_by_variant": quality_by_variant,
            "quality": legacy_quality["tensors"],
            "quality_passed": legacy_quality["passed"],
            "warm_speedups": speedups,
            "warm_speedup": speedups.get(legacy_variant, 0.0),
            "speed_passed_by_variant": {
                variant: value["speed_passed"]
                for variant, value in quality_by_variant.items()
            },
            "speed_passed": legacy_quality["speed_passed"],
            "warm_samples_passed_by_variant": {
                variant: value["warm_samples_passed"]
                for variant, value in quality_by_variant.items()
            },
            "stability_passed_by_variant": {
                variant: value["stability_passed"]
                for variant, value in quality_by_variant.items()
            },
            "retain_by_variant": {
                variant: value["retain"]
                for variant, value in quality_by_variant.items()
            },
            "retain": all(value["retain"]
                           for value in quality_by_variant.values()),
        }
        if gpu_taps_ab is not None:
            case["gpu_taps_ab_vs_fused"] = gpu_taps_ab
        for variant in variants:
            if variant != "staged":
                case[variant] = compact(results[variant])
        report["cases"].append(case)
        temporary = report_path.with_name(report_path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(report_path)
        print(json.dumps(case, sort_keys=True), flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
