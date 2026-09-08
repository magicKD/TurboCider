#!/usr/bin/env python3
"""Paired resident/streaming benchmark for TurboCider Z-Image GGUF.

Two independent TurboCider sessions stay alive for the entire measured run:
``A`` is the resident sd.cpp Metal path and ``B`` is the disk-backed streaming
path.  Requests are interleaved (ABBA by default), use the same model and
sampling inputs, and report both decoded-pixel parity and the child process's
lifetime peak physical footprint.

The default gates treat a slowdown above two percent as material, require at
least a 25 percent physical-footprint reduction, and require exact decoded RGB
pixels.  The raw thresholds and the stricter no-slowdown diagnostic are both
written to the report so a relaxed gate cannot be mistaken for exact parity.
"""

from __future__ import annotations

import argparse
import ctypes as C
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchmark_z_image_gguf import (
    component,
    consume,
    pixel_metrics,
    resolve_gguf,
    sha256,
)


def validate_order(order: str, runs: int, minimum_samples: int) -> str:
    value = order.upper()
    if runs < 1:
        raise ValueError("runs must be positive")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    if not value or set(value) - {"A", "B"}:
        raise ValueError("order must contain only A and B")
    if value.count("A") != value.count("B") or value.count("A") == 0:
        raise ValueError("order must contain the same non-zero number of A and B requests")
    if value.count("A") * runs < minimum_samples:
        raise ValueError(
            "order and runs do not provide the requested minimum samples per route"
        )
    return value


def configure_library(path: Path):
    library = C.CDLL(str(path))
    library.tc_string_free.argtypes = [C.c_void_p]
    library.tc_string_free.restype = None
    library.tc_engine_create_model.argtypes = [
        C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
    ]
    library.tc_engine_create_model.restype = C.c_int
    common = [
        C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p,
        C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
    ]
    library.tc_engine_generate.argtypes = common
    library.tc_engine_generate.restype = C.c_int
    library.tc_engine_prepare.argtypes = common[:2] + [C.c_int] + common[2:]
    library.tc_engine_prepare.restype = C.c_int
    library.tc_engine_free.argtypes = [C.c_void_p]
    library.tc_engine_free.restype = None
    return library


def create_engine(library, checkpoint: Path) -> C.c_void_p:
    engine, error = C.c_void_p(), C.c_void_p()
    status = library.tc_engine_create_model(
        b"z-image-turbo-gguf", str(checkpoint).encode(),
        C.byref(engine), C.byref(error),
    )
    message = consume(library, error)
    if status:
        raise RuntimeError(message)
    return engine


def request_value(args: argparse.Namespace, residency: str, output: Path) -> dict:
    streaming = residency == "streaming"
    value = {
        "model": "z-image-turbo-gguf",
        "model_variant": args.variant,
        "operation": "image.generate",
        "execution": "gpu",
        "residency": residency,
        "streaming_offload": streaming,
        "memory_budget_bytes": args.memory_budget_bytes if streaming else 0,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "frames": 1,
        "audio": False,
        "output": str(output.resolve()),
    }
    if args.lora:
        value["lora_strategy"] = "inference_time"
        value["loras"] = [{
            "path": str(args.lora.resolve()),
            "strength": args.lora_strength,
            "role": "transformer",
        }]
    return value


def call(library, engine: C.c_void_p, request: dict, *, prepare: bool = False) -> tuple[float, dict]:
    result, error = C.c_void_p(), C.c_void_p()
    started = time.perf_counter()
    if prepare:
        status = library.tc_engine_prepare(
            engine, json.dumps(request).encode(), 0, None, None,
            C.byref(result), C.byref(error),
        )
    else:
        status = library.tc_engine_generate(
            engine, json.dumps(request).encode(), None, None,
            C.byref(result), C.byref(error),
        )
    elapsed = time.perf_counter() - started
    payload, message = consume(library, result), consume(library, error)
    if status:
        raise RuntimeError(message)
    return elapsed, json.loads(payload)


def report_peak(report: dict, key: str) -> int:
    memory = report.get("memory", {})
    value = memory.get(key, 0) if isinstance(memory, dict) else 0
    return int(value) if isinstance(value, (int, float)) else 0


def comparison(
    resident_seconds: list[float],
    streaming_seconds: list[float],
    resident_reports: list[dict],
    streaming_reports: list[dict],
    parity: list[dict],
    *,
    maximum_performance_ratio: float,
    minimum_footprint_reduction: float,
    minimum_samples: int,
) -> dict:
    if not resident_seconds or not streaming_seconds:
        raise ValueError("both routes require measured samples")
    resident_median = statistics.median(resident_seconds)
    streaming_median = statistics.median(streaming_seconds)
    performance_ratio = streaming_median / resident_median
    resident_peak = max(
        (report_peak(value, "peak_physical_footprint_bytes") for value in resident_reports),
        default=0,
    )
    streaming_peak = max(
        (report_peak(value, "peak_physical_footprint_bytes") for value in streaming_reports),
        default=0,
    )
    footprint_ratio = streaming_peak / resident_peak if resident_peak else float("inf")
    footprint_reduction = 1.0 - footprint_ratio
    sample_gate = (
        len(resident_seconds) >= minimum_samples and
        len(streaming_seconds) >= minimum_samples
    )
    performance_gate = performance_ratio <= maximum_performance_ratio
    footprint_gate = (
        resident_peak > 0 and streaming_peak > 0 and
        footprint_reduction >= minimum_footprint_reduction
    )
    parity_gate = bool(parity) and all(value.get("pixel_exact") for value in parity)
    return {
        "resident_median_seconds": resident_median,
        "streaming_median_seconds": streaming_median,
        "streaming_over_resident": performance_ratio,
        "strict_no_slowdown_passed": performance_ratio <= 1.0,
        "resident_peak_physical_footprint_bytes": resident_peak,
        "streaming_peak_physical_footprint_bytes": streaming_peak,
        "streaming_footprint_ratio": footprint_ratio,
        "streaming_footprint_reduction": footprint_reduction,
        "minimum_pair_correlation": min(
            (float(value.get("correlation", 0.0)) for value in parity), default=0.0
        ),
        "maximum_pair_mae_255": max(
            (float(value.get("mae_255", float("inf"))) for value in parity),
            default=float("inf"),
        ),
        "gates": {
            "minimum_samples_per_route": minimum_samples,
            "sample_count_passed": sample_gate,
            "maximum_performance_ratio": maximum_performance_ratio,
            "performance_passed": performance_gate,
            "minimum_footprint_reduction": minimum_footprint_reduction,
            "footprint_passed": footprint_gate,
            "decoded_pixel_exact_required": True,
            "parity_passed": parity_gate,
        },
        "passed": sample_gate and performance_gate and footprint_gate and parity_gate,
    }


def machine_info() -> dict:
    def sysctl(name: str) -> str:
        try:
            return subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", name], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    memory = sysctl("hw.memsize")
    return {
        "platform": platform.platform(),
        "hardware_model": sysctl("hw.model"),
        "physical_memory_bytes": int(memory) if memory.isdigit() else None,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--variant", default="Q3_K_S")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--server", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", default="ABBA")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--minimum-samples", type=int, default=4)
    parser.add_argument("--memory-budget-bytes", type=int, default=8 << 30)
    parser.add_argument("--maximum-performance-ratio", type=float, default=1.02)
    parser.add_argument("--minimum-footprint-reduction", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        default="A cinematic red fox walking through fresh snow, soft morning light.",
    )
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    try:
        order = validate_order(args.order, args.runs, args.minimum_samples)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.warmups < 0:
        raise SystemExit("warmups must be nonnegative")
    if args.memory_budget_bytes < (1 << 30):
        raise SystemExit("memory budget must be at least 1 GiB")
    if args.maximum_performance_ratio <= 0:
        raise SystemExit("maximum performance ratio must be positive")
    if not 0.0 <= args.minimum_footprint_reduction < 1.0:
        raise SystemExit("minimum footprint reduction must be in [0, 1)")

    root = args.model_root.resolve()
    gguf = resolve_gguf(root, args.variant)
    component_root = root if root.is_dir() else root.parent
    vae = component(component_root, "split_files/vae/ae.safetensors")
    llm = component(component_root, "split_files/text_encoders/qwen_3_4b.safetensors")
    server = args.server
    if server is None:
        configured = os.environ.get("TURBOCIDER_SD_CPP_BIN")
        server = Path(configured) if configured else component_root / "bin/sd-server"
    server = server.resolve()
    library_path = args.library.resolve()
    if not server.is_file() or not os.access(server, os.X_OK):
        raise SystemExit(f"sd-server is missing or not executable: {server}")
    if not library_path.is_file():
        raise SystemExit(f"TurboCider library is missing: {library_path}")
    if args.lora and not args.lora.is_file():
        raise SystemExit(f"LoRA is missing: {args.lora}")

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["TURBOCIDER_SD_CPP_BIN"] = str(server)
    os.environ["TURBOCIDER_Z_IMAGE_VAE"] = str(vae)
    os.environ["TURBOCIDER_Z_IMAGE_LLM"] = str(llm)
    os.environ.pop("TURBOCIDER_Z_GGUF_NATIVE_GPU", None)

    library = configure_library(library_path)
    engines = {}
    seconds = {"resident": [], "streaming": []}
    reports = {"resident": [], "streaming": []}
    outputs = {"resident": [], "streaming": []}
    sequence = []
    prepare_reports = {}
    warmup_seconds = {"resident": [], "streaming": []}
    try:
        for residency in ("resident", "streaming"):
            engines[residency] = create_engine(library, gguf)
        for residency in ("resident", "streaming"):
            print(f"preparing {residency} session", file=sys.stderr, flush=True)
            request = request_value(
                args, residency, args.output / f"prepare-{residency}.png"
            )
            elapsed, report = call(
                library, engines[residency], request, prepare=True
            )
            prepare_reports[residency] = {
                "wall_seconds": elapsed,
                "report": report,
            }

        for index in range(args.warmups):
            for residency in ("resident", "streaming"):
                print(
                    f"warming {residency} session {index + 1}/{args.warmups}",
                    file=sys.stderr,
                    flush=True,
                )
                output = args.output / f"warmup-{residency}-{index:03d}.png"
                elapsed, _ = call(
                    library, engines[residency], request_value(args, residency, output)
                )
                warmup_seconds[residency].append(elapsed)

        counters = {"resident": 0, "streaming": 0}
        for symbol in order * args.runs:
            residency = "resident" if symbol == "A" else "streaming"
            index = counters[residency]
            counters[residency] += 1
            print(
                f"measuring {residency} request {index + 1}/"
                f"{order.count(symbol) * args.runs}",
                file=sys.stderr,
                flush=True,
            )
            output = args.output / f"{residency}-{index:03d}.png"
            elapsed, report = call(
                library, engines[residency], request_value(args, residency, output)
            )
            seconds[residency].append(elapsed)
            reports[residency].append(report)
            outputs[residency].append(output)
            sequence.append({
                "route": residency,
                "index": index,
                "seconds": elapsed,
                "output": output.name,
                "sha256": sha256(output),
                "runtime_backend": report.get("runtime_backend"),
                "memory": report.get("memory"),
            })
    finally:
        for engine in engines.values():
            library.tc_engine_free(engine)

    parity = []
    for resident, streaming in zip(outputs["resident"], outputs["streaming"]):
        metrics = pixel_metrics(resident, streaming)
        metrics.update({
            "resident": resident.name,
            "streaming": streaming.name,
            "resident_sha256": sha256(resident),
            "streaming_sha256": sha256(streaming),
        })
        parity.append(metrics)
    result_comparison = comparison(
        seconds["resident"], seconds["streaming"],
        reports["resident"], reports["streaming"], parity,
        maximum_performance_ratio=args.maximum_performance_ratio,
        minimum_footprint_reduction=args.minimum_footprint_reduction,
        minimum_samples=args.minimum_samples,
    )
    result = {
        "schema_version": 1,
        "model": "z-image-turbo-gguf",
        "checkpoint": {
            "filename": gguf.name,
            "sha256": sha256(gguf),
            "variant": args.variant,
        },
        "components": {
            "vae_sha256": sha256(vae),
            "text_encoder_sha256": sha256(llm),
        },
        "runtime": {
            "library_sha256": sha256(library_path),
            "sd_server_sha256": sha256(server),
            "resident_flags": [
                "--diffusion-fa", "--diffusion-conv-direct", "--clip-on-cpu",
                "--cfg-scale", "1.0",
            ],
            "streaming_extra_flags": [
                "--params-backend", "disk", "--mmap", "--stream-layers",
                "--max-vram", f"{args.memory_budget_bytes / (1 << 30):.3f}",
                "--vae-tiling",
            ],
        },
        "machine": machine_info(),
        "workload": {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "seed": args.seed,
            "prompt": args.prompt,
            "memory_budget_bytes": args.memory_budget_bytes,
            "order": order,
            "order_repetitions": args.runs,
            "warmups_per_route": args.warmups,
            "measured_samples_per_route": len(seconds["resident"]),
            "lora": None if not args.lora else {
                "filename": args.lora.name,
                "sha256": sha256(args.lora),
                "strength": args.lora_strength,
                "strategy": "inference_time",
            },
        },
        "prepare": prepare_reports,
        "warmup_seconds": warmup_seconds,
        "sequence": sequence,
        "resident_seconds": seconds["resident"],
        "streaming_seconds": seconds["streaming"],
        "parity": parity,
        "comparison": result_comparison,
        "notes": [
            "Both TurboCider sessions stayed alive for the measured interleaved sequence.",
            "Physical footprint is the child sd-server lifetime maximum; sampled RSS is recorded but is not the low-memory acceptance metric.",
            "The 1.02 default performance ratio is a material-regression gate; strict no-slowdown is reported separately and is not implied by passing it.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result_comparison["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
