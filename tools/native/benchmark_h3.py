#!/usr/bin/env python3
"""Compare the current h3.c CLI with TurboCider's native H3 Session.

The benchmark runs fresh processes sequentially in alternating AB/BA order.
It intentionally uses the same merged H3 Turbo model directory and exact
kernel flags on both sides, then requires byte-identical MP4 output unless the
caller explicitly permits an approximate output comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TurboCider H3 against the current h3.c CLI"
    )
    parser.add_argument("--h3-bin", required=True, type=Path)
    parser.add_argument("--turbocider-bin", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="A red fox walking through fresh snow, static camera")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--residency", choices=("resident", "streamed"), default="streamed")
    parser.add_argument(
        "--memory-budget-bytes",
        type=int,
        default=0,
        help="TurboCider streamed working-set target; zero preserves the original two-slot path",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-overhead-percent", type=float, default=5.0)
    parser.add_argument("--allow-approximation", action="store_true")
    parser.add_argument("--allow-output-difference", action="store_true")
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--lora-strength", type=float, default=0.0625)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for label in ("h3_bin", "turbocider_bin"):
        path = getattr(args, label).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"{label.replace('_', ' ')} is missing: {path}")
        setattr(args, label, path)
    args.model = args.model.expanduser().resolve()
    if not args.model.is_dir():
        raise ValueError(f"model directory is missing: {args.model}")
    if args.lora is not None:
        args.lora = args.lora.expanduser().resolve()
        if not args.lora.is_file():
            raise ValueError(f"LoRA file is missing: {args.lora}")
    if args.width < 64 or args.height < 64 or args.width % 32 or args.height % 32:
        raise ValueError("H3 dimensions must be at least 64 and multiples of 32")
    if args.frames < 22 or (args.frames - 5) % 17:
        raise ValueError("H3 frames must be 5+17n and at least 22")
    if args.fps != 24:
        raise ValueError("H3 requires 24 fps")
    if args.steps != 4:
        raise ValueError("the H3 Turbo comparison requires four steps")
    if args.rounds < 1:
        raise ValueError("rounds must be positive")
    if args.memory_budget_bytes < 0:
        raise ValueError("memory budget must be non-negative")
    if args.memory_budget_bytes and args.residency != "streamed":
        raise ValueError("memory budget requires streamed residency")
    if args.max_overhead_percent < 0:
        raise ValueError("max overhead percent must be non-negative")
    if args.allow_approximation and not args.allow_output_difference:
        raise ValueError(
            "--allow-approximation requires --allow-output-difference so an "
            "approximate run cannot be mislabeled as exact parity"
        )


def build_direct_command(args: argparse.Namespace, output: Path) -> list[str]:
    command = [
        str(args.h3_bin),
        "-d", str(args.model),
        "-p", args.prompt,
        "-o", str(output),
        "--width", str(args.width),
        "--height", str(args.height),
        "--frames", str(args.frames),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
        "--layers", "50",
        "--reuse", "1",
        "--use-reference-rope",
    ]
    if args.residency == "streamed":
        command.append("--ssd-streaming")
    if not args.allow_approximation:
        command.extend(
            [
                "--use-slower-bf16-mlp",
                "--use-slower-bf16-qkv",
                "--use-slower-bf16-attention-output",
            ]
        )
    return command


def build_turbocider_request(args: argparse.Namespace, output: Path) -> dict:
    request = {
        "schema_version": 2,
        "model": "minimax-h3-turbo",
        "operation": "video.generate",
        "inputs": [
            {"kind": "text", "role": "prompt", "text": args.prompt},
        ],
        "outputs": [
            {
                "kind": "video",
                "path": str(output),
                "width": args.width,
                "height": args.height,
                "frames": args.frames,
                "fps": args.fps,
                "audio": True,
            }
        ],
        "sampling": {"seed": args.seed, "steps": args.steps},
        "execution": {
            "policy": "gpu",
            "residency": args.residency,
            "allow_approximation": args.allow_approximation,
            "memory_budget_bytes": args.memory_budget_bytes,
        },
    }
    if args.lora is not None:
        request["loras"] = [
            {
                "path": str(args.lora),
                "role": "transformer",
                "strength": args.lora_strength,
            }
        ]
    return request


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_last_json_line(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise RuntimeError("TurboCider did not emit a result JSON object")


def run_process(command: list[str], cwd: Path) -> tuple[float, str, str]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    wall = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {command}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    return wall, completed.stdout, completed.stderr


def probe_media(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,size",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,avg_frame_rate,"
            "nb_frames,sample_rate,channels",
            "-of", "json", str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr}")
    return json.loads(completed.stdout)


def prepare_output_directory(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise ValueError(
            f"output directory is not empty: {path}; use a new directory or --overwrite"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def benchmark(args: argparse.Namespace) -> tuple[dict, bool]:
    output_dir = prepare_output_directory(args.output_dir, args.overwrite)
    doctor_wall, doctor_stdout, _ = run_process(
        [str(args.turbocider_bin), "doctor"], args.turbocider_bin.parent
    )
    doctor = parse_last_json_line(doctor_stdout)
    if not doctor.get("gpu_available"):
        raise RuntimeError("TurboCider doctor reports that no Metal GPU is available")

    orders = (("direct", "turbocider"), ("turbocider", "direct"))
    runs: list[dict] = []
    for round_index in range(args.rounds):
        pair: dict[str, dict] = {}
        for engine in orders[round_index % len(orders)]:
            output = output_dir / f"round-{round_index + 1}-{engine}.mp4"
            if output.exists() and not args.overwrite:
                raise ValueError(f"benchmark output already exists: {output}")
            if engine == "direct":
                command = build_direct_command(args, output)
                wall, stdout, stderr = run_process(command, args.h3_bin.parent)
                metrics = None
            else:
                request = build_turbocider_request(args, output)
                request_path = output_dir / f"round-{round_index + 1}-request.json"
                request_path.write_text(json.dumps(request, indent=2) + "\n")
                command = [
                    str(args.turbocider_bin), "generate", str(args.model),
                    str(request_path),
                ]
                wall, stdout, stderr = run_process(
                    command, args.turbocider_bin.parent
                )
                metrics = parse_last_json_line(stdout)
            (output_dir / f"round-{round_index + 1}-{engine}.stdout.log").write_text(stdout)
            (output_dir / f"round-{round_index + 1}-{engine}.stderr.log").write_text(stderr)
            pair[engine] = {
                "process_wall_seconds": wall,
                "output": str(output),
                "output_bytes": output.stat().st_size,
                "output_sha256": sha256_file(output),
                "media": probe_media(output),
                "metrics": metrics,
            }
            runs.append({"round": round_index + 1, "engine": engine, **pair[engine]})
        pair["direct"]["output_matches_peer"] = (
            pair["direct"]["output_sha256"] == pair["turbocider"]["output_sha256"]
        )
        pair["turbocider"]["output_matches_peer"] = pair["direct"]["output_matches_peer"]
        for run in runs:
            if run["round"] == round_index + 1:
                run["output_matches_peer"] = pair[run["engine"]][
                    "output_matches_peer"
                ]

    direct = [run["process_wall_seconds"] for run in runs if run["engine"] == "direct"]
    native = [run["process_wall_seconds"] for run in runs if run["engine"] == "turbocider"]
    direct_median = statistics.median(direct)
    native_median = statistics.median(native)
    overhead_percent = 100.0 * (native_median / direct_median - 1.0)
    output_exact = all(
        run["output_matches_peer"]
        for run in runs
        if run["engine"] == "direct"
    )
    performance_passed = overhead_percent <= args.max_overhead_percent
    output_passed = output_exact or args.allow_output_difference
    report = {
        "schema": "turbocider-h3-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Fresh-process sequential AB/BA; OS file/Metal caches are not flushed; "
            "process wall includes model preparation, denoise, decode, and MP4 export."
        ),
        "system": doctor,
        "doctor_wall_seconds": doctor_wall,
        "model": str(args.model),
        "request": build_turbocider_request(args, Path("{output}")),
        "runs": runs,
        "comparison": {
            "direct_median_seconds": direct_median,
            "turbocider_median_seconds": native_median,
            "turbocider_overhead_percent": overhead_percent,
            "max_overhead_percent": args.max_overhead_percent,
            "performance_passed": performance_passed,
            "byte_exact_output": output_exact,
            "output_passed": output_passed,
        },
        "passed": performance_passed and output_passed,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report, report["passed"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        report, passed = benchmark(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"benchmark_h3: {error}")
        return 2
    print(json.dumps(report["comparison"], sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
