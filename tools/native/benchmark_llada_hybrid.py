#!/usr/bin/env python3
"""Paired resident LLaDA MPS and experimental MPS/Core ML benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

from quality_gate import image_metrics, passes


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=root / "build/native/turbocider")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A red fox in fresh snow, cinematic photograph")
    parser.add_argument("--modes", nargs="+", choices=("gpu", "gpu_ane"),
                        default=("gpu", "gpu_ane"))
    parser.add_argument("--require-speedup", type=float, default=0.0)
    parser.add_argument("--require-quality", action="store_true",
                        help="fail when any paired GPU/ANE image misses the quality gate")
    parser.add_argument("--min-correlation", type=float, default=0.99,
                        help="minimum paired RGB correlation for --require-quality")
    parser.add_argument("--min-cosine", type=float, default=0.995,
                        help="minimum paired RGB cosine for --require-quality")
    parser.add_argument("--max-mae-255", type=float, default=5.0,
                        help="maximum paired RGB MAE for --require-quality")
    return parser.parse_args()


def run_mode(args: argparse.Namespace, mode: str, output: Path) -> dict:
    directory = output / mode
    directory.mkdir(parents=True, exist_ok=True)
    requests = []
    images = []
    for index in range(args.iterations):
        image = directory / f"request-{index}.png"
        request = {
            "schema_version": 2,
            "model": "llada-image-turbo",
            "operation": "image.generate",
            "inputs": [{"kind": "text", "role": "prompt", "text": args.prompt}],
            "outputs": [{
                "kind": "image", "path": str(image.resolve()),
                "width": args.width, "height": args.height, "frames": 1,
            }],
            "sampling": {"seed": args.seed, "steps": args.steps},
            "execution": {"policy": mode, "residency": "resident"},
        }
        if mode == "gpu_ane":
            request["execution"].update({
                "allow_approximation": True,
                "ane_manifest": str(args.manifest.resolve()),
            })
        path = directory / f"request-{index}.json"
        path.write_text(json.dumps(request, indent=2) + "\n")
        requests.append(path)
        images.append(image)

    command = [str(args.cli.resolve()), "batch", str(args.model.resolve())]
    command.extend(str(path.resolve()) for path in requests)
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["TURBOCIDER_LLADA_CONDITIONING_CACHE_DIR"] = str(
        (output / "conditioning-cache").resolve()
    )
    completed = subprocess.run(command, text=True, capture_output=True, env=environment)
    process_seconds = time.perf_counter() - started
    (directory / "stdout.log").write_text(completed.stdout)
    (directory / "stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(
            f"{mode} batch failed ({completed.returncode}):\n{completed.stderr[-8000:]}"
        )
    results = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("model") == "llada-image-turbo":
            results.append(value)
    if len(results) != args.iterations:
        raise RuntimeError(f"expected {args.iterations} {mode} results, received {len(results)}")
    walls = [float(value["timings_seconds"]["request_wall"]) for value in results]
    warm = walls[1:] if len(walls) > 1 else walls
    return {
        "mode": mode,
        "process_seconds": process_seconds,
        "request_wall_seconds": walls,
        "first_request_seconds": walls[0],
        "warm_median_seconds": statistics.median(warm),
        "outputs": [str(path.resolve()) for path in images],
        "last_result": results[-1],
    }


def main() -> int:
    args = arguments()
    if args.iterations < 2:
        raise ValueError("at least two iterations are required for a resident warm comparison")
    if args.width % 16 or args.height % 16 or args.steps != 4:
        raise ValueError("LLaDA hybrid dimensions must be multiples of 16 and use 4 steps")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "model": "llada-image-turbo",
        "workload": {
            "operation": "image.generate", "width": args.width, "height": args.height,
            "steps": args.steps, "seed": args.seed, "prompt": args.prompt,
        },
        "manifest": str(args.manifest.resolve()),
        "method": "separate native C++/MLX processes; first request excluded from warm median; wall includes PNG export",
        "modes": {},
    }
    for mode in args.modes:
        result = run_mode(args, mode, output)
        report["modes"][mode] = result
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"mode": mode, "warm_median_seconds": result["warm_median_seconds"]}),
              flush=True)
    if "gpu" in report["modes"] and "gpu_ane" in report["modes"]:
        gpu = report["modes"]["gpu"]
        hybrid = report["modes"]["gpu_ane"]
        speedup = gpu["warm_median_seconds"] / hybrid["warm_median_seconds"]
        parity = [
            image_metrics(Path(a), Path(b))
            for a, b in zip(gpu["outputs"], hybrid["outputs"])
        ]
        report["comparison"] = {
            "warm_speedup": speedup,
            "target_speedup": 1.2,
            "target_passed": speedup >= 1.2,
            "png_pairs": parity,
            "quality_thresholds": {
                "minimum_correlation": args.min_correlation,
                "minimum_cosine": args.min_cosine,
                "maximum_mae_255": args.max_mae_255,
            },
            "quality_gate_passed": all(
                passes(
                    item,
                    min_correlation=args.min_correlation,
                    min_cosine=args.min_cosine,
                    max_mae_255=args.max_mae_255,
                )
                for item in parity
            ),
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report.get("comparison", {}), indent=2))
    if args.require_speedup and report.get("comparison", {}).get("warm_speedup", 0) < args.require_speedup:
        return 2
    if args.require_quality and not report.get("comparison", {}).get("quality_gate_passed", False):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
