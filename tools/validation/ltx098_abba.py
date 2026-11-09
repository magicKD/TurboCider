#!/usr/bin/env python3
"""Fresh-process A/B/B/A GPU comparison and all-frame video diagnostics.

A is the public TurboCider CLI, B is an owned stock-ComfyUI HTTP instance.
The two matched seeds use identical image, prompt, geometry, sampler and VAE
mode. Different runtimes still use different decoder RNGs and MP4 encoders.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--comfy-python", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    cli, model, image = (p.resolve(strict=True) for p in (args.cli, args.model, args.input))
    # Preserve the virtualenv executable path: resolving its symlink can make
    # Python ignore the environment's pyvenv.cfg and lose its dependencies.
    comfy_python = args.comfy_python.absolute()
    if not comfy_python.is_file():
        parser.error("ComfyUI virtualenv Python is missing")
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if not args.prompt or not 0 <= args.seed < 0xffffffffffffffff or not 0 <= args.strength <= 1:
        parser.error("prompt, uint64 seed pair and strength in [0,1] required")
    tools = Path(__file__).resolve().parent
    output.mkdir(parents=True)
    report = {"schema": "ltx098-abba-v1", "status": "running",
              "scope": "A native CLI, B owned stock ComfyUI HTTP; fresh process each request, matched seeds; 512x512x121, 24fps, eight Euler steps; wall includes MP4 export; different decoder RNG/MP4 encoders",
              "prompt": args.prompt, "input": str(image), "strength": args.strength, "runs": [], "quality": []}

    def invoke(label, command):
        result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        (output / f"{label}.stdout.txt").write_text(result.stdout)
        (output / f"{label}.stderr.txt").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} exited {result.returncode}: {result.stderr[-4000:]}")

    try:
        for label, variant, seed in (("a0", "native", args.seed), ("b0", "comfy", args.seed),
                                     ("b1", "comfy", args.seed + 1), ("a1", "native", args.seed + 1)):
            destination = output / label
            if variant == "native":
                command = [sys.executable, str(tools / "ltx098_native_reference.py"), "--cli", str(cli),
                           "--model", str(model), "--input", str(image), "--output", str(destination), "--count", "1",
                           "--prompt", args.prompt, "--seed", str(seed), "--strength", str(args.strength)]
            else:
                command = [sys.executable, str(tools / "ltx098_comfy_reference.py"), "--python", str(comfy_python),
                           "--model", str(model), "--input", str(image), "--output", str(destination), "--count", "1",
                           "--decoder", "full", "--prompt", args.prompt, "--seed", str(seed), "--strength", str(args.strength)]
            invoke(label, command)
            summary = json.loads((destination / "summary.json").read_text())
            if variant == "native":
                if summary["status"] != "complete":
                    raise ValueError("native request incomplete")
                seconds = summary["runs"][0]["client_wall_seconds"]
            else:
                if summary["status"] != "complete_reference_only" or not summary["server_terminal"]:
                    raise ValueError("owned Comfy request/server incomplete")
                seconds = summary["report"]["runs"][0]["request_wall_including_export"]
            row = {"label": label, "variant": variant, "seed": seed, "wall_seconds": seconds}
            report["runs"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        for index in range(2):
            quality_path = output / f"quality-{index}.json"
            command = [sys.executable, str(tools / "ltx098_video_comparison.py"),
                       "--reference", str(output / f"b{index}/0.mp4"),
                       "--candidate", str(output / f"a{index}/0.mp4"), "--output", str(quality_path)]
            invoke(f"quality-{index}", command)
            quality = json.loads(quality_path.read_text())
            report["quality"].append({"seed": args.seed + index, "contract_passed": quality["contract_passed"],
                "aligned_rgb_gate_passed": quality["aligned_rgb_gate_passed"],
                "mean_mae_255": quality["metrics"]["mean_mae_255"],
                "mean_correlation": quality["metrics"]["mean_correlation"],
                "maximum_motion_relative_error": quality["metrics"]["maximum_motion_relative_error"]})
        native = [r["wall_seconds"] for r in report["runs"] if r["variant"] == "native"]
        comfy = [r["wall_seconds"] for r in report["runs"] if r["variant"] == "comfy"]
        report.update(native_median_seconds=statistics.median(native), comfy_median_seconds=statistics.median(comfy),
                      matched_speedups=[b / a for a, b in zip(native, comfy)],
                      all_native_faster=all(a < b for a, b in zip(native, comfy)),
                      all_quality_gates_passed=all(row["aligned_rgb_gate_passed"] for row in report["quality"]),
                      status="complete")
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("native_median_seconds", "comfy_median_seconds",
          "matched_speedups", "all_native_faster", "all_quality_gates_passed", "quality")}, indent=2))


if __name__ == "__main__":
    main()
