"""Serial local Z-Image quantization experiment, no automatic downloads.

Each variant runs in a new process. /usr/bin/time reports process peak RSS;
native metrics report MLX peak separately. Prompts/seeds/settings are matched.
Generated layouts use links and never alter source weights. Same-prompt warm
image runs measure prompt-cache reuse, not encoder throughput.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(command, folder):
    folder.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["/usr/bin/time", "-l", *map(str, command)], text=True, capture_output=True)
    (folder / "stdout.log").write_text(result.stdout)
    (folder / "stderr.log").write_text(result.stderr)
    match = re.search(r"(\d+)\s+maximum resident set size", result.stderr)
    record = {"command": list(map(str, command)), "returncode": result.returncode,
              "process_peak_rss_bytes": int(match[1]) if match else None}
    (folder / "process.json").write_text(json.dumps(record, indent=2))
    if result.returncode:
        raise RuntimeError(f"Benchmark failed: {folder}\n{result.stderr[-2000:]}")
    return result.stdout, record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--components", type=Path, required=True)
    p.add_argument("--q4", type=Path, required=True)
    p.add_argument("--q8", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runs", type=int, default=4)
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args()
    a.output = a.output.resolve()
    if a.output.exists():
        p.error("Use a new output directory to preserve prior evidence")
    a.output.mkdir(parents=True)
    components = a.components.resolve()
    split = components / "split_files"
    encoders = {"bf16": split / "text_encoders/qwen_3_4b.safetensors",
                "q4": a.q4.resolve() / "text_encoder/model.safetensors",
                "q8": a.q8.resolve() / "text_encoder/model.safetensors"}
    summary = {"encoder": {}, "image": {}}
    for label, weights in encoders.items():
        for tokens in (32, 128):
            folder = a.output / f"encoder-{label}-{tokens}"
            stdout, process = run([ROOT / "build/native/qwen3-quant-probe", weights, tokens, 3,
                                   folder / "conditioning.safetensors"], folder)
            metrics = json.loads(stdout)
            summary["encoder"][f"{label}-{tokens}"] = {**process, **metrics,
                "warm_median_seconds": statistics.median(s["seconds"] for s in metrics["samples"] if not s["warmup"])}
            print("encoder", label, tokens, summary["encoder"][f"{label}-{tokens}"]["warm_median_seconds"], flush=True)
            (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    for dit, text in (("bf16", "bf16"), ("int8_convrot", "bf16"), ("bf16", "q4"), ("bf16", "q8"), ("int8_convrot", "q4")):
        label = f"{dit}-text-{text}"
        folder = a.output / label
        layout = folder / "layout"
        (layout / "split_files/diffusion_models").mkdir(parents=True)
        (layout / "text_encoder").mkdir()
        (layout / "text_encoder/model.safetensors").symlink_to(encoders[text])
        (layout / "tokenizer").symlink_to(components / "tokenizer", target_is_directory=True)
        (layout / "split_files/vae").symlink_to(split / "vae", target_is_directory=True)
        name = f"z_image_turbo_{dit}.safetensors"
        (layout / "split_files/diffusion_models" / name).symlink_to(split / "diffusion_models" / name)
        stdout, process = run([sys.executable, ROOT / "tools/native/benchmark_native.py",
            "--library", ROOT / "build/native/libturbocider.dylib", "--model", layout,
            "--model-id", "z-image-turbo", "--request", ROOT / "examples/requests/z-image-turbo.json",
            "--output", folder / "images", "--runs", a.runs,
            "--width", a.size, "--height", a.size, "--steps", 9, "--seed", 42,
            "--execution", "gpu"], folder)
        report = json.loads((folder / "images/report.json").read_text())
        summary["image"][label] = {**process, "report": report}
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("image", label, "complete", flush=True)


if __name__ == "__main__":
    main()
