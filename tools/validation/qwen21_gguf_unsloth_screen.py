#!/usr/bin/env python3
"""Read-only Unsloth args builder -> actual pinned sd-cli model child screen."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled
from qwen21_gguf_hybrid_screen import PROMPTS

ROOT = Path(__file__).resolve().parents[2]


def measured_time(log, name):
    matches = re.findall(re.escape(name) + r"[^\n]*(?:taking |in )([0-9]+(?:\.[0-9]+)?)s", log)
    if len(matches) != 1 or float(matches[0]) <= 0:
        raise ValueError("missing or ambiguous actual native timing: " + name)
    return float(matches[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("fresh evidence directory required")
    reference, binary, model = (p.resolve(strict=True) for p in (args.reference, args.binary, args.model))
    backend = reference / "studio/backend"
    sys.dont_write_bytecode = True  # Reference source/cache remains read-only.
    sys.path.insert(0, str(backend))
    from core.inference.sd_cpp_args import (SdCppModelFiles, SdCppGenParams,
                                           build_sd_cpp_command, native_speed_flags)
    source_files = [binary, backend / "core/inference/sd_cpp_args.py", backend / "core/inference/diffusion_memory.py",
                    model / "diffusion_models/qwen-image-2.1-Q4_K_M.gguf",
                    model / "text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
                    model / "vae/qwen_image_2.1_vae_bf16.safetensors"]
    hashes = {str(p): sha256_file(p) for p in source_files}
    args.output.mkdir(parents=True)
    png = (args.output / "image.png").resolve()
    files = SdCppModelFiles(diffusion_model=str(source_files[3]), llm=str(source_files[4]), vae=str(source_files[5]))
    params = SdCppGenParams(prompt=PROMPTS[0], width=512, height=512, steps=40, seed=29, cfg_scale=1)
    command = build_sd_cpp_command(str(binary), files, params, output_path=str(png), threads=12, verbose=True,
                                   extra_args=native_speed_flags("max"))
    if ("--clip-on-cpu" not in command or "--diffusion-fa" not in command or
        "--diffusion-conv-direct" not in command):
        raise ValueError("original Apple CPU-text/default-max policy differs")
    summary = dict(schema="tc-qwen21-unsloth-native-screen-v1", status="running", qualification_passed=False,
                   source_identities=hashes, command=command, prompt=PROMPTS[0], steps=40,
                   scope="one fresh-process request; actual Unsloth default Apple CPU text / Metal DiT; no warm-lifecycle match")
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    start = time.perf_counter()
    with (args.output / "reference.stdout.txt").open("x") as stdout, (args.output / "reference.stderr.txt").open("x") as stderr:
        memory = run_sampled(command, repo=ROOT, output=args.output, stem="reference", env=benchmark_environment(),
                             stdout=stdout, stderr=stderr, timeout=600, interval_ms=100, max_gap_ms=500)
    summary["sampled_child_wall_seconds"] = time.perf_counter() - start
    log = (args.output / "reference.stdout.txt").read_text() + (args.output / "reference.stderr.txt").read_text()
    summary["native_timings_seconds"] = {
        "condition": measured_time(log, "get_learned_condition completed"),
        "sampling": measured_time(log, "sampling completed"),
        "generation": measured_time(log, "generate_image completed"),
    }
    if (not png.is_file() or any(sha256_file(p) != hashes[str(p)] for p in source_files) or
        "compute CPU, params cpu" not in log):
        raise ValueError("source changed, original text placement missing or output missing")
    summary.update(memory=memory, png_sha256=sha256_file(png), status="complete_diagnostic")
    destination.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
