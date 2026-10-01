#!/usr/bin/env python3
"""Real encoder component screen; correctness and source loss stay separate."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from compare_gguf_execution_outputs import load

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(reference, candidate):
    a, shape, sha = load(candidate)
    b, expected_shape, expected_sha = load(reference)
    if shape != expected_shape:
        raise ValueError("conditioning shape changed")
    denominator = max(np.linalg.norm(b), 1e-12)
    return {"shape": shape, "exact": sha == expected_sha, "reference_sha256": expected_sha,
            "candidate_sha256": sha, "rel_l2": float(np.linalg.norm(a - b) / denominator),
            "cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
            "max_abs": float(np.max(np.abs(a - b)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=ROOT / "build/quantized-execution/qwen3-gguf-probe")
    parser.add_argument("--q8", type=Path, required=True)
    parser.add_argument("--q4k", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    args.output.mkdir(parents=True)
    prompts = ["A studio photograph of an adult ceramic artist holding a blue cup.",
               "A wide photograph of a quiet coastal workshop at sunrise. An adult craftsperson repairs a wooden model boat. "
               "A blue notebook and three small brass tools lie to the left. A sleeping orange cat rests under the table. "
               "Tall windows reveal a harbor, fishing boats and a distant lighthouse. Preserve the spatial arrangement and every material. " * 4]
    sources = ["native/components/text/qwen3.cpp", "native/components/text/qwen3.hpp",
               "native/components/text/qwen3_gguf.cpp", "native/components/text/qwen3_gguf.hpp",
               "native/platform/apple/qwen3_gguf_config.mm", "native/core/gguf_directory.hpp",
               "native/runtime/streaming/gguf_weight_pager.cpp", "tools/native/qwen3_gguf_probe.cpp",
               "tools/validation/run_qwen3_gguf_validation.py"]
    report = {"schema": "tc-qwen3-gguf-component-screen-v1", "scope": "selected real inputs, not full R2/media/memory/performance qualification",
              "probe_sha256": digest(args.probe), "library_sha256": digest(args.probe.parent / "libturbocider.dylib"),
              "source_sha256": {path: digest(ROOT / path) for path in sources},
              "config_sha256": digest(args.config), "tokenizer_sha256": digest(args.tokenizer),
              "production_qualified": False, "cases": [], "negative_cases": []}

    def save():
        (args.output / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    with open("/tmp/turbocider-z-image-gpu-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, prompt in enumerate(prompts):
            for name, checkpoint in [("q8", args.q8), ("q4k", args.q4k)]:
                for prefetch in (0, 1):
                    directory = args.output / f"{name}-prompt{index}-p{prefetch}"
                    command = [str(args.probe), str(checkpoint), str(args.config), str(args.tokenizer),
                               str(directory), str(prefetch), str(args.reference) if prefetch == 0 else "-", prompt]
                    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
                    (directory.parent / (directory.name + ".stdout.txt")).write_text(result.stdout)
                    (directory.parent / (directory.name + ".stderr.txt")).write_text(result.stderr)
                    if result.returncode:
                        raise RuntimeError(result.stderr)
                    entry = {"name": name, "prompt": prompt, "prefetch": prefetch, "metrics": json.loads(result.stdout)}
                    if prefetch == 0:
                        entry["difference_from_original_bf16"] = compare(directory / "reference.safetensors", directory / "conditioning.safetensors")
                        # Source quantization loss is not a D1 implementation error.
                        entry["comparison_scope"] = "different source representation; not an O1 implementation oracle"
                    else:
                        previous = args.output / f"{name}-prompt{index}-p0"
                        entry["d1_slot_parity"] = compare(previous / "conditioning.safetensors", directory / "conditioning.safetensors")
                        if not entry["d1_slot_parity"]["exact"]:
                            raise ArithmeticError("single/double-slot output mismatch")
                    if entry["metrics"]["fills"] != 35 or entry["metrics"]["slots"] != 1 + prefetch:
                        raise ArithmeticError("actual compute/slot counts differ")
                    report["cases"].append(entry)
                    save()
                    print(name, index, prefetch, "PASS", flush=True)
        with tempfile.TemporaryDirectory(prefix="tc-qwen3-negative-") as raw:
            temporary = Path(raw)
            bad_config = temporary / "config.json"
            value = json.loads(args.config.read_text()); value["hidden_size"] = 2048
            bad_config.write_text(json.dumps(value))
            bad_tokenizer = temporary / "tokenizer.json"
            value = json.loads(args.tokenizer.read_text())
            keys = list(value["model"]["vocab"]); a, b = keys[:2]
            value["model"]["vocab"][a], value["model"]["vocab"][b] = value["model"]["vocab"][b], value["model"]["vocab"][a]
            bad_tokenizer.write_text(json.dumps(value))
            for name, config, tokenizer, prefetch, budget, setting, expected in [
                ("wrong_config", bad_config, args.tokenizer, 0, 8 << 30, {}, "qe_adapter_mismatch"),
                ("tokenizer_conflict", args.config, bad_tokenizer, 0, 8 << 30, {}, "tokenizer token-ID mapping differs"),
                ("budget_floor", args.config, args.tokenizer, 0, 1, {}, "qe_budget_floor"),
                ("unsupported_p2", args.config, args.tokenizer, 2, 8 << 30, {}, "qe_config_conflict"),
                ("lazy_interval", args.config, args.tokenizer, 0, 8 << 30, {"TURBOCIDER_QWEN3_EVAL_INTERVAL": "4"}, "qe_config_conflict")]:
                import os
                result = subprocess.run([str(args.probe), str(args.q8), str(config), str(tokenizer),
                    str(temporary / name), str(prefetch), "-", prompts[0], str(budget)],
                    cwd=ROOT, env={**os.environ, **setting}, text=True, capture_output=True, timeout=120)
                if result.returncode == 0 or expected not in result.stderr:
                    raise ArithmeticError("negative case failed to reject: " + name + result.stderr)
                report["negative_cases"].append({"name": name, "rejected": True, "code": expected})
                save()
    report["status"] = "completed"; save()


if __name__ == "__main__":
    main()
