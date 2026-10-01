#!/usr/bin/env python3
"""Resident/streamed 1/2/3-slot real encoder contract screen, not qualification."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

from compare_gguf_execution_outputs import load

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--q8", type=Path, required=True)
    p.add_argument("--q4k", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.is_symlink(): p.error("output already exists")
    a.output.mkdir(parents=True)
    binary = ROOT / "build/quantized-execution/qwen3-gguf-probe"
    sources = ["native/runtime/streaming/gguf_weight_pager.cpp", "native/components/text/qwen3_gguf.cpp",
               "native/models/z_image/gguf_execution.cpp", "native/core/quantized_execution.cpp",
               "tools/validation/run_gguf_streamed_validation.py"]
    report = {"schema": "tc-streamed-gguf-encoder-screen-v1", "production_qualified": False,
              "library_sha256": digest(binary.parent / "libturbocider.dylib"), "binary_sha256": digest(binary),
              "source_sha256": {s: digest(ROOT / s) for s in sources}, "cases": [], "negative_cases": []}
    prompt = "A studio photograph of an adult ceramic artist holding a blue cup."

    def save():
        (a.output / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    with open("/tmp/turbocider-z-image-gpu-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, checkpoint in [("q8", a.q8), ("q4k", a.q4k)]:
            golden = None
            for residency in ("packed_resident", "packed_streamed"):
                for prefetch in (0, 1, 2):
                    budget = 8 << 30 if residency == "packed_resident" else (prefetch + 1) * (256 << 20)
                    directory = a.output / f"{name}-{residency}-p{prefetch}"
                    command = [str(binary), str(checkpoint), str(a.config), str(a.tokenizer), str(directory),
                               str(prefetch), "-", prompt, str(budget), "simd", residency]
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                    (a.output / (directory.name + ".stdout.txt")).write_text(result.stdout)
                    (a.output / (directory.name + ".stderr.txt")).write_text(result.stderr)
                    if result.returncode: raise RuntimeError(result.stderr)
                    m = json.loads(result.stdout)
                    _, shape, sha = load(directory / "conditioning.safetensors")
                    if golden is None: golden = sha
                    if sha != golden or m["fills"] != 35 or m["slots"] != prefetch + 1:
                        raise ArithmeticError("wrong output or execution counts")
                    if residency == "packed_streamed" and (m["packed_capacity_bytes"] != 0 or m["read_buffer_bytes"] != 1 << 20):
                        raise ArithmeticError("hidden packed residency or extra read buffer")
                    if m["managed_peak_bytes"] > budget: raise ArithmeticError("managed budget exceeded")
                    report["cases"].append({"name": name, "residency": residency, "prefetch": prefetch,
                                            "budget": budget, "conditioning_sha256": sha, "shape": shape, "metrics": m})
                    print(name, residency, prefetch, "exact", flush=True); save()
            # Same small budget cannot hold the resident source; never silently
            # switch residency or reduce slots to make an unauthorized run pass.
            result = subprocess.run([str(binary), str(checkpoint), str(a.config), str(a.tokenizer),
                str(a.output / (name + "-resident-budget-rejection")), "0", "-", prompt,
                str(256 << 20), "simd", "packed_resident"], cwd=ROOT, capture_output=True, text=True, timeout=120)
            if not result.returncode or "qe_budget_floor" not in result.stderr:
                raise ArithmeticError("resident floor failed to reject")
            report["negative_cases"].append({"name": name, "resident_rejected_at_256_mib": True}); save()
        report["status"] = "completed"; save()


if __name__ == "__main__": main()
