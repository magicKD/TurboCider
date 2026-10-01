#!/usr/bin/env python3
"""Bounded mixed-K encoder scalar/SIMD comparison; small screening campaign."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--prefetch", type=int, choices=[0, 1], default=1)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.is_symlink(): p.error("output already exists")
    a.output.mkdir(parents=True)
    with open("/tmp/turbocider-z-image-gpu-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run(["bash", "tools/native/build_qwen3_source_probes.sh"], cwd=ROOT, check=True)
        binary = ROOT / "build/quantized-execution/qwen3-decoder-benchmark"
        result = subprocess.run([str(binary), str(a.checkpoint), str(a.config), str(a.tokenizer),
                                 str(a.output / "tensors"), str(a.prefetch), str(a.iterations)],
                                cwd=ROOT, text=True, capture_output=True, timeout=1800)
        (a.output / "raw.stdout.json").write_text(result.stdout)
        (a.output / "raw.stderr.txt").write_text(result.stderr)
        if result.returncode: raise RuntimeError(result.stderr)
        raw = json.loads(result.stdout)
        samples = [x for x in raw["samples"] if not x["warmup"]]
        medians = {mode: statistics.median(x["wall_seconds"] for x in samples if x["decoder"] == mode)
                   for mode in ["cpu_scalar", "cpu_simd"]}
        source_files = ["native/core/gguf_decode.cpp", "native/components/text/qwen3_gguf.cpp",
                        "native/runtime/streaming/gguf_weight_pager.cpp", "tools/native/qwen3_decoder_benchmark.cpp",
                        "tools/native/benchmark_qwen3_decoder.py"]
        receipt = {"scope": "same-source/math/budget whole encoder, hot process, reload weights; not full-request performance qualification",
                   "raw": raw, "median_wall_seconds": medians, "scalar_over_simd": medians["cpu_scalar"] / medians["cpu_simd"],
                   "library_sha256": digest(binary.parent / "libturbocider.dylib"), "binary_sha256": digest(binary),
                   "source_sha256": {s: digest(ROOT / s) for s in source_files},
                   "checkpoint_sha256": digest(a.checkpoint), "config_sha256": digest(a.config), "tokenizer_sha256": digest(a.tokenizer),
                   "raw_stdout_sha256": digest(a.output / "raw.stdout.json")}
        (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"median_seconds": medians, "scalar_over_simd": receipt["scalar_over_simd"]}, indent=2))


if __name__ == "__main__": main()
