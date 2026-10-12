#!/usr/bin/env python3
"""Matched scalar/SIMD CPU decode on real GGUF tensors; not generation qualification."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tensors", nargs="+", default=["layers.0.feed_forward.w1.weight", "layers.0.feed_forward.w2.weight"])
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--max-buffer-bytes", type=int, default=256 << 20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file(): parser.error("checkpoint missing")
    if args.output.exists() or args.output.is_symlink(): parser.error("output already exists")
    if not 8 <= args.iterations <= 100: parser.error("iterations must be 8..100")
    with open("/tmp/turbocider-gguf-decode-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix="tc-gguf-decode-benchmark-") as temporary:
            binary = Path(temporary)/"gguf-decode-probe"
            sources = [ROOT/"tools/native/gguf_decode_probe.cpp", ROOT/"native/core/gguf_decode.cpp"]
            command = ["xcrun", "clang++", "-std=c++20", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                       *(str(path) for path in sources), "-o", str(binary)]
            subprocess.run(command, check=True)
            receipt = {"schema_version": 1, "time_utc": datetime.now(timezone.utc).isoformat(),
                       "scope": "real GGUF checkpoint tensor CPU decode; not full model/memory/performance qualification",
                       "os": platform.platform(), "machine": platform.machine(),
                       "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                       "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "binary_sha256": digest(binary), "compiler_command": ["xcrun", "clang++", "-std=c++20", "-O2", "-ffp-contract=off"],
                       "sources": {str(path.relative_to(ROOT)): digest(path) for path in sources + [ROOT/"native/core/gguf_decode.hpp", ROOT/"native/core/gguf_directory.hpp"]},
                       "checkpoint_content_sha256": None, "samples": []}
            for tensor in args.tensors:
                raw = subprocess.check_output([str(binary), str(args.checkpoint), tensor, str(args.iterations),
                                               str(args.max_buffer_bytes)], text=True, timeout=180)
                sample = json.loads(raw)
                sample["scalar_median_ms"] = statistics.median(sample["scalar_ms"])
                sample["simd_median_ms"] = statistics.median(sample["simd_ms"])
                sample["scalar_over_simd"] = sample["scalar_median_ms"]/sample["simd_median_ms"]
                receipt["samples"].append(sample)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                json.dump(receipt, output, indent=2, sort_keys=True, allow_nan=False); output.write("\n")
            for sample in receipt["samples"]:
                print(f'{sample["tensor"]}: scalar={sample["scalar_median_ms"]:.3f}ms '
                      f'SIMD={sample["simd_median_ms"]:.3f}ms ratio={sample["scalar_over_simd"]:.3f}x exact={sample["exact"]}')
            print(args.output)


if __name__ == "__main__": main()
