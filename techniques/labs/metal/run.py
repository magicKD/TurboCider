#!/usr/bin/env python3
"""Compile/run the isolated Metal learning probe, never a production library."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
from common import emit, file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("vector", "fusion", "gemm", "norm-rope", "attention", "graph", "overlap"))
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--count", type=int, default=65539)
    parser.add_argument("--m", type=int, default=33)
    parser.add_argument("--n", type=int, default=41)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("actual Metal probe requires macOS; use labs/run.py for CPU references")
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error("output already exists; use a new report path")
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="tc-technique-metal-") as directory:
        binary = Path(directory) / "metal-lab"
        command = ["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-Wall", "-Wextra",
                   str(root / "bench.mm"), "-framework", "Foundation", "-framework", "Metal", "-o", str(binary)]
        built = subprocess.run(command, text=True, capture_output=True, timeout=120)
        if built.returncode:
            emit({"status": "build_failed", "stderr": built.stderr, "production_qualified": False}, args.output)
            return built.returncode
        result = subprocess.run([str(binary), str(root / "kernels.metal"), args.case, str(args.repeats), str(args.count),
                                 str(args.m), str(args.n), str(args.k), str(args.threads), str(args.seed)],
                                text=True, capture_output=True, timeout=120)
        try:
            value = json.loads(result.stdout)
        except ValueError:
            value = {"status": "invalid_native_report", "stdout": result.stdout}
        value.update(arguments={name: getattr(args, name) for name in ("case", "repeats", "count", "m", "n", "k", "threads", "seed")},
                     variant_threads={"baseline": args.threads, "candidate": 32 if args.case == "gemm" else args.threads},
                     binary_sha256=file_sha(binary), source_sha256={name: file_sha(root / name) for name in ("bench.mm", "kernels.metal")},
                     os=platform.mac_ver()[0], build_warnings=built.stderr, stderr=result.stderr, exit_code=result.returncode)
        emit(value, args.output)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
