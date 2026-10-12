#!/usr/bin/env python3
"""Real ConvRot CPU SIMD inverse-rotation screen, not a model/ANE speed claim."""
import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from run_runtime_w8a8_proof import convrot_slice
from convrot_w8a8_math import hadamard256


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def benchmark(args, native, prefix, shape):
    rows, columns = shape
    codes, scales, identity = convrot_slice(args.checkpoint, prefix, rows, columns)
    identity.pop("activation_source", None)
    fast = np.empty(shape, dtype=np.float16)
    scalar = np.empty_like(fast)

    def convert(output, scalar_only):
        start = time.perf_counter()
        status = native(codes.ctypes.data, codes.nbytes, scales.ctypes.data, scales.nbytes,
                        rows, columns, output.ctypes.data, output.size, scalar_only, args.headroom)
        seconds = time.perf_counter() - start
        if status:
            raise ArithmeticError(f"staging failed: {status}")
        return seconds

    for _ in range(3):
        convert(scalar, True)
        convert(fast, False)
    samples = []
    for iteration in range(args.iterations):
        order = (True, False) if iteration % 4 in (0, 3) else (False, True)
        for scalar_only in order:
            seconds = convert(scalar if scalar_only else fast, scalar_only)
            samples.append({"iteration": iteration, "arm": "scalar" if scalar_only else "simd",
                            "seconds": seconds})
    if not np.array_equal(fast.view(np.uint16), scalar.view(np.uint16)):
        raise ArithmeticError("SIMD/scalar staging mismatch")
    # Independent dense Kronecker contraction for fixed real source rows. This
    # is an oracle for this integer-first recipe, not legacy D*H FP32 rounding.
    h = (hadamard256() * np.float32(16)).astype(np.int64)
    checked = sorted(set([0, rows // 2, rows - 1]))
    for row in checked:
        dot = np.einsum("ij,jk->ik", codes[row].reshape(-1, 256).astype(np.int64), h, optimize=False)
        reference = (((dot.astype(np.float32) * np.float32(1 / 16)) * scales[row]) * np.float32(args.headroom)).astype(np.float16)
        if not np.array_equal(reference.reshape(-1).view(np.uint16), fast[row].view(np.uint16)):
            raise ArithmeticError("independent dense Hadamard oracle mismatch")
    medians = {arm: statistics.median(s["seconds"] for s in samples if s["arm"] == arm)
               for arm in ("scalar", "simd")}
    return {"source": identity, "shape": shape, "headroom": args.headroom, "warmups_per_arm": 3,
            "samples": samples, "median_seconds": medians,
            "scalar_over_simd": medians["scalar"] / medians["simd"],
            "whole_target_bit_exact": True, "independent_oracle_rows": checked,
            "output_sha256": hashlib.sha256(fast.tobytes()).hexdigest(),
            "input_target_capacity": codes.nbytes + scales.nbytes + fast.nbytes + scalar.nbytes,
            "per_worker_kernel_scratch_upper": 1024}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--headroom", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    if not 8 <= args.iterations <= 100 or not np.isfinite(args.headroom) or not 0 < args.headroom <= 1:
        parser.error("iterations must be 8..100; headroom must be finite in (0,1]")
    sources = ["native/backends/ane_runtime_packed.hpp", "native/backends/ane_runtime_convert.hpp",
               "native/core/gguf_decode.cpp", "tools/native/ane_packed_staging_probe.cpp",
               "tools/native/benchmark_convrot_staging.py", "tools/validation/run_runtime_w8a8_proof.py",
               "tools/validation/convrot_w8a8_math.py"]
    with open("/tmp/turbocider-z-image-gpu-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix="tc-convrot-staging-") as scratch:
            library = Path(scratch) / "probe.dylib"
            command = ["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-ffp-contract=off", "-dynamiclib", "tools/native/ane_packed_staging_probe.cpp",
                       "native/core/gguf_decode.cpp", "-o", str(library)]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            loaded = ctypes.CDLL(str(library))
            native = loaded.tc_convrot_stage
            native.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                               ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                               ctypes.c_bool, ctypes.c_float]
            native.restype = ctypes.c_int
            receipt = {"schema": "tc-convrot-cpu-staging-screen-v1", "scope": "serial CPU conversion only",
                       "recipe": "convrot-integer-h256-source-f32-scale-headroom-f16-rne-v1",
                       "time_utc": datetime.now(timezone.utc).isoformat(), "os": platform.platform(),
                       "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                       "binary_sha256": digest(library), "compiler_command": command[:-1] + ["<temporary-probe>"],
                       "source_sha256": {name: digest(ROOT / name) for name in sources},
                       "excludes": ["checkpoint I/O", "allocation", "IOSurface", "Core ML prediction",
                                    "GPU/ANE fork/join", "model/media quality", "whole-request memory qualification"],
                       "projections": []}
            for name, shape in [("w1", (10240, 3840)), ("w2", (3840, 10240))]:
                receipt["projections"].append(benchmark(args, native, "layers.0.feed_forward." + name, shape))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as handle:
                json.dump(receipt, handle, indent=2, allow_nan=False)
                handle.write("\n")
            print(json.dumps([{key: p[key] for key in ("shape", "median_seconds", "scalar_over_simd",
                                                       "whole_target_bit_exact")} for p in receipt["projections"]], indent=2))


if __name__ == "__main__":
    main()
