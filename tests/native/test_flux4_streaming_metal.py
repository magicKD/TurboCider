#!/usr/bin/env python3
"""Opt-in real 4B geometry/kernel smoke; sparse zero weights, not model quality."""
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
from test_flux_streaming_descriptor import write_fixture

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()


def main():
    if os.environ.get("TURBOCIDER_TEST_GPU") != "1":
        print("SKIP: TURBOCIDER_TEST_GPU=1 required (about 2 GiB of GPU slots/resident weights)")
        return
    mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx"
    with tempfile.TemporaryDirectory(prefix="tc-flux4-metal-") as raw:
        root = Path(raw)
        write_fixture(root / "transformer", klein4=True)
        binary = root / "probe"
        subprocess.run([
            "clang++", "-std=c++20", "-O2", "-I", str(ROOT / "native"),
            "-I", str(ROOT / "native/core"), "-isystem", str(mlx / "include"),
            str(ROOT / "tests/native/flux4_streaming_metal_test.cpp"),
            "-L" + str(NATIVE), "-lturbocider", "-L" + str(mlx / "lib"), "-lmlx",
            "-Wl,-rpath," + str(NATIVE), "-Wl,-rpath," + str(mlx / "lib"),
            "-o", str(binary),
        ], check=True)
        subprocess.run([str(binary), str(root / "transformer")], check=True, timeout=240)


if __name__ == "__main__":
    main()
