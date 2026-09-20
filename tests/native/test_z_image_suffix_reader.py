#!/usr/bin/env python3
"""Opt-in Metal buffer regression for the verified internal suffix reader."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sysconfig
import tempfile
from test_z_image_suffix_descriptor import fixture

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--native-dir", type=Path, default=ROOT / "build/native")
args = parser.parse_args()
if os.environ.get("TURBOCIDER_TEST_GPU") != "1":
    print("SKIP: set TURBOCIDER_TEST_GPU=1 to run Metal buffer checks")
    raise SystemExit(0)
mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx"
native = args.native_dir.resolve()
with tempfile.TemporaryDirectory(prefix="tc-suffix-reader-") as raw:
    root = Path(raw)
    path = root / "model.safetensors"
    header, payload = fixture(path, embedder_elements=64)
    with path.open("r+b") as stream:
        for branch in range(32):
            prefix = f"noise_refiner.{branch}" if branch < 2 else f"layers.{branch - 2}"
            for projection in (1, 2, 3):
                offset = header[f"{prefix}.feed_forward.w{projection}.weight"]["data_offsets"][0]
                indices = [row * 10240 + 10239 for row in (0, 1024, 3839)] if projection == 2 else [10239 * 3840 + col for col in (0, 2000, 3839)]
                for index in indices:
                    stream.seek(payload + offset + index * 2)
                    stream.write(struct.pack("<H", 0x3f00 + branch + projection))
    binary = root / "test"
    subprocess.run(["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
        "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core"), "-isystem", str(mlx / "include"),
        str(ROOT / "tests/native/z_image_suffix_reader_test.cpp"),
        "-L" + str(native), "-lturbocider", "-L" + str(mlx / "lib"), "-lmlx",
        "-Wl,-rpath," + str(native), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(binary)], check=True)
    subprocess.run([str(binary), str(path)], check=True, timeout=180)
