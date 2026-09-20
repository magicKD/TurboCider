#!/usr/bin/env python3
"""Real fd lifecycle and native hash proof using sparse Z-Image geometry."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import sysconfig
from test_z_image_suffix_descriptor import fixture

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--real-report", type=Path)
parser.add_argument("--native-dir", type=Path)
args = parser.parse_args()
if bool(args.checkpoint) != bool(args.real_report):
    parser.error("--checkpoint and --real-report must be provided together")
ROOT = Path(__file__).resolve().parents[2]
sources = ["tests/native/z_image_suffix_source_test.cpp", "native/platform/apple/z_image_streaming_descriptor.mm",
           "native/models/z_image/suffix_materialization.cpp", "native/runtime/streaming/config.cpp",
           "native/runtime/streaming/canonical_encoding.cpp", "native/runtime/streaming/layout.cpp",
           "native/runtime/streaming/source_lease.cpp", "native/runtime/memory_manifest.cpp",
           "native/runtime/memory_policy.cpp", "native/core/common.cpp"]
link_flags = ["-framework", "Foundation"]
if args.native_dir:
    native = args.native_dir.resolve()
    mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx/lib"
    sources = sources[:1]
    link_flags += ["-L" + str(native), "-lturbocider", "-Wl,-rpath," + str(native), "-Wl,-rpath," + str(mlx)]
flags = ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-fobjc-arc", "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core")]
if os.environ.get("TC_STREAMING_SANITIZER"):
    flags += ["-fsanitize=" + os.environ["TC_STREAMING_SANITIZER"], "-fno-omit-frame-pointer"]
with tempfile.TemporaryDirectory(prefix="tc-suffix-source-") as raw:
    root = Path(raw)
    paths = [root / f"copy{i}.safetensors" for i in range(2)]
    for path in paths:
        header, payload = fixture(path)
        with path.open("r+b") as stream:
            for branch in range(32):
                prefix = f"noise_refiner.{branch}" if branch < 2 else f"layers.{branch - 2}"
                offset = header[prefix + ".feed_forward.w2.weight"]["data_offsets"][0]
                for row in (0, 1024, 3839):
                    stream.seek(payload + offset + (row * 10240 + 10239) * 2)
                    stream.write(struct.pack("<H", 0x3f00 + branch))
    binary = root / "test"
    subprocess.run(["clang++", *flags, *(str(ROOT / s) for s in sources), *link_flags, "-o", str(binary)], check=True)
    subprocess.run([str(binary), *map(str, paths)], check=True, timeout=180)
    if args.checkpoint:
        with args.real_report.open("x") as output:
            subprocess.run([str(binary), str(args.checkpoint.resolve())], stdout=output, check=True, timeout=180)
