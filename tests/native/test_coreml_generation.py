#!/usr/bin/env python3
"""Host tests of private Core ML generation import; no device prediction."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifact", type=Path)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
sources = ["tests/native/coreml_generation_test.cpp", "native/models/z_image/coreml_generation.cpp",
           "native/runtime/streaming/source_lease.cpp", "native/runtime/streaming/canonical_encoding.cpp",
           "native/runtime/memory_manifest.cpp", "native/core/common.cpp"]
flags = ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core")]
if os.environ.get("TC_STREAMING_SANITIZER"):
    flags += ["-fsanitize=" + os.environ["TC_STREAMING_SANITIZER"], "-fno-omit-frame-pointer"]
with tempfile.TemporaryDirectory(prefix="tc-coreml-generation-test-") as raw:
    root = Path(raw).resolve()
    binary = root / "test"
    subprocess.run(["clang++", *flags, *(str(ROOT / s) for s in sources), "-framework", "Foundation", "-o", str(binary)], check=True)
    subprocess.run([str(binary), str(root), *([str(args.artifact.resolve())] if args.artifact else [])], check=True, timeout=180)
