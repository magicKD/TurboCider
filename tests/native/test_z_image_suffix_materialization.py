#!/usr/bin/env python3
"""Host-only H1 suffix oracle and deterministic I/O fault regression."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
flags = ["-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native")]
if os.environ.get("TC_STREAMING_SANITIZER"):
    flags += ["-fsanitize=" + os.environ["TC_STREAMING_SANITIZER"], "-fno-omit-frame-pointer"]
with tempfile.TemporaryDirectory(prefix="tc-suffix-host-") as raw:
    obj, binary = Path(raw) / "pack.o", Path(raw) / "test"
    subprocess.run(["clang++", *flags, "-Dpread=tc_test_pread", "-Dpwrite=tc_test_pwrite",
                    "-c", str(ROOT / "native/models/z_image/suffix_materialization.cpp"), "-o", str(obj)], check=True)
    subprocess.run(["clang++", *flags, str(ROOT / "tests/native/z_image_suffix_materialization_test.cpp"),
                    str(obj), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
