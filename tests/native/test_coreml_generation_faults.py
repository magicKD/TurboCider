#!/usr/bin/env python3
"""Deterministic import cancellation and syscall faults without production hooks."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
flags = ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core")]
if os.environ.get("TC_STREAMING_SANITIZER"):
    flags += ["-fsanitize=" + os.environ["TC_STREAMING_SANITIZER"], "-fno-omit-frame-pointer"]
sources = ["tests/native/coreml_generation_fault_test.cpp", "native/runtime/streaming/source_lease.cpp",
           "native/runtime/streaming/canonical_encoding.cpp", "native/runtime/memory_manifest.cpp", "native/core/common.cpp"]
with tempfile.TemporaryDirectory(prefix="tc-generation-faults-") as raw:
    root = Path(raw).resolve()
    obj, binary = root / "generation.o", root / "test"
    subprocess.run(["clang++", *flags, "-Dpread=tc_generation_pread", "-Dwrite=tc_generation_write", "-Dfchmod=tc_generation_fchmod",
                    "-c", str(ROOT / "native/models/z_image/coreml_generation.cpp"), "-o", str(obj)], check=True)
    subprocess.run(["clang++", *flags, *(str(ROOT / s) for s in sources), str(obj), "-framework", "Foundation", "-o", str(binary)], check=True)
    subprocess.run([str(binary), str(root)], check=True, timeout=180)
