#!/usr/bin/env python3
"""Host-only v1/v2 contract and ownership tests, using disposable fixtures only.

Requires the current native-only library. Never loads real weights or a GPU.
"""
from pathlib import Path
import json
import os
import shutil
import struct
import subprocess
import tempfile
from test_ltx_streaming_layout import fixture

ROOT = Path(__file__).resolve().parents[2]


def main():
    compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="tc-ltx-snapshot-") as raw:
        directory = Path(raw)
        tiny = directory / "header-failures.safetensors"
        header = json.dumps({"__metadata__": {"config": "{}", "gemma_config": "{}"},
                             "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
        tiny.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")
        clang = subprocess.check_output(["xcrun", "--find", "clang"], text=True).strip()
        sanitizer = os.environ.get("TC_STREAMING_SANITIZER", "")
        if sanitizer not in ("", "address,undefined"):
            raise ValueError("parser fault suite supports only address,undefined")
        flags = ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=" + sanitizer] if sanitizer else ["-O2"]
        parser_test = directory / "parser-failures"
        subprocess.run([clang, "-std=c11", "-fobjc-arc", "-Wall", "-Wextra", "-Werror",
                        "-isysroot", sdk, *flags,
                        str(ROOT / "tests/native/ltx_safetensors_failure_test.m"),
                        "-framework", "Foundation", "-o", str(parser_test)], check=True)
        subprocess.run([str(parser_test), str(tiny)], check=True, timeout=30)
        checkpoint = directory / "fixture.safetensors"
        other = directory / "other.safetensors"
        fixture(checkpoint, blocks=48)
        shutil.copyfile(checkpoint, other)
        binary = directory / "snapshot-test"
        subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-isysroot", sdk,
                        "-I", str(ROOT / "native/models/ltx_runtime"),
                        str(ROOT / "tests/native/ltx_streaming_snapshot_test.cpp"),
                        "-L", str(ROOT / "build/native"), "-lturbocider",
                        "-Wl,-rpath," + str(ROOT / "build/native"), "-o", str(binary)], check=True)
        subprocess.run([str(binary), str(checkpoint), str(other)], check=True, timeout=60)
        if sanitizer:
            print("SANITIZER SCOPE: parser/failure injection only; v1/v2 dylib tests use the release library")


if __name__ == "__main__":
    main()
