#!/usr/bin/env python3
"""Host-only compiler/config tests; no checkpoint or GPU is loaded."""
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    flags = ["-std=c++20", "-Wall", "-Wextra", "-Werror", "-isysroot", sdk]
    c_flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-isysroot", sdk]
    sanitizer = os.environ.get("TC_STREAMING_SANITIZER", "")
    if sanitizer:
        if sanitizer not in ("address,undefined", "thread"):
            raise ValueError("unsupported sanitizer")
        flags += ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=" + sanitizer]
        c_flags += ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=" + sanitizer]
    else:
        flags += ["-O2"]
        c_flags += ["-O2"]
    sources = [
        "native/runtime/streaming/config.cpp", "native/runtime/streaming/layout.cpp",
        "native/runtime/streaming/slot_pool.cpp", "native/runtime/streaming/io_executor.cpp",
        "native/runtime/streaming/context.cpp",
        "native/runtime/streaming/c_bridge.cpp",
        "native/runtime/memory_manifest.cpp", "native/runtime/memory_policy.cpp",
        "native/core/common.cpp", "native/core/json_keys.cpp",
    ]
    with tempfile.TemporaryDirectory(prefix="tc-streaming-layout-") as raw:
        for name in ("streaming_layout_test", "streaming_descriptor_test",
                     "streaming_executor_test", "streaming_c_bridge_failure_test"):
            binary = Path(raw) / name
            subprocess.run([compiler, *flags, "-pthread", "-I", str(ROOT / "native/runtime"),
                            str(ROOT / "tests/native" / (name + ".cpp")),
                            *[str(ROOT / s) for s in sources], "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True, timeout=30)
        c_object = Path(raw) / "c-bridge.o"
        clang = subprocess.check_output(["xcrun", "--find", "clang"], text=True).strip()
        subprocess.run([clang, *c_flags,
                        "-I", str(ROOT / "native/core"), "-c",
                        str(ROOT / "tests/native/streaming_c_bridge_test.c"), "-o", str(c_object)], check=True)
        binary = Path(raw) / "c-bridge-test"
        subprocess.run([compiler, *flags, "-pthread", str(c_object),
                        *[str(ROOT / s) for s in sources], "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=30)


if __name__ == "__main__":
    main()
