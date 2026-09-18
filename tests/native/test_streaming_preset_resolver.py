#!/usr/bin/env python3
"""Host-only public preset/catalog/context tests; no GPU or model weights."""

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="tc-streaming-preset-") as raw:
        binary = Path(raw) / "streaming-preset-resolver-test"
        flags = [
                compiler,
                "-std=c++20",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pthread",
                "-isysroot",
                sdk,
        ]
        sanitizer = os.environ.get("TC_STREAMING_SANITIZER", "")
        if sanitizer:
            if sanitizer not in ("address,undefined", "thread"):
                raise ValueError("unsupported sanitizer")
            flags += [
                "-O1",
                "-g",
                "-fno-omit-frame-pointer",
                "-fsanitize=" + sanitizer,
            ]
        subprocess.run(
            [
                *flags,
                "-I",
                str(ROOT / "native/core"),
                str(ROOT / "tests/native/streaming_preset_resolver_test.cpp"),
                str(ROOT / "native/core/common.cpp"),
                str(ROOT / "native/runtime/memory_manifest.cpp"),
                str(ROOT / "native/runtime/streaming/config.cpp"),
                str(ROOT / "native/runtime/streaming/canonical_encoding.cpp"),
                str(ROOT / "native/runtime/streaming/actual_receipt.cpp"),
                str(ROOT / "native/runtime/streaming/preset_catalog.cpp"),
                str(ROOT / "native/runtime/streaming/catalog_provider.cpp"),
                str(ROOT / "native/runtime/streaming/resolved_request.cpp"),
                str(ROOT / "native/runtime/streaming/source_lease.cpp"),
                str(ROOT / "native/runtime/streaming/preset_resolver.cpp"),
                str(ROOT / "native/runtime/streaming/public_request_validation.cpp"),
                str(ROOT / "native/runtime/streaming/public_runtime.cpp"),
                str(ROOT / "native/runtime/streaming/public_result.cpp"),
                str(ROOT / "native/runtime/streaming/run_context.cpp"),
                str(ROOT / "native/runtime/streaming/context.cpp"),
                str(ROOT / "native/runtime/streaming/io_executor.cpp"),
                str(ROOT / "native/runtime/streaming/slot_pool.cpp"),
                str(ROOT / "native/runtime/streaming/audit.cpp"),
                "-o",
                str(binary),
            ],
            check=True,
        )
        subprocess.run([str(binary)], check=True, timeout=30)


if __name__ == "__main__":
    main()
