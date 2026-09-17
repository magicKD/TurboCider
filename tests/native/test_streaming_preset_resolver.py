#!/usr/bin/env python3
"""Host-only public preset catalog/resolver tests; no GPU or model weights."""

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
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-isysroot",
                sdk,
                "-I",
                str(ROOT / "native/core"),
                str(ROOT / "tests/native/streaming_preset_resolver_test.cpp"),
                str(ROOT / "native/core/common.cpp"),
                str(ROOT / "native/runtime/memory_manifest.cpp"),
                str(ROOT / "native/runtime/streaming/config.cpp"),
                str(ROOT / "native/runtime/streaming/canonical_encoding.cpp"),
                str(ROOT / "native/runtime/streaming/preset_catalog.cpp"),
                str(ROOT / "native/runtime/streaming/catalog_provider.cpp"),
                str(ROOT / "native/runtime/streaming/resolved_request.cpp"),
                str(ROOT / "native/runtime/streaming/source_lease.cpp"),
                str(ROOT / "native/runtime/streaming/preset_resolver.cpp"),
                str(ROOT / "native/runtime/streaming/public_request_validation.cpp"),
                str(ROOT / "native/runtime/streaming/public_runtime.cpp"),
                str(ROOT / "native/runtime/streaming/public_result.cpp"),
                "-o",
                str(binary),
            ],
            check=True,
        )
        subprocess.run([str(binary)], check=True, timeout=30)


if __name__ == "__main__":
    main()
