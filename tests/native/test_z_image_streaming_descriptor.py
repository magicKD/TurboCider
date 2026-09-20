#!/usr/bin/env python3
"""Build the Z-Image metadata-only streaming descriptor with sparse fixtures."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_fixture(path: Path, *, missing: bool = False,
                  wrong_dtype: bool = False, wrong_shape: bool = False,
                  overlap: bool = False) -> None:
    records: list[tuple[str, str, list[int]]] = [
        ("x_embedder.weight", "BF16", [2, 2])
    ]
    for block in range(30):
        for field in range(13):
            if missing and block == 29 and field == 12:
                continue
            dtype = "F32" if wrong_dtype and block == 0 and field == 0 else "BF16"
            shape = [512, 2048] if wrong_shape and block == 1 and field == 0 else [1024, 1024]
            records.append((f"layers.{block}.tensor{field:02d}.weight", dtype, shape))

    header: dict[str, object] = {}
    cursor = 0
    for index, (name, dtype, shape) in enumerate(records):
        item_bytes = 4 if dtype == "F32" else 2
        size = item_bytes
        for dimension in shape:
            size *= dimension
        begin = 0 if overlap and index == 2 else cursor
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [begin, begin + size],
        }
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.truncate(8 + len(encoded) + cursor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    compiler = subprocess.check_output(
        ["xcrun", "--find", "clang++"], text=True
    ).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    sanitizer = os.environ.get("TC_STREAMING_SANITIZER", "")
    if sanitizer not in ("", "address,undefined", "thread"):
        raise ValueError("unsupported sanitizer")
    flags = [
        "-std=c++20", "-Wall", "-Wextra", "-Werror", "-fobjc-arc",
        "-isysroot", sdk, "-I", str(ROOT / "native"),
        "-I", str(ROOT / "native/core"),
    ]
    if sanitizer:
        flags += ["-O1", "-g", "-fno-omit-frame-pointer",
                  "-fsanitize=" + sanitizer]
    else:
        flags += ["-O2"]
    sources = (
        ROOT / "tests/native/z_image_streaming_descriptor_test.cpp",
        ROOT / "native/platform/apple/z_image_streaming_descriptor.mm",
        ROOT / "native/models/z_image/suffix_materialization.cpp",
        ROOT / "native/runtime/streaming/config.cpp",
        ROOT / "native/runtime/streaming/canonical_encoding.cpp",
        ROOT / "native/runtime/streaming/layout.cpp",
        ROOT / "native/runtime/streaming/source_lease.cpp",
        ROOT / "native/runtime/memory_manifest.cpp",
        ROOT / "native/runtime/memory_policy.cpp",
        ROOT / "native/core/common.cpp",
    )
    with tempfile.TemporaryDirectory(prefix="tc-z-image-descriptor-") as raw:
        directory = Path(raw)
        fixtures = []
        for name, options in (
            ("valid", {}),
            ("missing", {"missing": True}),
            ("wrong-dtype", {"wrong_dtype": True}),
            ("wrong-shape", {"wrong_shape": True}),
            ("overlap", {"overlap": True}),
        ):
            fixture = directory / f"{name}.safetensors"
            write_fixture(fixture, **options)
            fixtures.append(fixture)
        binary = directory / "z-image-streaming-descriptor-test"
        subprocess.run([
            compiler, *flags, *map(str, sources), "-framework", "Foundation",
            "-o", str(binary),
        ], check=True)
        subprocess.run([str(binary), *map(str, fixtures)], check=True,
                       timeout=180)
        if arguments.checkpoint:
            real = arguments.checkpoint.expanduser().resolve()
            if not real.is_file():
                raise FileNotFoundError(real)
            subprocess.run([str(binary), str(real)], check=True, timeout=120)


if __name__ == "__main__":
    main()
