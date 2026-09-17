#!/usr/bin/env python3
"""Compile and run the metadata-only H3 descriptor against sparse fixtures."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
H3 = ROOT / "native/models/h3_runtime"
INNER = 56 * 128
HIDDEN = 5376
FFN = 14336
MATRICES = (
    ("attn.qkv_proj.weight", 3 * INNER, HIDDEN),
    ("attn.out_proj.weight", HIDDEN, INNER),
    ("mlp.fc1.weight", 2 * FFN, HIDDEN),
    ("mlp.fc2.weight", HIDDEN, FFN),
)


def active_ids(active: int) -> list[int]:
    mask = [True] * 50
    skipped = 50 - active
    for index in range(skipped):
        block = ((2 * index + 1) * 50) // (2 * skipped)
        block = max(1, min(48, block))
        assert mask[block], (active, index, block)
        mask[block] = False
    return [index for index, enabled in enumerate(mask) if enabled]


def write_shard(path: Path, tensors: list[tuple[str, int, int]]) -> None:
    header: dict[str, object] = {}
    cursor = 0
    for name, rows, columns in tensors:
        size = rows * columns * 2
        header[name] = {
            "dtype": "BF16",
            "shape": [rows, columns],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.truncate(8 + len(encoded) + cursor)


def write_fixture(directory: Path, *, missing=False,
                  wrong_shape=False, duplicate=False) -> None:
    blocks = list(range(50))
    shards: list[list[tuple[str, int, int]]] = [[], [], [], []]
    for block in blocks:
        for field_index, (suffix, rows, columns) in enumerate(MATRICES):
            if missing and block == blocks[-1] and field_index == 3:
                continue
            if wrong_shape and block == blocks[0] and field_index == 0:
                rows -= 1
            name = f"blocks.{block}.{suffix}"
            shards[field_index].append((name, rows, columns))
            if duplicate and block == blocks[0] and field_index == 0:
                shards[(field_index + 1) % 4].append((name, rows, columns))
    for index, tensors in enumerate(shards, 1):
        write_shard(directory / f"model-{index:05d}.safetensors", tensors)


def compile_and_run() -> None:
    clang = subprocess.check_output(["xcrun", "--find", "clang"], text=True).strip()
    cxx = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    sanitizer = os.environ.get("TC_STREAMING_SANITIZER", "")
    if sanitizer not in ("", "address,undefined", "thread"):
        raise ValueError("unsupported sanitizer")
    common = ["-Wall", "-Wextra", "-Werror", "-isysroot", sdk,
              "-I", str(ROOT / "native/core"), "-I", str(H3)]
    if sanitizer:
        common += ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=" + sanitizer]
    else:
        common += ["-O2"]
    with tempfile.TemporaryDirectory(prefix="tc-h3-descriptor-") as raw:
        root = Path(raw)
        valid = root / "valid"
        valid.mkdir()
        write_fixture(valid)
        missing = root / "missing"
        missing.mkdir()
        write_fixture(missing, missing=True)
        wrong = root / "wrong"
        wrong.mkdir()
        write_fixture(wrong, wrong_shape=True)
        duplicate = root / "duplicate"
        duplicate.mkdir()
        write_fixture(duplicate, duplicate=True)

        objects: list[str] = []
        c_sources = (
            H3 / "h3_host.c",
            H3 / "h3_safetensors.c",
            H3 / "h3_weights.c",
            H3 / "h3_streaming_policy.c",
            ROOT / "native/runtime/block_residency.c",
        )
        for source in c_sources:
            target = root / (source.stem + ".o")
            subprocess.run([clang, "-std=c11", "-D_DARWIN_C_SOURCE", *common,
                            "-c", str(source), "-o", str(target)], check=True)
            objects.append(str(target))
        binary = root / "h3-descriptor-test"
        cpp_sources = (
            ROOT / "tests/native/h3_streaming_descriptor_test.cpp",
            ROOT / "native/models/h3_runtime/h3_streaming_descriptor.cpp",
            ROOT / "native/runtime/streaming/config.cpp",
            ROOT / "native/runtime/streaming/layout.cpp",
            ROOT / "native/runtime/streaming/slot_pool.cpp",
            ROOT / "native/runtime/streaming/io_executor.cpp",
            ROOT / "native/runtime/streaming/context.cpp",
            ROOT / "native/runtime/streaming/c_bridge.cpp",
            ROOT / "native/runtime/streaming/audit.cpp",
            ROOT / "native/runtime/memory_manifest.cpp",
            ROOT / "native/runtime/memory_policy.cpp",
            ROOT / "native/core/common.cpp",
        )
        subprocess.run([cxx, "-std=c++20", *common, "-pthread",
                        *map(str, cpp_sources), *objects,
                        "-framework", "Foundation", "-framework", "Accelerate",
                        "-o", str(binary)], check=True)
        subprocess.run([str(binary), str(valid), str(missing), str(wrong),
                        str(duplicate)], check=True, timeout=180)


if __name__ == "__main__":
    compile_and_run()
