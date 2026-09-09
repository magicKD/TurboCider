#!/usr/bin/env python3
"""Validate H3 quantized-shard schema and provenance without model weights."""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAGIC = 0x4833515354524D31
IDENTITY = 0x123456789ABCDEF0
BLOCKS, HIDDEN, INNER, FFN = 2, 4, 2, 3


def write_block(directory: Path, block: int, *, identity: int = IDENTITY,
                metadata_block: int | None = None, qkv_dtype: str = "I8") -> None:
    metadata = struct.pack(
        "<8Q", MAGIC, 1, identity,
        block if metadata_block is None else metadata_block,
        BLOCKS, HIDDEN, INNER, FFN,
    )
    descriptions = [
        ("qkv.weight", qkv_dtype, [INNER * 3, HIDDEN], INNER * 3 * HIDDEN),
        ("qkv.scales", "F32", [INNER * 3], INNER * 3 * 4),
        ("out.weight", "I8", [HIDDEN, INNER], HIDDEN * INNER),
        ("out.scales", "F32", [HIDDEN], HIDDEN * 4),
        ("fc1.weight", "I8", [FFN * 2, HIDDEN], FFN * 2 * HIDDEN),
        ("fc1.scales", "F32", [FFN * 2], FFN * 2 * 4),
        ("fc2.weight", "I8", [HIDDEN, FFN], HIDDEN * FFN),
        ("fc2.scales", "F32", [HIDDEN], HIDDEN * 4),
    ]
    header: dict[str, dict] = {
        "__turbocider_h3_quant_cache_v1__": {
            "dtype": "U64", "shape": [8],
            "data_offsets": [0, len(metadata)],
        }
    }
    payload = bytearray(metadata)
    offset = len(payload)
    for name, dtype, shape, size in descriptions:
        header[name] = {
            "dtype": dtype, "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        if dtype == "F32":
            payload.extend(struct.pack("<" + "f" * (size // 4),
                                       *([1.0] * (size // 4))))
        else:
            payload.extend(bytes(size))
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path = directory / f"block-{block:02d}.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def run(binary: Path, directory: Path, identity: int = IDENTITY) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(binary), str(directory), hex(identity), str(BLOCKS),
         str(HIDDEN), str(INNER), str(FFN)],
        capture_output=True, text=True,
    )


def main() -> int:
    compiler = subprocess.run(
        ["xcrun", "--find", "clang"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="turbocider-h3-quant-cache-") as raw:
        root = Path(raw)
        binary = root / "h3-quant-cache-test"
        subprocess.run(
            [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
             "-D_DARWIN_C_SOURCE",
             "-I", str(ROOT / "native/models/h3_runtime"),
             str(ROOT / "tests/native/h3_quant_cache_test.c"),
             str(ROOT / "native/models/h3_runtime/h3_quant_cache.c"),
             str(ROOT / "native/models/h3_runtime/h3_safetensors.c"),
             "-o", str(binary)],
            check=True,
        )
        cache = root / "valid"
        cache.mkdir()
        for block in range(BLOCKS):
            write_block(cache, block)
        valid = run(binary, cache)
        if valid.returncode:
            raise RuntimeError(valid.stderr)
        assert "PASS: H3 quantized cache" in valid.stdout

        wrong_identity = run(binary, cache, IDENTITY ^ 1)
        assert wrong_identity.returncode != 0
        assert "provenance/schema mismatch" in wrong_identity.stderr

        wrong_dtype = root / "wrong-dtype"
        wrong_dtype.mkdir()
        write_block(wrong_dtype, 0, qkv_dtype="U8")
        write_block(wrong_dtype, 1)
        invalid = run(binary, wrong_dtype)
        assert invalid.returncode != 0
        assert "wrong dtype/shape" in invalid.stderr

        wrong_block = root / "wrong-block"
        wrong_block.mkdir()
        write_block(wrong_block, 0, metadata_block=1)
        write_block(wrong_block, 1)
        invalid = run(binary, wrong_block)
        assert invalid.returncode != 0
        assert "provenance/schema mismatch" in invalid.stderr

        missing = root / "missing"
        missing.mkdir()
        write_block(missing, 0)
        invalid = run(binary, missing)
        assert invalid.returncode != 0
        assert "block-01.safetensors" in invalid.stderr
        assert "No such file or directory" in invalid.stderr

        truncated = root / "truncated"
        truncated.mkdir()
        for block in range(BLOCKS):
            write_block(truncated, block)
        second = truncated / "block-01.safetensors"
        second.write_bytes(second.read_bytes()[:-1])
        invalid = run(binary, truncated)
        assert invalid.returncode != 0
        assert "tensor data offsets exceed file" in invalid.stderr
    print(valid.stdout.strip())
    print("PASS: H3 quantized cache rejects identity, dtype, block and truncation errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
