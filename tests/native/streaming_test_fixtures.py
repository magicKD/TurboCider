"""Byte vocabulary and sparse BF16 fixtures for streaming contract tests."""

import json
from pathlib import Path
import struct


INNER = 56 * 128
HIDDEN = 5376
FFN = 14336
H3_MATRICES = (
    ("attn.qkv_proj.weight", 3 * INNER, HIDDEN),
    ("attn.out_proj.weight", HIDDEN, INNER),
    ("mlp.fc1.weight", 2 * FFN, HIDDEN),
    ("mlp.fc2.weight", HIDDEN, FFN),
)


def byte_vocab() -> dict[str, int]:
    result: dict[str, int] = {}
    extra = 0
    for value in range(256):
        if 33 <= value <= 126 or 161 <= value <= 172 or value >= 174:
            codepoint = value
        else:
            codepoint = 256 + extra
            extra += 1
        result[chr(codepoint)] = value
    return result


def write_bf16_shard(path: Path, tensors: list[tuple[str, int, int]]) -> None:
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
