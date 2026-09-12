#!/usr/bin/env python3
"""Validate the native H3 streamed safetensors reader on tiny fixtures."""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


def bf16(value: float) -> bytes:
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.pack("<H", word >> 16)


def write_safetensors(path: Path, records: list[tuple[str, str, list[int], bytes]]) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for name, dtype, shape, data in records:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        payload.extend(data)
        offset += len(data)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(raw_header)) % 8
    raw_header += b" " * padding
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def read_f32_safetensors(path: Path) -> list[float]:
    raw = path.read_bytes()
    (header_size,) = struct.unpack_from("<Q", raw)
    header = json.loads(raw[8 : 8 + header_size])
    meta = header["tensor"]
    assert meta["dtype"] == "F32"
    begin, end = meta["data_offsets"]
    payload = raw[8 + header_size + begin : 8 + header_size + end]
    return list(struct.unpack("<" + "f" * (len(payload) // 4), payload))


def close(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected), (len(actual), len(expected))
    for index, (left, right) in enumerate(zip(actual, expected)):
        assert math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6), (index, left, right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="turbocider-h3-shards-") as directory:
        root = Path(directory)
        values = [1.0, -2.0, 0.5, 3.25, -4.5, 8.0, 0.0, 16.0, 2.0, 4.0, 6.0, 10.0]
        f32 = b"".join(struct.pack("<f", value) for value in values)
        f16 = b"".join(struct.pack("<e", value) for value in values)
        b16 = b"".join(bf16(value) for value in values)
        write_safetensors(root / "model.safetensors", [
            ("weight_bf16", "BF16", [3, 4], b16),
            ("weight_f16", "F16", [3, 4], f16),
            ("weight_f32", "F32", [3, 4], f32),
        ])
        for key in ("weight_bf16", "weight_f16", "weight_f32"):
            output = root / f"{key}.safetensors"
            subprocess.run([str(args.probe), str(root), key, str(output)], check=True)
            close(read_f32_safetensors(output), values)
        rows = root / "rows.safetensors"
        subprocess.run([str(args.probe), str(root), "weight_bf16", str(rows), "2,0"], check=True)
        close(read_f32_safetensors(rows), values[8:12] + values[0:4])
        print("PASS: native H3 streamed safetensors BF16/F16/F32 and row gather")


if __name__ == "__main__":
    main()
