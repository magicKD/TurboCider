#!/usr/bin/env python3
"""Read only a bounded safetensors header; no payload evaluation or model import."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import struct

try:
    from .common import emit, report
except ImportError:
    from common import emit, report

DTYPES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
          "I16": 2, "U16": 2, "BF16": 2, "F16": 2, "I32": 4, "U32": 4, "F32": 4,
          "I64": 8, "U64": 8, "F64": 8}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON header key")
        result[key] = value
    return result


def inspect(path, block_prefix="blocks.", max_header=16 << 20):
    per_block, tensors, dtype_bytes = defaultdict(int), [], defaultdict(int)
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("short safetensors length")
        size = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= size <= max_header or size > before.st_size - 8:
            raise ValueError("header exceeds explicit cap/file bounds")
        header = stream.read(size)
        document = json.loads(header, object_pairs_hook=unique_object)
        if not isinstance(document, dict):
            raise ValueError("header must be an object")
        intervals = []
        for name, value in document.items():
            if name == "__metadata__":
                continue
            dtype, shape, offsets = value["dtype"], value["shape"], value["data_offsets"]
            if dtype not in DTYPES or not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
                raise ValueError("unknown dtype/invalid shape")
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
                raise ValueError("invalid offsets")
            start, end = offsets
            elements = 1
            for dim in shape:
                elements *= dim
                if elements > (1 << 63):
                    raise ValueError("shape overflow")
            if start < 0 or end < start or end - start != elements * DTYPES[dtype] or end > before.st_size - 8 - size:
                raise ValueError("tensor payload geometry/file bounds mismatch")
            intervals.append((start, end))
            absolute = 8 + size + start
            tensors.append({"name": name, "dtype": dtype, "shape": shape, "bytes": end - start,
                            "absolute_offset_mod16": absolute % 16, "absolute_offset_mod4096": absolute % 4096})
            dtype_bytes[dtype] += end - start
            if name.startswith(block_prefix):
                ordinal = name[len(block_prefix):].split(".", 1)[0]
                if not ordinal.isdigit():
                    raise ValueError("malformed block ordinal under selected prefix")
                per_block[int(ordinal)] += end - start
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("payload has gaps/overlap")
            cursor = end
        if cursor != before.st_size - 8 - size:
            raise ValueError("unaccounted trailing payload")
        after = os.fstat(stream.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("source changed while reading header")
    block_total = sum(per_block.values())
    return report("model-header", {"tensor_count": len(tensors), "payload_bytes": cursor,
                  "header_bytes_read": 8 + size, "header_sha256": hashlib.sha256(header).hexdigest(),
                  "payload_content_verified": False, "dtype_bytes": dict(dtype_bytes), "per_block_bytes": dict(per_block),
                  "widest_block_bytes": max(per_block.values(), default=0), "nonblock_payload_bytes": cursor - block_total,
                  "tensors": tensors, "limits": "Single shard metadata only. Header hash is NOT a checkpoint digest, RAM floor or semantic format proof."},
                  scope="actual bounded metadata-only file inspection", synthetic=False, backend="cpu_file_metadata")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--block-prefix", default="blocks.")
    parser.add_argument("--max-header-mib", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.block_prefix or not 1 <= args.max_header_mib <= 64:
        parser.error("explicit nonempty prefix and header cap 1..64 MiB required")
    emit(inspect(args.source, args.block_prefix, args.max_header_mib << 20), args.output)


if __name__ == "__main__":
    main()
