#!/usr/bin/env python3
"""Sparse metadata-only hybrid suffix planning oracle; never loads GPU weights."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]

def fixture(path, change=None):
    records = [("x_embedder.weight", [128])]
    for prefix in ["noise_refiner.0", "noise_refiner.1", "context_refiner.0", "context_refiner.1"]:
        for w in (1, 2, 3):
            records.append((f"{prefix}.feed_forward.w{w}.weight", [3840, 10240] if w == 2 else [10240, 3840]))
    for block in range(30):
        for field in range(10):
            records.append((f"layers.{block}.tensor{field:02d}.weight", [128]))
        for w in (1, 2, 3):
            records.append((f"layers.{block}.feed_forward.w{w}.weight", [3840, 10240] if w == 2 else [10240, 3840]))
    header, cursor = {}, 0
    for name, shape in records:
        size = 2
        for n in shape: size *= n
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [cursor, cursor + size]}
        cursor += size
    if change == "missing":
        header["noise_refiner.1.other.weight"] = header.pop("noise_refiner.1.feed_forward.w3.weight")
    if change == "geometry":
        header["noise_refiner.0.feed_forward.w2.weight"]["shape"] = [1920, 20480]
    encoded = json.dumps(header).encode()
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        stream.truncate(8 + len(encoded) + cursor)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path)
args = parser.parse_args()
sources = ["tests/native/z_image_suffix_descriptor_test.cpp", "native/platform/apple/z_image_streaming_descriptor.mm",
           "native/models/z_image/suffix_materialization.cpp", "native/runtime/streaming/config.cpp",
           "native/runtime/streaming/canonical_encoding.cpp", "native/runtime/streaming/layout.cpp",
           "native/runtime/streaming/source_lease.cpp", "native/runtime/memory_manifest.cpp",
           "native/runtime/memory_policy.cpp", "native/core/common.cpp"]
flags = ["-std=c++20", "-Wall", "-Wextra", "-Werror", "-fobjc-arc", "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core")]
if os.environ.get("TC_STREAMING_SANITIZER"):
    flags += ["-fsanitize=" + os.environ["TC_STREAMING_SANITIZER"], "-fno-omit-frame-pointer"]
with tempfile.TemporaryDirectory(prefix="tc-suffix-descriptor-") as raw:
    root = Path(raw)
    paths = [root / f"{name}.safetensors" for name in ("valid", "missing", "geometry")]
    for path, change in zip(paths, (None, "missing", "geometry")): fixture(path, change)
    binary = root / "test"
    subprocess.run(["clang++", *flags, *(str(ROOT / s) for s in sources), "-framework", "Foundation", "-o", str(binary)], check=True)
    subprocess.run([str(binary), str(paths[0]), "fixture", *map(str, paths[1:])], check=True, timeout=60)
    if args.checkpoint:
        subprocess.run([str(binary), str(args.checkpoint.resolve()), "real"], check=True, timeout=60)
