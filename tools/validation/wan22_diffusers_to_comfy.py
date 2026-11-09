#!/usr/bin/env python3
"""Create a ComfyUI-keyed clone of a Wan Diffusers safetensors checkpoint.

This only renames tensor keys in the safetensors header. Tensor payloads,
dtypes and offsets are unchanged. On APFS ``cp -c`` clones the large source
file; only the header is subsequently copied on write. A fresh output path is
required, and the source is never opened for writing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


TOP_LEVEL = {
    "scale_shift_table": "head.modulation",
    "proj_out.": "head.head.",
    "condition_embedder.time_embedder.linear_1.": "time_embedding.0.",
    "condition_embedder.time_embedder.linear_2.": "time_embedding.2.",
    "condition_embedder.time_proj.": "time_projection.1.",
    "condition_embedder.text_embedder.linear_1.": "text_embedding.0.",
    "condition_embedder.text_embedder.linear_2.": "text_embedding.2.",
}
BLOCKS = {
    "scale_shift_table": "modulation",
    "norm2.weight": "norm3.weight",
    "norm2.bias": "norm3.bias",
    "ffn.net.0.proj.weight": "ffn.0.weight",
    "ffn.net.0.proj.bias": "ffn.0.bias",
    "ffn.net.2.weight": "ffn.2.weight",
    "ffn.net.2.bias": "ffn.2.bias",
}
for number, prefix in ((1, "self_attn"), (2, "cross_attn")):
    for source, target in (("to_q", "q"), ("to_k", "k"), ("to_v", "v"), ("to_out.0", "o")):
        for suffix in ("weight", "bias"):
            BLOCKS[f"attn{number}.{source}.{suffix}"] = f"{prefix}.{target}.{suffix}"
    for part in ("norm_q", "norm_k"):
        BLOCKS[f"attn{number}.{part}.weight"] = f"{prefix}.{part}.weight"


def map_key(name):
    if name in ("patch_embedding.weight", "patch_embedding.bias"):
        return name
    for source, target in TOP_LEVEL.items():
        if name == source or (source.endswith(".") and name.startswith(source)):
            return target + name[len(source):]
    match = re.fullmatch(r"blocks\.(\d+)\.(.+)", name)
    if match and match.group(2) in BLOCKS:
        return f"blocks.{match.group(1)}.{BLOCKS[match.group(2)]}"
    raise ValueError(f"unsupported Wan Diffusers tensor name: {name}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source, destination, expected_sha256=None):
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(f"destination must be new: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("destination parent must already exist")
    source_hash = sha256(source)
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise ValueError("source checkpoint SHA-256 mismatch")
    with source.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors prefix")
        header_size = int.from_bytes(prefix, "little")
        if not 0 < header_size <= 64 * 1024 * 1024 or 8 + header_size > source.stat().st_size:
            raise ValueError("invalid safetensors header size")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict) or not header:
        raise ValueError("empty or invalid safetensors header")
    mapped = {}
    body_size = source.stat().st_size - 8 - header_size
    for name, item in header.items():
        target = name if name == "__metadata__" else map_key(name)
        if target in mapped:
            raise ValueError(f"tensor name collision: {target}")
        if name != "__metadata__":
            if not isinstance(item, dict) or not isinstance(item.get("shape"), list) or not isinstance(item.get("dtype"), str):
                raise ValueError(f"invalid tensor entry: {name}")
            offsets = item.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(x) is int for x in offsets) or not 0 <= offsets[0] <= offsets[1] <= body_size:
                raise ValueError(f"invalid tensor byte range: {name}")
        mapped[target] = item
    encoded = json.dumps(mapped, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > header_size:
        raise ValueError("renamed header does not fit the original header allocation")
    encoded += b" " * (header_size - len(encoded))
    with tempfile.TemporaryDirectory(prefix=".wan22-comfy-", dir=destination.parent) as temporary:
        clone = Path(temporary) / "checkpoint.safetensors"
        # On the target Mac this is an APFS clone, avoiding a second 10 GB copy.
        subprocess.run(["cp", "-c", str(source), str(clone)], check=True)
        with clone.open("r+b") as stream:
            stream.seek(8)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic no-overwrite publish; the temporary directory is ours alone.
        os.link(clone, destination)
    return {"source": str(source), "source_sha256": source_hash,
            "destination": str(destination), "tensor_count": len(mapped) - int("__metadata__" in mapped),
            "header_bytes": header_size, "payload_bytes": body_size,
            "payload_contract": "unchanged offsets and bytes; only cloned header renamed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.destination, args.expected_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
