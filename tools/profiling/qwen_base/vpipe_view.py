#!/usr/bin/env python3
"""APFS CoW header-only gate/up views for vpipe's permissive shard reader.

Not a portable safetensors export: the view headers leave unused payload gaps.
No fallback to a full copy, no original-file writes, no checkpoint conversion.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import struct

from collect import ROOT, model_snapshot, sha256_file, write_json


def split_header(header):
    gate, up = {}, {}
    index = {name: "source.safetensors" for name in header if name != "__metadata__"}
    for layer in range(32):
        prefix = f"transformer_blocks.{layer}.img_mlp."
        name = prefix + "gate_up.weight"
        value = header.get(name, {})
        begin, end = value.get("data_offsets", (0, 0))
        if value.get("dtype") != "BF16" or value.get("shape") != [24576, 4096] or end - begin != 24576 * 4096 * 2:
            raise ValueError("original32 contiguous [gate;up] BF16 matrices required")
        if prefix + "gate_layer.weight" in header or prefix + "proj.weight" in header:
            raise ValueError("ambiguous source gate/up names")
        half = (end - begin) // 2
        gate[prefix + "gate_layer.weight"] = dict(dtype="BF16", shape=[12288, 4096], data_offsets=[begin, begin + half])
        up[prefix + "proj.weight"] = dict(dtype="BF16", shape=[12288, 4096], data_offsets=[begin + half, end])
        del index[name]
        index[prefix + "gate_layer.weight"] = "gate-view.safetensors"
        index[prefix + "proj.weight"] = "up-view.safetensors"
    return gate, up, index


def validate(model, root):
    """Bound all view metadata and source generations; not a payload hash."""
    source = model / "diffusion_models/qwen_image_2.1_bf16.safetensors"
    evidence = json.loads((root / "view.json").read_text())
    if (evidence.get("schema") != "tc-vpipe-qwen-header-view-v1" or
            evidence.get("original_snapshot") != model_snapshot(model) or
            (root / "source.safetensors").resolve(strict=True) != source.resolve(strict=True)):
        raise ValueError("view must resolve to the unchanged original model")
    with source.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= size <= 16 << 20:
            raise ValueError("bounded original header required")
        gate, up, index = split_header(json.loads(stream.read(size)))
    if json.loads((root / "model.safetensors.index.json").read_text()) != dict(weight_map=index):
        raise ValueError("view's tensor-to-shard mapping changed")
    files = {}
    for filename, expected in (("gate-view.safetensors", gate), ("up-view.safetensors", up)):
        path = root / filename
        stat = path.stat()
        with path.open("rb") as view, source.open("rb") as original:
            if struct.unpack("<Q", view.read(8))[0] != size or stat.st_size != source.stat().st_size:
                raise ValueError("view moved/truncated the source data section")
            raw = view.read(size)
            if json.loads(raw) != expected:
                raise ValueError("view's shape/offset/name metadata changed")
            # Small independent payload boundary witnesses per projection.
            # APFS CoW made the full payload; these are NOT full payload hashes.
            for info in expected.values():
                start, end = info["data_offsets"]
                for offset in (start, end - 128):
                    view.seek(8 + size + offset)
                    original.seek(8 + size + offset)
                    if view.read(128) != original.read(128):
                        raise ValueError("view payload no longer matches the original range")
        files[filename] = dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                               mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)
    files["config.json"] = dict(sha256=sha256_file(root / "config.json"))
    files["view.json"] = dict(sha256=sha256_file(root / "view.json"))
    return files


def create(model, output):
    model = model.resolve(strict=True)
    source = model / "diffusion_models/qwen_image_2.1_bf16.safetensors"
    snapshot = model_snapshot(model)
    with source.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= size <= 16 << 20:
            raise ValueError("bounded original header required")
        header = json.loads(stream.read(size))
    gate, up, index = split_header(header)
    if shutil.disk_usage(output.parent).free < 1 << 30:
        raise ValueError("at least1GiB headroom required even for header-only clones")
    output.mkdir(exist_ok=False)
    libc = C.CDLL(None, use_errno=True)
    clone = libc.clonefile
    clone.argtypes = [C.c_char_p, C.c_char_p, C.c_int]
    clone.restype = C.c_int
    before_free = shutil.disk_usage(output).free
    for filename, values in (("gate-view.safetensors", gate), ("up-view.safetensors", up)):
        raw = json.dumps(values, separators=(",", ":")).encode()
        if len(raw) > size:
            raise ValueError("view must retain the original data-section start")
        path = output / filename
        if clone(os.fsencode(source), os.fsencode(path), 0):
            raise OSError(C.get_errno(), "APFS clonefile required; refusing a physical-copy fallback")
        # Alter ONLY the new clone's existing header. Every data offset and
        # byte stays at its original absolute position, shared copy-on-write.
        with path.open("r+b") as stream:
            stream.seek(8)
            stream.write(raw + b" " * (size - len(raw)))
        path.chmod(0o444)
    (output / "source.safetensors").symlink_to(source)
    write_json(output / "model.safetensors.index.json", dict(weight_map=index))
    write_json(output / "config.json", dict(_class_name="QwenImage21Transformer2DModel", num_layers=32,
               num_attention_heads=32, attention_head_dim=128, in_channels=64, out_channels=64, context_in_dim=4096,
               mlp_ratio=3, patch_size=1, axes_dims_rope=[16, 56, 56], eps=1e-6, causal_condition=True))
    if snapshot != model_snapshot(model):
        raise ValueError("original model generation changed; views cannot be admitted")
    result = dict(schema="tc-vpipe-qwen-header-view-v1", source=str(source), original_snapshot=snapshot,
                  original_data_start=8 + size, payload_transform="none; APFS clonefile, header-only writes",
                  layout="gate_up's first12288 rows=gate; second12288=up, matching native Transformer slicing",
                  portable_safetensors_export=False, reader="vpipe18ddb3b shard mmap; unused payload gaps retained",
                  free_bytes_before=before_free, free_bytes_after=shutil.disk_usage(output).free,
                  header_only_view_files=["gate-view.safetensors", "up-view.safetensors"],
                  recipe_sha256=sha256_file(Path(__file__)))
    write_json(output / "view.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.model, args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
