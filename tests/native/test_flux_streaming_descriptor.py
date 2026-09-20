#!/usr/bin/env python3
"""Build the FLUX.2 Klein 9B metadata-only streaming descriptor."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SHARD_NAMES = (
    "diffusion_pytorch_model-00001-of-00002.safetensors",
    "diffusion_pytorch_model-00002-of-00002.safetensors",
)
HIDDEN = 4096


def fixed_records(hidden: int = HIDDEN, context: int = HIDDEN * 3) -> list[tuple[str, list[int]]]:
    return [
        ("context_embedder.weight", [hidden, context]),
        ("double_stream_modulation_img.linear.weight", [hidden * 6, hidden]),
        ("double_stream_modulation_txt.linear.weight", [hidden * 6, hidden]),
        ("norm_out.linear.weight", [hidden * 2, hidden]),
        ("proj_out.weight", [128, hidden]),
        ("single_stream_modulation.linear.weight", [hidden * 3, hidden]),
        ("time_guidance_embed.timestep_embedder.linear_1.weight", [hidden, 256]),
        ("time_guidance_embed.timestep_embedder.linear_2.weight", [hidden, hidden]),
        ("x_embedder.weight", [hidden, 128]),
    ]


def dual_shapes(hidden: int = HIDDEN) -> list[tuple[str, list[int]]]:
    records = []
    for name in (
        "attn.add_k_proj.weight", "attn.add_q_proj.weight",
        "attn.add_v_proj.weight", "attn.to_add_out.weight",
        "attn.to_k.weight", "attn.to_out.0.weight",
        "attn.to_q.weight", "attn.to_v.weight",
    ):
        records.append((name, [hidden, hidden]))
    for name in (
        "attn.norm_added_k.weight", "attn.norm_added_q.weight",
        "attn.norm_k.weight", "attn.norm_q.weight",
    ):
        records.append((name, [128]))
    records.extend((
        ("ff.linear_in.weight", [hidden * 6, hidden]),
        ("ff.linear_out.weight", [hidden, hidden * 3]),
        ("ff_context.linear_in.weight", [hidden * 6, hidden]),
        ("ff_context.linear_out.weight", [hidden, hidden * 3]),
    ))
    return records


def single_shapes(hidden: int = HIDDEN) -> list[tuple[str, list[int]]]:
    return [
        ("attn.norm_k.weight", [128]),
        ("attn.norm_q.weight", [128]),
        ("attn.to_out.weight", [hidden, hidden * 4]),
        ("attn.to_qkv_mlp_proj.weight", [hidden * 9, hidden]),
    ]


def model_records(klein4: bool = False) -> list[tuple[str, list[int], int]]:
    hidden = 3072 if klein4 else HIDDEN
    records = [(name, shape, 0 if klein4 else index % 2)
               for index, (name, shape) in enumerate(fixed_records(hidden, 7680 if klein4 else hidden * 3))]
    for block in range(5 if klein4 else 8):
        for suffix, shape in dual_shapes(hidden):
            records.append((f"transformer_blocks.{block}.{suffix}", shape, 0))
    for block in range(20 if klein4 else 24):
        for suffix, shape in single_shapes(hidden):
            records.append((f"single_transformer_blocks.{block}.{suffix}", shape, 0 if klein4 else 1))
    return records


def write_fixture(directory: Path, *, missing: bool = False,
                  wrong_dtype: bool = False, wrong_shape: bool = False,
                  overlap: bool = False, klein4: bool = False) -> None:
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({
        "attention_head_dim": 128,
        "guidance_embeds": False,
        "in_channels": 128,
        "joint_attention_dim": 7680 if klein4 else HIDDEN * 3,
        "num_attention_heads": 24 if klein4 else 32,
        "num_layers": 5 if klein4 else 8,
        "num_single_layers": 20 if klein4 else 24,
    }, separators=(",", ":")))
    records = model_records(klein4)
    shards = ("diffusion_pytorch_model.safetensors",) if klein4 else SHARD_NAMES
    weight_map = {name: shards[shard]
                  for name, _shape, shard in records}
    shard_records: list[list[tuple[str, list[int]]]] = [[] for _ in shards]
    for name, shape, shard in records:
        if not (missing and name ==
                f"single_transformer_blocks.{19 if klein4 else 23}.attn.to_out.weight"):
            shard_records[shard].append((name, shape))

    total_size = 0
    for shard, entries in enumerate(shard_records):
        header: dict[str, object] = {}
        cursor = 0
        for index, (name, original_shape) in enumerate(entries):
            shape = list(original_shape)
            if (wrong_shape and name ==
                    "single_transformer_blocks.1.attn.norm_q.weight"):
                shape = [64]
            dtype = (
                "F32"
                if (wrong_dtype and name ==
                    "transformer_blocks.0.attn.to_q.weight")
                else "BF16"
            )
            item_bytes = 4 if dtype == "F32" else 2
            size = item_bytes
            for dimension in shape:
                size *= dimension
            begin = 0 if overlap and shard == 0 and index == 1 else cursor
            header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [begin, begin + size],
            }
            cursor += size
            total_size += size
        encoded = json.dumps(header, separators=(",", ":")).encode()
        path = directory / shards[shard]
        with path.open("wb") as stream:
            stream.write(struct.pack("<Q", len(encoded)))
            stream.write(encoded)
            stream.truncate(8 + len(encoded) + cursor)
    if not klein4:
        (directory / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_size},
                        "weight_map": weight_map}, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer", type=Path)
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
        ROOT / "tests/native/flux_streaming_descriptor_test.cpp",
        ROOT / "native/platform/apple/flux_streaming_descriptor.mm",
        ROOT / "native/runtime/streaming/config.cpp",
        ROOT / "native/runtime/streaming/layout.cpp",
        ROOT / "native/runtime/streaming/canonical_encoding.cpp",
        ROOT / "native/runtime/streaming/source_lease.cpp",
        ROOT / "native/runtime/memory_manifest.cpp",
        ROOT / "native/runtime/memory_policy.cpp",
        ROOT / "native/core/common.cpp",
    )
    with tempfile.TemporaryDirectory(prefix="tc-flux-descriptor-") as raw:
        directory = Path(raw)
        fixtures = []
        for name, options in (
            ("valid", {}),
            ("missing", {"missing": True}),
            ("wrong-dtype", {"wrong_dtype": True}),
            ("wrong-shape", {"wrong_shape": True}),
            ("overlap", {"overlap": True}),
        ):
            fixture = directory / name
            write_fixture(fixture, **options)
            fixtures.append(fixture)
        binary = directory / "flux-streaming-descriptor-test"
        subprocess.run([
            compiler, *flags, *map(str, sources), "-framework", "Foundation",
            "-o", str(binary),
        ], check=True)
        subprocess.run([str(binary), *map(str, fixtures)], check=True,
                       timeout=240)
        for name, options in (("valid", {}), ("missing", {"missing": True}),
                              ("dtype", {"wrong_dtype": True}), ("shape", {"wrong_shape": True}),
                              ("overlap", {"overlap": True})):
            fixture4 = directory / ("4b-" + name)
            write_fixture(fixture4, klein4=True, **options)
            run = subprocess.run([str(binary), "--4b", str(fixture4)], capture_output=True, text=True)
            if run.returncode != (0 if name == "valid" else 1):
                raise AssertionError((name, run.returncode, run.stdout, run.stderr))
            if name != "valid" and "flux_streaming_metadata:" not in run.stderr:
                raise AssertionError((name, run.stderr))
            if name == "valid": print(run.stdout, end="")
        if arguments.transformer:
            real = arguments.transformer.expanduser().resolve()
            if not real.is_dir():
                raise FileNotFoundError(real)
            subprocess.run([str(binary), str(real)], check=True, timeout=180)


if __name__ == "__main__":
    main()
