#!/usr/bin/env python3
"""Compile the real LTX metadata/reader with bounded synthetic model fixtures.

--checkpoint additionally inspects real checkpoint metadata (no GPU/full weights).
--metal additionally checks fixed Metal slot backing on the synthetic fixture.
"""
import argparse
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LTX = ROOT / "native/models/ltx_runtime"


def fixture(path, blocks=2):
    tensors = {}
    payload = bytearray()

    def add(name, dtype, shape, data):
        assert name not in tensors
        begin = len(payload)
        payload.extend(data)
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": [begin, len(payload)]}

    for block in range(blocks):
        prefix = f"model.diffusion_model.transformer_blocks.{block}"

        def linear(name):
            add(name + ".weight", "I8", [256, 256], bytes([block + 1]) * (256 * 256))
            add(name + ".weight_scale", "F32", [256], struct.pack("<f", 0.25 + block) * 256)
            add(name + ".bias", "BF16", [256], struct.pack("<H", 0x3F80 + block) * 256)
            quant = b'{"convrot":true,"convrot_groupsize":256}'
            add(name + ".comfy_quant", "U8", [len(quant)], quant)

        for module in ("attn1", "audio_attn1", "attn2", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"):
            name = prefix + "." + module
            for p in ("to_q", "to_k", "to_v", "to_out.0"):
                linear(name + "." + p)
            for suffix, shape in (("q_norm.weight", [256]), ("k_norm.weight", [256]),
                                  ("to_gate_logits.weight", [4, 256]), ("to_gate_logits.bias", [4])):
                count = 1
                for dim in shape:
                    count *= dim
                add(name + "." + suffix, "BF16", shape, struct.pack("<H", 0x3F00 + block) * count)
        for module in ("ff.net.0.proj", "ff.net.2", "audio_ff.net.0.proj", "audio_ff.net.2"):
            linear(prefix + "." + module)
        for suffix, rows in (("scale_shift_table", 9), ("audio_scale_shift_table", 9),
                             ("prompt_scale_shift_table", 2), ("audio_prompt_scale_shift_table", 2),
                             ("scale_shift_table_a2v_ca_video", 5), ("scale_shift_table_a2v_ca_audio", 5)):
            values = (0x3F808000, 0x3F818000, 0xBF808001, 0x3F7FFFFF, 0, 0x80000000)
            data = b"".join(struct.pack("<I", values[(i + block) % len(values)]) for i in range(rows * 256))
            add(prefix + "." + suffix, "F32", [rows, 256], data)
    header = json.dumps(tensors, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metal", action="store_true")
    args = parser.parse_args()
    clang = subprocess.check_output(["xcrun", "--find", "clang"], text=True).strip()
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-isysroot", sdk, "-I", str(LTX)]
    sanitizer = os.environ.get("TC_STREAMING_SANITIZER", "")
    if sanitizer:
        if sanitizer not in ("address,undefined", "thread"):
            raise ValueError("unsupported sanitizer")
        flags += ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=" + sanitizer]
    else:
        flags += ["-O2"]
    with tempfile.TemporaryDirectory(prefix="tc-ltx-streaming-") as raw:
        directory = Path(raw)
        checkpoint = directory / "fixture.safetensors"
        fixture(checkpoint)
        objects = []
        for source in (LTX / "ltx_streaming_layout.c", LTX / "ltx_safetensors.m", LTX / "ltx_weights.m"):
            target = directory / (source.stem + ".o")
            subprocess.run([clang, *flags, *(["-fobjc-arc"] if source.suffix == ".m" else []),
                            "-c", str(source), "-o", str(target)], check=True)
            objects.append(str(target))
        binary = directory / "layout-test"
        subprocess.run([clang, *flags, str(ROOT / "tests/native/ltx_streaming_layout_test.c"),
                        *objects, "-framework", "Foundation", "-o", str(binary)], check=True)
        subprocess.run([str(binary), str(checkpoint)], check=True, timeout=30)
        if args.checkpoint:
            subprocess.run([str(binary), str(args.checkpoint.resolve()), "--inspect"], check=True, timeout=120)
        if args.metal:
            metal = directory / "metal-test"
            gpu = directory / "gpu.o"
            subprocess.run([clang, *flags, "-fobjc-arc", "-c", str(LTX / "ltx_gpu.m"), "-o", str(gpu)], check=True)
            subprocess.run([clang, *flags, str(ROOT / "tests/native/ltx_streaming_slot_test.c"),
                            str(LTX / "ltx_streaming_slot.c"), *objects, str(gpu),
                            "-framework", "Foundation", "-framework", "Metal", "-framework", "MetalPerformanceShaders",
                            "-framework", "MetalPerformanceShadersGraph", "-o", str(metal)], check=True)
            subprocess.run([str(metal), str(checkpoint), str(LTX / "ltx_shaders.metal")], check=True, timeout=60)


if __name__ == "__main__":
    main()
