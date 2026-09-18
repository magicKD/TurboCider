#!/usr/bin/env python3
"""Build and run the host-only LTX public streaming adapter fixture."""

import json
import subprocess
import tempfile
from pathlib import Path

from test_ltx_streaming_layout import fixture

ROOT = Path(__file__).resolve().parents[2]


def tokenizer_fixture() -> dict:
    vocab = {"<pad>": 0, "<bos>": 1, "g": 2, "▁": 3}
    for value in range(256):
        vocab[f"<0x{value:02X}>"] = value + 4
    return {
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "normalizer": {
            "type": "Replace",
            "pattern": {"String": " "},
            "content": "▁",
        },
        "added_tokens": [],
    }


def write_model_root(root: Path) -> None:
    transformer = root / "diffusion_models"
    transformer.mkdir(parents=True)
    fixture(
        transformer /
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        blocks=48,
    )
    files = {
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": b"upsampler",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors": b"video-vae",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors": b"gemma",
        "connector.safetensors": b"connector",
        "config.json": b"{}",
        "embedded_config.json": b"{}",
    }
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    tokenizer = root / "gemma4-12b-ltx-v1/tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text(json.dumps(tokenizer_fixture(), separators=(",", ":")))


def main() -> None:
    compiler = subprocess.check_output(
        ["xcrun", "--find", "clang++"], text=True
    ).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="tc-ltx-public-") as raw:
        root = Path(raw) / "model"
        root.mkdir()
        write_model_root(root)
        binary = Path(raw) / "ltx-public-streaming-test"
        subprocess.run([
            compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-isysroot", sdk, "-I", str(ROOT / "native"),
            "-I", str(ROOT / "native/core"),
            str(ROOT / "tests/native/ltx_public_streaming_test.cpp"),
            "-L", str(ROOT / "build/native"), "-lturbocider",
            "-Wl,-rpath," + str(ROOT / "build/native"),
            "-o", str(binary),
        ], check=True)
        result = subprocess.run(
            [str(binary), str(root)], text=True, capture_output=True,
            timeout=180,
        )
        if result.returncode:
            raise SystemExit(result.stdout + result.stderr)
        print(result.stdout, end="")


if __name__ == "__main__":
    main()
