#!/usr/bin/env python3
"""Build and run the host-only Z-Image public adapter contract."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from test_z_image_streaming_descriptor import write_fixture


ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()


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


def main() -> None:
    compiler = subprocess.check_output(
        ["xcrun", "--find", "clang++"], text=True
    ).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    mlx_root = Path(subprocess.check_output([
        str(ROOT / ".venv/bin/python3"), "-I", "-c",
        "import sysconfig; print(sysconfig.get_paths()['purelib'] + '/mlx')",
    ], text=True).strip())
    with tempfile.TemporaryDirectory(prefix="tc-z-image-public-") as raw:
        root = Path(raw) / "model"
        transformer = (
            root / "split_files/diffusion_models/z_image_turbo_bf16.safetensors"
        )
        transformer.parent.mkdir(parents=True)
        write_fixture(transformer)
        for relative in (
            "split_files/text_encoders/qwen_3_4b.safetensors",
            "split_files/vae/ae.safetensors",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"public-source-fixture")
        tokenizer = root / "tokenizer/tokenizer.json"
        tokenizer.parent.mkdir(parents=True)
        tokenizer.write_text(json.dumps({
            "model": {"type": "BPE", "vocab": byte_vocab(), "merges": []},
            "pre_tokenizer": {
                "pretokenizers": [{"pattern": {"Regex": "."}}]
            },
            "added_tokens": [],
        }))

        binary = Path(raw) / "z-image-public-streaming-test"
        subprocess.run([
            compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-isysroot", sdk, "-I", str(ROOT / "native"),
            "-I", str(ROOT / "native/core"),
            "-isystem", str(mlx_root / "include"),
            str(ROOT / "tests/native/z_image_public_streaming_test.cpp"),
            "-L", str(NATIVE), "-lturbocider",
            "-Wl,-rpath," + str(NATIVE),
            "-o", str(binary),
        ], check=True)
        result = subprocess.run(
            [str(binary), str(root)], text=True, capture_output=True,
            timeout=180,
        )
        if result.returncode:
            print(result.stdout, end="")
            print(result.stderr, end="")
            raise SystemExit(result.returncode)
        if not result.stdout.startswith("PASS Z-Image public adapter:"):
            print(result.stdout, end="")
            print(result.stderr, end="")
            raise RuntimeError("Z-Image public adapter test lacks PASS marker")
        print(result.stdout, end="")


if __name__ == "__main__":
    main()
