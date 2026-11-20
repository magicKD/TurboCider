#!/usr/bin/env python3
"""Build and run the host-only H3 Turbo public adapter contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from streaming_test_fixtures import H3_MATRICES, byte_vocab, write_bf16_shard


ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()


def write_model_root(root: Path) -> None:
    transformer = root / "FL2VA/transformer"
    transformer.mkdir(parents=True)
    shards: list[list[tuple[str, int, int]]] = [[] for _ in range(13)]
    ordinal = 0
    for block in range(50):
        for suffix, rows, columns in H3_MATRICES:
            shards[ordinal % len(shards)].append(
                (f"blocks.{block}.{suffix}", rows, columns)
            )
            ordinal += 1
    for index, tensors in enumerate(shards, 1):
        write_bf16_shard(
            transformer / f"model-{index:05d}-of-00013.safetensors",
            tensors,
        )
    (transformer / "config.json").write_text("{}")
    (transformer / "model.safetensors.index.json").write_text("{}")
    (transformer / "h3-turbo-merge-manifest.json").write_text(json.dumps({
        "schema": "h3-turbo-merge-manifest-v2",
        "identity": {"profile": "lightx2v-4step"},
        "source": {"repository": "lightx2v/Minimax-h3-Turbo"},
        "mapping": {},
        "shards": {str(i): {} for i in range(13)},
    }, separators=(",", ":")))

    tokenizer = root / "FL2VA/tokenizer/tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text(json.dumps({
        "model": {
            "type": "BPE",
            "unk_token": None,
            "vocab": byte_vocab(),
            "merges": [],
        },
        "normalizer": {"type": "NFC"},
        "added_tokens": [],
    }, separators=(",", ":")))

    components = {
        "FL2VA/text_encoder": ("config.json", "model.safetensors"),
        "FL2VA/video_vae/source": ("config.json", "model.safetensors"),
        "FL2VA/audio_vae": ("config.json", "model.safetensors"),
    }
    for relative, names in components.items():
        directory = root / relative
        directory.mkdir(parents=True)
        (directory / names[0]).write_text("{}")
        (directory / names[1]).write_bytes(b"public-source-fixture")


def main() -> None:
    compiler = subprocess.check_output(
        ["xcrun", "--find", "clang++"], text=True
    ).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    mlx_root = Path(subprocess.check_output([
        sys.executable, "-I", "-c",
        "import sysconfig; print(sysconfig.get_paths()['purelib'] + '/mlx')",
    ], text=True).strip())
    with tempfile.TemporaryDirectory(prefix="tc-h3-public-") as raw:
        root = Path(raw) / "model"
        write_model_root(root)
        binary = Path(raw) / "h3-public-streaming-test"
        subprocess.run([
            compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-isysroot", sdk, "-I", str(ROOT / "native"),
            "-I", str(ROOT / "native/core"),
            "-isystem", str(mlx_root / "include"),
            str(ROOT / "tests/native/h3_public_streaming_test.cpp"),
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
        if not result.stdout.startswith("PASS H3 public adapter:"):
            print(result.stdout, end="")
            print(result.stderr, end="")
            raise RuntimeError("H3 public adapter test lacks PASS marker")
        print(result.stdout, end="")


if __name__ == "__main__":
    main()
