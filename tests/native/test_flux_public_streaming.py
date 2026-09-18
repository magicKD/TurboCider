#!/usr/bin/env python3
"""Build and run the host-only FLUX public adapter contract."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from test_flux_streaming_descriptor import HIDDEN, SHARD_NAMES, write_fixture


ROOT = Path(__file__).resolve().parents[2]


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


def write_model_root(root: Path) -> None:
    transformer = root / "transformer"
    write_fixture(transformer)
    (transformer / "config.json").write_text(json.dumps({
        "attention_head_dim": 128,
        "guidance_embeds": False,
        "in_channels": 128,
        "joint_attention_dim": HIDDEN * 3,
        "num_attention_heads": 32,
        "num_layers": 8,
        "num_single_layers": 24,
    }, separators=(",", ":")))

    text_encoder = root / "text_encoder"
    text_encoder.mkdir(parents=True)
    (text_encoder / "config.json").write_text(json.dumps({
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }, separators=(",", ":")))
    (text_encoder / "model.safetensors").write_bytes(b"text-encoder-fixture")

    vae = root / "vae"
    vae.mkdir(parents=True)
    (vae / "config.json").write_text(json.dumps({"latent_channels": 32}))
    (vae / "ae.safetensors").write_bytes(b"vae-fixture")

    tokenizer = root / "tokenizer"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": byte_vocab(), "merges": []},
        "pre_tokenizer": {"pretokenizers": [{"pattern": {"Regex": "."}}]},
        "added_tokens": [],
    }))


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
    with tempfile.TemporaryDirectory(prefix="tc-flux-public-") as raw:
        root = Path(raw) / "model"
        root.mkdir()
        write_model_root(root)

        missing = Path(raw) / "missing-vae-config"
        subprocess.run(["cp", "-R", str(root), str(missing)], check=True)
        (missing / "vae/config.json").unlink()
        binary = Path(raw) / "flux-public-streaming-test"
        subprocess.run([
            compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-isysroot", sdk, "-I", str(ROOT / "native"),
            "-I", str(ROOT / "native/core"),
            "-isystem", str(mlx_root / "include"),
            str(ROOT / "tests/native/flux_public_streaming_test.cpp"),
            "-L", str(ROOT / "build/native"), "-lturbocider",
            "-Wl,-rpath," + str(ROOT / "build/native"),
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
        if not result.stdout.startswith("PASS FLUX public adapter:"):
            print(result.stdout, end="")
            print(result.stderr, end="")
            raise RuntimeError("FLUX public adapter test lacks PASS marker")
        print(result.stdout, end="")

        missing_result = subprocess.run(
            [str(binary), str(missing)], text=True, capture_output=True,
            timeout=180,
        )
        missing_output = missing_result.stdout + missing_result.stderr
        if missing_result.returncode == 0 or "vae/config.json" not in missing_output:
            print(missing_result.stdout, end="")
            print(missing_result.stderr, end="")
            raise RuntimeError(
                "FLUX public adapter did not reject missing VAE closure"
            )


if __name__ == "__main__":
    main()
