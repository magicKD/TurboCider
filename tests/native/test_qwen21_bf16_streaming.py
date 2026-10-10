import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def fixture(path, tensors):
    metadata, payload = {}, bytearray()
    for ordinal, (name, shape) in enumerate(sorted(tensors.items())):
        count = math.prod(shape)
        if name.endswith("norm.weight") or "layernorm.weight" in name or name.endswith("norm_q.weight") or name.endswith("norm_k.weight"):
            values = [1.0] * count
        elif name == "txt_in.text_norm.weight":
            values = [0.0] * count
        else:
            values = [.03 * math.sin(index * .017 + ordinal * .31) for index in range(count)]
        start = len(payload)
        for value in values:
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            payload.extend(struct.pack("<H", (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16))
        metadata[name] = dict(dtype="BF16", shape=shape, data_offsets=[start, len(payload)])
    header = json.dumps(metadata, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit actual Metal opt-in required")
class Bf16StreamingTests(unittest.TestCase):
    def test_original_numerics_and_reader_lifetimes(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen-bf16-stream-") as raw:
            folder = Path(raw)
            text = {"model.embed_tokens.weight": [64, 128], "model.norm.weight": [128]}
            dit = {"img_in.weight": [128, 64], "modulation.1.weight": [512, 128],
                   "norm_out.linear.weight": [128, 128], "proj_out.weight": [64, 128],
                   "time_text_embed.timestep_embedder.linear_1.weight": [128, 256],
                   "time_text_embed.timestep_embedder.linear_2.weight": [128, 128],
                   "txt_in.text_norm.weight": [128], "txt_in.in_layer.weight": [128, 128],
                   "txt_in.out_layer.weight": [128, 128]}
            for layer in range(4):
                p = f"model.layers.{layer}."
                for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight",
                               "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                    text[p + suffix] = [128]
                for suffix in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                               "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
                    text[p + suffix] = [128, 128]
                p = f"transformer_blocks.{layer}."
                for suffix in ("attn.norm_k.weight", "attn.norm_q.weight"):
                    dit[p + suffix] = [128]
                for suffix in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "attn.to_out.0.weight", "img_mlp.out.weight"):
                    dit[p + suffix] = [128, 128]
                dit[p + "img_mlp.gate_up.weight"] = [256, 128]
            encoder, denoiser = folder / "encoder.safetensors", folder / "denoiser.safetensors"
            fixture(encoder, text)
            fixture(denoiser, dit)
            malformed = folder / "malformed.safetensors"
            fixture(malformed, {k: v for k, v in text.items() if k != "model.layers.3.mlp.up_proj.weight"})
            changed = folder / "owned-change.safetensors"
            shutil.copyfile(denoiser, changed)
            mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx"
            library = ROOT / os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR", "build/qwen21-bf16-stream-v1-private")
            binary = folder / "probe"
            subprocess.run(["xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                "-mmacosx-version-min=26.2", "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core"),
                "-isystem", str(mlx / "include"), str(ROOT / "tests/native/qwen21_bf16_streaming_test.cpp"),
                "-L", str(library), "-lturbocider", "-L", str(mlx / "lib"), "-lmlx",
                "-Wl,-rpath," + str(library), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary), str(encoder), str(denoiser), str(malformed), str(changed)],
                                    text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS Qwen original-BF16 streaming", result.stdout)
            print(result.stdout, end="")


if __name__ == "__main__":
    unittest.main()
