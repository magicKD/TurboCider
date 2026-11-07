import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CONVROT = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'


def projection(fields, stem, rows, columns):
    fields[stem + ".weight"] = ("I8", [rows, columns], None)
    fields[stem + ".weight_scale"] = ("F32", [rows, 1], None)
    fields[stem + ".comfy_quant"] = ("U8", [len(CONVROT)], CONVROT)


def fixed(shape):
    return "BF16", shape, None


def sources():
    encoder = {"model.norm.weight": fixed([256]), "model.visual.fixture": fixed([128])}
    projection(encoder, "model.embed_tokens", 128, 256)
    projection(encoder, "lm_head", 128, 256)
    denoiser = {
        "img_in.weight": fixed([256, 64]), "modulation.1.weight": fixed([1024, 256]),
        "norm_out.linear.weight": fixed([256, 256]), "proj_out.weight": fixed([64, 256]),
        "time_text_embed.timestep_embedder.linear_1.weight": fixed([256, 256]),
        "time_text_embed.timestep_embedder.linear_2.weight": fixed([256, 256]),
        "txt_in.text_norm.weight": fixed([256]), "txt_in.in_layer.weight": fixed([256, 256]),
        "txt_in.out_layer.weight": fixed([256, 256]),
    }
    for layer in range(4):
        stem = f"model.layers.{layer}."
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            projection(encoder, stem + name, 256, 256)
        for name in ("input_layernorm", "post_attention_layernorm"):
            encoder[stem + name + ".weight"] = fixed([256])
        for name in ("self_attn.q_norm", "self_attn.k_norm"):
            encoder[stem + name + ".weight"] = fixed([128])
        stem = f"transformer_blocks.{layer}."
        for name in ("attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0", "img_mlp.out"):
            projection(denoiser, stem + name, 256, 256)
        projection(denoiser, stem + "img_mlp.gate_up", 512, 256)
        denoiser[stem + "attn.norm_k.weight"] = fixed([128])
        denoiser[stem + "attn.norm_q.weight"] = fixed([128])
    marker = b'{"attention":"comfy_kitchen_int8"}'
    denoiser["transformer_blocks.1.attn.comfy_attention.config"] = ("U8", [len(marker)], marker)
    return encoder, denoiser


def write_fixture(path, fields):
    header, payload = {}, bytearray()
    for ordinal, (name, (dtype, shape, data)) in enumerate(sorted(fields.items())):
        count = math.prod(shape)
        if data is None and dtype == "I8":
            data = bytes(((i * 11 + (i // shape[1]) * 3 + ordinal * 17) % 127 - 63) & 255
                         for i in range(count))
        elif data is None and dtype == "F32":
            # Binary-exact scales keep the packed and raw embeddings equal.
            data = struct.pack(f"<{count}f", *((1 + i % 5) / 1024 for i in range(count)))
        elif data is None and dtype == "BF16":
            if len(shape) == 1:
                data = struct.pack("<H", 0 if name == "txt_in.text_norm.weight" else 0x3F80) * count
            else:
                data = struct.pack(f"<{count}H", *(
                    0x3B80 if (i + ordinal) % 5 < 2 else 0xBB80 for i in range(count)))
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


@unittest.skipUnless(sys.platform == "darwin", "requires the macOS native runtime")
class ConvRotStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = ROOT / os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR", "build/native")
        if not (library / "libturbocider.dylib").is_file():
            raise unittest.SkipTest("build the current native library first")
        directory = tempfile.TemporaryDirectory(prefix="tc-qwen-convrot-stream-")
        cls.addClassCleanup(directory.cleanup)
        folder = Path(directory.name)
        encoder, denoiser = sources()
        bad_marker = dict(encoder)
        marker = CONVROT.replace(b"256", b"128")
        bad_marker["model.layers.0.mlp.up_proj.comfy_quant"] = ("U8", [len(marker)], marker)
        missing_scale = dict(encoder)
        del missing_scale["model.layers.3.mlp.up_proj.weight_scale"]
        bad_attention = dict(denoiser)
        marker = b'{"attention":"unknown"}'
        bad_attention["transformer_blocks.1.attn.comfy_attention.config"] = ("U8", [len(marker)], marker)
        bad_visual = {**encoder, "model.visual.fixture": ("F32", [128], None)}
        cls.fixtures = []
        for name, fields in zip(("encoder", "denoiser", "bad-marker", "missing-scale", "bad-attention", "bad-visual"),
                                (encoder, denoiser, bad_marker, missing_scale, bad_attention, bad_visual)):
            fixture = folder / f"{name}.safetensors"
            write_fixture(fixture, fields)
            cls.fixtures.append(str(fixture))
        mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx"
        cls.binary = folder / "probe"
        subprocess.run([
            "xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-mmacosx-version-min=26.2", "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core"),
            "-isystem", str(mlx / "include"), str(ROOT / "tests/native/qwen21_convrot_streaming_test.cpp"),
            "-L", str(library), "-lturbocider", "-L", str(mlx / "lib"), "-lmlx",
            "-Wl,-rpath," + str(library), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(cls.binary),
        ], check=True)

    def run_probe(self, mode):
        result = subprocess.run([str(self.binary), *self.fixtures, mode], text=True, capture_output=True, timeout=90,
                                env={**os.environ, "TURBOCIDER_QWEN21_SOURCE_REGISTRATION": "0"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout, end="")
        return result.stdout

    def test_typed_source_validation_and_budget(self):
        self.assertIn("PASS ConvRot typed source closure", self.run_probe("cpu"))

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit actual Metal opt-in required")
    def test_streamed_resident_equivalence(self):
        result = self.run_probe("gpu")
        self.assertIn("PASS ConvRot Float-A streamed/resident numerics", result)
        if "SKIP ConvRot W8A8 GPU unavailable" not in result:
            self.assertIn("PASS ConvRot W8A8 streamed/resident numerics", result)


if __name__ == "__main__":
    unittest.main()
