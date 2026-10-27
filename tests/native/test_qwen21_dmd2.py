"""DMD2 public API admission, immutable asset geometry and metadata contracts."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from test_contract import plan

ROOT = Path(__file__).resolve().parents[2]
FILENAME = "Qwen-Image-2.1-DMD2-2steps.safetensors"
SHA256 = "3a50c28d2f70170a06011e7b5d93a12418edbc4a0fc4de4acee8d9d227a38c21"


def request(**changes):
    return dict(dict(model="qwen-image-2.1", operation="image.generate", prompt="A teapot",
                     width=512, height=512, steps=2, frames=1, audio=False, execution="gpu",
                     residency="component_staged", allow_approximation=True,
                     lora_strategy="inference_time",
                     loras=[dict(path=FILENAME, role="transformer", strength=1)]), **changes)


class DMD2Tests(unittest.TestCase):
    def assert_admitted(self, value):
        code, result, error = plan(value)
        self.assertEqual(code, 0, error)
        self.assertEqual(result["qwen21_sampler"], "dmd2")
        self.assertIn("qwen21_dmd2_fixed_two_step_distillation", result["algorithm_approximations"])

    def test_canonical_and_explicit_sampler_v1_v2(self):
        self.assert_admitted(request())
        self.assert_admitted(request(qwen21_sampler="dmd2", loras=[dict(
            path="pytorch_lora_weights.safetensors", role="transformer", strength=1)]))
        self.assert_admitted(dict(schema_version=2, model="qwen-image-2.1", operation="image.generate",
            inputs=[dict(kind="text", role="prompt", text="A teapot")],
            outputs=[dict(kind="image", path="result.png", width=512, height=512, frames=1, audio=False)],
            sampling=dict(steps=2),
            execution=dict(policy="gpu", residency="component_staged", allow_approximation=True),
            parameters=dict(qwen21_sampler="dmd2"), lora_strategy="inference_time",
            loras=[dict(path="pytorch_lora_weights.safetensors", role="transformer", strength=1)]))

    def test_edit_and_aligned_canvas_domain(self):
        for size in ((512, 512), (768, 512), (1024, 1024), (2048, 512)):
            for reference_size in (512, 1024):
                self.assert_admitted(request(width=size[0], height=size[1], operation="image.edit",
                    qwen21_reference_size=reference_size,
                    inputs=[dict(kind="image", role="reference", path="reference.png")]))

    def test_other_recipes_and_unqualified_requests_fail_closed(self):
        changes = [dict(steps=n) for n in (1, 3, 6, 8, 20, 40)] + [
            dict(qwen21_sampler="unknown"), dict(allow_approximation=False), dict(execution="gpu_ane"),
            dict(hybrid_mlp_mode="runtime"), dict(qwen21_denoiser_backend="native"),
            dict(qwen21_dit_cache="fast"), dict(qwen21_w8a8=True), dict(qwen21_gpu_w8a16=True),
            dict(width=513), dict(width=2048, height=1024), dict(frames=2), dict(audio=True),
            dict(memory_budget_bytes=1 << 30), dict(prompt_enhance=True),
            dict(lora_strategy="in_memory_merge"), dict(operation="image.edit", inputs=[]),
            dict(qwen21_sampler="dmd2", loras=[]), dict(qwen21_reference_size=256),
            dict(loras=[dict(path=FILENAME, role="text_encoder", strength=1)]),
            dict(loras=[dict(path=FILENAME, role="transformer", strength=.5)]),
            dict(qwen21_sampler="dmd2", model="z-image-turbo"),
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(plan(request(**change))[0], 0)
        self.assertNotEqual(plan(request(loras=request()["loras"] * 2, qwen21_sampler="dmd2"))[0], 0)
        self.assertNotEqual(plan(request(operation="image.edit", inputs=[dict(
            kind="image", role="reference", path="reference.png")] * 4))[0], 0)

    def test_peft_metadata_defaults_and_active_variants(self):
        with tempfile.TemporaryDirectory(prefix="tc-lora-metadata-") as directory:
            binary = Path(directory) / "metadata"
            subprocess.run(["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "native"), str(ROOT / "tests/native/lora_metadata_contract_test.cpp"),
                "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_QWEN21_DMD2"), "set DMD2 asset path for the opt-in audit")
    def test_installed_adapter_identity_and_geometry(self):
        path = Path(os.environ["TURBOCIDER_TEST_QWEN21_DMD2"])
        self.assertEqual(path.stat().st_size, 67144520)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), SHA256)
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            self.assertLess(length, 1024 * 1024)
            header = json.loads(stream.read(length))
        metadata = json.loads(header.pop("__metadata__")["lora_adapter_metadata"])
        self.assertEqual(metadata["transformer.r"], 32)
        self.assertEqual(metadata["transformer.lora_alpha"], 32)
        expected = {}
        for block in range(32):
            for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                stem = f"transformer.transformer_blocks.{block}.attn.{projection}"
                expected[stem + ".lora.down.weight"] = [32, 4096]
                expected[stem + ".lora.up.weight"] = [4096, 32]
        self.assertEqual(set(header), set(expected))
        for key, shape in expected.items():
            self.assertEqual(header[key]["shape"], shape)
            self.assertEqual(header[key]["dtype"], "BF16")


if __name__ == "__main__":
    unittest.main()
