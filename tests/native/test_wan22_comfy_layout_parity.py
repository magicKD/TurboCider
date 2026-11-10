"""Mixed FP16/FP32 Comfy keys must reproduce the canonical BF16 graph."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.validation.wan22_diffusers_to_comfy import map_key


PROBE = Path(os.environ.get("TURBOCIDER_WAN22_TRANSFORMER_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file(), "native Wan2.2 transformer probe required")
class Wan22ComfyLayoutParityTests(unittest.TestCase):
    def test_mixed_comfy_checkpoint_matches_diffusers_and_rejects_unknown_keys(self):
        import torch
        from safetensors.torch import load_file, save_file

        torch.manual_seed(22)
        channels, hidden, heads, intermediate, layers, text_dim, frequency = 4, 32, 4, 64, 2, 16, 8

        def values(shape):
            # Exact in BF16 and FP16, so conversion itself introduces no discrepancy.
            return (torch.randint(-64, 65, shape).float() / 4096).to(torch.bfloat16)

        weights = {"patch_embedding.weight": values((hidden, channels, 1, 2, 2)),
                   "patch_embedding.bias": values((hidden,)), "scale_shift_table": values((1, 2, hidden)),
                   "proj_out.weight": values((channels * 4, hidden)), "proj_out.bias": values((channels * 4,))}
        for name, rows, cols in (("condition_embedder.time_embedder.linear_1", hidden, frequency),
                                 ("condition_embedder.time_embedder.linear_2", hidden, hidden),
                                 ("condition_embedder.time_proj", 6 * hidden, hidden),
                                 ("condition_embedder.text_embedder.linear_1", hidden, text_dim),
                                 ("condition_embedder.text_embedder.linear_2", hidden, hidden)):
            weights[name + ".weight"], weights[name + ".bias"] = values((rows, cols)), values((rows,))
        for layer in range(layers):
            prefix = f"blocks.{layer}."
            weights[prefix + "scale_shift_table"] = values((1, 6, hidden))
            weights[prefix + "norm2.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
            weights[prefix + "norm2.bias"] = values((hidden,))
            for attention in ("attn1", "attn2"):
                for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                    name = prefix + attention + "." + projection
                    weights[name + ".weight"], weights[name + ".bias"] = values((hidden, hidden)), values((hidden,))
                for norm in ("norm_q", "norm_k"):
                    weights[prefix + attention + "." + norm + ".weight"] = torch.ones(hidden, dtype=torch.bfloat16)
            for name, rows, cols in (("ffn.net.0.proj", intermediate, hidden), ("ffn.net.2", hidden, intermediate)):
                weights[prefix + name + ".weight"], weights[prefix + name + ".bias"] = values((rows, cols)), values((rows,))

        comfy = {map_key(name): value.to(torch.float16 if index % 2 else torch.float32).contiguous()
                 for index, (name, value) in enumerate(weights.items())}
        self.assertEqual(len(comfy), len(weights))
        inputs = {"latents": values((1, channels, 3, 4, 4)), "text": values((1, 5, text_dim))}
        with tempfile.TemporaryDirectory(prefix="tc-wan22-comfy-layout-") as temporary:
            root = Path(temporary)
            save_file(weights, str(root / "diffusers.safetensors"))
            save_file(comfy, str(root / "comfy.safetensors"))
            save_file(inputs, str(root / "input.safetensors"))

            def run(source, output, *extra):
                return subprocess.run([str(PROBE), str(source), str(root / "input.safetensors"), str(output),
                                       str(channels), str(hidden), str(heads), str(intermediate), str(layers),
                                       str(text_dim), str(frequency), "960", *extra],
                                      capture_output=True, text=True, timeout=120)

            baseline = run(root / "diffusers.safetensors", root / "diffusers-output.safetensors")
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            converted = run(root / "comfy.safetensors", root / "comfy-output.safetensors", "--comfy")
            self.assertEqual(converted.returncode, 0, converted.stderr)
            expected = load_file(str(root / "diffusers-output.safetensors"))["output"]
            actual = load_file(str(root / "comfy-output.safetensors"))["output"]
            self.assertTrue(torch.equal(actual, expected))
            cached = run(root / "comfy.safetensors", root / "cached-output.safetensors", "--comfy", "--cache-cross")
            self.assertEqual(cached.returncode, 0, cached.stderr)
            self.assertTrue(torch.equal(load_file(str(root / "cached-output.safetensors"))["output"], expected))
            fused = run(root / "comfy.safetensors", root / "fused-output.safetensors", "--comfy", "--cache-cross", "--fused-rope")
            self.assertEqual(fused.returncode, 0, fused.stderr)
            self.assertTrue(torch.equal(load_file(str(root / "fused-output.safetensors"))["output"], expected))

            comfy["unknown.weight"] = torch.ones(1)
            save_file(comfy, str(root / "unknown.safetensors"))
            rejected = run(root / "unknown.safetensors", root / "unknown-output.safetensors", "--comfy")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("unsupported Wan2.2 Comfy weight", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
