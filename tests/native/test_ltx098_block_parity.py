"""Independent tiny PyTorch reference for the LTX 0.9.8 MLX block.

Set TURBOCIDER_LTX098_BLOCK_PROBE to a binary built from the current sources.
No model checkpoint is needed. This is a component test, not video validation.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_LTX098_BLOCK_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file(), "native LTX 0.9.8 block probe required")
class BlockParityTests(unittest.TestCase):
    def test_interleaved_rope_text_mask_and_per_token_modulation(self):
        import torch
        import torch.nn.functional as F
        from safetensors.torch import load_file, save_file

        torch.manual_seed(55)
        dtype, hidden_size, heads, rows, text_rows = torch.bfloat16, 32, 4, 7, 5
        head_dim = hidden_size // heads
        prefix = "model.diffusion_model.transformer_blocks.0."
        weights = {}

        def rand(shape, scale=0.06):
            return (torch.randn(shape, dtype=torch.float32) * scale).to(dtype)

        def linear(name, output, input_size):
            weights[name + ".weight"] = rand((output, input_size))
            weights[name + ".bias"] = rand((output,))

        for layer in ("attn1", "attn2"):
            for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                linear(layer + "." + projection, hidden_size, hidden_size)
            for norm in ("q_norm", "k_norm"):
                weights[layer + "." + norm + ".weight"] = (1 + rand((hidden_size,), 0.02)).to(dtype)
        linear("ff.net.0.proj", hidden_size * 4, hidden_size)
        linear("ff.net.2", hidden_size, hidden_size * 4)
        weights["scale_shift_table"] = rand((6, hidden_size), .1)
        hidden, context = rand((1, rows, hidden_size), .5), rand((1, text_rows, hidden_size), .5)
        angles = torch.randn((1, heads, rows, head_dim // 2), dtype=torch.float32)
        cosine, sine = angles.cos().to(dtype), angles.sin().to(dtype)
        text_mask = torch.tensor([0., 0., 0., -10000., -10000.], dtype=dtype).reshape(1, 1, 1, text_rows)

        def project(x, name):
            return F.linear(x, weights[name + ".weight"], weights[name + ".bias"])

        def attention(x, memory, name, rope, mask):
            q = F.rms_norm(project(x, name + ".to_q"), (hidden_size,), weights[name + ".q_norm.weight"], 1e-5)
            k = F.rms_norm(project(memory, name + ".to_k"), (hidden_size,), weights[name + ".k_norm.weight"], 1e-5)
            v = project(memory, name + ".to_v")
            q = q.reshape(1, q.shape[1], heads, head_dim).transpose(1, 2)
            k = k.reshape(1, k.shape[1], heads, head_dim).transpose(1, 2)
            v = v.reshape(1, v.shape[1], heads, head_dim).transpose(1, 2)
            if rope:
                def rotate(value):
                    pairs = value.reshape(1, heads, rows, head_dim // 2, 2)
                    a, b = pairs.unbind(dim=-1)
                    return torch.stack((torch.addcmul(a * cosine, -sine, b),
                                        torch.addcmul(a * sine, cosine, b)), dim=-1).reshape_as(value)
                q, k = rotate(q), rotate(k)
            result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            result = result.transpose(1, 2).reshape(1, x.shape[1], hidden_size)
            return project(result, name + ".to_out.0")

        def norm(value):
            return F.rms_norm(value.float(), (hidden_size,), eps=1e-6).to(dtype)

        def reference(modulation):
            values = (modulation.reshape(1, modulation.shape[1], 6, hidden_size) +
                      weights["scale_shift_table"].reshape(1, 1, 6, hidden_size)).unbind(2)
            x = hidden + attention(norm(hidden) * (1 + values[1]) + values[0],
                                   norm(hidden) * (1 + values[1]) + values[0], "attn1", True, None) * values[2]
            x = x + attention(x, context, "attn2", False, text_mask)
            normalized = norm(x)
            feed = torch.addcmul(normalized, normalized, values[4]) + values[3]
            feed = project(F.gelu(project(feed, "ff.net.0.proj"), approximate="tanh"), "ff.net.2")
            return torch.addcmul(x, feed, values[5])

        with tempfile.TemporaryDirectory(prefix="tc-ltx098-block-") as temporary:
            root = Path(temporary)
            save_file({prefix + key: value.contiguous() for key, value in weights.items()}, str(root / "weights.safetensors"))
            for mod_rows in (1, rows):
                with self.subTest(modulation_rows=mod_rows):
                    modulation = rand((1, mod_rows, 6 * hidden_size), .1)
                    inputs = {"hidden": hidden, "context": context, "modulation": modulation,
                              "cosine": cosine, "sine": sine, "text_mask": text_mask}
                    save_file({key: value.contiguous() for key, value in inputs.items()}, str(root / "input.safetensors"))
                    command = [str(PROBE), str(root / "weights.safetensors"), str(root / "input.safetensors"),
                               str(root / "output.safetensors"), "0", str(heads)]
                    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    actual = load_file(str(root / "output.safetensors"))["output"]
                    self.assertEqual(actual.dtype, dtype, "block must preserve BF16 across all 28 layers")
                    actual = actual.float()
                    expected = reference(modulation).float()
                    relative_l2 = (actual - expected).norm() / expected.norm()
                    cosine_similarity = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
                    self.assertLess(relative_l2.item(), .04)
                    self.assertGreater(cosine_similarity.item(), .999)


if __name__ == "__main__":
    unittest.main()
