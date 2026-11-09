"""Tiny Wan2.2 BF16 block with first-frame versus motion-frame timesteps."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_WAN22_BLOCK_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file(), "native Wan2.2 block probe required")
class Wan22BlockParityTests(unittest.TestCase):
    def test_two_timestep_groups_against_pytorch(self):
        import torch
        import torch.nn.functional as F
        from safetensors.torch import load_file, save_file

        torch.manual_seed(292)
        hidden_size, heads, intermediate, rows, text_rows, first = 32, 4, 64, 7, 5, 2
        head_dim = hidden_size // heads
        dtype = torch.bfloat16
        def rand(shape, scale=.05):
            return (torch.randn(shape, dtype=torch.float32) * scale).to(dtype)
        weights = {"scale_shift_table": rand((1, 6, hidden_size), .08),
                   "norm2.weight": (1 + rand((hidden_size,), .01)).to(dtype),
                   "norm2.bias": rand((hidden_size,), .01)}
        for attention in ("attn1", "attn2"):
            for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                weights[f"{attention}.{projection}.weight"] = rand((hidden_size, hidden_size))
                weights[f"{attention}.{projection}.bias"] = rand((hidden_size,))
            for part in ("norm_q", "norm_k"):
                weights[f"{attention}.{part}.weight"] = (1 + rand((hidden_size,), .01)).to(dtype)
        weights["ffn.net.0.proj.weight"] = rand((intermediate, hidden_size))
        weights["ffn.net.0.proj.bias"] = rand((intermediate,))
        weights["ffn.net.2.weight"] = rand((hidden_size, intermediate))
        weights["ffn.net.2.bias"] = rand((hidden_size,))
        hidden, context = rand((1, rows, hidden_size), .5), rand((1, text_rows, hidden_size), .5)
        modulation = rand((1, 2, 6, hidden_size), .08)
        angles = torch.randn((rows, head_dim // 2), dtype=torch.float32)
        cosine, sine = angles.cos().repeat_interleave(2, -1), angles.sin().repeat_interleave(2, -1)

        def project(value, prefix):
            return F.linear(value, weights[prefix + ".weight"], weights[prefix + ".bias"])

        def apply_rope(value):
            value = value.float().reshape(1, rows, heads, head_dim)
            pairs = value.reshape(1, rows, heads, head_dim // 2, 2)
            even, odd = pairs.unbind(-1)
            rotated = torch.stack((-odd, even), -1).reshape_as(value)
            return (value * cosine[None, :, None, :] + rotated * sine[None, :, None, :]).to(dtype).reshape(1, rows, hidden_size)

        def attention(query, memory, prefix, use_rope):
            q = F.rms_norm(project(query, prefix + ".to_q"), (hidden_size,), weights[prefix + ".norm_q.weight"], 1e-6)
            k = F.rms_norm(project(memory, prefix + ".to_k"), (hidden_size,), weights[prefix + ".norm_k.weight"], 1e-6)
            v = project(memory, prefix + ".to_v")
            if use_rope:
                q, k = apply_rope(q), apply_rope(k)
            def split(value):
                return value.reshape(1, -1, heads, head_dim).transpose(1, 2)
            result = F.scaled_dot_product_attention(split(q), split(k), split(v))
            return project(result.transpose(1, 2).reshape(1, query.shape[1], hidden_size), prefix + ".to_out.0")

        packed = modulation + weights["scale_shift_table"].unsqueeze(1)
        packed = torch.cat((packed[:, :1].expand(1, first, 6, hidden_size),
                            packed[:, 1:].expand(1, rows - first, 6, hidden_size)), dim=1)
        shifts = packed.unbind(2)
        def modulate(value, shift, scale):
            normalized = F.layer_norm(value, (hidden_size,), eps=1e-6)
            return torch.addcmul(shift, normalized, 1 + scale)
        x = torch.addcmul(hidden, attention(modulate(hidden, shifts[0], shifts[1]),
                                             modulate(hidden, shifts[0], shifts[1]), "attn1", True), shifts[2])
        cross = F.layer_norm(x, (hidden_size,), weights["norm2.weight"], weights["norm2.bias"], eps=1e-6)
        x = x + attention(cross, context, "attn2", False)
        ff = project(F.gelu(project(modulate(x, shifts[3], shifts[4]), "ffn.net.0.proj"), approximate="tanh"), "ffn.net.2")
        expected = torch.addcmul(x, ff, shifts[5]).float()

        with tempfile.TemporaryDirectory(prefix="tc-wan22-block-") as temporary:
            root = Path(temporary)
            save_file({"blocks.0." + key: value.contiguous() for key, value in weights.items()}, str(root / "weights.safetensors"))
            save_file({"hidden": hidden, "context": context, "modulation": modulation,
                       "cosine": cosine, "sine": sine}, str(root / "input.safetensors"))
            result = subprocess.run([str(PROBE), str(root / "weights.safetensors"), str(root / "input.safetensors"),
                                     str(root / "output.safetensors"), str(hidden_size), str(heads), str(intermediate), str(first)],
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = load_file(str(root / "output.safetensors"))["output"].float()
        error = (actual - expected).norm() / expected.norm()
        cosine_similarity = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        self.assertLess(error.item(), .025)
        self.assertGreater(cosine_similarity.item(), .999)


if __name__ == "__main__":
    unittest.main()
