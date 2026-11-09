"""Tiny full Wan2.2 TI2V transformer against Diffusers' independent model."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_WAN22_TRANSFORMER_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file(), "native Wan2.2 transformer probe required")
class Wan22TransformerParityTests(unittest.TestCase):
    def test_clean_first_frame_and_noisy_video_timestep(self):
        import torch
        import torch.nn.functional as F
        from diffusers import WanTransformer3DModel
        from safetensors.torch import load_file, save_file

        torch.manual_seed(73)
        channels, hidden, heads, intermediate, layers, text_dim, frequency = 4, 32, 4, 64, 2, 16, 8
        model = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=heads,
                    attention_head_dim=hidden // heads, in_channels=channels, out_channels=channels,
                    text_dim=text_dim, freq_dim=frequency, ffn_dim=intermediate, num_layers=layers,
                    rope_max_seq_len=32).to(dtype=torch.bfloat16).eval()
        latents = torch.randn((1, channels, 3, 4, 4), dtype=torch.bfloat16)
        text = torch.randn((1, 5, text_dim), dtype=torch.bfloat16)
        timestep = 757.0
        patch_tokens_per_frame = 4
        times = torch.tensor([[0.0] * patch_tokens_per_frame + [timestep] * (2 * patch_tokens_per_frame)])
        with torch.inference_mode():
            reference = model(latents, times, text).sample.float()
        with tempfile.TemporaryDirectory(prefix="tc-wan22-transformer-") as temporary:
            root = Path(temporary)
            state = {key: value.contiguous().clone() for key, value in model.state_dict().items()}
            save_file(state, str(root / "weights.safetensors"))
            save_file({"latents": latents, "text": text}, str(root / "input.safetensors"))
            result = subprocess.run([str(PROBE), str(root / "weights.safetensors"), str(root / "input.safetensors"),
                                     str(root / "output.safetensors"), str(channels), str(hidden), str(heads),
                                     str(intermediate), str(layers), str(text_dim), str(frequency), str(timestep)],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = load_file(str(root / "output.safetensors"))["output"].float()
        self.assertEqual(actual.shape, reference.shape)
        relative = (actual - reference).norm() / reference.norm().clamp_min(1e-30)
        cosine = F.cosine_similarity(actual.flatten(), reference.flatten(), dim=0)
        self.assertLess(relative.item(), .08)
        self.assertGreater(cosine.item(), .995)


if __name__ == "__main__":
    unittest.main()
