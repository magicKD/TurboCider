"""Small, weight-independent T5/UMT5 parity checks against Transformers.

Set TURBOCIDER_UMT5_PROBE to a native probe built from the current tree and run
with a Python environment containing torch, transformers, and safetensors.
The fixtures intentionally have incompatible relative-bias layouts: standard
T5 shares block 0's table, whereas UMT5 owns one table in every block.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_UMT5_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file(), "native UMT5 probe required")
class EncoderParityTests(unittest.TestCase):
    def check_encoder(self, t5, comfy_sdpa=False, comfy_fp32=False):
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import T5Config, T5EncoderModel, UMT5Config, UMT5EncoderModel

        torch.manual_seed(31)
        config_class, model_class = (T5Config, T5EncoderModel) if t5 else (UMT5Config, UMT5EncoderModel)
        config = config_class(vocab_size=128, d_model=32, d_kv=8, d_ff=64,
                              num_layers=2, num_heads=4, dropout_rate=0.0,
                              feed_forward_proj="gated-gelu")
        model = model_class(config).to(dtype=torch.bfloat16)
        if comfy_fp32:
            model = model.to(dtype=torch.float32)
        model = model.eval()
        ids = torch.tensor([[4, 14, 7, 52, 1, 0, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
        with torch.inference_mode():
            reference = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        state = model.state_dict()
        shared_key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        second_key = "encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight"
        self.assertIn(shared_key, state)
        self.assertEqual(second_key in state, not t5)

        with tempfile.TemporaryDirectory(prefix="tc-ltx098-t5-") as temporary:
            root = Path(temporary)
            weights = root / "weights"
            weights.mkdir()
            # Safetensors rejects tied storage; clone each tensor independently.
            save_file({key: value.detach().contiguous().clone() for key, value in state.items()},
                      str(weights / "model.safetensors"))
            save_file({"ids": ids[0].to(torch.int32), "valid": torch.tensor(5, dtype=torch.int32)},
                      str(root / "input.safetensors"))
            command = [str(PROBE), str(weights), str(root / "input.safetensors"),
                       str(root / "output.safetensors"), "--t5-fixture" if t5 else "--umt5-fixture"]
            if comfy_sdpa:
                command.append("--comfy-sdpa")
            if comfy_fp32:
                command.append("--comfy-fp32")
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            native = load_file(str(root / "output.safetensors"))

        self.assertEqual(native["output"].dtype, torch.float32 if comfy_fp32 else torch.bfloat16)
        self.assertEqual(native["output"].shape, reference.last_hidden_state.shape)
        # Comfy/Transformers retain padded T5 output and pass its mask to the
        # denoiser. Wan's existing UMT5 boundary intentionally zeroes padding.
        if t5:
            self.assertGreater(native["output"][:, 5:].float().abs().max().item(), 0)
        else:
            self.assertEqual(native["output"][:, 5:].float().abs().max().item(), 0)
        for actual, expected in ((native["block_0"], reference.hidden_states[1]),
                                 (native["output"], reference.last_hidden_state)):
            a, b = actual[:, :5].float(), expected[:, :5].float()
            relative_l2 = (a - b).norm() / b.norm()
            cosine = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
            self.assertLess(relative_l2.item(), 0.04)
            self.assertGreater(cosine.item(), 0.999)

    def test_standard_t5_shared_relative_bias(self):
        self.check_encoder(t5=True)

    def test_standard_t5_comfy_sdpa(self):
        self.check_encoder(t5=True, comfy_sdpa=True)

    def test_standard_t5_comfy_fp32(self):
        self.check_encoder(t5=True, comfy_sdpa=True, comfy_fp32=True)

    def test_wan_umt5_per_layer_bias_and_zero_padding(self):
        self.check_encoder(t5=False)


if __name__ == "__main__":
    unittest.main()
