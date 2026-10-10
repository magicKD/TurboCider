from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen_ffn_phase_screen import SHARE_MODES, phase_policy, selected_channels
from qwen_encoder_residency_screen import validate_rows


class PrefillShareScreenTests(unittest.TestCase):
    def test_explicit_shares_keep_prefill_only_and_existing_baseline(self):
        for mode, channels in (
            ("gpu", 5120), ("prefill_a4096", 4096), ("prefill", 5120),
            ("prefill_a6144", 6144), ("prefill_a7168", 7168),
        ):
            with self.subTest(mode=mode):
                self.assertIn(mode, SHARE_MODES)
                self.assertEqual(selected_channels(mode, 5120), channels)
                self.assertEqual(phase_policy(mode), "gpu" if mode == "gpu" else "prefill")
        self.assertEqual(selected_channels("prefill", 4608), 4608)

    def test_actual_arm_channel_receipt_cannot_be_relabelled(self):
        for channels in (4096, 5120, 6144, 7168):
            with self.subTest(channels=channels):
                row = dict(
                    prompt_cache_hit=False,
                    runtime_backend="mlx_cpp_metal+private_ane_runtime_weight_experimental",
                    steps=6, actual_denoise_steps=6, encoder_execution="gpu",
                    encoder_hybrid={}, encoder_runtime_reuse=None,
                    timings_seconds=dict(request_wall=12, text_encode=1, denoise=10),
                    encoder_weight_residency=dict(
                        enabled=True, weights_retained=True, weights_reused=False,
                        loads_session_total=1, retained_bytes=17 << 30, decline_reason="",
                    ),
                    hybrid=dict(
                        runtime_failed=False, runtime_calls_session_total=32,
                        runtime_weight=dict(
                            executor_backend="private_ane", data_path="w8a8_hadamard",
                            fallback_blocks_session_total=0, ane_channels=channels,
                        ),
                    ),
                )
                validate_rows([row], "dit_weights", 1, True, None, channels)
                with self.assertRaises(ValueError):
                    validate_rows([row], "dit_weights", 1, True, None, channels + 512)

    def test_scope_errors_precede_fixtures_and_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "never-created"
            base = [
                sys.executable, "-S", str(ROOT / "tools/validation/qwen_ffn_phase_screen.py"),
                "--cli", "unused", "--model", "unused", "--dit-manifest", "unused",
                "--reference", "unused", "--prompt", "one", "--prompt", "two",
                "--prompt", "three", "--output", str(output),
            ]
            valid = ["--prefill-share-screen", "--lora", "unused", "--joint-ab"]
            invalid_flags = [
                ["--modes", "gpu,prefill_a4096"], ["--prefill-share-screen"],
                ["--prefill-share-screen", "--lora", "unused"],
                [*valid, "--channels", "4096"], [*valid, "--generation"],
                [*valid, "--prefill-layer-screen"],
                [*valid, "--prefill-code-cache-bytes", "512"],
                [*valid, "--frozen-manifest", "unused"], [*valid, "--fused-b"],
                [*valid, "--modes", "gpu,decode"],
                [*valid, "--modes", "gpu,prefill", "--order", "gpu,prefill_a4096"],
            ]
            for flags in invalid_flags:
                with self.subTest(flags=flags):
                    result = subprocess.run(
                        [*base, *flags], capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
