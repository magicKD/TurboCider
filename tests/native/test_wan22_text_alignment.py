"""Opt-in hardware policy and real-weight text alignment checks."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def configured(name):
    value = os.environ.get(name)
    return Path(value).resolve() if value else None


POLICY = configured("TURBOCIDER_WAN22_TEXT_POLICY_PROBE")
PROBE = configured("TURBOCIDER_WAN22_TEXT_PROBE")
CHECKPOINT = configured("TURBOCIDER_UMT5_CHECKPOINT")
TOKENIZER = configured("TURBOCIDER_UMT5_TOKENIZER_ROOT")
CAPTURE = configured("TURBOCIDER_WAN22_TEXT_CAPTURE")


@unittest.skipUnless(POLICY and POLICY.is_file(), "native text policy probe required")
class Wan22TextPolicyTests(unittest.TestCase):
    def test_only_qualified_device_and_short_padded_prompts(self):
        def policy(gpu="Apple M4 Max", memory=64 << 30, major=26, minor=6, patch=2,
                   mlx="0.32.0", valid=50, sequence=512):
            args = [gpu, memory, major, minor, patch, mlx, valid, sequence]
            return int(subprocess.check_output([str(POLICY), *map(str, args)], text=True))
        self.assertEqual(policy(), 256)
        self.assertEqual(policy(valid=256), 256)
        for options in ({"gpu": "Apple M3 Max"}, {"memory": 32 << 30}, {"major": 25}, {"minor": 7}, {"patch": 3},
                        {"mlx": "0.32.1"}, {"valid": 0}, {"valid": 257}, {"sequence": 513},
                        {"valid": 526, "sequence": 526}):
            with self.subTest(options=options):
                self.assertEqual(policy(**options), 0)


@unittest.skipUnless(PROBE and PROBE.is_file() and CHECKPOINT and CHECKPOINT.is_file() and
                     TOKENIZER and TOKENIZER.is_dir() and CAPTURE and CAPTURE.is_file(),
                     "native text probe, real UMT5 weights and short Comfy fixture required")
class Wan22TextAlignmentTests(unittest.TestCase):
    def test_real_short_prompt_matches_full_padded_encoding(self):
        import torch
        from safetensors.torch import load_file

        fixture = load_file(str(CAPTURE))
        self.assertLessEqual(fixture["valid"].item(), 256)
        with tempfile.TemporaryDirectory(prefix="tc-wan22-text-alignment-") as temporary:
            baseline, optimized = Path(temporary) / "baseline.safetensors", Path(temporary) / "optimized.safetensors"
            args = [str(PROBE), str(CHECKPOINT), str(TOKENIZER), str(CAPTURE)]
            subprocess.run([*args, str(baseline)], check=True, capture_output=True, text=True, timeout=180)
            subprocess.run([*args, str(optimized), "--align256"], check=True, capture_output=True, text=True, timeout=180)
            a = load_file(str(baseline))["output"]
            b = load_file(str(optimized))["output"]
        self.assertEqual(a.shape, (1, 512, 4096))
        self.assertTrue(torch.equal(a, b), "aligned text differed from full 512-token encoding")


if __name__ == "__main__":
    unittest.main()
