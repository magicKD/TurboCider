"""Paired RoPE/head packing must preserve the existing BF16 results."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_WAN22_ROPE_HEADS_PROBE", "/nonexistent")).resolve()
POLICY = Path(os.environ.get("TURBOCIDER_WAN22_ROPE_POLICY_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(POLICY.is_file(), "native Wan2.2 RoPE policy probe required")
class Wan22RoPEPolicyTests(unittest.TestCase):
    def test_only_qualified_device_and_large_97_frame_canvases(self):
        def policy(gpu="Apple M4 Max", memory=64 << 30, major=26, minor=6, patch=2,
                   mlx="0.32.0", width=512, height=512, frames=97):
            args = [gpu, memory, major, minor, patch, mlx, width, height, frames]
            return int(subprocess.check_output([str(POLICY), *map(str, args)], text=True))
        for size in ((512, 512), (512, 768), (768, 512), (768, 768), (640, 640)):
            self.assertEqual(policy(width=size[0], height=size[1]), 1)
        for options in ({"gpu": "Apple M3 Max"}, {"memory": 32 << 30}, {"major": 25}, {"minor": 7},
                        {"patch": 3}, {"mlx": "0.32.1"}, {"frames": 9}, {"width": 480},
                        {"height": 800}, {"width": 513}):
            with self.subTest(options=options):
                self.assertEqual(policy(**options), 0)


@unittest.skipUnless(PROBE.is_file(), "native Wan2.2 RoPE probe required")
class Wan22FusedRoPETests(unittest.TestCase):
    def test_independent_frequency_channels_and_tail_rows(self):
        import torch
        from safetensors.torch import load_file, save_file

        torch.manual_seed(22)
        for rows, heads, dim in ((7, 4, 8), (65, 4, 32), (257, 24, 128)):
            with self.subTest(rows=rows, heads=heads, dim=dim), tempfile.TemporaryDirectory(prefix="tc-wan22-rope-") as temporary:
                root = Path(temporary)
                shape = (1, rows, heads * dim)
                values = {name: torch.randn(shape).to(torch.bfloat16) for name in ("q", "k", "v")}
                # Do not assume the two channels in a pair have equal frequencies.
                values["cosine"], values["sine"] = torch.randn(rows, dim), torch.randn(rows, dim)
                save_file(values, str(root / "input.safetensors"))
                for mode in ("baseline", "fused"):
                    subprocess.run([str(PROBE), str(root / "input.safetensors"), str(root / f"{mode}.safetensors"),
                                    str(heads), mode], check=True, capture_output=True, text=True, timeout=120)
                baseline, fused = (load_file(str(root / f"{mode}.safetensors")) for mode in ("baseline", "fused"))
                for name in ("q", "k", "v"):
                    self.assertTrue(torch.equal(baseline[name].view(torch.uint8), fused[name].view(torch.uint8)),
                                    f"{name} bits differ")
                values["cosine"] = torch.randn(rows, dim + 2)
                save_file(values, str(root / "invalid.safetensors"))
                rejected = subprocess.run([str(PROBE), str(root / "invalid.safetensors"), str(root / "invalid-output.safetensors"),
                                           str(heads), "fused"], capture_output=True, text=True, timeout=120)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("frequency geometry", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
