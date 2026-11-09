"""Seeded CPU noise must preserve generator state between DMD draws."""
import os
from pathlib import Path
import subprocess
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_WAN_NOISE_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file(), "native Wan noise probe required")
class Wan22SeededNoiseTests(unittest.TestCase):
    def test_three_draws_match_torch_cpu_generator(self):
        import numpy as np
        import torch

        count, seed = 48 * 3 * 8 * 8, 42
        data = subprocess.check_output([str(PROBE), str(count), str(seed), "3"])
        actual = torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy().reshape(3, count))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        expected = torch.stack([torch.randn(count, generator=generator) for _ in range(3)])
        self.assertTrue(torch.allclose(actual, expected, rtol=2e-6, atol=2e-6),
                        f"maximum error {(actual - expected).abs().max().item()}")


if __name__ == "__main__":
    unittest.main()
