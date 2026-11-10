"""Compare saved baseline/fused Wan2.2 latent trajectories from real weights."""
import os
from pathlib import Path
import unittest


def configured(name):
    value = os.environ.get(name)
    return Path(value).resolve() if value else None


BASELINE = configured("TURBOCIDER_WAN22_BASELINE_LATENT")
FUSED = configured("TURBOCIDER_WAN22_FUSED_LATENT")


@unittest.skipUnless(BASELINE and BASELINE.is_file() and FUSED and FUSED.is_file(),
                     "matched baseline and fused real-weight latent files required")
class Wan22FusedTrajectoryTests(unittest.TestCase):
    def test_final_latents_match_exactly(self):
        import torch
        from safetensors.torch import load_file

        baseline, fused = (load_file(str(path))["latents"] for path in (BASELINE, FUSED))
        self.assertEqual(baseline.shape, fused.shape)
        self.assertEqual(baseline.dtype, fused.dtype)
        self.assertTrue(torch.isfinite(baseline).all() and torch.isfinite(fused).all())
        self.assertTrue(torch.equal(baseline.view(torch.uint8), fused.view(torch.uint8)),
                        "fused RoPE changed final video latent bits")


if __name__ == "__main__":
    unittest.main()
