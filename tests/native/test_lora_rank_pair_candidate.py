import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class LoRAPairedRankCandidateTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit Metal opt-in required")
    def test_two_original_sources_contiguous_ranks_and_physical_bounds(self):
        with tempfile.TemporaryDirectory(prefix="tc-lora-rank-pair-") as temporary:
            subprocess.run(
                ["bash", "tools/native/build_qwen_lora_rank_pair_probe.sh"],
                cwd=ROOT, check=True,
                env={**os.environ, "TURBOCIDER_NATIVE_OUT": temporary},
                capture_output=True, text=True, timeout=180,
            )
            result = subprocess.run(
                [str(Path(temporary) / "qwen-lora-rank-pair-probe")],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(
                "PASS 216 paired rank Metal numeric cases and 192 malformed contracts",
                result.stdout,
            )


if __name__ == "__main__":
    unittest.main()
