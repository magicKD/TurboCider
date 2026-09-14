import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class VDNMLXSolveTests(unittest.TestCase):
    def test_gpu_solve_matches_cpu_reference(self):
        probe = ROOT / "build/native/h3-mlx-vdn-solve-probe"
        if not probe.is_file():
            self.skipTest("native MLX solve probe has not been built")
        completed = subprocess.run(
            [str(probe)], cwd=ROOT, check=False, text=True,
            capture_output=True,
        )
        if completed.returncode != 0 and "No Metal device available" in completed.stderr:
            self.skipTest("Metal is unavailable in this test environment")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["dimensions"], 6)
        self.assertEqual(payload["batch"], 3)
        self.assertLess(payload["max_abs"], 5e-4)
        self.assertLess(payload["relative_rmse"], 5e-4)
        self.assertLess(payload["max_residual"], 1e-3)


if __name__ == "__main__":
    unittest.main()
