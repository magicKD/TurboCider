import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "z_image_w8a8_sq_search", ROOT / "tools/validation/z_image_w8a8_sq_search.py")
SEARCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEARCH)


class ZImageSQSearchTests(unittest.TestCase):
    def test_crossval_folds_do_not_mix_capture_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for pid in (111, 222):
                for step in range(2):
                    np.save(directory / f"sample-{pid}-{step}-20.npy",
                            np.full((8, 16), pid + step, dtype=np.float16))
            folds, pids, digest = SEARCH.capture_folds(directory, 8, 16, np)
            self.assertEqual(pids, ["111", "222"])
            self.assertEqual(list(map(len, folds)), [2, 2])
            self.assertEqual(float(folds[0][0][0, 0]), 111.)
            self.assertEqual(float(folds[1][0][0, 0]), 222.)
            self.assertEqual(len(digest), 64)

    def test_crossval_requires_two_pid_tagged_captures(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory / "sample-111-0-20.npy", np.ones((8, 16), np.float16))
            with self.assertRaisesRegex(ValueError, "two independent"):
                SEARCH.capture_folds(directory, 8, 16, np)
            np.save(directory / "no-process-id.npy", np.ones((8, 16), np.float16))
            with self.assertRaisesRegex(ValueError, "PID-tagged"):
                SEARCH.capture_folds(directory, 8, 16, np)


if __name__ == "__main__":
    unittest.main()
