"""No GPU work: reject malformed replay inputs before native creation."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "experimental/transformer/build/ltx-stage2-replay"


@unittest.skipUnless(EXE.is_file(), "optional native replay not built")
class ReplayCLITests(unittest.TestCase):
    def test_invalid_decoder_mode_before_gpu(self):
        env = {**os.environ, "TURBOCIDER_LTX_REPLAY_VIDEO_MODE": "typo"}
        result = subprocess.run([str(EXE)] + ["missing"] * 5 + ["2"],
                                env=env, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"invalid replay video mode", result.stderr)

    def test_wrong_argument_count(self):
        result = subprocess.run([str(EXE)], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"usage:", result.stderr)

    def test_invalid_run_count_before_gpu(self):
        for value in ("0", "-1", "21", "1junk", "nan"):
            with self.subTest(value=value):
                result = subprocess.run([str(EXE)] + ["missing"] * 5 + [value],
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 2)

    def test_missing_decoder_preflight_before_tensor_or_gpu_load(self):
        with tempfile.TemporaryDirectory(prefix="ltx-replay-cli-") as path:
            directory = Path(path)
            placeholder = directory / "placeholder"
            placeholder.write_bytes(b"not a checkpoint")
            missing = directory / "missing-decoder"
            args = [str(placeholder)] * 5 + ["2", str(placeholder),
                    str(placeholder), str(missing), str(placeholder)]
            result = subprocess.run([str(EXE), *args], capture_output=True,
                                    timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn(b"missing regular input file:", result.stderr)
            self.assertIn(str(missing).encode(), result.stderr)
            self.assertNotIn(b'"kind":"load"', result.stdout)


if __name__ == "__main__":
    unittest.main()
