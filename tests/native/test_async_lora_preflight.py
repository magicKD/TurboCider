"""CPU/file-I/O only: no MLX, Core ML, App, checkpoint parse or model load."""
from pathlib import Path
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == "darwin", "CommonCrypto/macOS file generation test")
class AsyncLoraPreflightTests(unittest.TestCase):
    def test_provenance_cancellation_and_ownership(self):
        with tempfile.TemporaryDirectory(prefix="tc-async-lora-", dir="/private/tmp") as directory:
            folder = Path(directory)
            binary = folder / "test"
            command = [os.environ.get("CXX", "clang++"), "-std=c++20", "-O2", "-pthread",
                       "-Wall", "-Wextra", "-Werror", "-Wno-deprecated-declarations",
                       str(ROOT / "tests/native/async_lora_preflight_test.cpp"),
                       str(ROOT / "native/core/common.cpp"), "-o", str(binary)]
            build = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            # Only ~1 MiB: exercise a second read without a checkpoint-sized fixture.
            payload = bytes(range(256)) * 4096 + b"second chunk ends"
            (folder / "chunks.bin").write_bytes(payload)
            run = subprocess.run([str(binary), str(folder), hashlib.sha256(payload).hexdigest()],
                                 capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("PASS async LoRA preflight: 14 CPU cases", run.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
