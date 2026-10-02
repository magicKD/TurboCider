"""Lossless fused affine CPU packing versus the preserved scalar implementation."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class FusedAffineTests(unittest.TestCase):
    def test_exhaustive_legacy_parity_and_spans(self):
        with tempfile.TemporaryDirectory(prefix="tc-affine-fused-") as raw:
            binary=Path(raw)/"probe"
            flags=["-std=c++20","-O2","-ffp-contract=off","-Wall","-Wextra","-Werror"]
            if os.environ.get("TC_STREAMING_SANITIZER"):
                flags += ["-fsanitize="+os.environ["TC_STREAMING_SANITIZER"],"-fno-omit-frame-pointer"]
            subprocess.run(["xcrun","clang++",*flags,"-I",str(ROOT/"native/core"),
                str(ROOT/"tests/native/gguf_affine_fused_test.cpp"),str(ROOT/"native/core/gguf_affine.cpp"),
                str(ROOT/"native/core/gguf_decode.cpp"),"-o",str(binary)],check=True)
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS fused affine",result.stdout);print(result.stdout,end="")


if __name__=="__main__":unittest.main()
