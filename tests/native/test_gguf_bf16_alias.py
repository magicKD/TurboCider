"""Pure CPU exhaustive same-width legacy importer RNE conversion."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
class Bf16AliasTests(unittest.TestCase):
    def test_exhaustive_rne_and_failure(self):
        with tempfile.TemporaryDirectory(prefix="tc-bf16-alias-") as raw:
            binary=Path(raw)/"probe"
            subprocess.run(["xcrun","clang++","-std=c++20","-O2","-ffp-contract=off","-Wall","-Wextra","-Werror",
                "-I",str(ROOT/"native/core"),str(ROOT/"tests/native/gguf_bf16_alias_test.cpp"),str(ROOT/"native/core/gguf_decode.cpp"),
                "-o",str(binary)],check=True)
            subprocess.run([str(binary)],check=True)
if __name__=="__main__":unittest.main()
