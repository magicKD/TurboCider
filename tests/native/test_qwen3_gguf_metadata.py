"""Real verified GGUF metadata/cache tests; no synthetic hardware qualification."""
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit actual Metal/local fixture opt-in required")
class Qwen3MetadataTests(unittest.TestCase):
    def test_verified_metadata_reuse_without_weight_retention(self):
        gguf=ROOT/"models/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf"
        config=ROOT/"models/Tongyi-MAI-Z-Image-Turbo/text_encoder/config.json"
        tokenizer=ROOT/"models/z-image-runtime-gguf-q8/tokenizer/tokenizer.json"
        self.assertTrue(all(path.is_file() for path in (gguf,config,tokenizer)),"required actual fixtures missing")
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        native=ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/quantized-execution")
        with tempfile.TemporaryDirectory(prefix="tc-qwen3-metadata-") as raw:
            folder=Path(raw);binary=folder/"probe"
            subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min=26.2","-I",str(ROOT/"native"),"-I",str(ROOT/"native/core"),
                "-isystem",str(mlx/"include"),str(ROOT/"tests/native/qwen3_gguf_metadata_test.cpp"),
                "-L",str(native),"-lturbocider","-L",str(mlx/"lib"),"-lmlx",
                "-Wl,-rpath,"+str(native),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(binary)],check=True)
            result=subprocess.run([str(binary),str(gguf),str(config),str(tokenizer),str(folder)],
                capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS Qwen3 metadata",result.stdout);print(result.stdout,end="")


if __name__=="__main__":unittest.main()
