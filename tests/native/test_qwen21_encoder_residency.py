from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class EncoderResidencyHostTests(unittest.TestCase):
    def test_generation_admission_and_channel_override(self):
        with tempfile.TemporaryDirectory(prefix="tc-encoder-residency-") as directory:
            root=Path(directory);fixture=root/"source";fixture.write_bytes(b"payload")
            probe=root/"probe"
            subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                str(ROOT/"tests/native/qwen21_encoder_residency_test.cpp"),str(ROOT/"native/core/common.cpp"),
                "-o",str(probe)],check=True,cwd=ROOT,capture_output=True,text=True,timeout=60)
            result=subprocess.run([str(probe),str(fixture)],check=True,capture_output=True,text=True,timeout=30)
            self.assertIn("PASS encoder source generation",result.stdout);print(result.stdout)


if __name__=="__main__":unittest.main()
