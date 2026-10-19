"""Actual MLX student nodes, six/five/six restoration and invalid schedules."""
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform=="darwin","requires Apple MLX GPU")
class QwenSchedulerTests(unittest.TestCase):
    def test_authored_nodes_and_default_restoration(self):
        native=Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR",ROOT/"build/native")).resolve()
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        with tempfile.TemporaryDirectory(prefix="tc-qwen-scheduler-") as directory:
            output=Path(directory); binary=output/"probe"
            subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra",
                "-mmacosx-version-min="+platform.mac_ver()[0],
                str(ROOT/"tests/native/qwen21_media_schedule_test.cpp"),
                "-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),
                "-L"+str(native),"-lturbocider","-L"+str(mlx/"lib"),"-lmlx",
                "-Wl,-rpath,"+str(native),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(binary)],
                check=True,cwd=ROOT)
            subprocess.run([str(binary),str(output)],check=True,cwd=ROOT,timeout=60)
            self.assertTrue((output/"schedules.safetensors").is_file())


if __name__=="__main__": unittest.main()
