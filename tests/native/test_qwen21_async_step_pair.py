"""Real MLX scheduled states, independent masks, odd tail and scope-exit drain."""
import json
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
class AsyncStepPairTests(unittest.TestCase):
    def test_state_masks_progress_capacity_and_aborted_scope(self):
        native=Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR",ROOT/"build/native")).resolve()
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        with tempfile.TemporaryDirectory(prefix="tc-qwen-async-step-") as directory:
            binary=Path(directory)/"probe"
            subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min="+platform.mac_ver()[0],str(ROOT/"tests/native/qwen21_async_step_pair_test.cpp"),
                "-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),"-L"+str(native),"-lturbocider",
                "-L"+str(mlx/"lib"),"-lmlx","-Wl,-rpath,"+str(native),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(binary)],check=True,cwd=ROOT)
            env={key:value for key,value in os.environ.items() if not key.startswith("TURBOCIDER_QWEN21_") and key!="MLX_DISABLE_COMPILE"}
            result=subprocess.run([str(binary)],capture_output=True,text=True,env=env,cwd=ROOT,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            values=json.loads(result.stdout)
            for key in ("five_six_state_bits_exact","per_step_nonfinite_recovery_rejected","scope_exit_drained","bounded_capacity_and_progress"):
                self.assertIs(values[key],True)


if __name__=="__main__":unittest.main()
