"""Real joint-BF16 QKV prefix import/export and first-step target decode."""
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
class QkvPrefixSnapshotTests(unittest.TestCase):
    def test_direct_prefix_tensor_parity_and_explicit_ownership_guards(self):
        self.run_fixture(False)

    def test_small_gemm_counterexample_is_retained_not_given_equivalence(self):
        self.run_fixture(True)

    def test_typed_swiglu_and_direct_qkv_prefix_full_down_lora_are_bitwise_exact(self):
        self.run_fixture(False,True,False)

    def test_typed_swiglu_with_ordinary_qkv_prefix_is_bitwise_exact(self):
        self.run_fixture(False,True,True)

    def run_fixture(self,counterexample,swiglu=False,ordinary=False):
        native=Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR",ROOT/"build/native")).resolve()
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        with tempfile.TemporaryDirectory(prefix="tc-qwen-qkv-prefix-") as directory:
            binary=Path(directory)/"probe"
            macros=(["-DTC_QKV_PREFIX_COUNTEREXAMPLE=1"] if counterexample else [])+(["-DTC_LORA_SWIGLU_PREFIX_FIXTURE=1"] if swiglu else [])+(["-DTC_LORA_SWIGLU_ORDINARY_QKV=1"] if ordinary else [])
            subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra",*macros,
                "-mmacosx-version-min="+platform.mac_ver()[0],str(ROOT/"tests/native/qwen21_qkv_prefix_snapshot_test.cpp"),
                "-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),"-L"+str(native),"-lturbocider",
                "-L"+str(mlx/"lib"),"-lmlx","-Wl,-rpath,"+str(native),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(binary)],check=True,cwd=ROOT)
            env={key:value for key,value in os.environ.items() if not key.startswith("TURBOCIDER_QWEN21_") and key!="MLX_DISABLE_COMPILE"}
            result=subprocess.run([str(binary),directory],capture_output=True,text=True,env=env,cwd=ROOT,timeout=90)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            measured=json.loads(result.stdout)
            keys=("small_shape_not_equivalent","native_base_counterfactual_exact") if counterexample else ("same_prefix_tensor_exact","first_step_decode_counted","source_destroyed","callback_and_budget_rejected")
            for key in keys:
                self.assertIs(measured[key],True)


if __name__=="__main__":unittest.main()
