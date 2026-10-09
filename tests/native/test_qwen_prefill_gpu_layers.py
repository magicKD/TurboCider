import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from qwen_ffn_phase_screen import validate_phases
from tests.native.test_qwen_ffn_phase_screen import receipt


class PrefillGpuLayerTests(unittest.TestCase):
    def test_actual_host_policy_and_scope(self):
        with tempfile.TemporaryDirectory(prefix="tc-prefill-layer-policy-") as folder:
            probe=Path(folder)/"probe"
            build=subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                str(ROOT/"tests/native/qwen_prefill_gpu_layer_config_test.cpp"),str(ROOT/"native/core/common.cpp"),"-o",str(probe)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(build.returncode,0,build.stdout+build.stderr)
            result=subprocess.run([str(probe)],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn("PASS Qwen prefill GPU layer policy",result.stdout)

    @staticmethod
    def rows(layers):
        result=[]
        for index in range(3):
            row=receipt("prefill",index,3168);blocks=32-len(layers)
            row["qwen_ffn_phases"]["prefill"].update(runtime_calls_this_request=blocks,completed_channel_blocks_this_request=blocks)
            row["shared_lora_ranks"]["completed_hybrid_blocks_this_request"]=blocks
            row["hybrid"]["runtime_calls_session_total"]=blocks*(index+1)
            row["hybrid"]["runtime_weight"].update(channel_blocks_session_total=blocks*(index+1),
                async_hybrid_blocks_session_total=blocks*(index+1),forced_gpu_blocks_session_total=len(layers)*(index+1),
                unsplit_gpu_blocks_session_total=len(layers)*(index+1),requested_gpu_layers=list(layers))
            row["acceleration_selection"]="experimental runtime prefill complete-GPU FFN blocks="+",".join(map(str,layers))
            result.append(row)
        return result

    def test_first_last_and_mixed_actual_layer_work(self):
        for layers in ((0,),tuple(range(8)),tuple(range(24,32)),(0,31)):
            validate_phases(self.rows(layers),"prefill",True,layers)

    def test_counters_policy_markers_and_gpu_leaks_rejected(self):
        layers=tuple(range(8));rows=self.rows(layers)
        for key,value in (("requested_gpu_layers",list(range(24,32))),("forced_gpu_blocks_session_total",0),
                          ("unsplit_gpu_blocks_session_total",0),("channel_blocks_session_total",32),
                          ("async_hybrid_blocks_session_total",0)):
            bad=copy.deepcopy(rows);bad[0]["hybrid"]["runtime_weight"][key]=value
            with self.assertRaises(ValueError):validate_phases(bad,"prefill",True,layers)
        bad=copy.deepcopy(rows);bad[0]["acceleration_selection"]=""
        with self.assertRaises(ValueError):validate_phases(bad,"prefill",True,layers)
        bad=copy.deepcopy(rows);bad[1]["hybrid"]["runtime_calls_session_total"]=24
        with self.assertRaises(ValueError):validate_phases(bad,"prefill",True,layers)
        bad=copy.deepcopy(rows);bad[0]["qwen_ffn_phases"]["decode"]["runtime_calls_this_request"]=1
        with self.assertRaises(ValueError):validate_phases(bad,"prefill",True,layers)
        for policy in ("gpu","decode","all"):
            with self.assertRaises(ValueError):validate_phases(rows,policy,True,layers)
        for bad_layers in ([0],(1,0),(0,0),(32,),tuple(range(32))):
            with self.assertRaises(ValueError):validate_phases(rows,"prefill",True,bad_layers)

    def test_layer_cli_errors_precede_fixture_access(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"never-created"
            base=[sys.executable,"-S",str(ROOT/"tools/validation/qwen_ffn_phase_screen.py"),"--cli","unused","--model","unused",
                "--dit-manifest","unused","--reference","unused","--prompt","one","--prompt","two","--prompt","three","--output",str(output)]
            for flags in (["--prefill-layer-screen","--modes","gpu,decode"],["--joint-ab"],
                          ["--joint-ab","--fused-b","--lora","unused"],["--defer-prefill-join"]):
                result=subprocess.run([*base,*flags],capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,2,result.stdout+result.stderr);self.assertNotIn("Traceback",result.stderr)
                self.assertFalse(output.exists())


if __name__=="__main__":unittest.main()
