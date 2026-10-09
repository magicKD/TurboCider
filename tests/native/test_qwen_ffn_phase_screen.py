import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"tools/validation"))
SPEC=importlib.util.spec_from_file_location("qwen_ffn_phase_screen",ROOT/"tools/validation/qwen_ffn_phase_screen.py")
SCREEN=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(SCREEN)


def receipt(policy,index=0):
    row=dict(width=512,height=512,steps=6,actual_denoise_steps=6,text_tokens=72,reference_tokens=2048,timings_seconds=dict(denoise=12))
    prefill=policy in ("prefill","all");decode=policy in ("decode","all")
    calls=96*prefill+160*decode;blocks=32*prefill+160*decode
    row["qwen_ffn_phases"]=dict(policy=policy,
        prefill=dict(steps_this_request=1,actual_rows=3144,step_seconds=4,runtime_calls_this_request=96*prefill,completed_channel_blocks_this_request=32*prefill),
        decode=dict(steps_this_request=5,actual_rows=1024,step_seconds=7,runtime_calls_this_request=160*decode,completed_channel_blocks_this_request=160*decode))
    row["shared_lora_ranks"]=dict(completed_hybrid_blocks_this_request=blocks)
    if policy!="gpu":row["hybrid"]=dict(bucket=1056,runtime_failed=False,runtime_failures_session_total=0,runtime_calls_session_total=calls*(index+1),
        runtime_weight=dict(fallback_blocks_session_total=0,overflow_retries_session_total=0,headroom_scale=1,executor_backend="private_ane",
            partition_axis="intermediate_channels",data_path="w8a8_hadamard",forced_gpu_blocks_session_total=0,
            channel_blocks_session_total=blocks*(index+1),async_hybrid_blocks_session_total=blocks*(index+1)))
    return row


class PhaseScreenTests(unittest.TestCase):
    def test_actual_phase_counts_and_cumulative_progress(self):
        for policy in SCREEN.MODES:SCREEN.validate_phases([receipt(policy,i) for i in range(3)],policy,True)

    def test_disabled_phase_cannot_hide_runtime_calls(self):
        for policy,name in (("prefill","decode"),("decode","prefill"),("gpu","prefill")):
            row=receipt(policy);row["qwen_ffn_phases"][name]["runtime_calls_this_request"]=1
            with self.assertRaises(ValueError):SCREEN.validate_phases([row],policy)

    def test_missing_forged_reset_failure_and_nonfinite_rejected(self):
        changes=[lambda r:r.pop("qwen_ffn_phases"),lambda r:r["qwen_ffn_phases"].update(policy="all"),
            lambda r:r["qwen_ffn_phases"]["prefill"].update(steps_this_request=True),
            lambda r:r["qwen_ffn_phases"]["decode"].update(actual_rows=3144),
            lambda r:r["qwen_ffn_phases"]["prefill"].update(step_seconds=float("nan")),
            lambda r:r["qwen_ffn_phases"]["prefill"].update(step_seconds=20),
            lambda r:r["hybrid"].update(runtime_calls_session_total=0),
            lambda r:r["hybrid"]["runtime_weight"].update(overflow_retries_session_total=1),
            lambda r:r["hybrid"]["runtime_weight"].update(async_hybrid_blocks_session_total=0),
            lambda r:r["shared_lora_ranks"].update(completed_hybrid_blocks_this_request=192),
            lambda r:r.update(student_ffn_reuse={})]
        for mutate in changes:
            row=receipt("prefill");mutate(row)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):SCREEN.validate_phases([row],"prefill",True)
        with self.assertRaises(ValueError):SCREEN.validate_phases([receipt("all"),receipt("all")],"all")
        with self.assertRaises(ValueError):SCREEN.validate_phases([],"all")

    def test_invalid_cli_arguments_precede_fixture_access(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"never-created"
            base=[sys.executable,"-S",str(ROOT/"tools/validation/qwen_ffn_phase_screen.py"),"--cli","unused","--model","unused",
                "--dit-manifest","unused","--reference","unused","--prompt","one","--prompt","two","--prompt","three","--output",str(output)]
            for flags in (["--modes","prefill,decode"],["--modes","gpu,gpu"],["--order","all,prefill,decode,gpu,gpu"],
                ["--channels","0"],["--fused-b"]):
                result=subprocess.run([*base,*flags],capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,2,result.stdout+result.stderr);self.assertNotIn("Traceback",result.stderr);self.assertFalse(output.exists())


if __name__=="__main__":unittest.main()
