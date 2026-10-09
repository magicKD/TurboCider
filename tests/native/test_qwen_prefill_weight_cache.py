import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"tools/validation"))
from qwen_ffn_phase_screen import validate_prefill_weight_cache,phase_policy,prefill_cache_policy
from qwen_encoder_residency_screen import validate_weight_code_cache

class PrefillWeightCacheTests(unittest.TestCase):
    @staticmethod
    def rows(storage="surface",budget=1342177280):
        rows=[]
        for index in range(3):
            hits=63*index;misses=96+33*index
            cache=dict(enabled=True,budget_bytes=budget,native_surface_storage=storage=="surface",hits_session_total=hits,
                copy_hits_session_total=hits if storage=="copy" else 0,surface_bind_hits_session_total=hits if storage=="surface" else 0,
                misses_session_total=misses,fills_session_total=63,failed_fills_session_total=0,entries=63,ready_entries=63,
                retained_bytes=1340473344,live_capacity_bytes=1340473344,peak_capacity_bytes=1340473344,
                evictions_session_total=0,declines_session_total=33*(index+1),ineligible_session_total=12+32*(index+1))
            rows.append(dict(hybrid=dict(runtime_weight=dict(weight_code_cache=cache)),qwen_ffn_phases=dict(policy="prefill",
                prefill=dict(completed_channel_blocks_this_request=32,runtime_calls_this_request=32),decode=dict(completed_channel_blocks_this_request=0,runtime_calls_this_request=0))))
        return rows

    def test_actual_cold_fill_zero_hit_then_warm_reuse(self):
        for storage in ("copy","surface"):
            rows=self.rows(storage);validate_prefill_weight_cache(rows,1342177280,storage)
            with self.assertRaises(ValueError):validate_weight_code_cache(rows,1342177280,storage)
        off=self.rows("copy")
        for row in off:
            cache=row["hybrid"]["runtime_weight"]["weight_code_cache"]
            for key in cache:cache[key]=False if type(cache[key]) is bool else 0
        validate_prefill_weight_cache(off,0)
        with self.assertRaises(ValueError):validate_prefill_weight_cache([],0)

    def test_forged_work_counters_or_disabled_decode_work_rejected(self):
        for key,value in (("hits_session_total",0),("misses_session_total",96),("fills_session_total",64),
            ("ready_entries",62),("entries",64),("declines_session_total",33),("evictions_session_total",1),
            ("ineligible_session_total",1),("copy_hits_session_total",63),("surface_bind_hits_session_total",0)):
            bad=self.rows();bad[1]["hybrid"]["runtime_weight"]["weight_code_cache"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_prefill_weight_cache(bad,1342177280,"surface")
        bad=self.rows();bad[0]["hybrid"]["runtime_weight"]["weight_code_cache"]["hits_session_total"]=1
        with self.assertRaises(ValueError):validate_prefill_weight_cache(bad,1342177280,"surface")
        bad=self.rows();bad[1]["qwen_ffn_phases"]["decode"]["runtime_calls_this_request"]=1
        with self.assertRaises(ValueError):validate_prefill_weight_cache(bad,1342177280,"surface")

    def test_a8_shared_stager_scope_follows_actual_prediction_calls(self):
        rows=self.rows()
        for index,row in enumerate(rows):
            row["qwen_ffn_phases"]["prefill"]["runtime_calls_this_request"]=64
            row["hybrid"]["runtime_weight"]["weight_code_cache"]["ineligible_session_total"]=12+64*(index+1)
        validate_prefill_weight_cache(rows,1342177280,"surface")
        rows[1]["hybrid"]["runtime_weight"]["weight_code_cache"]["ineligible_session_total"]=76
        with self.assertRaises(ValueError):validate_prefill_weight_cache(rows,1342177280,"surface")

    def test_cache_arms_share_prefill_only_and_explicit_policy(self):
        for mode,expected in (("gpu",(0,"copy")),("prefill",(0,"copy")),("prefill_copy_cache",(512,"copy")),("prefill_surface_cache",(512,"surface"))):
            self.assertEqual(prefill_cache_policy(mode,512),expected)
            self.assertEqual(phase_policy(mode),"gpu" if mode=="gpu" else "prefill")
        self.assertEqual(phase_policy("decode"),"decode");self.assertEqual(phase_policy("all"),"all")

    def test_scope_errors_precede_source_access(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"never-created"
            base=[sys.executable,"-S",str(ROOT/"tools/validation/qwen_ffn_phase_screen.py"),"--cli","unused","--model","unused",
                "--dit-manifest","unused","--reference","unused","--prompt","one","--prompt","two","--prompt","three","--output",str(output)]
            for flags in (["--prefill-code-cache-bytes","0"],["--prefill-code-cache-bytes","2147483649"],
                ["--prefill-code-cache-bytes","512"],["--prefill-code-cache-bytes","512","--lora","unused"],
                ["--prefill-code-cache-bytes","512","--lora","unused","--joint-ab","--generation"],
                ["--prefill-code-cache-bytes","512","--lora","unused","--joint-ab","--prefill-layer-screen"],
                ["--prefill-code-cache-bytes","512","--lora","unused","--joint-ab","--modes","gpu,all"],
                ["--modes","gpu,prefill_surface_cache"]):
                result=subprocess.run([*base,*flags],capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,2,result.stdout+result.stderr);self.assertNotIn("Traceback",result.stderr);self.assertFalse(output.exists())

if __name__=="__main__":unittest.main()
