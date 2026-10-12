import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from qwen_ffn_phase_screen import validate_frozen_base


class FrozenBaseScreenTests(unittest.TestCase):
    @staticmethod
    def rows():
        rows=[]
        for index in range(3):
            rows.append(dict(model="qwen-image-2.1",operation="image.generate",width=512,height=512,steps=40,actual_denoise_steps=40,
                reference_tokens=0,text_tokens=31,lora_applied_projections=0,prompt_cache_hit=False,
                runtime_backend="mlx_cpp_metal+coreml",encoder_execution="gpu",
                timings_seconds=dict(request_wall=33.0,text_encode=1.0,denoise=30.2),
                encoder_weight_residency=dict(enabled=True,weights_retained=True,loads_session_total=1,weights_reused=index>0,
                    retained_bytes=17534247392,decline_reason=""),
                hybrid=dict(bucket=1024,hidden=4096,mlp_width=12288,block_count=32,output_channels=4096,ane_mlp_range=[0,6144],weight_variant="int8_pc",
                    checkpoint_sha256_verified=True,runtime_failed=False,runtime_weight=None,compute_units="cpuAndNeuralEngine",
                    runtime_failures_session_total=0,runtime_calls_session_total=1248*(index+1),
                    warmup_calls_session_total=32,calls_session_total=32+1248*(index+1)),
                qwen_ffn_phases=dict(policy="gpu",prefill=dict(steps_this_request=1,actual_rows=1055,step_seconds=1.1,runtime_calls_this_request=0,
                    completed_channel_blocks_this_request=0),decode=dict(steps_this_request=39,actual_rows=1024,
                    step_seconds=29.0,runtime_calls_this_request=0,completed_channel_blocks_this_request=0))))
        return rows

    def test_real_cached_coreml_work_and_equal_source_retention(self):
        validate_frozen_base(self.rows())
        local=self.rows()
        for index,row in enumerate(local):row["encoder_weight_residency"]=dict(enabled=False,weights_retained=False,
            weights_reused=False,retained_bytes=0,loads_session_total=index+1,source_bytes=17534247392,decline_reason="")
        validate_frozen_base(local,False)
        with self.assertRaises(ValueError):validate_frozen_base(local,True)
        with self.assertRaises(ValueError):validate_frozen_base(self.rows(),False)
        for key,value in (("enabled",True),("weights_reused",True),("retained_bytes",1),("loads_session_total",1)):
            bad=copy.deepcopy(local);bad[1]["encoder_weight_residency"][key]=value
            with self.assertRaises(ValueError):validate_frozen_base(bad,False)

    def test_wrong_source_backend_geometry_and_fake_runtime_callbacks_rejected(self):
        rows=self.rows()
        for key,value in (("checkpoint_sha256_verified",False),("bucket",1056),("ane_mlp_range",[0,5120]),
                          ("weight_variant","fp16"),("compute_units","all"),("runtime_weight",{}),
                          ("runtime_calls_session_total",0),("calls_session_total",1248),("warmup_calls_session_total",0)):
            bad=copy.deepcopy(rows);bad[0]["hybrid"][key]=value
            with self.assertRaises(ValueError):validate_frozen_base(bad)
        for key,value in (("lora_applied_projections",227),("reference_tokens",1024),("operation","image.edit"),
                          ("runtime_backend","mlx_cpp_metal+private_ane_runtime_weight_experimental"),("prompt_cache_hit",True)):
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError):validate_frozen_base(bad)
        bad=copy.deepcopy(rows);bad[1]["hybrid"]["runtime_calls_session_total"]=1248
        with self.assertRaises(ValueError):validate_frozen_base(bad)
        bad=copy.deepcopy(rows);bad[0]["qwen_ffn_phases"]["decode"]["runtime_calls_this_request"]=1248
        with self.assertRaises(ValueError):validate_frozen_base(bad)
        bad=copy.deepcopy(rows);bad[1]["encoder_weight_residency"]["weights_reused"]=False
        with self.assertRaises(ValueError):validate_frozen_base(bad)
        with self.assertRaises(ValueError):validate_frozen_base([])

    def test_frozen_scope_errors_precede_source_access(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"never-created"
            base=[sys.executable,"-S",str(ROOT/"tools/validation/qwen_ffn_phase_screen.py"),"--cli","unused","--model","unused",
                "--dit-manifest","unused","--prompt","one","--prompt","two","--prompt","three","--output",str(output)]
            for flags in (["--generation","--modes","gpu,frozen"],["--frozen-manifest","unused"],
                          ["--generation","--frozen-manifest","unused","--lora","unused"],
                          ["--generation","--frozen-manifest","unused","--prefill-layer-screen"]):
                result=subprocess.run([*base,*flags],capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,2,result.stdout+result.stderr);self.assertNotIn("Traceback",result.stderr)
                self.assertFalse(output.exists())

    def test_native_base_omits_optional_zero_lora_counter(self):
        rows=self.rows()
        for row in rows:del row["lora_applied_projections"]
        validate_frozen_base(rows)

    def test_invalid_or_unenclosed_frozen_timings_rejected(self):
        for name in ("request_wall","text_encode","denoise"):
            for value in (True,0,-1,float("inf"),float("nan"),None,"30"):
                bad=self.rows();bad[0]["timings_seconds"][name]=value
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):validate_frozen_base(bad)
        for name in ("prefill","decode"):
            for value in (True,0,-1,float("inf"),float("nan"),None,"30"):
                bad=self.rows();bad[0]["qwen_ffn_phases"][name]["step_seconds"]=value
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):validate_frozen_base(bad)
        bad=self.rows();bad[0]["qwen_ffn_phases"]["decode"]["step_seconds"]=31.0
        with self.assertRaises(ValueError):validate_frozen_base(bad)

    def test_unadmitted_retention_and_nonoriginal_work_rejected(self):
        for key,value in (("enabled",False),("retained_bytes",0),("retained_bytes",True),
                          ("retained_bytes",(20<<30)+1),("loads_session_total",True),
                          ("weights_reused",0),("decline_reason","growth_limit")):
            bad=self.rows();bad[0]["encoder_weight_residency"][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_frozen_base(bad)
        for key,value in (("steps",6),("width",512.0),("reference_tokens",False),
                          ("lora_applied_projections",False),("student_ffn_reuse",{}),
                          ("encoder_hybrid",{"runtime_calls_session_total":1}),("encoder_runtime_reuse",{})):
            bad=self.rows();bad[0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_frozen_base(bad)
        for key,value in (("block_count",24),("output_channels",10240),("hidden",4096.0)):
            bad=self.rows();bad[0]["hybrid"][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_frozen_base(bad)


if __name__=="__main__":unittest.main()
