import copy
import importlib.util
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
SPEC=importlib.util.spec_from_file_location("encoder_residency_screen",ROOT/"tools/validation/qwen_encoder_residency_screen.py")
SCREEN=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(SCREEN)


def receipt(mode,index):
    data=dict(prompt_cache_hit=False,runtime_backend="mlx_cpp_metal",steps=6,actual_denoise_steps=6,
        encoder_execution="gpu",encoder_hybrid={},encoder_runtime_reuse=None,
        timings_seconds=dict(request_wall=12.0,text_encode=1.2,denoise=10.0))
    if mode!="gpu":
        retained=mode=="encoder_retained"
        data.update(encoder_execution="gpu_ane_experimental",
            encoder_runtime_reuse=dict(enabled=retained,executor_retained=retained,executor_reused=retained and index>0,
                actual_calls_this_request=36,retained_estimated_bytes=240000000 if retained else 0),
            encoder_hybrid=dict(runtime_failed=False,runtime_calls_session_total=36*(index+1) if retained else 36,
                session_released_after_encoding=not retained,
                runtime_weight=dict(executor_backend="private_ane",fallback_blocks_session_total=0,
                    data_path="w8a8_hadamard",partition_axis="intermediate_channels",ane_channels=3072)))
    return data


class EncoderResidencyScreenTests(unittest.TestCase):
    def test_split_down_rank_screen_requires_actual_progress(self):
        def row(blocks,arrays):return dict(hybrid=dict(runtime_weight=dict(
            split_down_rank_blocks_session_total=blocks,split_down_rank_arrays_session_total=arrays)))
        SCREEN.validate_down_ranks([row(192,192),row(384,384)],True)
        SCREEN.validate_down_ranks([row(0,0),row(0,0)],False)
        for rows in ([{}],[row(True,1)],[row(0,0)],[row(1,0)],[row(192,192),row(192,192)]):
            with self.assertRaises(ValueError):SCREEN.validate_down_ranks(rows,True)
        with self.assertRaises(ValueError):SCREEN.validate_down_ranks([row(1,1)],False)

    def test_weight_code_cache_requires_completed_hits_and_bounded_leases(self):
        cache=dict(enabled=True,budget_bytes=1024,hits_session_total=10,misses_session_total=3,
            native_surface_storage=False,copy_hits_session_total=10,surface_bind_hits_session_total=0,
            fills_session_total=3,failed_fills_session_total=0,entries=3,ready_entries=3,
            retained_bytes=900,live_capacity_bytes=900,peak_capacity_bytes=1024,
            evictions_session_total=0,declines_session_total=0,ineligible_session_total=0)
        def row(value):return dict(hybrid=dict(runtime_weight=dict(weight_code_cache=value)))
        SCREEN.validate_weight_code_cache([row(cache),row({**cache,"hits_session_total":20,"copy_hits_session_total":20})],1024)
        surface={**cache,"native_surface_storage":True,"copy_hits_session_total":0,"surface_bind_hits_session_total":10}
        SCREEN.validate_weight_code_cache([row(surface)],1024,"surface")
        for change in (dict(native_surface_storage=False),dict(surface_bind_hits_session_total=0),dict(copy_hits_session_total=1)):
            with self.assertRaises(ValueError):SCREEN.validate_weight_code_cache([row({**surface,**change})],1024,"surface")
        for field,value in (("enabled",False),("budget_bytes",True),("hits_session_total",True),
            ("hits_session_total",0),("peak_capacity_bytes",1025),("live_capacity_bytes",899),
            ("ready_entries",4),("failed_fills_session_total",1),("fills_session_total",4)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                SCREEN.validate_weight_code_cache([row({**cache,field:value})],1024)
        with self.assertRaises(ValueError):SCREEN.validate_weight_code_cache([row(cache),row(cache)],1024)
        off={name:0 for name in cache};off.update(enabled=False,native_surface_storage=False)
        SCREEN.validate_weight_code_cache([row(off)],0)
        off["retained_bytes"]=1
        with self.assertRaises(ValueError):SCREEN.validate_weight_code_cache([row(off)],0)
        with self.assertRaises(ValueError):SCREEN.validate_weight_code_cache([{}],1024)

    def test_rank_screen_argument_errors_precede_fixture_or_output_access(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"never-created"
            base=[sys.executable,"-S",str(ROOT/"tools/validation/qwen_encoder_residency_screen.py"),
                "--cli","unused","--model","unused","--manifest","unused","--reference","unused",
                "--prompt","one","--prompt","two","--prompt","three","--output",str(output)]
            for flags in (["--lora-ranks-gpu-control"],["--lora-ranks-screen"],
                ["--lora-ranks-screen","--lora","unused","--dit-manifest","unused","--global-channels","0"]):
                result=subprocess.run([*base,*flags],capture_output=True,text=True,timeout=10)
                with self.subTest(flags=flags):
                    self.assertEqual(result.returncode,2,result.stdout+result.stderr)
                    self.assertNotIn("Traceback",result.stderr)
                    self.assertFalse(output.exists())

    def test_shared_ranks_requires_real_completed_dual_consumer_work(self):
        metrics=dict(enabled=True,prepared_sets_this_request=192,
            completed_hybrid_blocks_this_request=192,completed_adapter_rank_arrays_this_request=384)
        SCREEN.validate_shared_ranks([dict(shared_lora_ranks=metrics)],True)
        for field,value in (("enabled",False),("prepared_sets_this_request",True),
                            ("completed_hybrid_blocks_this_request",0),
                            ("completed_hybrid_blocks_this_request",193),
                            ("completed_adapter_rank_arrays_this_request",191)):
            invalid=dict(metrics);invalid[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                SCREEN.validate_shared_ranks([dict(shared_lora_ranks=invalid)],True)
        off=dict.fromkeys(tuple(metrics)[1:],0);off["enabled"]=False
        SCREEN.validate_shared_ranks([dict(shared_lora_ranks=off)],False)
        off["prepared_sets_this_request"]=1
        with self.assertRaises(ValueError):SCREEN.validate_shared_ranks([dict(shared_lora_ranks=off)],False)
        with self.assertRaises(ValueError):SCREEN.validate_shared_ranks([{}],True)

    def test_model_snapshot_is_bounded_and_generation_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ("diffusion_models/qwen_image_2.1_bf16.safetensors",
                         "text_encoders/qwen3vl_8b_bf16.safetensors","vae/qwen_image_2.1_vae_bf16.safetensors"):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes((2).to_bytes(8,"little")+b"{}"+b"payload")
            snapshot=SCREEN.model_snapshot(root)
            self.assertEqual(len(snapshot),3)
            path=root/"text_encoders/qwen3vl_8b_bf16.safetensors"
            path.write_bytes((2).to_bytes(8,"little")+b"{}"+b"changed")
            self.assertNotEqual(SCREEN.model_snapshot(root),snapshot)
            path.write_bytes((32<<20).to_bytes(8,"little"))
            with self.assertRaises(ValueError):SCREEN.model_snapshot(root)

    def test_request_preserves_order_and_real_adapter(self):
        references=[Path("one.png"),Path("two.png")]
        gpu=SCREEN.make_request("fresh",references,Path("gpu.png"))
        candidate=SCREEN.make_request("fresh",references,Path("ane.png"),Path("manifest.json"),Path("adapter.safetensors"))
        self.assertEqual([row["path"] for row in candidate["inputs"]],["one.png","two.png"])
        self.assertEqual(gpu["steps"],40);self.assertEqual(candidate["steps"],6)
        self.assertEqual(candidate["execution"],"gpu")
        self.assertEqual(candidate["lora_strategy"],"inference_time")
        self.assertNotIn("encoder_ane_manifest",gpu)

    def test_three_actual_lifecycles_and_cumulative_counters(self):
        for mode in SCREEN.MODES:
            SCREEN.validate_rows([receipt(mode,i) for i in range(3)],mode,3,True,3072)

    def test_gpu_and_ane_weight_retention_require_actual_reuse(self):
        for mode in ("gpu_weights","encoder_weights"):
            base="gpu" if mode=="gpu_weights" else "encoder_retained"
            rows=[receipt(base,i) for i in range(3)]
            for index,row in enumerate(rows):
                row["encoder_weight_residency"]=dict(enabled=True,weights_retained=True,weights_reused=index>0,
                    loads_session_total=1,retained_bytes=17<<30,decline_reason="")
            SCREEN.validate_rows(rows,mode,3,True,3072)
            rows[1]["encoder_weight_residency"]["loads_session_total"]=2
            with self.assertRaises(ValueError):SCREEN.validate_rows(rows,mode,3,True,3072)

    def test_combined_encoder_and_dit_shares_are_independent(self):
        request=SCREEN.make_request("fresh",[Path("reference.png")],Path("output.png"),
            Path("encoder.json"),Path("adapter.safetensors"),Path("dit.json"))
        self.assertEqual(request["execution"],"gpu_ane")
        self.assertEqual(request["hybrid_mlp_mode"],"runtime")
        for mode in ("dit_weights","dit_encoder_weights"):
            base="gpu" if mode=="dit_weights" else "encoder_retained"
            rows=[receipt(base,i) for i in range(3)]
            for index,row in enumerate(rows):
                row["runtime_backend"]="mlx_cpp_metal+private_ane_runtime_weight_experimental"
                row["encoder_weight_residency"]=dict(enabled=True,weights_retained=True,weights_reused=index>0,
                    loads_session_total=1,retained_bytes=17<<30,decline_reason="")
                row["hybrid"]=dict(runtime_failed=False,runtime_calls_session_total=192*(index+1),
                    runtime_weight=dict(executor_backend="private_ane",data_path="w8a8_hadamard",
                        fallback_blocks_session_total=0,ane_channels=5120))
            SCREEN.validate_rows(rows,mode,3,True,3072,5120)
            rows[1]["hybrid"]["runtime_weight"]["ane_channels"]=3072
            with self.assertRaises(ValueError):SCREEN.validate_rows(rows,mode,3,True,3072,5120)

    def test_cache_hit_fallback_wrong_source_or_forged_reuse_rejected(self):
        mutations=[lambda r:r.update(prompt_cache_hit=True),
            lambda r:r.update(runtime_backend="mlx_cpp_metal+private_ane_runtime_weight_experimental"),
            lambda r:r["encoder_runtime_reuse"].update(actual_calls_this_request=0),
            lambda r:r["encoder_runtime_reuse"].update(executor_reused=False),
            lambda r:r["encoder_hybrid"].update(runtime_calls_session_total=36),
            lambda r:r["encoder_hybrid"]["runtime_weight"].update(fallback_blocks_session_total=1),
            lambda r:r["encoder_hybrid"]["runtime_weight"].update(data_path="fp16"),
            lambda r:r["encoder_hybrid"]["runtime_weight"].update(ane_channels=5120),
            lambda r:r["timings_seconds"].update(request_wall=float("nan"))]
        for mutate in mutations:
            rows=[receipt("encoder_retained",i) for i in range(3)];mutate(rows[1])
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                SCREEN.validate_rows(rows,"encoder_retained",3,True,3072)

    def test_gpu_baseline_cannot_be_relabelled_as_encoder_work(self):
        rows=[receipt("gpu",i) for i in range(3)]
        rows[0]["encoder_runtime_reuse"]=copy.deepcopy(receipt("encoder_retained",0)["encoder_runtime_reuse"])
        with self.assertRaises(ValueError):SCREEN.validate_rows(rows,"gpu",3,True)


if __name__=="__main__":unittest.main()
