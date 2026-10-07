import copy
import importlib.util
from pathlib import Path
import sys
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
