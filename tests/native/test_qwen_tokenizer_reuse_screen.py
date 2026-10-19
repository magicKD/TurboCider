"""Host-only CPU-owner lifecycle, unchanged work and fresh-prompt controls."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_tokenizer_reuse_screen as screen


class TokenizerScreenTests(unittest.TestCase):
    def report(self,reuse):
        runs=[]
        for i,point in enumerate(screen.points()):
            selected="gpu; joint BF16 LoRA A/B operands; six-step student schedule"
            if i>=5:selected+="; reused reference encodings=1"
            m=dict(model="qwen-image-2.1",operation="image."+point["case"],width=512,height=512,steps=6,
                actual_denoise_steps=6,seed=73,runtime_backend="mlx_cpp_metal",runtime_precision="bf16",
                encoder_runtime_backend="mlx_cpp_metal",encoder_runtime_precision="bf16",
                encoder_gpu_graph="qwen3_vl_deepstack_gpu_only",lora_strategy="inference_time",lora_applied_projections=227,
                prompt_cache_hit=False,acceleration_selection=selected,timings_seconds=dict(text_encode=.4),
                encoder_weight_residency=dict(enabled=False,weights_retained=False,weights_reused=False,loads_session_total=i+1))
            if reuse:m["qwen_tokenizer_reuse"]=dict(enabled=True,retained=True,reused=i>0,constructed_this_request=i==0,
                tokenized_this_request=True,processor_sha256=screen.PROCESSOR_SHA,processor_source_bytes=11422654,
                validation_seconds_this_request=.0001,tokenization_seconds_this_request=.0001,
                construction_seconds_this_request=.12 if i==0 else 0.)
            runs.append(dict(metrics=m,request_wall_including_export=8.))
        return dict(runs=runs)

    def test_both_receipts_require_same_full_work(self):
        for enabled in (False,True):screen.validate(self.report(enabled),enabled)
        for key,value in (("actual_denoise_steps",5),("prompt_cache_hit",True),
                          ("encoder_gpu_graph","qwen3_vl_fused"),("lora_applied_projections",226),
                          ("runtime_precision","bf16+vae_fp16_conv_fp32_state")):
            bad=self.report(True);bad["runs"][2]["metrics"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):screen.validate(bad,True)
        with self.assertRaises(ValueError):screen.validate(self.report(False),True)
        with self.assertRaises(ValueError):screen.validate(self.report(True),False)

    def test_retained_weights_or_skipped_tokenization_cannot_masquerade_as_cpu_cache(self):
        for key,value in (("constructed_this_request",True),("reused",False),("tokenized_this_request",False),
                          ("processor_sha256","bad"),("construction_seconds_this_request",.1)):
            bad=self.report(True);bad["runs"][2]["metrics"]["qwen_tokenizer_reuse"][key]=value
            with self.assertRaises(ValueError):screen.validate(bad,True)
        bad=self.report(True);bad["runs"][2]["metrics"]["encoder_weight_residency"]["weights_retained"]=True
        with self.assertRaises(ValueError):screen.validate(bad,True)

    def test_all_prompts_are_fresh_and_edit_prime_is_not_scored(self):
        points=screen.points()
        self.assertEqual(len({x["prompt"] for x in points}),8)
        self.assertEqual([i for i,x in enumerate(points) if not x["score"]],[0,4])
        self.assertEqual([i for i,x in enumerate(points) if x["case"]=="edit"],[4,5,6,7])
        args=SimpleNamespace(lora=Path("/tmp/r128"),reference=Path("/tmp/ref.png"),output=Path("/tmp/run"))
        requests=screen.requests(args)
        for i,r in enumerate(requests):
            self.assertEqual(r["steps"],6);self.assertNotIn("dump_tensors",r)
            self.assertEqual("inputs" in r,i>=4);self.assertEqual("qwen21_reference_size" in r,i>=4)

    def test_missing_and_nonfinite_wall_fail_closed(self):
        bad=self.report(True);bad["runs"].pop()
        with self.assertRaises(ValueError):screen.validate(bad,True)
        for value in (True,0,-1,float("nan"),float("inf")):
            bad=copy.deepcopy(self.report(True));bad["runs"][2]["request_wall_including_export"]=value
            with self.assertRaises(ValueError):screen.validate(bad,True)


if __name__=="__main__":unittest.main()
