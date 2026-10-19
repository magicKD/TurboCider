"""Host-only full-request/four-arm/cache-schedule admission evidence."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_lora_swiglu_request_screen as screen


class SwigluRequestScreenTests(unittest.TestCase):
    def report(self,seq):
        rows=[]
        for p in seq:
            hit=p["prefix_hit"] is True;labels=[]
            if p["prefix"] and p["qkv"]:labels.append("qwen21_direct_qkv_immutable_prefix_composition_experimental")
            if p["ffn"]:labels.append("qwen21_typed_lora_swiglu_decode_experimental")
            selection="joint BF16 LoRA A/B operands; "+(screen.prefix_screen.FIVE if p["steps"]==5 else screen.prefix_screen.SIX)
            if p["prefix"]:selection+="; edit prefix KV snapshot "+("hit" if hit else "miss")
            if p["ffn"]:selection+="; typed LoRA B/SwiGLU decode fusion"
            m=dict(model="qwen-image-2.1",operation="image."+p["case"],width=512,height=512,steps=p["steps"],actual_denoise_steps=p["steps"],
                seed=p["seed"],lora_applied_projections=227,runtime_backend="mlx_cpp_metal",runtime_precision="bf16",encoder_runtime_backend="mlx_cpp_metal",
                lora_strategy="inference_time",prompt_cache_hit=p["conditioning_hit"],acceleration_selection=selection,text_tokens=35,reference_tokens=1024,
                plan=dict(algorithm_approximations=labels))
            blocks=32*(p["steps"]-(0 if hit else 1))
            if p["qkv"]:m["qwen_corrected_qkv"]=dict(enabled=True,recipe="decode-only-corrected-qkv-v3-original-buffers",weight_storage="original-buffers",
                bank_bytes=0,bank_reused=False,pack_seconds_this_request=0.,completed_decode_blocks=blocks,completed_sampler_steps=p["steps"],completed_prefill_steps=0 if hit else 1,
                prefix_snapshot_enabled=p["prefix"],prefix_snapshot_hit=hit,prefix_snapshot_bytes=1059*2*32*32*128*2 if p["prefix"] else 0,
                rank_math="f32",online_lora=True,prefill_changed=False)
            if p["ffn"]:m["qwen_lora_swiglu"]=dict(enabled=True,recipe=screen.RECIPE,bm=16,bn=128,completed_decode_blocks=blocks,
                completed_sampler_steps=p["steps"],completed_prefill_steps=0 if hit else 1,prefix_snapshot_enabled=p["prefix"],prefix_snapshot_hit=hit,
                extra_weight_bytes=0,native_base_gemm=True,online_independent_ranks=True,rank_math="f32",complete_down_lora=True,prefill_changed=False)
            rows.append(dict(metrics=m,request_wall_including_export=8.))
        return dict(runs=rows)

    def test_full_raw_matrix_and_both_tier_broader_quality_sequences(self):
        for mode,count in (("raw",16),("quality",184),("restoration",42)):
            seq=screen.points(mode);self.assertEqual(len(seq),count);screen.validate(self.report(seq),seq)
        seq=screen.points("quality")
        self.assertEqual({(p["case_id"],p["steps"]) for p in seq},{(cid,steps) for cid,_,_ in screen.QUALITY_CASES for steps in (5,6)})

    def test_all_twelve_timing_windows_are_fixed_complete_work(self):
        for case,prefixes in (("generate",(False,)),("edit",(False,True))):
            for steps in (5,6):
                for prefix in prefixes:
                    for fused in (False,True):
                        seq=screen.points("timing",case,steps,fused,prefix);screen.validate(self.report(seq),seq)
                        self.assertEqual(len(seq),6)
                        if prefix:self.assertEqual([p["prefix_hit"] for p in seq],[False,True,True,True,True,True])

    def test_conditioning_and_kv_are_separate_and_seed_does_not_invalidate(self):
        s=screen.Sequence();s.add();s.add(seed=74);self.assertTrue(s.points[-1]["prefix_hit"])
        s.add("generate",6,True,True,False);s.add()
        self.assertTrue(s.points[-1]["conditioning_hit"]);self.assertIs(s.points[-1]["prefix_hit"],False)
        s.add(ffn=False);self.assertIs(s.points[-1]["prefix_hit"],False)
        s.add(ffn=False,seed=74);self.assertTrue(s.points[-1]["prefix_hit"])
        s.add(reference="alternate");self.assertFalse(s.points[-1]["conditioning_hit"]);self.assertFalse(s.points[-1]["prefix_hit"])

    def test_actual_coverage_and_full_down_lora_cannot_be_omitted(self):
        seq=screen.points("timing","edit",5,True,True)
        for key,value in (("completed_decode_blocks",128),("completed_decode_blocks",160.),("completed_prefill_steps",1),("completed_sampler_steps",6),
                          ("extra_weight_bytes",1),("native_base_gemm",False),("complete_down_lora",False),("online_independent_ranks",False),("bm",32)):
            r=self.report(seq);r["runs"][1]["metrics"]["qwen_lora_swiglu"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):screen.validate(r,seq)
        r=self.report(seq);r["runs"][1]["metrics"]["plan"]["algorithm_approximations"]=["qwen21_direct_qkv_immutable_prefix_composition_experimental"]
        with self.assertRaises(ValueError):screen.validate(r,seq)

    def test_control_cannot_claim_fused_work(self):
        seq=screen.points("timing");r=self.report(seq);r["runs"][0]["metrics"]["qwen_lora_swiglu"]={"enabled":True}
        with self.assertRaises(ValueError):screen.validate(r,seq)


if __name__=="__main__":unittest.main()
