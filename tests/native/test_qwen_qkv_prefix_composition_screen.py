"""Host-only workload, real hit coverage and refusal gates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_qkv_prefix_composition_screen as screen


class PrefixCompositionScreenTests(unittest.TestCase):
    def report(self,sequence):
        rows=[]
        for p in sequence:
            direct,prefix=screen.ARMS[p["arm"]];hit=p["prefix_hit"] is True
            selection="joint BF16 LoRA A/B operands; "+(screen.FIVE if p["steps"]==5 else screen.SIX)
            if prefix:selection+="; edit prefix KV snapshot "+("hit" if hit else "miss")
            m=dict(model="qwen-image-2.1",operation="image."+p["case"],width=512,height=512,steps=p["steps"],actual_denoise_steps=p["steps"],
                seed=p["seed"],lora_applied_projections=227,runtime_backend="mlx_cpp_metal",runtime_precision="bf16",encoder_runtime_backend="mlx_cpp_metal",
                lora_strategy="inference_time",prompt_cache_hit=p["conditioning_hit"],acceleration_selection=selection,text_tokens=35,reference_tokens=1024,
                plan=dict(algorithm_approximations=["qwen21_direct_qkv_immutable_prefix_composition_experimental"] if direct and prefix else []))
            if direct:m["qwen_corrected_qkv"]=dict(enabled=True,recipe="decode-only-corrected-qkv-v3-original-buffers",weight_storage="original-buffers",
                bank_bytes=0,bank_reused=False,pack_seconds_this_request=0.,completed_decode_blocks=32*(p["steps"]-(0 if hit else 1)),
                completed_sampler_steps=p["steps"],completed_prefill_steps=0 if hit else 1,prefix_snapshot_enabled=prefix,prefix_snapshot_hit=hit,
                prefix_snapshot_bytes=1059*2*32*32*128*2 if prefix else 0,rank_math="f32",online_lora=True,prefill_changed=False)
            rows.append(dict(metrics=m,request_wall_including_export=8.))
        return dict(runs=rows)

    def test_all_eight_fixed_arm_windows_have_five_hot_complete_requests(self):
        for steps in (5,6):
            for arm in screen.ARMS:
                seq=screen.points("timing",steps,arm);self.assertEqual(len(seq),6);screen.validate(self.report(seq),seq)
                if screen.ARMS[arm][1]:self.assertEqual([p["prefix_hit"] for p in seq],[False,True,True,True,True,True])

    def test_quality_and_restoration_sequences_are_not_silent_scope_reductions(self):
        seq=screen.points("quality");self.assertEqual(len(seq),60);screen.validate(self.report(seq),seq)
        self.assertEqual({(p["steps"],p["case_id"]) for p in seq},{(steps,cid) for steps in (5,6) for cid,_,_ in screen.EDITS})
        seq=screen.points("restoration");self.assertEqual(len(seq),27);screen.validate(self.report(seq),seq)
        self.assertEqual({p["case"] for p in seq},{"generate","edit"})
        self.assertIs(seq[25]["conditioning_hit"],True)
        self.assertIs(seq[25]["prefix_hit"],False)
        r=self.report(seq);r["runs"][25]["metrics"]["prompt_cache_hit"]=False
        with self.assertRaises(ValueError):screen.validate(r,seq)

    def test_prefix_hit_must_count_first_step_and_actual_sampler_separately(self):
        seq=screen.points("timing",5,"combined")
        for key,value in (("completed_decode_blocks",128),("completed_decode_blocks",160.),("completed_sampler_steps",6),("completed_prefill_steps",1),
                          ("prefix_snapshot_hit",False),("prefix_snapshot_bytes",0),("prefix_snapshot_bytes",555745281),("bank_bytes",3<<30)):
            r=self.report(seq);r["runs"][1]["metrics"]["qwen_corrected_qkv"][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):screen.validate(r,seq)
        r=self.report(seq);r["runs"][1]["metrics"]["plan"]["algorithm_approximations"]=[]
        with self.assertRaises(ValueError):screen.validate(r,seq)


if __name__=="__main__":unittest.main()
