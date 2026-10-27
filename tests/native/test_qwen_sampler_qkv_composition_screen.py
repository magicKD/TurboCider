"""Host-only full workload and quality-tier separation gates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_sampler_qkv_composition_screen as screen


class SamplerQkvCompositionTests(unittest.TestCase):
    def report(self,sequence):
        rows=[]
        for p in sequence:
            steps=p["steps"];labels=[]
            if steps==5:labels.append("qwen21_viggle_v021_r128_5step_sampler_experimental")
            if steps==5 and p["on"]:labels.append("qwen21_five_step_direct_qkv_composition_experimental")
            m=dict(model="qwen-image-2.1",operation="image."+p["case"],width=512,height=512,steps=steps,actual_denoise_steps=steps,
                seed=p["seed"],lora_applied_projections=227,runtime_backend="mlx_cpp_metal",runtime_precision="bf16",
                encoder_runtime_backend="mlx_cpp_metal",lora_strategy="inference_time",prompt_cache_hit=p["conditioning_hit"],
                acceleration_selection="joint BF16 LoRA A/B operands; "+(screen.base.FIVE if steps==5 else screen.base.SIX),
                plan=dict(algorithm_approximations=labels))
            if p["on"]:m["qwen_corrected_qkv"]=dict(enabled=True,recipe=screen.base.DIRECT_RECIPE,completed_decode_blocks=32*(steps-1),
                completed_sampler_steps=steps,bank_bytes=0,bank_reused=False,pack_seconds_this_request=0.,weight_storage="original-buffers",
                prefill_changed=False,online_lora=True,rank_math="f32")
            rows.append(dict(request_wall_including_export=10.,metrics=m))
        return dict(runs=rows)

    def test_four_frozen_arms_and_both_task_timing_sequences(self):
        self.assertEqual(set(screen.ARMS),{"six-original","six-direct","five-original","five-direct"})
        for case in ("generate","edit"):
            for arm in screen.ARMS:
                seq=screen.points("timing",case,arm);self.assertEqual(len(seq),6);screen.validate(self.report(seq),seq)

    def test_restoration_and_quality_cover_both_sampler_tiers_without_prefix_hits(self):
        seq=screen.points("raw");self.assertEqual(len(seq),18);screen.validate(self.report(seq),seq)
        self.assertEqual([p["steps"] for p in seq[:9]],[6,6,5,5,5,5,6,6,6])
        seq=screen.points("quality");self.assertEqual(len(seq),56);screen.validate(self.report(seq),seq)
        self.assertEqual(len({p["case_id"] for p in seq}),7)
        for start in range(0,56,8):
            self.assertEqual({p["arm"] for p in seq[start:start+8]},set(screen.ARMS))
            self.assertEqual({p["seed"] for p in seq[start:start+8]},{73,74})

    def test_five_cannot_claim_six_steps_or_160_decode_blocks(self):
        seq=screen.points("raw")
        for key,value in (("completed_decode_blocks",160),("completed_sampler_steps",6),("completed_sampler_steps",5.)):
            report=self.report(seq);report["runs"][3]["metrics"]["qwen_corrected_qkv"][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):screen.validate(report,seq)
        report=self.report(seq);report["runs"][3]["metrics"]["actual_denoise_steps"]=6
        with self.assertRaises(ValueError):screen.validate(report,seq)

    def test_sampler_and_composition_labels_cannot_be_omitted_or_relabelled(self):
        seq=screen.points("raw")
        for labels in ([],["qwen21_viggle_v021_r128_6step_distillation"],["qwen21_viggle_v021_r128_5step_sampler_experimental"]):
            report=self.report(seq);report["runs"][3]["metrics"]["plan"]["algorithm_approximations"]=labels
            with self.subTest(labels=labels),self.assertRaises(ValueError):screen.validate(report,seq)


if __name__=="__main__":unittest.main()
