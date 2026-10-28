"""Host-only authored sampler labels, matched workloads and cache lifecycle."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_student_sampler_screen as screen


class SamplerScreenTests(unittest.TestCase):
    def report(self,mode="timing"):
        runs=[]
        for point in screen.points(73,mode):
            steps=point["steps"]
            selected="gpu; joint BF16 LoRA A/B operands; "+(screen.FIVE if steps==5 else screen.SIX)
            if point["prefix"] is not None:
                selected+="; edit prefix KV snapshot "+("hit" if point["prefix"] else "miss")
            tag="qwen21_viggle_v021_r128_"+("5step_sampler_experimental" if steps==5 else "6step_distillation")
            runs.append(dict(request_wall_including_export=7.,metrics=dict(
                model="qwen-image-2.1",operation="image."+point["case"],width=512,height=512,
                steps=steps,actual_denoise_steps=steps,seed=point["seed"],
                runtime_backend="mlx_cpp_metal",encoder_runtime_backend="mlx_cpp_metal",runtime_precision="bf16",
                lora_applied_projections=227,lora_strategy="inference_time",acceleration_selection=selected,
                prompt_cache_hit=point["conditioning_hit"],plan=dict(algorithm_approximations=[tag]))))
        return dict(runs=runs)

    def test_complete_receipts_and_precise_tier_labels(self):
        for mode in ("timing","quality"):
            screen.validate(self.report(mode),73,mode)
        for key,value in (("steps",6),("actual_denoise_steps",6),("lora_applied_projections",226),
                          ("runtime_precision","bf16+vae_fp16_conv_fp32_state"),
                          ("encoder_runtime_backend","coreml"),("prompt_cache_hit",False),
                          ("hybrid",{"active":True}), ("student_ffn_reuse",{"enabled":True})):
            bad=self.report(); bad["runs"][3]["metrics"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                screen.validate(bad,73)
        bad=self.report(); bad["runs"][3]["metrics"]["plan"]["algorithm_approximations"]=[
            "qwen21_viggle_v021_r128_6step_distillation"]
        with self.assertRaises(ValueError): screen.validate(bad,73)

    def test_each_tier_switch_repopulates_edit_prefix_before_scoring(self):
        sequence=screen.points(73)
        previous=None
        for point in sequence[8:]:
            if point["steps"]!=previous:
                self.assertIs(point["prefix"],False)
                self.assertFalse(point["score"])
            if point["score"]: self.assertIs(point["prefix"],True)
            previous=point["steps"]
        for index in (9,11,14,17):
            bad=self.report()
            bad["runs"][index]["metrics"]["acceleration_selection"]=bad["runs"][index]["metrics"][
                "acceleration_selection"].replace("snapshot hit","snapshot miss")
            with self.assertRaises(ValueError): screen.validate(bad,73)

    def test_requests_change_only_sampler_not_precision_or_inputs(self):
        args=SimpleNamespace(seed=73,mode="timing",output=Path("/tmp/results"),
                             lora=Path("/tmp/r128.safetensors"),reference=Path("/tmp/ref.png"))
        values=screen.requests(args)
        for a,b in ((2,3),(5,4),(9,11),(14,12)):
            normalize=lambda x:{k:v for k,v in x.items() if k not in ("steps","output")}
            self.assertEqual(normalize(values[a]),normalize(values[b]))
            self.assertEqual((values[a]["steps"],values[b]["steps"]),(6,5))
        for i,request in enumerate(values):
            self.assertEqual("inputs" in request,i>=8)
            self.assertEqual("qwen21_reference_size" in request,i>=8)
            self.assertNotIn("dump_tensors",request)
        for steps in (5,6):
            for case in ("generate","edit"):
                self.assertEqual(sum(x["score"] and x["steps"]==steps and x["case"]==case
                                     for x in screen.points(73)),3)

    def test_quality_matrix_is_frozen_paired_and_not_a_speed_cohort(self):
        points=screen.points(73,"quality")
        self.assertEqual(len(points),28)
        self.assertEqual(len({p["case_id"] for p in points}),7)
        self.assertTrue(all(x["prefix"] is None and not x["score"] for x in points))
        for i in range(0,len(points),4):
            self.assertEqual([x["steps"] for x in points[i:i+4]],[6,5,5,6])
            self.assertEqual([x["seed"] for x in points[i:i+4]],[73,73,74,74])
            self.assertEqual([x["conditioning_hit"] for x in points[i:i+4]],[False,True,True,True])

    def test_missing_and_nonfinite_results_fail_closed(self):
        bad=self.report(); bad["runs"].pop()
        with self.assertRaises(ValueError): screen.validate(bad,73)
        for value in (True,0,-1,float("nan"),float("inf")):
            bad=copy.deepcopy(self.report()); bad["runs"][3]["request_wall_including_export"]=value
            with self.assertRaises(ValueError): screen.validate(bad,73)


if __name__=="__main__": unittest.main()
