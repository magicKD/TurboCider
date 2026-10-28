"""Host-only whole-block denominator, complete LoRA and numerical gates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_qkv_lora_block_screen as screen


class CorrectedQkvScreenTests(unittest.TestCase):
    def report(self):
        def q(exact):return dict(relative_l2=0. if exact else .0005,cosine=1.,max_abs=0. if exact else .01,
                                 changed=0 if exact else 100,elements=1024*4096)
        result=dict(status="measured",runtime_lora_applied_terms=227,image_export=False,borrowed_values_unchanged=True,
            fresh_arguments_verified=True,all_outputs_lazy_before_eval=True,completed_full_dit_steps=1,stopped_step=1,stopped_layer=0,
            packed_base_bytes=100663296,initial_quality=[dict(arm=name,quality=q(name=="gpu-replay"),first_replay_seconds=.1) for name in screen.NAMES[1:]],samples=[])
        for i in range(-3,12):
            for visit in range(6):
                name=screen.NAMES[(visit+i+6)%6]
                result["samples"].append(dict(iteration=i,arm=name,seconds=.035 if name=="gpu-original" else .033,
                                               quality=q(name in screen.NAMES[:2])))
        return result

    def joint_report(self):
        r=self.report();r.update(qkv_recipe_version="joint-base-a-v1",packed_joint_base_a_bytes=103809024,measured_samples_per_arm=18)
        def q(exact,elements=1024*4096):return dict(relative_l2=0. if exact else .0005,cosine=1.,max_abs=0. if exact else .01,
                                                   changed=0 if exact else 100,elements=elements)
        r["initial_quality"]=[dict(arm=name,quality=q(name=="gpu-replay"),first_replay_seconds=.1) for name in screen.JOINT_NAMES[1:]]
        r["rank_boundary_quality"]=[dict(projection=i,mlx=q(True,1024*128),mpp=q(True,1024*128)) for i in range(3)]
        r["samples"]=[]
        for i in range(-3,18):
            for visit in range(9):
                name=screen.JOINT_NAMES[(visit+i+9)%9]
                r["samples"].append(dict(iteration=i,arm=name,seconds=.035 if name=="gpu-original" else .033,
                                          quality=q(name in screen.NAMES[:2])))
        return r

    def test_joint_source_boundary_and_counter_scope_are_explicit(self):
        result=screen.analyze(self.joint_report())
        self.assertEqual(result["primary_recipe"],"joint-base-a-mpp-b32")
        self.assertTrue(all(len(v)==18 for v in result["whole_block_seconds"].values()))
        for key,value in (("packed_joint_base_a_bytes",0),("measured_samples_per_arm",12),("qkv_recipe_version","other")):
            r=self.joint_report();r[key]=value
            with self.assertRaises(ValueError):screen.analyze(r)
        r=self.joint_report();r["rank_boundary_quality"][0]["mpp"]["relative_l2"]=.0051
        with self.assertRaises(ValueError):screen.analyze(r)
        r=self.joint_report();r["rank_boundary_quality"][0]["mpp"]["cosine"]=True
        with self.assertRaises(ValueError):screen.analyze(r)

    def test_original_block_is_the_only_performance_denominator(self):
        r=screen.analyze(self.report())
        self.assertEqual(r["primary_recipe"],"fused-b32");self.assertTrue(r["primary_local_gate_passed"])
        self.assertAlmostEqual(r["reduction_vs_original_gpu_percent"]["fused-b32"],100*(1-.033/.035))
        self.assertTrue(all(len(v)==12 for v in r["whole_block_seconds"].values()))

    def test_incomplete_or_mutated_borrow_and_cached_output_rejected(self):
        for key,value in (("runtime_lora_applied_terms",226),("image_export",True),("fresh_arguments_verified",False),
                          ("borrowed_values_unchanged",False),("all_outputs_lazy_before_eval",False),("stopped_step",2),
                          ("packed_base_bytes",0)):
            r=self.report();r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):screen.analyze(r)

    def test_every_initial_warm_and_measured_quality_is_required(self):
        r=self.report();r["initial_quality"][0]["quality"]["changed"]=1
        with self.assertRaises(ValueError):screen.analyze(r)
        for value in (float("nan"),float("inf"),0,-1,True):
            r=self.report();r["samples"][0]["seconds"]=value
            with self.assertRaises(ValueError):screen.analyze(r)
        for index in (0,6,89):
            r=self.report();r["samples"][index]["quality"]["relative_l2"]=.00101
            with self.assertRaises(ValueError):screen.analyze(r)

    def test_missing_or_reordered_arms_do_not_pass(self):
        r=self.report();r["samples"].pop()
        with self.assertRaises(ValueError):screen.analyze(r)
        r=self.report();r["samples"].reverse()
        with self.assertRaises(ValueError):screen.analyze(r)


if __name__=="__main__":unittest.main()
