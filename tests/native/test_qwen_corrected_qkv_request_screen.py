"""Host-only whole-request work/bank lifecycle gates, not model qualification."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_corrected_qkv_request_screen as screen
import qwen_corrected_qkv_quality_matrix as matrix
import qwen_corrected_qkv_storage_switch as storage_switch


class CorrectedQkvRequestTests(unittest.TestCase):
    def report(self,sequence):
        rows=[]
        for i,p in enumerate(sequence):
            m=dict(model="qwen-image-2.1",operation="image."+p["case"],width=512,height=512,steps=6,actual_denoise_steps=6,
                seed=p["seed"],lora_applied_projections=227,runtime_backend="mlx_cpp_metal",runtime_precision="bf16",
                encoder_runtime_backend="mlx_cpp_metal",lora_strategy="inference_time",prompt_cache_hit=p["conditioning_hit"],
                acceleration_selection="joint BF16 LoRA A/B operands; "+screen.SIX)
            if p["on"]:m["qwen_corrected_qkv"]=dict(enabled=True,recipe="decode-only-corrected-qkv-v1",completed_decode_blocks=160,
                bank_bytes=3<<30,bank_reused=p["bank_reused"],pack_seconds_this_request=0. if p["bank_reused"] else .1,
                prefill_changed=False,online_lora=True,rank_math="f32")
            if p["on"] and p.get("storage")=="direct":m["qwen_corrected_qkv"].update(recipe=screen.DIRECT_RECIPE,bank_bytes=0,
                bank_reused=False,pack_seconds_this_request=0.,weight_storage="original-buffers")
            if p["on"] and p.get("storage")=="packed":m["qwen_corrected_qkv"]["recipe"]=screen.RECIPE
            rows.append(dict(index=i,request_wall_including_export=10.,metrics=m))
        return dict(runs=rows)

    def test_raw_sequence_includes_both_tasks_optout_and_seed_restoration(self):
        sequence=screen.points("raw","generate","off");self.assertEqual(len(sequence),10)
        self.assertEqual([p["on"] for p in sequence[:5]],[False,True,True,True,False])
        self.assertEqual([p["seed"] for p in sequence[:5]],[73,73,74,73,73])
        self.assertEqual({p["case"] for p in sequence},{"generate","edit"})
        screen.validate(self.report(sequence),sequence)

    def test_fixed_arm_timing_keeps_cold_and_five_completed_hot_requests(self):
        for case in ("generate","edit"):
            for arm in ("off","on"):
                sequence=screen.points("timing",case,arm);self.assertEqual(len(sequence),6)
                screen.validate(self.report(sequence),sequence)

    def test_rounding_fixed_recipe_is_explicit_and_old_receipts_stay_auditable(self):
        sequence=screen.points("raw","generate","off");report=self.report(sequence)
        screen.validate(report,sequence)
        with self.assertRaises(ValueError):screen.validate(report,sequence,screen.RECIPE)
        for row in report["runs"]:
            if row["metrics"].get("qwen_corrected_qkv"):row["metrics"]["qwen_corrected_qkv"]["recipe"]=screen.RECIPE
        screen.validate(report,sequence,screen.RECIPE)

    def test_broad_matrix_freezes_four_generation_three_edit_tasks_and_both_seeds(self):
        sequence=matrix.points();self.assertEqual(len(sequence),28)
        self.assertEqual(len({p["case_id"] for p in sequence if p["case"]=="generate"}),4)
        self.assertEqual(len({p["case_id"] for p in sequence if p["case"]=="edit"}),3)
        for start in range(0,28,4):
            group=sequence[start:start+4]
            self.assertEqual([p["on"] for p in group],[False,True,True,False])
            self.assertEqual([p["seed"] for p in group],[73,73,74,74])
            self.assertEqual([p["bank_reused"] for p in group],[False,False,True,False])
            self.assertEqual(len({p["prompt"] for p in group}),1)
        report=self.report(sequence)
        for row in report["runs"]:
            if row["metrics"].get("qwen_corrected_qkv"):row["metrics"]["qwen_corrected_qkv"]["recipe"]=screen.RECIPE
        screen.validate(report,sequence,screen.RECIPE)

    def test_missing_blocks_or_wrong_bank_cost_and_identity_are_rejected(self):
        sequence=screen.points("raw","generate","off")
        for key,value in (("completed_decode_blocks",32),("completed_decode_blocks",160.),("bank_bytes",0),
                          ("bank_reused",True),("pack_seconds_this_request",0),("pack_seconds_this_request",-.1),
                          ("prefill_changed",True),("online_lora",False),("rank_math","fp16")):
            report=self.report(sequence);report["runs"][1]["metrics"]["qwen_corrected_qkv"][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):screen.validate(report,sequence)

    def test_control_cannot_claim_fusion_and_complete_png_request_cannot_be_skipped(self):
        sequence=screen.points("raw","generate","off")
        report=self.report(sequence);report["runs"][0]["metrics"]["qwen_corrected_qkv"]={"enabled":True}
        with self.assertRaises(ValueError):screen.validate(report,sequence)
        for key,value in (("actual_denoise_steps",5),("lora_applied_projections",226),("prompt_cache_hit",False),("hybrid",{"enabled":True})):
            report=self.report(sequence);report["runs"][1]["metrics"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):screen.validate(report,sequence)
        report=self.report(sequence);report["runs"].pop()
        with self.assertRaises(ValueError):screen.validate(report,sequence)

    def test_direct_receipt_cannot_claim_packing_or_a_bank_hit(self):
        sequence=screen.points("raw","generate","off",True);screen.validate(self.report(sequence),sequence,screen.DIRECT_RECIPE)
        for key,value in (("bank_bytes",3<<30),("bank_reused",True),("pack_seconds_this_request",.1),("weight_storage","packed-bank"),("recipe",screen.RECIPE)):
            report=self.report(sequence);report["runs"][1]["metrics"]["qwen_corrected_qkv"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):screen.validate(report,sequence,screen.DIRECT_RECIPE)
        sequence=matrix.points(True);self.assertEqual(len(sequence),28);screen.validate(self.report(sequence),sequence,screen.DIRECT_RECIPE)

    def test_mixed_storage_switch_rebuilds_after_direct_instead_of_claiming_a_hit(self):
        sequence=storage_switch.points();self.assertEqual([p["storage"] for p in sequence],["off","packed","direct","direct","direct","packed","off"])
        self.assertTrue(all(p["bank_reused"] is False for p in sequence));screen.validate(self.report(sequence),sequence)
        report=self.report(sequence);report["runs"][5]["metrics"]["qwen_corrected_qkv"]["bank_reused"]=True
        with self.assertRaises(ValueError):screen.validate(report,sequence)


if __name__=="__main__":unittest.main()
