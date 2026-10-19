"""Host-only exact stage/control and complete block evidence gates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_qkv_numeric_screen as screen


class QkvNumericTests(unittest.TestCase):
    def report(self):
        def q():return dict(relative_l2=0.,cosine=1.,max_abs=0.,changed=0,elements=1024*4096)
        r=dict(status="measured",runtime_lora_applied_terms=227,image_export=False,actual_full_expression_exact=True,
            borrowed_values_unchanged=True,fresh_arguments_verified=True,completed_full_dit_steps=1,stopped_step=1,stopped_layer=0,stage_quality=[],samples=[])
        for recipe in screen.RECIPES:
            for stage in range(4):
                for i in range(3):r["stage_quality"].append(dict(recipe=recipe,stage=stage,projection=i,quality=q()))
            for name,stage in (("norm",2),("rope",3)):
                for i in range(3):r["stage_quality"].append(dict(recipe=recipe+"-matched-"+name,stage=stage,projection=i,quality=q()))
        for iteration in range(-3,15):
            for visit in range(5):r["samples"].append(dict(iteration=iteration,arm=screen.NAMES[(visit+iteration+5)%5],seconds=.035,quality=q()))
        return r

    def test_complete_balanced_stages_and_strong_whole_block_controls(self):
        result=screen.analyze(self.report());self.assertEqual(result["primary"],"native-base-b16")
        self.assertEqual(len(result["stage_summaries"]),4)
        self.assertTrue(all(len(values)==15 for values in result["whole_block_seconds"].values()))

    def test_missing_duplicate_or_nonfinite_stage_is_rejected(self):
        for kind in range(4):
            r=self.report()
            if kind==0:r["stage_quality"].pop()
            if kind==1:r["stage_quality"].append(r["stage_quality"][0])
            if kind==2:r["stage_quality"][0]["quality"]["relative_l2"]=float("nan")
            if kind==3:next(row for row in r["stage_quality"] if row["recipe"]=="native-base-b16" and row["stage"]==0)["quality"]["changed"]=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):screen.analyze(r)

    def test_actual_full_expression_ownership_and_fresh_arguments_are_required(self):
        for key,value in (("actual_full_expression_exact",False),("borrowed_values_unchanged",False),("fresh_arguments_verified",False),
                          ("runtime_lora_applied_terms",226),("image_export",True)):
            r=self.report();r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):screen.analyze(r)
        r=self.report();next(row for row in r["samples"] if row["arm"]=="gpu-replay")["quality"]["changed"]=1
        with self.assertRaises(ValueError):screen.analyze(r)

    def rope_report(self):
        r=self.report();r["numeric_recipe"]="rope-contract-v2"
        for recipe in screen.RECIPES:
            for i in range(3):r["stage_quality"].append(dict(recipe=recipe+"-no-contract",stage=3,projection=i,
                quality=dict(relative_l2=0.,cosine=1.,max_abs=0.,changed=0,elements=1024*4096)))
        r["samples"]=[]
        for iteration in range(-3,12):
            for visit in range(6):r["samples"].append(dict(iteration=iteration,arm=screen.ROPE_NAMES[(visit+iteration+6)%6],seconds=.035,
                quality=dict(relative_l2=0.,cosine=1.,max_abs=0.,changed=0,elements=1024*4096)))
        return r

    def test_scoped_rope_control_does_not_erase_previous_expression(self):
        r=self.rope_report();result=screen.analyze(r);self.assertEqual(result["primary"],"no-contract-prepare")
        self.assertTrue(all(len(values)==12 for values in result["whole_block_seconds"].values()))
        self.assertTrue(all(row["no_contract_changed"]==0 for row in result["stage_summaries"]))
        r["stage_quality"].pop()
        with self.assertRaises(ValueError):screen.analyze(r)


if __name__=="__main__":unittest.main()
