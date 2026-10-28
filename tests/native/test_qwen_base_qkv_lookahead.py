"""CPU-only strict lookahead environment, receipt and state-probe gates."""
from collections import Counter
import copy
from pathlib import Path
import sys
import unittest
import tempfile
import json
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import collect
import qkv_lookahead_collect as state
import qkv_lookahead_audit as audit
from test_qwen_base_joint_profile import metrics
from test_qwen_base_profiling import fixture


def receipts(enabled=True,prefix=False):
    rows=metrics(prefix);expected=previous=0
    for row in rows:
        current=row["qwen_private_qkv"]["predictions_session_total"]
        if enabled:expected+=current-previous-(current-previous==1280)
        previous=current
        row["qwen_private_qkv_prefetch"]=dict(requested=enabled,enabled=enabled,future_pending=False,
            **{k+"_session_total":3*expected if k=="producer_submissions" else expected
               for k in ("attempts","submissions","producer_submissions","claims","weight_completions","completed_hits")},
            discards_session_total=0,retired_claims_session_total=0,failures_session_total=0,
            weight_join_wait_seconds_session_total=current/10000,producer_lifetime_seconds_session_total=current/1000 if enabled else 0)
    return rows


class LookaheadTests(unittest.TestCase):
    def test_legacy_environment_unchanged_explicit_off_on_changes_only_new_key(self):
        kwargs=dict(private_qkv=True,fixed_program=True)
        legacy=collect.environment("runtime","stock",6144,**kwargs)
        off=collect.environment("runtime","stock",6144,**kwargs,private_qkv_prefetch=False)
        on=collect.environment("runtime","stock",6144,**kwargs,private_qkv_prefetch=True)
        self.assertNotIn("TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH",legacy)
        self.assertEqual(set(off)-set(legacy),{"TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH"})
        self.assertEqual({k for k in off if off[k]!=on[k]},{"TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH"})
        for route,mode,private,fixed,resident in (("gpu","stock",True,True,False),("cache","stock",True,True,False),
            ("runtime","runtime-layers",True,True,False),("runtime","stock",False,True,False),
            ("runtime","stock",True,False,False),("runtime","stock",True,True,True)):
            with self.assertRaises(ValueError):
                collect.environment(route,mode,4096,False,resident,private,fixed,private_qkv_prefetch=True)
        with self.assertRaises(ValueError):collect.environment("runtime","stock",private_qkv_prefetch=1)

    def test_complete_regular_and_target_only_prefill_balance(self):
        for enabled in (False,True):
            for prefix in (False,True):collect.validate_private_qkv_prefetch(receipts(enabled,prefix),enabled)
        self.assertEqual(receipts(True,True)[-1]["qwen_private_qkv_prefetch"]["completed_hits_session_total"],1248+1279*2)
        with self.assertRaises(ValueError):collect.validate_private_qkv_prefetch(receipts(False),True)

    def test_all_counters_drains_failure_and_times_are_strict(self):
        original=receipts()
        values=original[1]["qwen_private_qkv_prefetch"]
        for name,value in values.items():
            bad=copy.deepcopy(original)
            bad[1]["qwen_private_qkv_prefetch"][name]=(not value) if type(value) is bool else float("nan") if type(value) is float else value+1
            with self.subTest(name=name),self.assertRaises(ValueError):collect.validate_private_qkv_prefetch(bad,True)
        bad=receipts();del bad[0]["qwen_private_qkv_prefetch"]["claims_session_total"]
        with self.assertRaises(ValueError):collect.validate_private_qkv_prefetch(bad,True)
        bad=receipts();bad[1]["qwen_private_qkv"]["predictions_session_total"]+=32
        with self.assertRaises(ValueError):collect.validate_private_qkv_prefetch(bad,True)
        bad=receipts(False);bad[1]["qwen_private_qkv_prefetch"]["producer_lifetime_seconds_session_total"]=.1
        with self.assertRaises(ValueError):collect.validate_private_qkv_prefetch(bad,False)

    def test_synthetic_state_receipt_has_no_silent_fallback_or_incomplete_cleanup(self):
        report=dict(status="complete",synthetic_only=True,complete_request=False,image_export=False,predictions=5,validation_calls=2,
                    no_fallback=True,source_unchanged=True,old_output_unchanged=True,same_program_key="fixed",
                    prefetch=dict(state.COUNTS,attempts=6,requested=True,enabled=False,future_pending=False))
        state.validate(report)
        for key,value in report["prefetch"].items():
            bad=copy.deepcopy(report);bad["prefetch"][key]=not value if type(value) is bool else value+1
            with self.subTest(key=key),self.assertRaises(ValueError):state.validate(bad)
        for key in ("no_fallback","source_unchanged","old_output_unchanged"):
            bad=copy.deepcopy(report);bad[key]=False
            with self.assertRaises(ValueError):state.validate(bad)

    def test_synthetic_actual_selector_scopes_are_exact(self):
        raw,_=fixture();template=next(r for r in raw["events"] if r["name"]=="ane.evaluate_api")
        raw["events"]=[r for r in raw["events"] if r["request"]==-2]
        for i in range(1,6):raw["events"].append(dict(template,request=i))
        raw["counts"]=dict(Counter(r["name"] for r in raw["events"]))
        state.observer_contract(raw)
        for key,value in (("model",9),("request",6),("name","ane.load_api")):
            bad=copy.deepcopy(raw);bad["events"][-1][key]=value;bad["counts"]=dict(Counter(r["name"] for r in bad["events"]))
            with self.assertRaises(ValueError):state.observer_contract(bad)

    def test_full_diagonal_oracle_checks_all_rows_and_restore_arrays(self):
        a=np.full((1,1024,4096),.5,dtype=np.float32);b=np.full_like(a,.375)
        x=np.concatenate([a*g for g in (.125,.25,.5)],axis=-1)
        y=np.concatenate([b*g for g in (.1875,.375,.75)],axis=-1)
        values=dict(input_a=a,input_b=b,output_a=x,output_b=y,restored_a=x,mismatch_a=x,generation_a=x)
        audit.synthetic_arrays(values)
        bad=dict(values);bad["generation_a"]=x.copy();bad["generation_a"][0,1023,12287]=3
        with self.assertRaises(ValueError):audit.synthetic_arrays(bad)
        bad=dict(values);bad["output_b"]=y.copy();bad["output_b"][0,1023,12287]=1000
        with self.assertRaises(ValueError):audit.synthetic_arrays(bad)
        bad=dict(values);bad["output_b"]=y.copy();bad["output_b"][0,0,0]+=.001
        with self.assertRaises(ValueError):audit.synthetic_arrays(bad)

    def test_matched_full_rgba_gate_rejects_one_pixel_and_unrelated_environment_change(self):
        with tempfile.TemporaryDirectory() as directory:
            values=[]
            for enabled in (False,True):
                root=Path(directory)/str(enabled);(root/"trial").mkdir(parents=True)
                for i in range(3):(root/f"trial/request-{i}.json").write_text(json.dumps(dict(seed=(73,74,73)[i],output=str(root/f"{i}.png"))))
                values.append(dict(root=root,images=[np.full((2,2,4),i,dtype=np.uint8) for i in range(3)],
                    summary=dict(mode="stock",environment=collect.environment("runtime","stock",6144,private_qkv=True,fixed_program=True,private_qkv_prefetch=enabled),
                                 assets=dict(library="same",manifest="same",reference="same")),receipt=dict(hot_median_seconds=30)))
            audit.pair(*values)
            bad=copy.deepcopy(values[1]);bad["images"][1][1,1,3]+=1
            with self.assertRaises(ValueError):audit.pair(values[0],bad)
            bad=copy.deepcopy(values[1]);bad["summary"]["environment"]["TURBOCIDER_PRIVATE_ANE_CHANNELS"]="5120"
            with self.assertRaises(ValueError):audit.pair(values[0],bad)
            bad=copy.deepcopy(values[1]);bad["images"].pop()
            with self.assertRaises(ValueError):audit.pair(values[0],bad)

    def test_frozen_controls_are_only_matched_requests_and_exact_complete_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            current=Path(directory)/"current";previous=Path(directory)/"frozen"
            for root in (current,previous):
                (root/"trial").mkdir(parents=True);(root/"summary.json").write_text("{}")
                for i in range(3):(root/f"trial/request-{i}.json").write_text(json.dumps(dict(seed=(73,74,73)[i],output=str(root/f"{i}.png"))))
            images=[np.full((512,512,4),i,dtype=np.uint8) for i in range(3)]
            for i,x in enumerate(images):Image.fromarray(x).save(previous/f"trial/{i}.png")
            value=dict(root=current,images=images);audit.frozen(value,previous)
            wrong=images[2].copy();wrong[0,0,0]+=1;Image.fromarray(wrong).save(previous/"trial/2.png")
            with self.assertRaises(ValueError):audit.frozen(value,previous)
            Image.fromarray(images[2]).save(previous/"trial/2.png")
            (previous/"trial/request-1.json").write_text(json.dumps(dict(seed=73)))
            with self.assertRaises(ValueError):audit.frozen(value,previous)


if __name__=="__main__":unittest.main()
