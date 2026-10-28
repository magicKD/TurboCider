"""Host contracts for explicit two-program QKV+FFN profiling."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/profiling/qwen_base"))
import analyze
import collect
import joint_audit
from test_qwen_base_profiling import fixture


def metrics(prefix=False):
    result=[]
    predictions=gpu=0
    for i in range(3):
        predictions+=1280 if prefix and i else 1248
        gpu+=0 if prefix and i else 32
        result.append(dict(hybrid=dict(runtime_weight=dict(estimated_bytes=1000 << 20)),
            qwen_private_qkv=dict(failed=False,failure_reason="",validation_passed=True,
                validation_calls_session_total=1,validation_relative_l2=.0035,
                predictions_session_total=predictions,completed_blocks_session_total=predictions,
                weight_stages_session_total=predictions,gpu_blocks_session_total=gpu,fallback_blocks_session_total=0,
                slot_bytes=65814528,estimated_bytes=288 << 20,program_key="fixed-qkv",
                stage_lifetime_seconds_session_total=i+1,activation_seconds_session_total=i+1,join_seconds_session_total=i+1)))
    return result


class JointProfileTests(unittest.TestCase):
    def test_only_qkv_switch_changes_matched_environments(self):
        for prefix in (False,True):
            off=collect.environment("runtime","stock",6144,prefix)
            on=collect.environment("runtime","stock",6144,prefix,False,True)
            self.assertEqual(set(off),set(on))
            self.assertEqual({k for k in off if off[k]!=on[k]}, {"TURBOCIDER_QWEN21_PRIVATE_QKV"})
        for route,mode,resident in (("gpu","stock",False),("cache","stock",False),("runtime","runtime-layers",False),("runtime","stock",True)):
            with self.assertRaises(ValueError):
                collect.environment(route,mode,4096,False,resident,True)

    def test_cumulative_decode_and_prefix_hits_are_explicit(self):
        collect.validate_private_qkv(metrics())
        collect.validate_private_qkv(metrics(True),True)
        with self.assertRaises(ValueError):
            collect.validate_private_qkv(metrics(True))

    def test_fallback_bad_validation_wrong_counts_budget_and_new_models_rejected(self):
        for name,value in (("failed",True),("failure_reason","fallback"),("validation_passed",False),
                           ("validation_calls_session_total",2),("validation_relative_l2",float("nan")),
                           ("validation_relative_l2",.03),("predictions_session_total",1),
                           ("completed_blocks_session_total",1),("weight_stages_session_total",1),
                           ("fallback_blocks_session_total",1),("gpu_blocks_session_total",0),
                           ("slot_bytes",0),("estimated_bytes",2 << 30),("program_key","new-key"),
                           ("activation_seconds_session_total",-1)):
            rows=metrics();rows[1]["qwen_private_qkv"][name]=value
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                collect.validate_private_qkv(rows)

    def test_two_models_require_explicit_contract_and_native_per_model_counts(self):
        value,runs=fixture()
        for i,run in enumerate(runs):
            run["metrics"]=dict(hybrid=dict(runtime_calls_session_total=1 if i==0 else 2),
                                qwen_private_qkv=dict(predictions_session_total=1 if i==0 else 2))
            event=next(r for r in value["events"] if r["request"]==i and r["name"]=="ane.evaluate_api")
            qkv=copy.deepcopy(event);qkv["model"]=2;value["events"].append(qkv)
            if i==0:
                value["events"].extend(copy.deepcopy(event) for _ in range(3))
        value["counts"]=dict(Counter(r["name"] for r in value["events"]))
        with self.assertRaises(ValueError):
            analyze.observer_contract(value,runs,True)
        result=analyze.observer_contract(value,runs,True,expected_models=2)
        self.assertEqual(result["runtime_model_count"],2)
        self.assertTrue(result["hot_no_compile_observed"])
        runs[1]["metrics"]["qwen_private_qkv"]["predictions_session_total"]+=1
        with self.assertRaises(ValueError):
            analyze.observer_contract(value,runs,True,expected_models=2)

    def test_independent_audit_requires_the_same_requests_and_only_qkv_switch(self):
        with tempfile.TemporaryDirectory(prefix="tc-joint-match-") as directory:
            records=[]
            for enabled in (False,True):
                root=Path(directory)/str(enabled);(root/"trial").mkdir(parents=True)
                for i in range(3):
                    (root/f"trial/request-{i}.json").write_text(json.dumps(dict(seed=(73,74,73)[i],steps=40,output=str(root/f"{i}.png"))))
                records.append(dict(root=root,summary=dict(environment=collect.environment("runtime","stock",6144,False,False,enabled))))
            joint_audit.matched(*records)
            records[1]["summary"]["environment"]["TURBOCIDER_PRIVATE_ANE_CHANNELS"]="5120"
            with self.assertRaises(ValueError):
                joint_audit.matched(*records)
            records[1]["summary"]["environment"]["TURBOCIDER_PRIVATE_ANE_CHANNELS"]="6144"
            request=records[1]["root"]/"trial/request-1.json"
            request.write_text(json.dumps(dict(seed=75,steps=40)))
            with self.assertRaises(ValueError):
                joint_audit.matched(*records)


if __name__=="__main__":
    unittest.main()
