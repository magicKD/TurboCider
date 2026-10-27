"""CPU-only explicit text prefix, real chunk counts and prompt-change controls."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import collect
import text_prefix_audit
import text_qkv_step0_audit
from test_qwen_base_joint_profile import metrics as qkv_metrics


def fixture(matrix=False):
    hits=(False,True,False,False) if matrix else (False,True,True)
    result=[];calls=0
    for i,hit in enumerate(hits):
        text=48 if matrix and i==2 else 24;calls+=1280 if hit else 1312
        result.append(dict(operation="image.generate",reference_tokens=0,text_tokens=text,
            acceleration_selection="; runtime text prefix KV snapshot "+("hit" if hit else "miss"),
            qwen_runtime_prefix_snapshot=dict(admitted=True,hit=hit,published=not hit,bytes=text*524288,recipe_sha256="a"*64,decline_reason=""),
            hybrid=dict(runtime_calls_session_total=calls),
            qwen_ffn_phases=dict(policy="all",prefill=dict(steps_this_request=0 if hit else 1,
                completed_channel_blocks_this_request=0 if hit else 32,actual_rows=0 if hit else 1024+text),
                decode=dict(steps_this_request=40 if hit else 39,completed_channel_blocks_this_request=1280 if hit else 1248,actual_rows=1024))))
    return result


class TextPrefixTests(unittest.TestCase):
    def test_new_flag_changes_only_one_environment_key_and_keeps_legacy_edit_option(self):
        options=dict(runtime_text_prefix=False)
        off=collect.environment("runtime","stock",6144,False,False,True,True,"descriptor","sigmoid-complement",**options)
        on=collect.environment("runtime","stock",6144,False,False,True,True,"descriptor","sigmoid-complement",runtime_text_prefix=True)
        self.assertEqual({k for k in off if off[k]!=on[k]},{"TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_SNAPSHOT"})
        self.assertEqual(on["TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT"],"0")
        self.assertNotIn("TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_SNAPSHOT",collect.environment("runtime","stock",6144))
        for route,mode,fixed,edit,resident in (("gpu","stock",True,False,False),("runtime","gpu-ops",True,False,False),
            ("runtime","stock",False,False,False),("runtime","stock",True,True,False),("runtime","stock",True,False,True)):
            with self.assertRaises(ValueError):collect.environment(route,mode,6144,edit,resident,True,fixed,runtime_text_prefix=True)

    def test_actual_miss_hit_and_matrix_invalidation_coverage(self):
        collect.validate_runtime_text_prefix(fixture())
        collect.validate_runtime_text_prefix(fixture(True),matrix=True)
        for change in (lambda r:r[1]["qwen_runtime_prefix_snapshot"].update(hit=False),
            lambda r:r[0]["qwen_runtime_prefix_snapshot"].update(published=False),
            lambda r:r[1]["qwen_runtime_prefix_snapshot"].update(bytes=4096*524288),
            lambda r:r[1]["qwen_runtime_prefix_snapshot"].update(recipe_sha256="b"*64),
            lambda r:r[1].update(reference_tokens=4096),lambda r:r[1].update(operation="image.edit"),
            lambda r:r[1]["hybrid"].update(runtime_calls_session_total=2624),
            lambda r:r[1]["qwen_ffn_phases"]["decode"].update(steps_this_request=39),
            lambda r:r[0]["qwen_ffn_phases"]["prefill"].update(actual_rows=1024)):
            changed=fixture();change(changed)
            with self.assertRaises(ValueError):collect.validate_runtime_text_prefix(changed)
        changed=fixture(True);changed[2]["qwen_runtime_prefix_snapshot"].update(hit=True,published=False)
        with self.assertRaises(ValueError):collect.validate_runtime_text_prefix(changed,matrix=True)

    def test_gpu_step0_is_separate_policy_and_keeps_only_39_private_qkv_steps(self):
        old=collect.environment("runtime","stock",6144,False,False,True,True,runtime_text_prefix=True,text_prefix_qkv_step0="ane")
        gpu=collect.environment("runtime","stock",6144,False,False,True,True,runtime_text_prefix=True,text_prefix_qkv_step0="gpu")
        self.assertEqual({k for k in old if old[k]!=gpu[k]},{"TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_QKV_STEP0"})
        rows=fixture()
        for row in rows:row["acceleration_selection"]+="; runtime text prefix preserves first-step GPU QKV"
        collect.validate_runtime_text_prefix(rows,qkv_step0="gpu")
        with self.assertRaises(ValueError):collect.validate_runtime_text_prefix(rows)
        with self.assertRaises(ValueError):collect.validate_runtime_text_prefix(fixture(),qkv_step0="gpu")
        collect.validate_private_qkv(qkv_metrics(),True,hit_pattern=[False,False,False])
        with self.assertRaises(ValueError):collect.validate_private_qkv(qkv_metrics(True),True,hit_pattern=[False,False,False])
        for text,private in ((False,True),(True,False)):
            with self.assertRaises(ValueError):collect.environment("runtime","stock",6144,False,False,private,True,runtime_text_prefix=text,text_prefix_qkv_step0="gpu")

    def test_prompt_matrix_uses_real_changed_condition_and_restores_original_request(self):
        args=argparse.Namespace(route="runtime",mode="timeline",channels=6144,case="generate",requests=4,
            scenario="text-prefix-matrix",runtime_text_prefix_snapshot=True,output=Path("/synthetic"),prompt=None,
            model=Path("/model"),manifest=Path("/shape.json"),reference=Path("/reference.png"))
        values=collect.requests(args)
        self.assertEqual([r["seed"] for r in values],[73,74,73,73])
        self.assertEqual(values[0]["prompt"],values[1]["prompt"])
        self.assertNotEqual(values[1]["prompt"],values[2]["prompt"])
        self.assertEqual({k:v for k,v in values[0].items() if k!="output"},{k:v for k,v in values[3].items() if k!="output"})
        self.assertTrue(all(r["steps"]==40 and not r.get("inputs") for r in values))

    def test_independent_audit_rejects_replay_changes_and_false_prompt_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            roots=[Path(directory)/name for name in ("stock","timeline","matrix")]
            for root in roots:(root/"trial").mkdir(parents=True)
            for root in roots:
                for i in range(4 if root==roots[2] else 3):
                    seed=74 if i==1 else 73;prompt="changed" if root==roots[2] and i==2 else "original"
                    (root/f"trial/request-{i}.json").write_text(json.dumps(dict(seed=seed,prompt=prompt,output=str(root/f"{i}.png"))))
            a=np.zeros((2,2,4),dtype=np.uint8);b=np.ones_like(a);c=np.full_like(a,2)
            base=dict(root=roots[0],images=[a,b,c]);same=dict(root=roots[1],images=[a.copy(),b.copy(),c.copy()])
            matrix=dict(root=roots[2],images=[a.copy(),b.copy(),np.full_like(a,3),a.copy()])
            text_prefix_audit.replay(base,same);text_prefix_audit.prompt_matrix(base,matrix)
            changed=copy.deepcopy(same);changed["images"][1][0,0,0]=4
            with self.assertRaises(ValueError):text_prefix_audit.replay(base,changed)
            with self.assertRaises(ValueError):text_prefix_audit.replay(base,{**same,"images":same["images"][:2]})
            changed=copy.deepcopy(matrix);changed["images"][3][0,0,0]=4
            with self.assertRaises(ValueError):text_prefix_audit.prompt_matrix(base,changed)
            path=roots[2]/"trial/request-2.json";value=json.loads(path.read_text());value["seed"]=74;path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):text_prefix_audit.prompt_matrix(base,matrix)

    def test_first_qkv_counterfactual_and_exact_cache_pair_have_distinct_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            values=[]
            for label,text,policy in (("off",False,"ane"),("ane",True,"ane"),("gpu",True,"gpu")):
                root=Path(directory)/label;(root/"trial").mkdir(parents=True)
                for i in range(3):(root/f"trial/request-{i}.json").write_text(json.dumps(dict(prompt="same",seed=(73,74,73)[i],output=str(root/f"{i}.png"))))
                values.append(dict(root=root,images=[np.full((2,2,4),i,dtype=np.uint8) for i in range(3)],summary=dict(
                    environment=collect.environment("runtime","stock",6144,False,False,True,True,runtime_text_prefix=text,text_prefix_qkv_step0=policy),
                    assets=dict(library="same",manifest="same",reference="same"))))
            off,ane,gpu=values
            text_qkv_step0_audit.only_policy(ane,gpu)
            text_qkv_step0_audit.cached_gpu_pair(off,gpu,require_exact=True)
            bad=copy.deepcopy(gpu);bad["images"][2][0,0,0]=9
            with self.assertRaises(ValueError):text_qkv_step0_audit.cached_gpu_pair(off,bad,require_exact=True)
            # Incremental quality may still be recorded when exact acceptance
            # is not requested; cold miss always stays exact.
            text_qkv_step0_audit.cached_gpu_pair(off,bad)
            bad=copy.deepcopy(gpu);bad["summary"]["assets"]["manifest"]="other"
            with self.assertRaises(ValueError):text_qkv_step0_audit.only_policy(ane,bad)
            bad=copy.deepcopy(gpu);bad["summary"]["environment"]["TURBOCIDER_PRIVATE_ANE_CHANNELS"]="5120"
            with self.assertRaises(ValueError):text_qkv_step0_audit.cached_gpu_pair(off,bad)


if __name__=="__main__":unittest.main()
