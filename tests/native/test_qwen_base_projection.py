"""CPU-only FFN projection admission/recipe/full-block sample controls."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import collect
import ffn_projection_collect as screen
import ffn_projection_audit as block_audit
import projection_request_audit as requests


def fixture():
    w=dict(partition_axis="intermediate_channels",ane_channels=6144,gpu_channels=6144,executor_backend="private_ane",
        data_path="w8a8_hadamard",fp32_channel_join_enabled=True,prefetch_enabled=False,weight_code_cache=dict(enabled=False),
        fixed_program_requested=True,fixed_program_enabled=True,program_disabled=False,program_instances_session_total=1,
        blocked_rebuilds_session_total=0,overflow_retries_session_total=0,headroom_scale=1,fallback_blocks_session_total=0,
        hybrid_blocks_session_total=15,source_recipe="sylvester:w8-projection-single-v1+fp32-partial-join-v1")
    return dict(status="complete",complete_request=False,image_export=False,projection="single",repeats=8,warmups=2,
        completed_full_dit_steps=1,stopped_step=1,stopped_layer=0,gpu_staged_exact=True,source_unchanged=True,
        fresh_weights_inputs_outputs_verified=True,
        actual_source_request=dict(model="qwen-image-2.1",operation="image.generate",width=512,height=512,steps=40,seed=73,audio=False,lora_count=0,reference_count=0,reference_size=1024),
        metrics=dict(runtime_failed=False,runtime_calls_session_total=15,runtime_weight=w),
        **{name:dict(relative_l2=.004,elements=4194304) for name in ("complete_block_quality","alternate_quality","other_weight_quality")},
        samples=[dict(arm=screen.ARMS[(j+rep+3)%3],rep=rep,warmup=rep<0,input_variant=bool(rep&1),seconds=.03)
                 for rep in range(-2,8) for j in range(3)])


class ProjectionProfileTests(unittest.TestCase):
    def test_only_projection_changes_in_same_fixed_environment(self):
        original=collect.environment("runtime","stock",6144,False,False,True,True)
        for projection in collect.FFN_PROJECTIONS[1:]:
            changed=collect.environment("runtime","stock",6144,False,False,True,True,projection)
            self.assertEqual(set(original),set(changed))
            self.assertEqual({k for k in original if original[k]!=changed[k]},{"TURBOCIDER_PRIVATE_ANE_W8_PROJECTION"})
        for route,mode,fixed in (("gpu","stock",False),("cache","stock",True),("runtime","runtime-layers",True),("runtime","stock",False)):
            with self.assertRaises(ValueError):collect.environment(route,mode,4096,False,False,False,fixed,"single")

    def test_actual_recipe_must_match_and_default_cannot_hide_new_reduction(self):
        metric=dict(hybrid=dict(runtime_weight=dict(source_recipe="original")))
        collect.validate_projection_receipt([metric],"descriptor")
        for name in collect.FFN_PROJECTIONS[1:]:
            with self.assertRaises(ValueError):collect.validate_projection_receipt([metric],name)
        metric["hybrid"]["runtime_weight"]["source_recipe"]="original:w8-projection-single-v1+fp32-partial-join-v1"
        collect.validate_projection_receipt([metric],"single")
        with self.assertRaises(ValueError):collect.validate_projection_receipt([metric],"descriptor")

    def test_complete_block_samples_do_not_claim_full_images(self):
        report=fixture();result=screen.validate(report,8,"single")
        self.assertEqual(result["measured_blocks"],24)
        for change in (lambda r:r.update(complete_request=True),lambda r:r.update(completed_full_dit_steps=40),
            lambda r:r["metrics"]["runtime_weight"].update(program_instances_session_total=2),
            lambda r:r["metrics"]["runtime_weight"].update(source_recipe="original"),
            lambda r:r["actual_source_request"].update(lora_count=1),lambda r:r["samples"][0].update(warmup=False),
            lambda r:r["samples"].reverse()):
            value=copy.deepcopy(report);change(value)
            with self.assertRaises(ValueError):screen.validate(value,8,"single")

    def test_complete_requests_change_only_ffn_reduction_not_quality_or_geometry(self):
        with tempfile.TemporaryDirectory(prefix="tc-projection-request-audit-") as directory:
            records=[]
            for projection in collect.FFN_PROJECTIONS:
                root=Path(directory)/projection;(root/"trial").mkdir(parents=True)
                for i in range(3):
                    (root/f"trial/request-{i}.json").write_text(json.dumps(dict(seed=(73,74,73)[i],steps=40,width=512,
                        output=str(root/f"{i}.png"),allow_approximation=True)))
                records.append(dict(root=root,summary=dict(ffn_projection=projection,case="generate",mode="stock",
                    channels=6144,private_qkv=True,fixed_program=True,resident_prefix=False,runtime_prefix_snapshot=False,
                    environment=collect.environment("runtime","stock",6144,False,False,True,True,projection),
                    assets=dict(library="same",manifest="same",reference="same"))))
            for value in records[1:]:requests.matched(records[0],value,value["summary"]["ffn_projection"])
            for key,value in (("case","edit"),("fixed_program",False),("private_qkv",False),("channels",5120),
                              ("ffn_projection","descriptor"),("runtime_prefix_snapshot",True),("mode","timeline")):
                changed=copy.deepcopy(records[:2]);changed[1]["summary"][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):requests.matched(*changed,"k4096")
            changed=copy.deepcopy(records[:2]);changed[1]["summary"]["environment"]["TURBOCIDER_PRIVATE_ANE_PREFETCH"]="1"
            with self.assertRaises(ValueError):requests.matched(*changed,"k4096")
            changed=copy.deepcopy(records[:2]);changed[1]["summary"]["assets"]["reference"]="different"
            with self.assertRaises(ValueError):requests.matched(*changed,"k4096")
            request=records[1]["root"]/"trial/request-1.json"
            value=json.loads(request.read_text());value["seed"]=75;request.write_text(json.dumps(value))
            with self.assertRaises(ValueError):requests.matched(*records[:2],"k4096")

    def test_mixed_block_recipes_are_explicit_not_an_alias_for_whole_single(self):
        for name in ("up-single","down-single"):
            report=fixture();report["projection"]=name
            report["metrics"]["runtime_weight"]["source_recipe"]="sylvester:w8-projection-"+name+"-v1+fp32-partial-join-v1"
            self.assertEqual(screen.validate(report,8,name)["native_workload_predictions"],15)
            with self.assertRaises(ValueError):screen.validate(report,8,"single")
            for other in collect.FFN_PROJECTIONS:
                if other!=name:
                    with self.assertRaises(ValueError):collect.validate_projection_receipt([dict(hybrid=report["metrics"])],other)

    def test_same_recipe_timeline_requires_exact_requests_and_all_pngs(self):
        with tempfile.TemporaryDirectory(prefix="tc-projection-replay-audit-") as directory:
            records=[]
            for mode in ("stock","timeline"):
                root=Path(directory)/mode;(root/"trial").mkdir(parents=True)
                for i in range(3):(root/f"trial/request-{i}.json").write_text(json.dumps(dict(seed=(73,74,73)[i],output=str(root/f"{i}.png"))))
                records.append(dict(root=root,images=[np.zeros((2,2,4),dtype=np.uint8) for _ in range(3)],
                    summary=dict(ffn_projection="single",case="edit",mode=mode,channels=6144,private_qkv=True,
                        fixed_program=True,resident_prefix=False,runtime_prefix_snapshot=True,
                        environment=collect.environment("runtime",mode,6144,True,False,True,True,"single"),
                        assets=dict(library="same",manifest="same",reference="same"))))
            requests.repeated(*records,"single","timeline")
            changed=copy.deepcopy(records);changed[1]["images"][2][0,0,0]=1
            with self.assertRaises(ValueError):requests.repeated(*changed,"single","timeline")
            changed=copy.deepcopy(records);changed[1]["summary"]["assets"]["library"]="different"
            with self.assertRaises(ValueError):requests.repeated(*changed,"single","timeline")

    def test_legacy_frozen_gpu_requires_exact_envelope_and_complete_ordered_runs(self):
        runs=[dict(index=i,request_wall_including_export=40.) for i in range(3)]
        legacy=dict(engine="gpu",constructor_seconds=.01,runs=runs)
        current=dict(status="complete",route="gpu",runs=runs)
        for value in (legacy,current):requests.frozen_gpu_report_contract(value)
        for value in (dict(runs=runs),{**legacy,"engine":"runtime"},{**legacy,"unknown":True},
                      {**current,"status":"failed"},{**current,"route":"runtime"},{**current,"runs":runs[:2]},
                      {**current,"runs":list(reversed(runs))},
                      {**current,"runs":[{**runs[0],"request_wall_including_export":float("nan")},*runs[1:]]}):
            with self.assertRaises(ValueError):requests.frozen_gpu_report_contract(value)

    def test_factorial_uses_only_matched_stock_and_exact_source_replays(self):
        records=[]
        for policy,seconds in (("descriptor",.025),("up-single",.026),("down-single",.027),("single",.029)):
            records.append(dict(cohort=policy,projection=policy,case="generate",mode="stock",
                array_sha256={**dict.fromkeys(("ffn_input","block_reference","alternate_reference","other_weight_reference"),"same"),
                              "block_candidate":policy},analysis=dict(measured_blocks=24,median_seconds={"private-restaged":seconds})))
        repeated=copy.deepcopy(records[1]);repeated["cohort"]="up-repeat";repeated["analysis"]["measured_blocks"]=48
        records.append(repeated)
        value=block_audit.factorial(records)
        self.assertAlmostEqual(value["up_only_delta_seconds"],.001)
        self.assertAlmostEqual(value["down_only_delta_seconds"],.002)
        self.assertAlmostEqual(value["interaction_seconds"],.001)
        self.assertTrue(value["same_policy_replays_exact"])
        for change in (lambda r:r.reverse(),lambda r:r[0].update(mode="timeline"),
            lambda r:r[1]["analysis"].update(measured_blocks=48),
            lambda r:r[2]["array_sha256"].update(ffn_input="changed"),
            lambda r:r[4]["array_sha256"].update(block_candidate="changed"),lambda r:r[2].update(case="edit")):
            changed=copy.deepcopy(records);change(changed)
            with self.assertRaises(ValueError):block_audit.factorial(changed)


if __name__=="__main__":unittest.main()
