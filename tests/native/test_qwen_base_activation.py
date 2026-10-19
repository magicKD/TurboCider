"""CPU-only activation recipe/geometry/source controls; no device inference."""
import copy
import json
from pathlib import Path
import sys
import unittest
import tempfile

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import collect
import ffn_projection_collect as screen
import ffn_projection_audit as audit
import fixed_program_collect as negative
import activation_request_audit as request_audit
import sequence_audit
import quality_collect
from test_qwen_base_projection import fixture
from test_qwen_base_fixed_program import negative_fixture


class ActivationProfileTests(unittest.TestCase):
    def test_quality_gpu_control_remains_unmodified_and_joint_captures_activation(self):
        self.assertEqual(quality_collect.execution_options("gpu",6144,"sigmoid-complement"),[])
        flags=quality_collect.execution_options("joint",6144,"sigmoid-complement")
        self.assertIn("--private-qkv",flags);self.assertIn("--fixed-program",flags)
        self.assertEqual(flags[flags.index("--ffn-activation")+1],"sigmoid-complement")
        self.assertNotIn("--fixed-program",quality_collect.execution_options("joint",6144,"exp"))

    def test_variable_sequence_does_not_hide_a_changed_activation_recipe(self):
        self.assertEqual(sequence_audit.arithmetic_recipe({},{}),("descriptor","exp"))
        a=dict(ffn_projection="descriptor",ffn_activation="sigmoid-complement",fixed_program=True)
        self.assertEqual(sequence_audit.arithmetic_recipe(a,a),("descriptor","sigmoid-complement"))
        for summary,report in ((a,{}),({**a,"fixed_program":False},a),
                               ({**a,"ffn_projection":"single"},{**a,"ffn_projection":"single"}),
                               (a,{**a,"ffn_activation":"silu"})):
            with self.assertRaises(ValueError):sequence_audit.arithmetic_recipe(summary,report)

    def test_complete_requests_only_change_activation_and_same_recipe_replays_exact(self):
        with tempfile.TemporaryDirectory(prefix="tc-activation-requests-") as directory:
            values=[]
            for name,mode in (("exp","stock"),("sigmoid-complement","stock"),("sigmoid-complement","timeline")):
                root=Path(directory)/(name+mode);(root/"trial").mkdir(parents=True)
                for i in range(3):(root/f"trial/request-{i}.json").write_text(json.dumps(dict(seed=(73,74,73)[i],steps=40,output=str(root/f"{i}.png"))))
                values.append(dict(root=root,images=[np.zeros((2,2,4),dtype=np.uint8) for _ in range(3)],
                    summary=dict(ffn_activation=name,ffn_projection="descriptor",fixed_program=True,private_qkv=True,
                        channels=6144,resident_prefix=False,case="generate",mode=mode,runtime_prefix_snapshot=False,
                        assets=dict(library="same",manifest="same",reference="same"),
                        environment=collect.environment("runtime",mode,6144,False,False,True,True,"descriptor",name))))
            request_audit.matched(values[0],values[1],"sigmoid-complement")
            request_audit.repeated(values[1],values[2],"sigmoid-complement","timeline")
            for key,value in (("ffn_projection","single"),("private_qkv",False),("fixed_program",False),("runtime_prefix_snapshot",True)):
                changed=copy.deepcopy(values[:2]);changed[1]["summary"][key]=value
                with self.assertRaises(ValueError):request_audit.matched(*changed,"sigmoid-complement")
            changed=copy.deepcopy(values[1:]);changed[1]["images"][2][0,0,0]=1
            with self.assertRaises(ValueError):request_audit.repeated(*changed,"sigmoid-complement","timeline")

    def test_exactly_one_immutable_activation_switch_with_no_other_approximation(self):
        original=collect.environment("runtime","stock",6144,False,False,True,True)
        for name in collect.FFN_ACTIVATIONS[1:]:
            value=collect.environment("runtime","stock",6144,False,False,True,True,"descriptor",name)
            self.assertEqual(set(original),set(value))
            self.assertEqual({k for k in original if original[k]!=value[k]},{"TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION"})
        for route,mode,fixed,projection in (("gpu","stock",False,"descriptor"),("cache","stock",True,"descriptor"),
            ("runtime","runtime-layers",True,"descriptor"),("runtime","stock",False,"descriptor"),("runtime","stock",True,"single")):
            with self.assertRaises(ValueError):collect.environment(route,mode,4096,False,False,False,fixed,projection,"silu")

    def test_actual_recipe_cannot_hide_activation_or_combine_with_other_reductions(self):
        for name in collect.FFN_ACTIVATIONS:
            recipe="sylvester"+((":w8-activation-"+name+"-v1") if name!="exp" else "")+"+fp32-partial-join-v1"
            metric=dict(hybrid=dict(runtime_weight=dict(source_recipe=recipe)))
            collect.validate_activation_receipt([metric],name)
            for other in collect.FFN_ACTIVATIONS:
                if other!=name:
                    with self.assertRaises(ValueError):collect.validate_activation_receipt([metric],other)
            if name!="exp":
                metric["hybrid"]["runtime_weight"]["source_recipe"]+=":w8-projection-single-v1"
                with self.assertRaises(ValueError):collect.validate_activation_receipt([metric],name)

    def test_full_block_activation_receipt_and_source_must_match(self):
        for name in collect.FFN_ACTIVATIONS[1:]:
            report=fixture();report.update(projection="descriptor",activation=name)
            report["metrics"]["runtime_weight"]["source_recipe"]="sylvester:w8-activation-"+name+"-v1+fp32-partial-join-v1"
            self.assertEqual(screen.validate(report,8,"descriptor",name)["native_workload_predictions"],15)
            with self.assertRaises(ValueError):screen.validate(report,8,"descriptor")
            changed=copy.deepcopy(report);changed["metrics"]["runtime_weight"]["source_recipe"]="sylvester"
            with self.assertRaises(ValueError):screen.validate(changed,8,"descriptor",name)

    def test_capture_controls_and_same_recipe_replays_are_full_array_exact(self):
        records=[]
        for name in ("exp","sigmoid","silu"):
            records.append(dict(cohort=name,case="generate",projection="descriptor",activation=name,
                array_sha256={**dict.fromkeys(("ffn_input","block_reference","alternate_reference","other_weight_reference"),"same"),
                              "block_candidate":name}))
        records.append(copy.deepcopy(records[1]))
        self.assertTrue(audit.same_capture(records)["same_recipe_replays_exact"])
        for change in (lambda r:r[1]["array_sha256"].update(ffn_input="changed"),
                       lambda r:r[3]["array_sha256"].update(block_candidate="changed"),lambda r:r[1].update(case="edit")):
            changed=copy.deepcopy(records);change(changed)
            with self.assertRaises(ValueError):audit.same_capture(changed)

    def test_activation_overflow_must_keep_captured_owner_and_gpu_fallback_recipe(self):
        report,observed=negative_fixture();report["activation"]="sigmoid"
        report["direct"]["captured_activation_survived_env_change"]=True
        for phase in ("first_metrics","second_metrics"):
            report["hybrid"][phase]["runtime_weight"]["source_recipe"]="sylvester:w8-activation-sigmoid-v1+fp32-partial-join-v1"
        self.assertTrue(negative.validate(report,observed,True,"sigmoid")["hot_no_compile"])
        changed=copy.deepcopy(report);changed["direct"]["captured_activation_survived_env_change"]=False
        with self.assertRaises(ValueError):negative.validate(changed,observed,True,"sigmoid")
        changed=copy.deepcopy(report);changed["hybrid"]["second_metrics"]["runtime_weight"]["source_recipe"]="sylvester"
        with self.assertRaises(ValueError):negative.validate(changed,observed,True,"sigmoid")


if __name__=="__main__":unittest.main()
