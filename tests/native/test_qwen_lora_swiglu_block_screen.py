"""Host-only exact consumer, full denominator and strong-control gates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_lora_swiglu_block_screen as screen


class SwigluBlockScreenTests(unittest.TestCase):
    def native(self,order="forward"):
        q=dict(changed=0,elements=4194304,relative_l2=0.,max_abs=0.,cosine=1.)
        rows=[]
        for iteration in range(-3,21):
            for visit in range(7):
                arm=screen.ARMS[((6-visit if order=="reverse" else visit)+iteration+7)%7]
                wall=.034 if arm==screen.PRIMARY else .035 if arm==screen.CONTROL else .036
                rows.append(dict(iteration=iteration,visit=visit,arm=arm,seconds=wall,quality=dict(q),output_sha256="a"*64))
        return dict(status="measured",image_export=False,order=order,runtime_lora_applied_terms=227,extra_owned_weight_bytes=0,
            completed_full_dit_steps=1,stopped_step=1,stopped_layer=0,fresh_arguments_verified=True,borrowed_values_unchanged=True,
            all_outputs_lazy_before_eval=True,bitwise_outputs_checked=True,ordinary_block_sha256="a"*64,
            actual_layer_source_sha256={str(i):"a"*64 for i in range(22)},samples=rows,
            first_replays=[dict(arm=arm,seconds=.1,quality=dict(q),output_sha256="a"*64) for arm in screen.ARMS[1:]])

    def test_both_orders_keep_all_actual_complete_calls_and_strong_direct_control(self):
        for order in ("forward","reverse"):
            a=screen.analyze(self.native(order),order)
            self.assertEqual((a["measured_calls"],a["warm_calls"]),(147,21));self.assertEqual(a["strong_control"],"direct-qkv")
            self.assertTrue(a["primary_local_signal"])

    def test_complete_output_equality_cannot_be_replaced_by_close_l2(self):
        for key,value in (("changed",1),("relative_l2",1e-9),("elements",1024),("max_abs",.001),("cosine",float("nan"))):
            n=self.native();n["samples"][17]["quality"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):screen.analyze(n,"forward")

    def test_missing_numerical_lifecycle_and_full_denominators_are_rejected(self):
        for key,value in (("runtime_lora_applied_terms",226),("image_export",True),("completed_full_dit_steps",6),
                          ("extra_owned_weight_bytes",1),("fresh_arguments_verified",False),("borrowed_values_unchanged",False),("all_outputs_lazy_before_eval",False)):
            n=self.native();n[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):screen.analyze(n,"forward")
        n=self.native();n["samples"].pop()
        with self.assertRaises(ValueError):screen.analyze(n,"forward")
        n=self.native();n["samples"][0]["seconds"]=float("inf")
        with self.assertRaises(ValueError):screen.analyze(n,"forward")

    def test_wrong_visit_order_and_first_replay_omission_are_rejected(self):
        with self.assertRaises(ValueError):screen.analyze(self.native("reverse"),"forward")
        n=self.native();n["first_replays"].pop()
        with self.assertRaises(ValueError):screen.analyze(n,"forward")

    def test_numerically_equal_signed_zero_cannot_bypass_bitwise_gate(self):
        n=self.native();n["samples"][20]["output_sha256"]="b"*64
        with self.assertRaises(ValueError):screen.analyze(n,"forward")
        n=self.native();n["bitwise_outputs_checked"]=False
        with self.assertRaises(ValueError):screen.analyze(n,"forward")


if __name__=="__main__":unittest.main()
