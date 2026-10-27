"""Host-only full-consumer/strong-denominator and no-readback protocol guards."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
import qwen_base_swiglu_block_screen as screen


def receipt(order="forward"):
    q = dict(bits_exact=True, changed=0, elements=4194304, relative_l2=0., max_abs=0., cosine=1.)
    value = dict(status="measured", image_export=False, order=order, runtime_lora_applied_terms=227,
                 extra_owned_weight_bytes=0, completed_full_dit_steps=1, stopped_step=1, stopped_layer=0,
                 fresh_arguments_verified=True, borrowed_values_unchanged=True, all_outputs_lazy_before_eval=True,
                 timing_readback_policy="first-fresh-restored-and-final-only-v1",
                 actual_layer_source_sha256={str(i): "0" * 64 for i in range(22)}, ordinary_block_sha256="1" * 64)
    value["first_replays"] = [dict(arm=arm, seconds=.05, quality=copy.deepcopy(q)) for arm in screen.ARMS[1:]]
    value["final_quality"] = [dict(arm=arm, quality=copy.deepcopy(q)) for arm in screen.ARMS]
    value["samples"] = [dict(iteration=i, visit=j, arm=screen.ARMS[((3-j if order=="reverse" else j)+i+4)%4],
                             seconds=.04 if screen.ARMS[((3-j if order=="reverse" else j)+i+4)%4]==screen.PRIMARY else .05)
                        for i in range(-screen.WARMUPS, screen.SAMPLES) for j in range(4)]
    return value


class BaseSwigluBlockScreenTests(unittest.TestCase):
    def test_both_orders_keep_all_completed_calls_and_fixed_strong_primary(self):
        for order in ("forward", "reverse"):
            result = screen.analyze(receipt(order), order)
            self.assertEqual(result["measured_calls"], 64)
            self.assertEqual(result["warm_calls"], 12)
            self.assertEqual(result["strong_control"], "strong-control")
            self.assertEqual(result["primary"], "base-paired-b64")
            self.assertAlmostEqual(result["primary_reduction_percent"], 20.)

    def test_missing_first_final_or_timed_output_is_rejected(self):
        for key in ("first_replays", "final_quality", "samples"):
            value = receipt()
            value[key].pop()
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.analyze(value, "forward")

    def test_work_ownership_and_timing_readback_policy_cannot_be_relabelled(self):
        for key, bad in (("runtime_lora_applied_terms", 226), ("extra_owned_weight_bytes", 100),
                         ("completed_full_dit_steps", 6), ("stopped_step", 0), ("stopped_layer", 1),
                         ("fresh_arguments_verified", False), ("borrowed_values_unchanged", False),
                         ("all_outputs_lazy_before_eval", False), ("image_export", True),
                         ("timing_readback_policy", "per-sample-hash")):
            value = receipt()
            value[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.analyze(value, "forward")

    def test_numeric_zero_cannot_substitute_for_bitwise_or_full_output_equality(self):
        for key, bad in (("bits_exact", False), ("changed", 1), ("elements", 1024),
                         ("relative_l2", .00001), ("max_abs", .01), ("cosine", float("nan"))):
            value = receipt()
            value["first_replays"][1]["quality"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.analyze(value, "forward")

    def test_nonfinite_partial_timing_and_changed_rotation_are_rejected(self):
        for bad in (0., -1., float("nan"), float("inf"), True):
            value = receipt()
            value["samples"][0]["seconds"] = bad
            with self.subTest(wall=bad), self.assertRaises(ValueError):
                screen.analyze(value, "forward")
        value = receipt()
        value["samples"][0]["arm"] = screen.PRIMARY
        with self.assertRaises(ValueError):
            screen.analyze(value, "forward")


if __name__ == "__main__":
    unittest.main()
