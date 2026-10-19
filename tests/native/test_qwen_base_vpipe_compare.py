"""Portable component-route accounting; no reference build/device required."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/profiling/qwen_base"))
import vpipe_compare as compare


def fixture(case="edit", arm="ffn-qkv"):
    active = case == "edit" and arm != "gpu"
    samples = []
    for i in range(8):
        work = active and not i % 2
        samples.append(dict(index=i, rep=i // 2, warmup=i < 2, phase="decode" if i % 2 else "prefill",
                            seconds=1.0, gpu_flops=100, ane_flops=100 if work else 0,
                            ane_predict_records=32 if work else 0,
                            ffn_armed=active, qkv_armed=active and arm == "ffn-qkv", ffn_failed=False, qkv_failed=False))
    return dict(schema="tc-vpipe-qwen-dit-v1", status="complete", complete_request=False, synthetic_inputs=True,
                joint_rows=5184 if case == "edit" else 1088, reference_rows=4096 if case == "edit" else 0,
                decode_rows=1024, edit=int(case == "edit"), arm=compare.ARMS[arm], samples=samples)


class VpipeComparisonTests(unittest.TestCase):
    def test_all_arms_and_cases_have_explicit_component_denominator(self):
        for case in ("generate", "edit"):
            for arm in compare.ARMS:
                report = fixture(case, arm)
                result = compare.validate_report(report, case, arm, 3)
                self.assertEqual(result["decode"]["median_seconds"], 1)
                self.assertEqual(result["decode"]["ane_predict_records"], [0] * 4)
                report["complete_request"] = True
                with self.assertRaises(ValueError):
                    compare.validate_report(report, case, arm, 3)

    def test_armed_is_not_actual_decode_acceleration(self):
        report = fixture()
        self.assertTrue(report["samples"][1]["ffn_armed"])
        compare.validate_report(report, "edit", "ffn-qkv", 3)
        report["samples"][1].update(ane_flops=100, ane_predict_records=1)
        with self.assertRaises(ValueError):
            compare.validate_report(report, "edit", "ffn-qkv", 3)

    def test_empty_ane_work_failed_route_and_forged_warmup_are_rejected(self):
        for mutate in (lambda r: [s.update(ane_flops=0, ane_predict_records=0) for s in r["samples"]],
                       lambda r: r["samples"][2].update(ffn_failed=True),
                       lambda r: r["samples"][2].update(warmup=True),
                       lambda r: r["samples"].pop(),
                       lambda r: r["samples"][2].update(seconds=float("nan")),
                       lambda r: r["samples"][2].update(ane_predict_records=-1)):
            report = copy.deepcopy(fixture())
            mutate(report)
            with self.assertRaises(ValueError):
                compare.validate_report(report, "edit", "ffn-qkv", 3)

    def test_generation_decline_cannot_be_reported_as_successful_ane(self):
        report = fixture("generate")
        report["samples"][0]["ffn_armed"] = True
        with self.assertRaises(ValueError):
            compare.validate_report(report, "generate", "ffn-qkv", 3)

    def test_environment_drops_experimental_reference_flags(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"VPIPE_ANE_FAIL": "0", "VPIPE_QWEN_IMAGE21_ANE": "1"}):
            self.assertFalse(any(k.startswith("VPIPE_") for k in compare.environment("stock")))
            self.assertEqual([k for k in compare.environment("timeline") if k.startswith("VPIPE_")],
                             ["VPIPE_QWEN_IMAGE21_ANE_PROFILE"])


if __name__ == "__main__":
    unittest.main()
