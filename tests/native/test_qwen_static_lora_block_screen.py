"""Host-only boundaries: original GPU denominator and complete static replay."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
import qwen_static_lora_block_screen as screen


class StaticLoraBlockScreenTests(unittest.TestCase):
    def report(self):
        rows = []
        for i in range(-3, 12):
            for visit in range(4):
                arm = screen.NAMES[(visit+i+4) % 4]
                rows.append(dict(iteration=i, arm=arm, seconds={"gpu-original": .035, "gpu-staged": .036,
                    "private-serial": .041, "private-hybrid": .032}[arm], quality=dict(
                        relative_l2=0 if arm.startswith("gpu-") else .0005,
                        cosine=1., max_abs=0 if arm.startswith("gpu-") else .01,
                        changed=0 if arm.startswith("gpu-") else 2000, elements=1024*4096)))
                rows[-1]["ffn_quality"] = None if arm == "gpu-original" else dict(
                    relative_l2=0 if arm == "gpu-staged" else .002, cosine=1., max_abs=.01,
                    changed=0 if arm == "gpu-staged" else 3000, elements=1024*4096)
        return dict(status="measured", runtime_lora_applied_terms=227, successful_private_calls=30,
                    image_export=False, borrowed_values_unchanged=True, samples=rows)

    def test_original_complete_gpu_not_staged_or_serial_is_denominator(self):
        result = screen.analyze(self.report())
        self.assertAlmostEqual(result["reduction_vs_original_gpu_percent"], 100*(1-.032/.035))
        self.assertTrue(all(len(values) == 12 for values in result["whole_block_seconds"].values()))

    def test_incomplete_adapter_private_work_or_mutated_borrow_rejected(self):
        for key, value in (("status", "failed"), ("runtime_lora_applied_terms", 226),
                           ("successful_private_calls", 29), ("image_export", True),
                           ("borrowed_values_unchanged", False)):
            bad = self.report(); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.analyze(bad)

    def test_missing_unbalanced_or_nonexact_gpu_replays_rejected(self):
        for change in (lambda r: r["samples"].pop(),
                       lambda r: r["samples"].reverse(),
                       lambda r: r["samples"][3]["quality"].update(changed=1)):
            bad = self.report(); change(bad)
            with self.assertRaises(ValueError):
                screen.analyze(bad)

    def test_nonfinite_zero_and_boolean_metrics_rejected(self):
        for value in (float("nan"), float("inf"), 0, -1, True):
            bad = self.report(); bad["samples"][0]["seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                screen.analyze(bad)
        for key, value in (("relative_l2", float("nan")), ("relative_l2", .00101),
                           ("cosine", .998), ("elements", True), ("changed", True)):
            bad = copy.deepcopy(self.report()); bad["samples"][0]["quality"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                screen.analyze(bad)


if __name__ == "__main__":
    unittest.main()
