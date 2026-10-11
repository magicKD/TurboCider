import copy
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from z_mflux_framework_evidence import audit, cohort


def reference_fixture():
    trials = []
    for index in range(2):
        report = dict(constructor_seconds=1, required_weight_coverage={name: dict(required_targets=2, covered_targets=2, missing=[])
            for name in ("vae", "transformer", "text_encoder")}, runs=[dict(index=request,
                request_wall_including_export=10 + index + request / 10, actual_steps=list(range(8)),
                actual_quantization=None, loras=[], prompt_cache_hit=False) for request in range(4)])
        trials.append(dict(index=index, route="mflux-bf16", report=report,
            memory=dict(verified=dict(tree_peak_phys_footprint_bytes=100))))
    samples = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
    return dict(status="complete_diagnostic", steps=8, count=4, order=["mflux-bf16"] * 2,
        trials=trials, medians={"mflux-bf16": dict(samples=samples, fresh_warm_seconds=statistics.median(samples))})


class ZFrameworkEvidenceTests(unittest.TestCase):
    def test_actual_fresh_samples_exclude_cold_and_repeat(self):
        rows = cohort(reference_fixture())
        self.assertEqual(rows[0]["fresh_warm_seconds"], [10.1, 10.2, 11.1, 11.2])
        self.assertEqual(rows[0]["cold_seconds"], [10, 11])
        self.assertEqual(rows[0]["repeat_seconds"], [10.3, 11.3])

    def test_running_short_or_missing_route_is_not_complete(self):
        for name, value in (("status", "running"), ("steps", 2), ("count", 1), ("order", ["mflux-bf16"]), ("trials", [])):
            data = reference_fixture()
            data[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                cohort(data)

    def test_missing_empty_or_forged_coverage_is_rejected(self):
        data = reference_fixture()
        for coverage in (None, {}, {name: dict(required_targets=True, covered_targets=True, missing=[])
                for name in ("vae", "transformer", "text_encoder")}, {name: dict(required_targets=2, covered_targets=1, missing=["weight"])
                for name in ("vae", "transformer", "text_encoder")}):
            altered = copy.deepcopy(data)
            altered["trials"][0]["report"]["required_weight_coverage"] = coverage
            with self.subTest(coverage=coverage), self.assertRaises(ValueError):
                cohort(altered)

    def test_actual_steps_quantization_and_stock_cache_policy_are_checked(self):
        for name, value in (("actual_steps", [0]), ("actual_quantization", 4), ("loras", ["adapter"]),
                ("prompt_cache_hit", True), ("request_wall_including_export", float("nan"))):
            data = reference_fixture()
            data["trials"][0]["report"]["runs"][0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                cohort(data)

    def test_claimed_median_cannot_replace_all_actual_samples(self):
        data = reference_fixture()
        data["medians"]["mflux-bf16"]["fresh_warm_seconds"] = 1
        with self.assertRaisesRegex(ValueError, "median"):
            cohort(data)

    def test_running_audit_never_opens_model_or_build(self):
        with tempfile.TemporaryDirectory(prefix="tc-z-framework-evidence-") as raw:
            path = Path(raw)
            (path / "summary.json").write_text(json.dumps(dict(status="running")))
            with self.assertRaises(ValueError):
                audit(path, path / "absent-model", path / "absent-build")


if __name__ == "__main__":
    unittest.main()
