import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux9_mflux_quant_evidence import audit, cohort
from flux9_mflux_quant_screen import BITS


def fixture():
    order = ["bf16", "q4", "q8", "q8", "q4", "bf16"]
    trials = []
    for index, mode in enumerate(order):
        report = dict(quantization=BITS[mode], constructor_seconds=1,
            required_weight_coverage={name: dict(required_targets=2, covered_targets=2, missing=[])
                for name in ("vae", "transformer", "text_encoder")},
            runs=[dict(index=request, request_wall_including_export=10 + index + request / 10,
                actual_steps=list(range(4)), actual_quantization=BITS[mode], loras=[]) for request in range(4)])
        trials.append(dict(index=index, mode=mode, report=report, memory=dict(verified=dict(tree_peak_phys_footprint_bytes=100))))
    routes = []
    for mode in BITS:
        values = [row["request_wall_including_export"] for trial in trials if trial["mode"] == mode for row in trial["report"]["runs"][1:3]]
        routes.append(dict(mode=mode, fresh_warm_seconds=values, fresh_warm_median=statistics.median(values)))
    return dict(status="complete_diagnostic", order=order, trials=trials, routes=routes)


class FluxStockEvidenceTests(unittest.TestCase):
    def test_precision_cohort_is_separate_and_fresh_is_not_repeat(self):
        rows = cohort(fixture())
        self.assertEqual([row["route"] for row in rows], ["mflux-bf16", "mflux-q4", "mflux-q8"])
        self.assertEqual(rows[1]["fresh_warm_seconds"], [11.1, 11.2, 14.1, 14.2])
        self.assertEqual(rows[1]["repeat_seconds"], [11.3, 14.3])

    def test_incomplete_and_misordered_cohort_rejected(self):
        for name, value in (("status", "running"), ("order", ["q4"] * 6), ("trials", [])):
            data = fixture()
            data[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                cohort(data)

    def test_weights_steps_adapter_and_nonfinite_time_are_rejected(self):
        for name, value in (("actual_steps", [0]), ("actual_quantization", 8), ("loras", ["adapter"]),
                ("request_wall_including_export", True), ("request_wall_including_export", float("inf"))):
            data = fixture()
            data["trials"][1]["report"]["runs"][0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                cohort(data)
        for coverage in (None, {}, {name: dict(required_targets=0, covered_targets=0, missing=[])
                for name in ("vae", "transformer", "text_encoder")}):
            data = fixture()
            data["trials"][0]["report"]["required_weight_coverage"] = coverage
            with self.subTest(coverage=coverage), self.assertRaises(ValueError):
                cohort(data)

    def test_claimed_minimum_does_not_replace_median(self):
        data = fixture()
        data["routes"][0]["fresh_warm_median"] = 1
        with self.assertRaisesRegex(ValueError, "median"):
            cohort(data)

    def test_running_audit_rejected_before_model_access(self):
        with tempfile.TemporaryDirectory(prefix="tc-flux-stock-evidence-") as raw:
            path = Path(raw)
            (path / "summary.json").write_text(json.dumps(dict(status="running")))
            with self.assertRaises(ValueError):
                audit(path)


if __name__ == "__main__":
    unittest.main()
