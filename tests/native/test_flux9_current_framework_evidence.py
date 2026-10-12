import copy
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux9_current_framework_evidence import audit, cohort
from flux9_current_framework_screen import ORDER, QUANTIZED


def fixture():
    # Cohort structure fixture only. Actual native/reference execution fields
    # are separately required by validate_report and its focused tests.
    data = dict(status="complete_diagnostic", count=4, order=list(ORDER),
        mflux_package_version="0.22.0", reference_cache_mib=0,
        conversion=dict(status="complete_conversion_only", converted_components=["transformer", "text_encoder"],
                        components={name: dict(full_compressed_payload_equal=True) for name in ("transformer", "text_encoder")}),
        trials=[], medians={})
    for index, route in enumerate(ORDER):
        data["trials"].append(dict(index=index, route=route,
            report=dict(quantized_components=["transformer", "text_encoder"] if route in QUANTIZED else [],
                        runs=[dict(request_wall_including_export=10. + index + request / 10) for request in range(4)])))
    for route in dict.fromkeys(ORDER):
        samples = [row["request_wall_including_export"] for trial in data["trials"] if trial["route"] == route
                   for row in trial["report"]["runs"][1:3]]
        data["medians"][route] = dict(samples=samples, fresh_warm_seconds=statistics.median(samples))
    return data


class CurrentFluxEvidenceTests(unittest.TestCase):
    def test_complete_cohort_uses_all_fresh_samples_not_cold_or_repeat(self):
        data = fixture()
        with patch("flux9_current_framework_evidence.validate_report") as validator:
            self.assertEqual(cohort(data), data["medians"])
            self.assertEqual(validator.call_count, 14)
        self.assertEqual(data["medians"]["native-bf16"]["samples"], [10.1, 10.2, 23.1, 23.2])

    def test_short_running_wrong_version_or_partial_controls_never_promoted(self):
        for name, value in (("status", "running"), ("status", "complete_precheck_only"), ("count", 1),
                            ("mflux_package_version", "0.19.2"), ("reference_cache_mib", 1024),
                            ("order", ["mflux-bf16", "mflux-bf16"]), ("trials", [])):
            data = fixture()
            data[name] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "complete exact current seven-route"):
                cohort(data)

    def test_actual_route_component_scope_and_entire_payload_proof_are_required(self):
        with patch("flux9_current_framework_evidence.validate_report"):
            bad = fixture()
            bad["trials"][2]["report"]["quantized_components"] = ["transformer"]
            with self.assertRaisesRegex(ValueError, "component scope"):
                cohort(bad)
            bad = fixture()
            bad["conversion"]["components"]["text_encoder"]["full_compressed_payload_equal"] = False
            with self.assertRaisesRegex(ValueError, "whole matched"):
                cohort(bad)
            bad = fixture()
            bad["trials"][2]["route"] = "native-bf16"
            with self.assertRaisesRegex(ValueError, "measured route order"):
                cohort(bad)

    def test_declared_fast_median_cannot_replace_four_actual_fresh_values(self):
        data = fixture()
        data["medians"]["mflux-fp16"]["fresh_warm_seconds"] = 1
        with patch("flux9_current_framework_evidence.validate_report"):
            with self.assertRaisesRegex(ValueError, "four actual fresh"):
                cohort(data)

    def test_incomplete_audit_rejects_before_build_or_model_reads(self):
        with tempfile.TemporaryDirectory(prefix="tc-current-flux-evidence-") as raw:
            root = Path(raw)
            (root / "summary.json").write_text(json.dumps(dict(status="running")))
            with self.assertRaisesRegex(ValueError, "complete exact current seven-route"):
                audit(root, root / "absent-build.json", "2026-10-12")
        with self.assertRaises(ValueError):
            audit(Path("not-read"), Path("not-read"), "2026-13-99")


if __name__ == "__main__":
    unittest.main()
