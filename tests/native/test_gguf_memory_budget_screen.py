"""Evidence validation for empirical budget ranking; no RAM-cap certification."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("screen", ROOT / "tools/validation/screen_gguf_memory_budgets.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reports(native=False):
    result = []
    arms = [("packed_streamed", 10., 2 << 30), ("packed_resident", 5., 9 << 30)]
    if native: arms.append(("native", 2., 11 << 30))
    for residency, seconds, peak in arms:
        for mode in ("timing", "memory"):
            report = {"status": "completed", "measurement": mode, "library_sha256": "a" * 64,
                "loaded_libraries": {}, "prompt_cache_policy": "miss", "encoder_environment": {},
                "native_packed_math_profile": "z-mlx-compat-affine-v1", "dit_source_sha256": "b" * 64, "runs": []}
            for i in range(4 if mode == "timing" else 1):
                qe = {"schema_version": 1, "enabled": True, "precision_profile": "z-mlx-compat-affine-v1",
                      "prefetch_layers": 1, "source_residency": residency}
                row = {"prefetch": -1 if residency == "native" else 1, "status": 0, "error": None,
                    "measurement": mode, "warmup": False, "wall_seconds": seconds + i % 2,
                    "png_sha256": "d" * 64, "request": {"schema_version": 2, "model": "z-image-turbo-gguf",
                        "sampling": {"steps": 4, "seed": 42}, "inputs": [{"kind": "text", "text": "fixed prompt"}],
                        "outputs": [{"kind": "image", "path": "different.png", "width": 512, "height": 512}],
                        "execution": {"policy": "gpu", "quantized_execution": qe}},
                    "metrics": {"prompt_cache_hit": False, "quantized_execution": {
                        "source_sha256": "b" * 64, "slot_count": 2, "fill_count": 120, "source_residency": residency,
                        "layout_digest": ("1" if residency == "packed_streamed" else "2") * 64,
                        "managed_peak_bytes": 400 << 20}, "encoder_quantized_execution": {
                        "source_sha256": "c" * 64, "layout_digest": "e" * 64,
                        "precision_profile": "qwen3-z-source-mixed-v1", "managed_peak_bytes": 600 << 20}},
                    "vm_deltas": {"Swapouts": 0}, "memory_samples": []}
                if residency == "native":
                    row["request"]["execution"].pop("quantized_execution")
                    row["metrics"].pop("quantized_execution")
                if mode == "memory":
                    row["memory_samples"] = [{"phys_footprint_bytes": peak,
                        "lifetime_max_phys_footprint_bytes": peak}]
                    row["memory_sample_max_gap_seconds"] = .01
                report["runs"].append(row)
            result.append(report)
    return result


class BudgetScreenTests(unittest.TestCase):
    def test_encoder_metadata_lifecycle_and_controls_are_bound(self):
        data=reports()
        for report in data:
            report["encoder_environment"]["TURBOCIDER_QWEN3_GGUF_METADATA_CACHE"]="1"
            for row in report["runs"]:
                row["metrics"]["encoder_quantized_execution"].update(source_metadata_policy="engine-verified-cpu-metadata-only-v1",
                    source_metadata_reused=False,source_metadata_preparations=1,conditioning_producer_generation=1)
        result=MODULE.screen(data)
        self.assertEqual(result["identity"]["encoder_metadata_policy"],"engine-verified-cpu-metadata-only-v1")
        for fault in ("policy","missing","reuse_type","counter","control","claimed_reuse"):
            bad=copy.deepcopy(data);encoder=bad[1]["runs"][0]["metrics"]["encoder_quantized_execution"]
            if fault=="policy":encoder["source_metadata_policy"]="unknown"
            if fault=="missing":encoder.pop("source_metadata_policy")
            if fault=="reuse_type":encoder["source_metadata_reused"]=1
            if fault=="counter":encoder["conditioning_producer_generation"]=1.5
            if fault=="control":bad[1]["encoder_environment"]["TURBOCIDER_QWEN3_GGUF_METADATA_CACHE"]="0"
            if fault=="claimed_reuse":encoder.update(source_metadata_policy="reconstruct-per-request-v1",source_metadata_reused=True)
            with self.subTest(fault=fault),self.assertRaises(ValueError):MODULE.screen(bad)

    def test_fastest_observed_fit_depends_on_budget_not_smallest_memory(self):
        result = MODULE.screen(reports(native=True))
        self.assertEqual([b["fastest_observed_candidate"] for b in result["budgets"]],
                         ["packed_streamed:p1", "packed_streamed:p1", "packed_resident:p1", "native_packed"])
        self.assertFalse(result["production_qualified"])
        self.assertEqual(result["whole_request_memory"], "unknown")
        self.assertTrue(all(not b["bounded_certified"] for b in result["budgets"]))

    def test_uses_conservative_lifetime_peak_not_current_rss(self):
        data = reports()
        data[1]["runs"][0]["memory_samples"][0]["lifetime_max_phys_footprint_bytes"] = 8 << 30
        result = MODULE.screen(data)
        self.assertIsNone(result["budgets"][0]["fastest_observed_candidate"])

    def test_decimal_gb_is_not_silently_treated_as_gib(self):
        result = MODULE.screen(reports(native=True), budget_unit="GB")
        self.assertEqual(result["budgets"][2]["limit_bytes"], 10_000_000_000)
        self.assertEqual(result["budgets"][2]["fastest_observed_candidate"], "packed_streamed:p1")
        self.assertNotIn("budget_gib", result["budgets"][2])

    def test_direct_packed_import_plan_and_ceiling_are_bound(self):
        data=reports(native=True)
        for report in data[-2:]:
            report["native_import"]="cpu_direct"
            for row in report["runs"]:
                row["metrics"]["gguf_import"]={"recipe":"gguf-mlx-compat-affine-packed-bank-v1",
                    "source_sha256":"b"*64,"plan_digest":"f"*64,"allocator_cache_limit_bytes":0,
                    "managed_peak_bytes":7<<30}
        result=MODULE.screen(data)
        self.assertEqual(result["budgets"][-1]["fastest_observed_candidate"],"native_packed:cpu_direct")
        data[-1]["runs"][0]["metrics"]["gguf_import"]["plan_digest"]="1"*64
        with self.assertRaisesRegex(ValueError,"import plan changed"): MODULE.screen(data)

    def test_warmups_are_retained_but_not_ranked(self):
        data = reports()
        row = copy.deepcopy(data[0]["runs"][0]); row["warmup"] = True; row["wall_seconds"] = 1000
        data[0]["runs"].insert(0, row)
        self.assertEqual(MODULE.screen(data)["candidates"]["packed_streamed:p1"]["warm_median_seconds"], 10.5)

    def test_swap_and_bad_sampling_are_not_budget_admissions(self):
        for field, value in (("vm_deltas", {"Swapouts": 1}), ("memory_sample_max_gap_seconds", .1)):
            data = reports(); data[1]["runs"][0][field] = value
            result = MODULE.screen(data)
            self.assertIsNone(result["budgets"][0]["fastest_observed_candidate"])

    def test_incomplete_reports_and_insufficient_samples_rejected(self):
        for mode in ("timing", "memory"):
            data = [r for r in reports() if r["measurement"] == mode]
            with self.assertRaises(ValueError): MODULE.screen(data)
        data = reports(); data[0]["runs"] = data[0]["runs"][:3]
        with self.assertRaises(ValueError): MODULE.screen(data)

    def test_identity_tampering_is_rejected(self):
        changes = [
            lambda r: r.update(png_sha256="f" * 64),
            lambda r: r["request"]["sampling"].update(seed=123),
            lambda r: r["request"]["outputs"][0].update(width=1024),
            lambda r: r["request"]["execution"]["quantized_execution"].update(precision_profile="other"),
            lambda r: r["metrics"]["quantized_execution"].update(source_sha256="f" * 64),
            lambda r: r["metrics"]["quantized_execution"].update(layout_digest="f" * 64),
            lambda r: r["metrics"]["encoder_quantized_execution"].update(layout_digest="f" * 64),
            lambda r: r["metrics"].update(prompt_cache_hit=True),
        ]
        for change in changes:
            data = reports(); change(data[1]["runs"][0])
            with self.subTest(change=change), self.assertRaises(ValueError): MODULE.screen(data)
        for key, value in (("library_sha256", "f" * 64), ("status", "running")):
            data = reports(); data[1][key] = value
            with self.assertRaises(ValueError): MODULE.screen(data)
        data = reports()
        for report in data: report["prompt_cache_policy"] = "hit"
        with self.assertRaisesRegex(ValueError, "did not hit"): MODULE.screen(data)

    def test_invalid_evidence_rejected_without_dropping_bad_samples(self):
        changes = [
            lambda r: r.update(status=1), lambda r: r.update(wall_seconds=float("nan")),
            lambda r: r.update(png_sha256=""), lambda r: r.update(warmup="false"),
            lambda r: r.update(prefetch=True), lambda r: r.update(cancellation_triggered=True),
            lambda r: r["metrics"]["quantized_execution"].update(fill_count=119),
            lambda r: r["metrics"]["quantized_execution"].update(slot_count=1),
            lambda r: r["request"].update(dump_tensors="dump"),
            lambda r: r.update(memory_samples=[{}]),
        ]
        for change in changes:
            data = reports(); change(data[0]["runs"][0])
            with self.subTest(change=change), self.assertRaises(ValueError): MODULE.screen(data)
        data = reports(); data[1]["runs"][0]["sampling_errors"] = ["failed"]
        with self.assertRaises(ValueError): MODULE.screen(data)

    def test_invalid_budget_inputs_rejected(self):
        for kwargs in ({"budgets": [0]}, {"budgets": [True]}, {"buffer_percent": 100}, {"min_samples": 1}, {"budget_unit": "unknown"}):
            with self.assertRaises(ValueError): MODULE.screen(reports(), **kwargs)


if __name__ == "__main__": unittest.main()
