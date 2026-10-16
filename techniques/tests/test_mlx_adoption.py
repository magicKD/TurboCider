"""Offline MLX adoption evidence checks; never import MLX or Apple frameworks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
import mlx_adoption as adoption

SPEC = importlib.util.spec_from_file_location("mlx_adoption_curator", ROOT / "scripts/curate_mlx_adoption.py")
curator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curator)


class AdoptionMath(unittest.TestCase):
    def test_periodic_stream_digest_covers_every_logical_element(self):
        for seed in (17, 29):
            for generation in range(3):
                values = [((i * 7 + generation * 5 + seed * 11) % 37 - 18) / 16. for i in range(32 * 33)]
                expected = {name: hashlib.sha256(struct.pack("<" + dtype * len(items), *items)).hexdigest()
                            for name, dtype, items in (("input_sha256", "e", values), ("consumer_sha256", "f", [2*x+.25 for x in values]))}
                self.assertEqual(adoption.witness(32, 33, generation, seed), expected)
        for count in (1, 36, 37, 38, 1001):
            values = [index / 8 for index in range(37)]
            expected = hashlib.sha256(struct.pack("<" + "f" * count, *(values[i % 37] for i in range(count)))).hexdigest()
            self.assertEqual(adoption.digest_pattern(values, "f", count), expected)

    def test_bool_shape_generation_and_unbounded_recipe_are_refused(self):
        for args in ((True, 33, 0, 17), (32, 32, 0, 17), (32, 33, True, 17), (32, 33, 3, 17), (32, 33, 0, 10001)):
            with self.subTest(args=args), self.assertRaises(ValueError): adoption.witness(*args)

    def test_default_wrapper_prepares_without_installed_mlx_or_build(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            child = subprocess.run([sys.executable, "-E", "-B", str(ROOT / "labs/metal/mlx_adoption_probe.py"),
                                    "--mlx-root", str(Path(directory) / "missing-mlx"), "--output-dir", str(output)],
                                   text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(child.stdout)["status"], "prepared-only-no-device")
            value = json.loads((output / "report.json").read_text())
            self.assertIsNone(value["native"])
            self.assertNotIn("mlx_assets", value)
            self.assertFalse((output / "mlx-adoption").exists())


class AdoptionEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "review/mlx-adoption-evidence-20261016.json").read_text())

    def record(self):
        return copy.deepcopy(self.data["runs"][-1]["report"])

    def test_four_protocols_32_consumers_and_original_failure_are_all_retained(self):
        self.assertEqual(len(self.data["runs"]), 4); self.assertEqual(len(self.data["failures"]), 1)
        self.assertEqual(self.data["curator_sha256"], hashlib.sha256((ROOT / "scripts/curate_mlx_adoption.py").read_bytes()).hexdigest())
        self.assertEqual(self.data["contract_sha256"], hashlib.sha256((ROOT / "labs/mlx_adoption.py").read_bytes()).hexdigest())
        positive, negatives = 0, 0
        for run in self.data["runs"]:
            value = adoption.verify(run["report"])
            self.assertEqual(value, run["independent_validation"])
            positive += value["full_positive_consumer_hashes"]
            negatives += sum(value[key] for key in ("cached_result_rejections", "materialized_snapshot_rejections", "packed_stride_rejections"))
            self.assertTrue(run["child_exit_confirmed"])
            self.assertEqual(len(run["report"]["mlx_assets"]["headers"]), 383)
            for name, digest in run["report"]["source_sha256"].items():
                self.assertEqual(digest, hashlib.sha256((ROOT / "labs/metal" / name).read_bytes()).hexdigest())
        self.assertEqual((positive, negatives), (32, 20))
        failure = self.data["failures"][0]
        self.assertEqual(adoption.verify_failure(failure["report"]), failure["independent_validation"])
        self.assertTrue(failure["child_exit_confirmed"])
        self.assertNotEqual(failure["report"]["source_sha256"]["mlx_adoption_probe.mm"], self.record()["source_sha256"]["mlx_adoption_probe.mm"])

    def test_8mib_is_complete_synthetic_geometry_not_real_qwen(self):
        for row in self.data["runs"]:
            value = row["report"]
            self.assertFalse(value["real_qwen"]); self.assertFalse(value["formal_performance_score"])
            self.assertEqual(value["macos_build"], "25G83")
            if value["case"] == "port":
                self.assertEqual(value["native"]["payload_bytes"], 8 * 1024**2)
                self.assertEqual(value["native"]["logical_shape"], [1024, 4096])
                self.assertEqual(row["independent_validation"]["logical_elements_per_consumer"], 4194304)

    def test_model_performance_physical_or_unknown_version_claims_are_refused(self):
        for key, value in (("real_qwen", True), ("formal_performance_score", True), ("physical_overlap_qualified", True),
                           ("runtime_internal_copy_behavior", "none"), ("source_inputs_unchanged", False)):
            row = self.record(); row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): adoption.verify(row)
        row = self.record(); row["native"]["mlx_version"] = "future"
        with self.assertRaises(ValueError): adoption.verify(row)

    def test_early_late_or_bool_deleter_and_alias_changes_are_refused(self):
        for kind in ("early", "late", "bool", "pointer", "alias", "snapshot"):
            row = self.record(); record = row["native"]["records"][0]
            if kind == "early": record["deleter_calls_after_flat_scope"] = 1
            elif kind == "late": record["deleter_calls_after_last_view_release"] = 0
            elif kind == "bool": record["deleter_calls_after_last_view_release"] = True
            elif kind == "pointer": record["deleter_pointer_matches_surface"] = False
            elif kind == "alias": record["view_alias"]["buffer_contents_matches_surface"] = False
            else: record["snapshot_has_distinct_backing"] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError): adoption.verify(row)

    def test_missing_generation_hash_guard_clock_and_completion_changes_are_refused(self):
        for kind in ("generation", "hash", "input", "guard", "clock", "duration", "completion", "layout"):
            row = self.record(); record = row["native"]["records"][0]; sample = record["samples"][0]
            if kind == "generation": record["samples"].pop()
            elif kind == "hash": sample["consumer"]["consumer_sha256"] = "0" * 64
            elif kind == "input": sample["input_sha256"] = "0" * 64
            elif kind == "guard": sample["tail_guard_pass"] = False
            elif kind == "clock": sample["host_events"][0]["clock_domain"] = "gpu-clock"
            elif kind == "duration": sample["host_events"][0]["duration_ns"] = True
            elif kind == "completion": sample["mlx_stream_drained"] = False
            else: row["native"]["pitch_elements"] = 8192
            with self.subTest(kind=kind), self.assertRaises(ValueError): adoption.verify(row)

    def test_same_dtype_alias_cached_output_and_snapshot_are_not_relabelled(self):
        for kind in ("alias", "copy-consumer", "cached", "snapshot", "ratio"):
            row = self.record(); record = row["native"]["records"][0]
            if kind == "alias": record["same_dtype_copy_alias"]["allocator_raw_pointer_matches_surface"] = False
            elif kind == "copy-consumer": record["new_consumer_of_same_dtype_copy"]["mismatched_elements"] = 1
            elif kind == "cached": record["stale_controls"]["cached_evaluated_output"]["mismatched_elements"] = 0
            elif kind == "snapshot": record["stale_controls"]["new_consumer_of_materialized_snapshot"]["maximum_finite_abs_error"] = 0
            else: record["stale_controls"]["used_for_speed_ratio"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError): adoption.verify(row)

    def test_no_fabricated_padded_negative_or_address_based_proof(self):
        row = self.record(); row["native"]["records"][0]["packed_stride_control"] = {"mismatched_elements": 1}
        with self.assertRaises(ValueError): adoption.verify(row)
        values = {key: True for key in adoption.ALIASES}; values["same_metal_device"] = 1
        with self.assertRaises(ValueError): adoption.alias(values)

    def test_failed_snapshot_assumption_cannot_be_promoted_or_renamed_unsupported(self):
        for kind in ("success", "unsupported", "partial"):
            row = copy.deepcopy(self.data["failures"][0]["report"])
            if kind == "success": row["status"] = "actual-adoption-and-lifetime-validated"
            elif kind == "unsupported": row["native"]["errors"] = ["hardware unsupported"]
            else: row["native"]["records"][0]["snapshot_has_distinct_backing"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError): adoption.verify_failure(row)

    def test_curator_rejects_duplicates_and_escaping_inputs_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError): curator.curate(root, root, root, [], [])
            with self.assertRaises(ValueError): curator.curate(root, root, root, ["same"], ["same"])
            for value in ("../outside", "/outside", "missing"):
                with self.assertRaises(ValueError): curator.local(root, value)


if __name__ == "__main__":
    unittest.main()
