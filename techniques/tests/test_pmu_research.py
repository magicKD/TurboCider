"""Offline PMU provenance and full synthetic-output witnesses; no native imports."""
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
import pmu


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


curator = module("pmu_data_verifier", "scripts/curate_pmu_campaign.py")
driver = module("pmu_bounded_builder", "labs/ane/pmu_probe.py")


class PMUContracts(unittest.TestCase):
    def test_missing_empty_zeroed_and_unknown_payload_are_distinct(self):
        examples = [({"object_present": False, "payload_bytes": None, "raw_words": None}, "missing-object"),
                    ({"object_present": True, "payload_bytes": 0, "raw_words": []}, "empty-payload"),
                    ({"object_present": True, "payload_bytes": 232, "raw_words": [0] * 29}, "zero-filled-unqualified"),
                    ({"object_present": True, "payload_bytes": 8, "raw_words": [1]}, "unknown-payload-schema")]
        for data, state in examples:
            with self.subTest(state=state):
                result = pmu.classify_stats(data)
                self.assertEqual(result["state"], state)
                self.assertIsNone(result["physical_duration_seconds"])
                self.assertFalse(result["counter_semantics_qualified"])

    def test_duplicate_names_do_not_merge_or_relabel_raw_indices(self):
        data = {"object_present": True, "payload_bytes": 232, "raw_words": list(range(29)),
                "labels_by_index": ["claimed-cycles"] * 24 + ["kANE_UNKNOWN"] * 5}
        result = pmu.classify_stats(data)
        self.assertEqual(result["state"], "raw-available-unqualified")
        self.assertEqual([row["value"] for row in result["raw_by_index"][-5:]], [24, 25, 26, 27, 28])
        self.assertTrue(all(row["unit"] is None for row in result["raw_by_index"]))

    def test_invalid_types_truncation_and_fabricated_data_are_rejected(self):
        for data in ({"object_present": 1}, {"object_present": False, "payload_bytes": 0, "raw_words": []},
                     {"object_present": True, "payload_bytes": 231, "raw_words": [0] * 29},
                     {"object_present": True, "payload_bytes": 232, "raw_words": [True] * 29},
                     {"object_present": True, "payload_bytes": 232, "raw_words": [-1] * 29},
                     {"object_present": True, "payload_bytes": 232, "raw_words": [2**64] * 29}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                pmu.classify_stats(data)

    def test_reset_and_wrap_are_not_silently_zero_or_automatically_64bit(self):
        for kwargs in ({"bits": None}, {"bits": True}, {"bits": 8}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pmu.counter_delta(254, 2, **kwargs)
        self.assertEqual(pmu.counter_delta(254, 2, bits=8, wrap_verified=True), 4)
        self.assertIsNone(pmu.counter_delta(100, 2, bits=8, reset_observed=True))
        self.assertEqual(pmu.counter_delta(100, 100, bits=64), 0)  # arithmetic, not per-request activity

    def test_witness_matches_direct_nonperiodic_small_mathematics(self):
        c, w, generation = 3, 5, 2
        source = [[((i * 7 + x * 3 + generation * 5) % 17 - 8) / 8. for x in range(w)] for i in range(c)]
        result = [[math.fsum(source[i][x] * (((o * 7 + i * 3) % 11 - 5) / 64.) for i in range(c))
                   for x in range(w)] for o in range(c)]
        raw = b"".join(struct.pack("<e", v) for row in result for v in row)
        self.assertEqual(pmu.witness_hashes("conv", c, w, generation)["output_sha256"], hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError): pmu.witness_hashes("conv", 129, w, generation)

    def test_mil_builder_and_blob_are_bounded_and_not_an_upstream_dependency(self):
        self.assertIn("[1, 32, 1, 64]", driver.mil(32, 64))
        self.assertNotIn("BLOBFILE", driver.mil(32, 64))
        self.assertIn("BLOBFILE", driver.mil(32, 64, "conv"))
        blob = driver.weights_blob(32)
        self.assertEqual(struct.unpack_from("<II", blob), (1, 2))
        magic, dtype, size, offset = struct.unpack_from("<IIQQ", blob, 64)
        self.assertEqual((magic, dtype, size, offset), (0xDEADBEEF, 1, 32 * 32 * 2, 128))
        self.assertEqual(len(blob), 128 + size)

    def test_help_needs_no_optional_modules_or_device(self):
        script = ROOT / "labs/ane/pmu_probe.py"
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(script), "--help"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_prepare_needs_no_optional_modules_or_device(self):
        script = ROOT / "labs/ane/pmu_probe.py"
        with tempfile.TemporaryDirectory() as directory:
            if shutil.disk_usage(directory).free < 256 * 1024**2:
                self.skipTest("PMU prepare requires 256 MiB free; retain the runtime admission gate")
            target = Path(directory) / "owned"
            result = subprocess.run([sys.executable, "-I", "-S", "-B", str(script), "--output-dir", str(target)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((target / "report.json").read_text())
            self.assertEqual(report["status"], "prepared-only-no-device")
            self.assertFalse(report["device_requested"])
            self.assertIsNone(report["native"])
            self.assertFalse((target / "pmu-probe").exists())
            self.assertEqual(report["contract_helper_sha256"], hashlib.sha256((target / "source/pmu.py").read_bytes()).hexdigest())

    def test_low_disk_refuses_before_source_model_or_device_work(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "owned"
            usage = shutil.disk_usage(directory)._replace(free=256 * 1024**2 - 1)
            with patch.object(sys, "argv", ["pmu_probe.py", "--output-dir", str(target)]), \
                    patch.object(driver.shutil, "disk_usage", return_value=usage), \
                    patch.object(driver.subprocess, "run") as native, \
                    patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(driver.main(), 1)
            report = json.loads((target / "report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["errors"], [{"type": "ValueError", "message": "256 MiB disk headroom required; no data deleted"}])
            self.assertIsNone(report["native"])
            self.assertFalse((target / "source").exists())
            self.assertFalse((target / "model").exists())
            self.assertFalse((target / "pmu-probe").exists())
            native.assert_not_called()


class PMUEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "review/ane-pmu-evidence-20261016.json").read_text())

    def sample(self):
        return copy.deepcopy(self.data["runs"][-1]["report"])

    def test_all_81_calls_independently_reconstructed_without_backend(self):
        self.assertEqual(len(self.data["runs"]), 9)
        self.assertEqual(self.data["curator_sha256"], hashlib.sha256((ROOT / "scripts/curate_pmu_campaign.py").read_bytes()).hexdigest())
        self.assertEqual(self.data["contract_helper_sha256"], hashlib.sha256((ROOT / "labs/pmu.py").read_bytes()).hexdigest())
        self.assertEqual(sum(len(run["report"]["native"]["samples"]) for run in self.data["runs"]), 81)
        for run in self.data["runs"]:
            self.assertEqual(curator.verify(run["report"]), run["independent_validation"])
            self.assertEqual(run["independent_validation"]["payload_state_counts"], {"missing-object": 9})
            self.assertIsNone(run["independent_validation"]["counter_rates"])
            self.assertTrue(run["child_exit_confirmed"])

    def test_counter_absence_is_exit3_not_a_numerical_or_permission_failure(self):
        last = self.data["runs"][-1]
        self.assertEqual(last["child_returncode"], 3)
        self.assertEqual(last["report"]["status"], "execution-passed-pmu-unavailable")
        self.assertEqual(last["report"]["native_returncode"], 0)
        self.assertEqual(last["report"]["counter_capture_status"], "unavailable-for-this-run")
        self.assertEqual(last["report"]["native"]["mask_translation_readonly"],
                         [{"anef_mask": 0, "driver_mask_returned": 0}, {"anef_mask": 2, "driver_mask_returned": 4},
                          {"anef_mask": 15, "driver_mask_returned": 15}])

    def test_upstream_false_positive_and_skipped_branch_are_preserved(self):
        controls = {run["id"]: run for run in self.data["upstream_cli_controls"]}
        missing = controls["upstream-missing"]
        self.assertTrue(missing["model_missing_reported"])
        self.assertTrue(missing["claimed_pmu_active"])
        self.assertEqual(missing["returncode"], 0)
        self.assertTrue(controls["upstream-owned-conv"]["no_stats_object_reported"])
        self.assertTrue(controls["upstream-debug-conv"]["no_stats_object_reported"])
        self.assertFalse(controls["capacity-owned"]["pmu_branch_entered_marker"])
        self.assertFalse(self.data["capacity_export"]["ane_bundle_present"])
        self.assertEqual(self.data["debug_signature_entitlements_readonly"], {"com.apple.security.get-task-allow": True})

    def test_hash_generation_layout_time_and_counter_promotions_refused(self):
        for change in ("input", "output", "generation", "time", "stride", "finite", "promote", "source", "offset", "mask"):
            value = self.sample()
            row = value["native"]["samples"][-1]
            if change == "input": row["input_sha256"] = "a" * 64
            elif change == "output": row["output_sha256"] = "a" * 64
            elif change == "generation": row["input_generation"] = 0
            elif change == "time": row["api_wall_ns"] = True
            elif change == "stride": value["native"]["input_descriptor"]["PlaneStride"] = 1
            elif change == "finite": row["all_outputs_finite"] = False
            elif change == "promote": value["counter_capture_status"] = "raw-only-unqualified"
            elif change == "source": value["source_inputs_unchanged"] = False
            elif change == "offset": value["native"]["input_descriptor"]["Offset"] = 32
            else: value["requested_mask"] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                curator.verify(value)


if __name__ == "__main__":
    unittest.main()
