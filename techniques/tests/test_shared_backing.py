"""Offline complete-tensor, backing, stride and lifecycle tests; no device imports."""
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
import shared_backing as backing

SPEC = importlib.util.spec_from_file_location("shared_surface_curator", ROOT / "scripts/curate_shared_surface.py")
curator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curator)


class BackingContracts(unittest.TestCase):
    def test_full_dyadic_witness_includes_real_channels_and_tail(self):
        source = backing.source_values(32, 33, 0, 17)
        self.assertEqual(len(source), 1056)
        self.assertEqual(source[0], -1.)
        self.assertNotEqual(source[:33], source[33:66])
        expected = hashlib.sha256(struct.pack("<" + "e" * len(source), *source)).hexdigest()
        self.assertEqual(backing.witness(32, 33, 0, 17)["input_sha256"], expected)
        self.assertNotEqual(expected, backing.witness(32, 33, 1, 17)["input_sha256"])
        for args in ((True, 33, 0, 17), (32, 32, 0, 17), (32, 33, True, 17), (32, 33, 0, 10001)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                backing.witness(*args)

    def test_offsets_pitch_dtype_and_bool_layouts_are_not_guessed(self):
        base = {"Batches": 1, "Channels": 32, "Height": 1, "Width": 33, "Type": "Float16",
                "RowStride": 128, "PlaneStride": 128, "BatchStride": 4096}
        self.assertEqual(backing.physical_layout(base, 32, 33), 4096)
        for key, value in (("Batches", True), ("Type", "Float32"), ("RowStride", 64),
                           ("PlaneStride", 129), ("BatchStride", 4000), ("Offset", 2), ("Depth", 2), ("PlaneCount", 1)):
            row = dict(base); row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                backing.physical_layout(row, 32, 33)

    def test_identity_requires_actual_input_and_output_aliases(self):
        values = {key: True for key in backing.IDENTITY_KEYS}
        backing.identity(values)
        for key in values:
            row = dict(values); row[key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                backing.identity(row)

    def test_default_cli_only_prepares_and_never_runs_a_device(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            child = subprocess.run([sys.executable, "-E", "-B", str(ROOT / "labs/ane/shared_surface_probe.py"),
                                    "--output-dir", str(output)], text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(child.stdout)["status"], "prepared-only-no-device")
            value = json.loads((output / "report.json").read_text())
            self.assertIsNone(value["native"])
            self.assertFalse(value["device_requested"])
            self.assertFalse((output / "shared-probe").exists())
            self.assertIn("[1, 32, 1, 33]", (output / "model/model.mil").read_text())
            self.assertTrue(value["source_inputs_unchanged"])


class SharedEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "review/private-shared-evidence-20261016.json").read_text())

    def record(self):
        return copy.deepcopy(self.data["runs"][-1]["report"])

    def test_all_four_processes_and_120_full_calls_are_retained(self):
        self.assertEqual(len(self.data["runs"]), 4)
        self.assertEqual(sum(len(row["report"]["native"]["samples"]) for row in self.data["runs"]), 120)
        self.assertEqual(self.data["curator_sha256"], hashlib.sha256((ROOT / "scripts/curate_shared_surface.py").read_bytes()).hexdigest())
        self.assertEqual(self.data["contract_sha256"], hashlib.sha256((ROOT / "labs/shared_backing.py").read_bytes()).hexdigest())
        for run in self.data["runs"]:
            self.assertTrue(run["child_exit_confirmed"])
            self.assertEqual(backing.verify(run["report"]), run["independent_validation"])
            for name, digest in run["report"]["source_sha256"].items():
                self.assertEqual(digest, hashlib.sha256((ROOT / "labs/ane" / name).read_bytes()).hexdigest())
        self.assertEqual({run["report"]["seed"] for run in self.data["runs"]}, {17, 29})
        self.assertEqual({tuple(run["report"]["shape_nchw"]) for run in self.data["runs"]}, {(1, 32, 1, 33), (1, 64, 1, 65)})

    def test_booleans_cannot_upgrade_backing_to_physical_or_runtime_proof(self):
        for key, value in (("production_qualified", True), ("physical_overlap_qualified", True),
                           ("runtime_internal_copy_behavior", "none"), ("full_system_zero_copy", True), ("native_returncode", False)):
            row = self.record(); row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): backing.verify(row)
        row = self.record(); row["native"]["initial_identity"]["request_output_surface_matches"] = False
        with self.assertRaises(ValueError): backing.verify(row)

    def test_missing_calls_changed_source_and_hashes_are_refused(self):
        for kind in ("call", "source", "shape", "input", "output", "consumer", "metric-bool", "order"):
            row = self.record(); native = row["native"]; sample = native["samples"][0]
            if kind == "call": native["samples"].pop()
            elif kind == "source": row["source_inputs_unchanged"] = False
            elif kind == "shape": native["shape_nchw"][0] = True
            elif kind == "input": sample["input_sha256"] = "0" * 64
            elif kind == "output": sample["ane_output_sha256"] = "0" * 64
            elif kind == "consumer": sample["consumer"]["consumer_sha256"] = "0" * 64
            elif kind == "metric-bool": sample["consumer"]["mismatched_elements"] = False
            else: native["samples"][:2] = native["samples"][:2][::-1]
            with self.subTest(kind=kind), self.assertRaises(ValueError): backing.verify(row)

    def test_copy_boundary_completion_owner_and_guard_changes_are_refused(self):
        for kind in ("count", "bytes", "owner", "quarantine", "guard", "poison", "completion", "page", "plane"):
            row = self.record(); native = row["native"]
            sample = next(s for s in native["samples"] if s["arm"] == "shared")
            if kind == "count": sample["external_feature_copy_count"] = 2
            elif kind == "bytes": sample["external_feature_copy_bytes"] = 1
            elif kind == "owner": native["unload_succeeded"] = False
            elif kind == "quarantine": native["mapping_reuse_after_failure"] = True
            elif kind == "guard": sample["tail_guards_pass"] = False
            elif kind == "poison": sample["output_poison_overwritten"] = False
            elif kind == "completion": sample["gpu_completions_checked"] = False
            elif kind == "page": native["surface_allocation_bytes"][0] -= 1
            else: native["output_descriptor"]["PlaneStride"] = 130
            with self.subTest(kind=kind), self.assertRaises(ValueError): backing.verify(row)

    def test_same_process_host_order_and_all_copy_events_are_required(self):
        for kind in ("clock", "overlap", "duration", "copy", "wall"):
            row = self.record(); sample = next(s for s in row["native"]["samples"] if s["arm"] == "copy")
            if kind == "clock": sample["host_events"][0]["clock_domain"] = "unmatched-gpu-clock"
            elif kind == "overlap": sample["host_events"][-1]["start_ns"] = 0
            elif kind == "duration": sample["host_events"][0]["duration_ns"] = True
            elif kind == "copy": sample["host_events"].pop(1)
            else: sample["wall_ns"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError): backing.verify(row)

    def test_finite_wrong_stride_and_actual_stale_output_still_fail(self):
        for kind in ("stride", "stale", "hash", "guard", "ratio"):
            row = self.record(); control = row["native"]["negative_controls"]
            if kind == "stride": control["packed_stride"]["mismatched_elements"] = 0
            elif kind == "stale": control["stale_generation"]["mismatched_elements"] = 0
            elif kind == "hash": control["stale_generation"]["consumer_sha256"] = "0" * 64
            elif kind == "guard": control["consumer_tail_guard_pass"] = False
            else: control["used_for_speed_ratio"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError): backing.verify(row)
        self.assertEqual(self.record()["native"]["negative_controls"]["packed_stride"]["nonfinite_elements"], 0)

    def test_curator_does_not_discover_other_runs_or_escaping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError): curator.curate(root, [])
            with self.assertRaises(ValueError): curator.curate(root, ["same", "same"])
            for value in ("../report.json", "/report.json", "missing.json"):
                with self.subTest(value=value), self.assertRaises(ValueError): curator.local(root, value)


if __name__ == "__main__":
    unittest.main()
