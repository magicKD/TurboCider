"""Offline real-private-FFN source/gate/clock regressions; no GPU/ANE imports."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
import silu_domain
SPEC = importlib.util.spec_from_file_location("private_qwen_data", ROOT / "scripts/curate_qwen_private_ffn.py")
curator = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(curator)


class StableDomain(unittest.TestCase):
    def test_extreme_source_has_no_exp_overflow_in_the_cpu_formula(self):
        for value in (-1000., -100., -12., -1., 0., 1., 12., 100., 1000.):
            result = silu_domain.stable_silu(value)
            self.assertTrue(math.isfinite(result))
            if abs(value) < 100:
                self.assertAlmostEqual(result, value / (1 + math.exp(-value)), places=14)
        self.assertEqual(silu_domain.stable_silu(0.), 0.)
        for value in (float("nan"),float("inf")):
            with self.assertRaises(ValueError): silu_domain.stable_silu(value)

    def test_gate_up_and_down_partition_uses_the_original_paired_source(self):
        value = silu_domain.channel_slices(12288,2048)
        self.assertEqual(value["gate"],[0,2048]); self.assertEqual(value["up"],[12288,14336])
        self.assertEqual(value["down_columns"],[0,2048]); self.assertEqual(value["gpu_down_columns"],[2048,12288])
        for width,share in ((12,0),(12,12),(12,True),(True,1)):
            with self.assertRaises(ValueError): silu_domain.channel_slices(width,share)

    def test_stdlib_npy_reader_checks_dtype_shape_order_and_byte_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.npy"
            def data(dtype="<f2", fortran=False, payload=b"\0"*8):
                text = repr({"descr":dtype,"fortran_order":fortran,"shape":(2,2)}).encode("ascii")+b"\n"
                return b"\x93NUMPY\x01\x00"+struct.pack("<H",len(text))+text+payload
            path.write_bytes(data())
            self.assertEqual(curator.npy_span(path,(2,2)),path.stat().st_size-8)
            for raw in (data(dtype="<f4"),data(fortran=True),data(payload=b"\0")):
                path.write_bytes(raw)
                with self.assertRaises(ValueError): curator.npy_span(path,(2,2))
            with self.assertRaises(ValueError): curator.local(Path(directory),"../fixture.npy")


class QwenPrivateEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT/"review/qwen-private-ffn-evidence-20261016.json").read_text())

    def record(self):
        return copy.deepcopy(self.data["runs"][-1])

    def test_all_complete_consumers_and_prototype_failure_are_retained(self):
        self.assertEqual(len(self.data["runs"]),7); self.assertEqual(len(self.data["failures"]),1)
        self.assertEqual(sum(len(r["report"]["samples"]) for r in self.data["runs"]),297)
        self.assertEqual(self.data["curator_sha256"],hashlib.sha256((ROOT/"scripts/curate_qwen_private_ffn.py").read_bytes()).hexdigest())
        for run in self.data["runs"]:
            self.assertEqual(curator.verify_report(run["report"],run["group"]),run["independent_data_validation"])
            self.assertNotIn("checkpoint_stat_before",run["report"])
            self.assertTrue(run["child_exit_confirmed"])
        failure = self.data["failures"][0]["report"]
        self.assertEqual(failure["samples"],[]); self.assertIsNone(failure["comparisons"])
        self.assertIn("PEP 3118",failure["errors"][0]["message"])

    def test_protected_witness_and_domain_code_were_frozen_for_all_numeric_runs(self):
        for run in self.data["runs"]:
            for name,path in (("qwen_witness.py","labs/ane/qwen_witness.py"),("silu_domain.py","labs/silu_domain.py")):
                snapshot = next(p for p in run["report"]["source_snapshots"] if p["path"].endswith(name))
                self.assertEqual(snapshot["sha256"],hashlib.sha256((ROOT/path).read_bytes()).hexdigest())

    def test_confirmation_has_frozen_share_and_held_out_timesteps(self):
        confirmed = [r for r in self.data["runs"] if r["group"]=="confirmation"]
        self.assertEqual({r["report"]["point"] for r in confirmed},{"layer-0-step-10","layer-0-step-19"})
        self.assertEqual({r["report"]["share"] for r in confirmed},{2048})
        self.assertEqual(confirmed[-1]["report"]["measured_groups"],24)
        self.assertTrue(confirmed[-1]["report"]["pre_arm_validation_drain"])
        self.assertTrue(all(r["report"]["comparisons"][1]["median_paired_host_ratio"]>1 for r in confirmed))
        self.assertFalse(any(r["report"]["production_qualified"] for r in confirmed))

    def test_weak_or_changed_gpu_denominator_and_missing_prefix_gate_are_refused(self):
        for change in ("gpu-math","missing-prefix","false-marker","budget","dtype","source","abs-error"):
            row=self.record(); value=row["report"]
            if change=="gpu-math": value["samples"][0]["metrics"]["full_original"]["relative_l2"] = .001
            elif change=="missing-prefix": del value["samples"][1]["metrics"]["prefix_original"]
            elif change=="false-marker": value["samples"][1]["metrics"]["prefix_original"]["relative_l2"] = .031
            elif change=="budget": value["gates"]["relative_l2_max"] = .1
            elif change=="dtype": value["samples"][1]["output_dtype"] = "float32"
            elif change=="source": value["source_inputs_unchanged"] = False
            else: value["samples"][1]["metrics"]["full_original"]["max_abs"] = True
            with self.subTest(change=change),self.assertRaises(ValueError): curator.verify_report(value,row["group"])

    def test_order_point_slice_and_copy_boundary_changes_are_refused(self):
        for change in ("order","point","slice","boundary","copy","residency"):
            row=self.record(); value=row["report"]
            if change=="order": value["samples"][-3:] = value["samples"][-3:][::-1]
            elif change=="point": value["samples"][-1]["point"]="layer-0-step-1"
            elif change=="slice": value["channel_slices"]["up"]=[0,2048]
            elif change=="boundary": value["boundary"]="private-execute-only"
            elif change=="copy": value["zero_copy"]=True
            else: value["logical_gpu_weight_bytes"]=0
            with self.subTest(change=change),self.assertRaises(ValueError): curator.verify_report(value,row["group"])

    def test_same_process_host_intersections_do_not_grant_physical_overlap(self):
        record=self.record(); validation=curator.verify_report(record["report"],record["group"])
        events=validation["host_intersections"]
        self.assertTrue(all(r["api_host_intersection_ns"]==0 for r in events if r["arm"]=="serial"))
        self.assertTrue(any(r["api_host_intersection_ns"]>0 for r in events if r["arm"]=="hybrid"))
        self.assertTrue(all(r["physical_overlap_qualified"] is False for r in events))
        for change in ("clock","time","join"):
            value=self.record()["report"]
            row=next(r for r in value["samples"] if r["arm"]=="hybrid")
            if change=="clock": row["host_events"][0]["clock_domain"]="unmatched-gpu-clock"
            elif change=="time": row["host_events"][0]["duration_ns"]=True
            else: next(e for e in row["host_events"] if "join" in e["name"])["start_ns"]=0
            with self.assertRaises(ValueError): curator.verify_report(value,"confirmation")

    def test_relabelled_ratio_or_confirmation_parameters_are_refused(self):
        value=self.record()["report"]; value["comparisons"][1]["median_paired_host_ratio"]+=.1
        with self.assertRaises(ValueError): curator.verify_report(value,"confirmation")
        value=copy.deepcopy(self.data["runs"][1]["report"])
        with self.assertRaises(ValueError): curator.verify_report(value,"confirmation")

    def test_all_row_domain_witness_is_not_a_private_intermediate_or_request_proof(self):
        value=self.record()["report"]["full_prefix_gpu_gate_domain"]
        self.assertEqual(value["elements"],1024*2048)
        self.assertEqual(value["abs_over_11"],0)
        self.assertTrue(value["not_a_proof_of_private_fp16_intermediate_values"])


if __name__=="__main__": unittest.main()
