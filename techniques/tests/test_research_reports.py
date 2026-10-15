"""Evidence cannot silently become a qualified performance or device claim."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


curator = module("curate_ffn_campaign", "scripts/curate_ffn_campaign.py")
traces = module("host_trace", "scripts/host_trace.py")


class ReportTests(unittest.TestCase):
    def document(self):
        samples = [{"trial": index, "warmup": False, "pair": index // 2, "point": "synthetic",
                    "variant": "candidate" if index % 4 in (1, 2) else "gpu", "component_numeric_pass": True,
                    "wall_seconds": 1. if index % 4 in (1, 2) else 2.} for index in range(8)]
        return {"status": "component_numeric_pass", "samples": samples}

    def test_paired_stats_deterministic_and_derived(self):
        value = curator.paired_stats(self.document())
        self.assertEqual(value["median_paired_speedup"], 2.)
        self.assertEqual(value["exploratory_bootstrap_95_percentile_interval"], [2., 2.])
        self.assertEqual(value["source_kind"], "derived")
        self.assertEqual(curator.paired_stats(self.document()), value)

    def test_failed_numerics_withhold_speedups(self):
        value = self.document()
        value["samples"][1]["component_numeric_pass"] = False
        self.assertIsNone(curator.paired_stats(value))
        value["status"] = "failed"
        self.assertIsNone(curator.paired_stats(value))

    def test_missing_duplicate_mismatched_and_nonfinite_samples_rejected(self):
        base = self.document()
        variants = []
        value = copy.deepcopy(base); value["samples"].pop(); variants.append(value)
        value = copy.deepcopy(base); value["samples"][-1] = copy.deepcopy(value["samples"][0]); variants.append(value)
        value = copy.deepcopy(base); value["samples"][1]["point"] = "other"; variants.append(value)
        for invalid in (True, 0, -1, float("nan"), float("inf")):
            value = copy.deepcopy(base); value["samples"][0]["wall_seconds"] = invalid; variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                curator.paired_stats(value)

    def test_generated_source_index_matches_canonical_registry(self):
        child = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/generate_source_index.py"), "--check"],
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_curated_real_evidence_has_no_failed_run_speedup(self):
        evidence = json.loads((ROOT / "review/qwen-ffn-tiling-evidence-20261015.json").read_text())
        self.assertFalse(evidence["production_qualified"])
        self.assertEqual(evidence["physical_ane_placement"], "unverified")
        for run in evidence["runs"]:
            if run["status"] == "failed":
                self.assertIsNone(run["paired_statistics"])
            if run.get("plan_summary"):
                self.assertIsNone(run["plan_summary"]["physical_device_timing"])

    def test_policy_diagnostic_collection_is_not_numeric_qualification(self):
        evidence=json.loads((ROOT/"review/qwen-ffn-tiling-evidence-20261015.json").read_text())
        value=evidence["diagnostics"][0]["data"]
        self.assertEqual(value["status"],"collected")
        self.assertEqual(value["compiled_assets_before"],value["compiled_assets_after"])
        self.assertEqual([call["component_sample_gate_pass"] for call in value["calls"]],[True,False,False,True])
        self.assertTrue(all(call["input_unchanged_after_call"] for call in value["calls"]))
        self.assertTrue(all(call["timing_qualified"] is False for call in value["calls"]))
        self.assertFalse(value["production_qualified"])

    def test_host_trace_unit_and_scope_not_physical_device(self):
        value = {"schema": "tc-coreml-tiling-experiment-v1", "route": "hybrid", "runner_sha256": "demo",
                 "samples": [{"trial": 1, "warmup": False, "variant": "candidate", "point": "fixture",
                              "component_numeric_pass": True, "host_events": [{"name": "coreml-predict-host-call-and-output-copy",
                                  "start_ns": 1000, "duration_ns": 3000, "source_kind": "measured",
                                  "clock_domain": "this-process-perf-counter-host-only"}]}]}
        result = traces.convert(value, 1)
        event = result["traceEvents"][-1]
        self.assertEqual((event["ts"], event["dur"], event["tid"]), (1., 3., 2))
        self.assertIn("no GPU/ANE", result["metadata"]["scope"])
        value["samples"][0]["host_events"][0]["clock_domain"] = "unknown-gpu-clock"
        with self.assertRaises(ValueError):
            traces.convert(value, 1)


if __name__ == "__main__":
    unittest.main()
