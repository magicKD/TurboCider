"""Untrusted timing exports cannot turn heuristic values into device evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/normalize_gputrace.py"
SPEC = importlib.util.spec_from_file_location("normalize_gputrace", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
sys.path.insert(0, str(SCRIPT.parent))
import curate_profiling_campaign as curator
sys.path.pop(0)


class TraceNormalizationTests(unittest.TestCase):
    def document(self, source="synthetic", approximate=True):
        return {"timing_source": source, "timing_approximate": approximate,
                "total_duration": 5000000, "total_encoders": 1,
                "encoder_timings": [{"label": "gemm_scalar", "duration_ns": 5000000,
                                     "duration_ms": 5., "start_timestamp": 100, "end_timestamp": 5000100}]}

    def normalize(self, document):
        return adapter.normalize(document, adapter.AUDITED_COMMIT)

    def test_synthetic_and_extracted_are_predicted_even_with_false_marker(self):
        for source in ("synthetic", "extracted", "store0"):
            for approximate in (True, False, None):
                with self.subTest(source=source, approximate=approximate):
                    value = self.normalize(self.document(source, approximate))
                    self.assertEqual(value["events"][0]["reported_duration"]["source_kind"], "predicted")
                    self.assertIsNone(value["physical_gpu_duration_seconds"])
                    self.assertFalse(value["formal_performance_score"])
                    self.assertEqual(bool(value["issues"]), approximate is not True)

    def test_profiler_claim_does_not_establish_physical_endpoints(self):
        value = self.normalize(self.document("profiler", False))
        self.assertEqual(value["events"][0]["reported_duration"]["source_kind"], "unknown")
        self.assertIsNone(value["events"][0]["physical_gpu_duration_seconds"])
        self.assertTrue(value["issues"])
        self.assertFalse(value["production_qualified"])

    def test_empty_successful_export_is_missing_not_zero_gpu_time(self):
        value = self.normalize({"timing_source": "synthetic", "timing_approximate": True,
                                "total_duration": 0, "total_encoders": 0, "encoder_timings": None})
        self.assertEqual(value["status"], "no-usable-timing")
        self.assertEqual(value["reported_total_duration"], 0)
        self.assertIsNone(value["physical_gpu_duration_seconds"])
        self.assertIsNone(value["physical_dispatch_count"])

    def test_zero_duration_and_unknown_parser_cannot_be_promoted(self):
        document = self.document()
        document["encoder_timings"][0]["duration_ns"] = 0
        value = self.normalize(document)
        self.assertIsNone(value["events"][0]["reported_duration"]["value"])
        value = adapter.normalize(self.document(), "0" * 40)
        self.assertEqual(value["events"][0]["reported_duration"]["source_kind"], "unknown")
        self.assertIn("parser-commit-not-audited", value["issues"])

    def test_invalid_uint64_duration_and_metadata_rejected(self):
        for invalid in (True, -1, 2**64, .5, float("nan"), float("inf"), None):
            document = self.document()
            document["encoder_timings"][0]["duration_ns"] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.normalize(document)
        for invalid in (0, "false", [], {}):
            document = self.document()
            document["timing_approximate"] = invalid
            with self.subTest(approximate=invalid), self.assertRaises(ValueError):
                self.normalize(document)

    def test_raw_metadata_paths_and_addresses_are_not_forwarded(self):
        document = self.document()
        document.update(metadata={"uuid": "private-device-id"}, trace_path="/private/machine/run", command="do not execute")
        document["encoder_timings"][0].update(label="/private/machine/shader", queue_id=12345)
        before = copy.deepcopy(document)
        value = self.normalize(document)
        encoded = json.dumps(value)
        for private in ("private-device-id", "/private/machine", "queue_id", "do not execute"):
            self.assertNotIn(private, encoded)
        self.assertEqual(document, before)
        self.assertEqual(value["events"][0]["label"], "<redacted-label>")
        self.assertTrue(value["events"][0]["label_redacted"])

    def test_missing_source_and_count_mismatch_are_explicit(self):
        document = self.document()
        del document["timing_source"]
        document["total_encoders"] = 5
        value = self.normalize(document)
        self.assertIn("missing-or-unknown-timing-source", value["issues"])
        self.assertIn("reported-encoder-count-does-not-match-rows", value["issues"])

    def test_identifier_like_and_credential_like_labels_are_redacted(self):
        for label in ("00112233-4455-6677-8899-aabbccddeeff", "sk-" + "syntheticcanary" * 3):
            document = self.document()
            document["encoder_timings"][0]["label"] = label
            value = self.normalize(document)
            self.assertNotIn(label, json.dumps(value))
            self.assertTrue(value["events"][0]["label_redacted"])

    def test_cli_reads_only_explicit_json_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "timing.json"
            output = Path(directory) / "normalized.json"
            source.write_text(json.dumps(self.document()), encoding="utf-8")
            command = [sys.executable, "-I", "-S", "-B", str(SCRIPT), str(source),
                       "--parser-commit", adapter.AUDITED_COMMIT, "--output", str(output)]
            child = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(child.returncode, 0, child.stderr)
            encoded = output.read_bytes()
            self.assertFalse(json.loads(encoded)["raw_metadata_exported"])
            child = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(child.returncode, 0)
            self.assertEqual(output.read_bytes(), encoded)
            source.write_text('{"total_duration": NaN}', encoding="utf-8")
            child = subprocess.run(command[:-2], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(child.returncode, 0)


class NativeProfilingEvidenceTests(unittest.TestCase):
    def document(self, mode="none"):
        def sample(trial):
            candidate = trial % 4 in (1, 2)
            counter = None
            if mode.startswith("counter-"):
                counter = {"sampling_point": "stage-boundary", "duration_seconds": None,
                           "samples": [{"start_raw": 100, "end_raw": 200, "delta_raw": 100, "valid": True}]}
            return {"trial": trial, "warmup": trial < 0, "pair": trial // 2,
                    "variant": "candidate" if candidate else "baseline", "input_generation": trial // 2 + 2,
                    "wall_seconds": 1. if candidate else 2., "gpu_command_seconds_sum": .1,
                    "counter_diagnostic": counter}
        return {"schema": "turbocider-metal-teaching-lab-v1", "actual_metal_dispatch": True,
                "all_outputs_finite": True, "tail_guards_intact": True, "source_inputs_unchanged": True,
                "arguments": {"repeats": 4}, "diagnostic_mode": mode, "formal_performance_score": True,
                "samples": [sample(trial) for trial in range(4)],
                "warmup_samples": [sample(trial) for trial in range(-4, 0)]}

    def test_profiled_report_never_receives_formal_ratio(self):
        for mode in ("none", "counter-stage", "capture"):
            result = curator.native_summary(self.document(mode))
            self.assertEqual(result["median_paired_host_ratio"], 2. if mode == "none" else None)
            self.assertEqual(result["formal_operator_score"], mode == "none")
            self.assertFalse(result["production_qualified"])

    def test_zero_timestamps_missing_pairs_and_changed_sources_rejected(self):
        for field, invalid in (("wall_seconds", 0), ("gpu_command_seconds_sum", 0),
                               ("gpu_command_seconds_sum", float("nan")), ("wall_seconds", True)):
            value = self.document()
            value["samples"][0][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                curator.native_summary(value)
        value = self.document()
        value["samples"][1]["input_generation"] = 19
        with self.assertRaises(ValueError):
            curator.native_summary(value)
        value = self.document()
        value["source_inputs_unchanged"] = False
        with self.assertRaises(ValueError):
            curator.native_summary(value)

    def test_counter_error_sentinel_is_not_a_zero_or_a_valid_measurement(self):
        value = self.document("counter-stage")
        counter = value["samples"][0]["counter_diagnostic"]["samples"][0]
        counter["end_raw"] = 2**64 - 1
        with self.assertRaises(ValueError):
            curator.native_summary(value)
        counter.update(valid=False, delta_raw=None)
        result = curator.native_summary(value)
        self.assertIsNone(result["median_paired_host_ratio"])

    def test_counter_units_cannot_silently_become_seconds(self):
        value = self.document("counter-stage")
        value["samples"][0]["counter_diagnostic"]["duration_seconds"] = .001
        with self.assertRaises(ValueError):
            curator.native_summary(value)

    def test_curated_corpus_keeps_failures_sources_and_counter_units(self):
        evidence = json.loads((ROOT / "review/gpu-profiling-evidence-20261016.json").read_text())
        self.assertEqual(len(evidence["native_runs"]), 11)
        self.assertEqual(len(evidence["captures"]), 10)
        self.assertFalse(evidence["production_qualified"])
        self.assertEqual(evidence["curator_sha256"], hashlib.sha256((ROOT / "scripts/curate_profiling_campaign.py").read_bytes()).hexdigest())
        self.assertEqual(evidence["normalizer_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
        failures = [run for run in evidence["native_runs"] if run["launch"]["returncode"] != 0]
        self.assertEqual(len(failures), 1)
        self.assertIsNone(failures[0]["summary"])
        for run in evidence["native_runs"]:
            if run["launch"]["returncode"] != 0:
                continue
            expected = curator.native_summary({**run, "schema": "turbocider-metal-teaching-lab-v1"})
            self.assertEqual(run["summary"], expected)
            for sample in run["samples"] + run["warmup_samples"]:
                if sample["counter_diagnostic"]:
                    self.assertIsNone(sample["counter_diagnostic"]["duration_seconds"])
        self.assertEqual(sum(run["stats_launch"]["returncode"] != 0 for run in evidence["captures"]), 5)
        for capture in evidence["captures"]:
            self.assertEqual(capture["normalized_timing"]["reported_source"], "synthetic")
            self.assertIsNone(capture["normalized_timing"]["physical_gpu_duration_seconds"])


if __name__ == "__main__":
    unittest.main()
