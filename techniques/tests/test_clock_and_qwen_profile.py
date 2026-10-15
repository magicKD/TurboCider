"""CPU-only clock hypotheses and real-profile provenance/rejection contracts."""
import copy
import hashlib
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


clock = module("clock_calibration", "scripts/calibrate_metal_timestamps.py")
profiles = module("qwen_host_curator", "scripts/curate_qwen_profile.py")
traces = module("qwen_host_trace", "scripts/host_trace.py")
sys.path.insert(0, str(ROOT / "labs/metal"))
probe = module("qwen_host_probe", "labs/metal/qwen_ffn_profile.py")
sys.path.pop(0)


class ClockCalibrationTests(unittest.TestCase):
    def sample(self, unit="nanoseconds"):
        def reference(ns, gpu):
            ticks = ns * 3 // 125
            return {"cpu_raw": ns if unit=="nanoseconds" else ticks, "gpu_raw": gpu,
                    "mach_ticks_before":ticks-1,"mach_ticks_after":ticks+1,
                    "uptime_ns_before":ns-10,"uptime_ns_after":ns+10,
                    "mach_timebase_numer":125,"mach_timebase_denom":3,
                    "source_kind":"measured", "reference_clock":"mach_absolute_time and CLOCK_UPTIME_RAW; same process"}
        before, after = reference(10000000,5000000), reference(11000000,5500000)
        return {"trial":0,"warmup":False,"variant":"baseline","counter_diagnostic":{
            "source_kind":"measured","sampling_point":"stage-boundary",
            "reference_before":before,"reference_after":after,
            "cpu_pair_before_raw":before["cpu_raw"],"cpu_pair_after_raw":after["cpu_raw"],
            "gpu_pair_before_raw":before["gpu_raw"],"gpu_pair_after_raw":after["gpu_raw"],
            "samples":[{"label":"gemm","start_raw":5100000,"end_raw":5150000,"delta_raw":50000,"valid":True}],
            "completed_command_windows":[{"start_host_seconds":.0102,"end_host_seconds":.0103,"source_kind":"measured"}]}}

    def test_ns_and_mach_hypotheses_both_map_with_measured_timebase(self):
        for unit in ("nanoseconds","mach-ticks"):
            result = clock.calibrate_sample(self.sample(unit))
            self.assertEqual(result["status"],"calibration-and-window-pass")
            self.assertEqual(result["cpu_unit"],unit)
            self.assertEqual(result["gpu_ns_per_tick"],2.)
            self.assertAlmostEqual(result["events"][0]["qualified_duration_seconds"],.0001)
            self.assertFalse(result["formal_performance_score"])

    def test_correct_unit_with_wrong_trial_window_is_rejected(self):
        value = self.sample()
        window = value["counter_diagnostic"]["completed_command_windows"][0]
        window.update(start_host_seconds=.0106,end_host_seconds=.0107)
        result = clock.calibrate_sample(value)
        self.assertEqual(result["cpu_unit"],"nanoseconds")
        self.assertEqual(result["status"],"window-validation-failed")
        self.assertIsNone(result["events"][0]["qualified_duration_seconds"])

    def test_missing_historical_reference_is_not_retrofitted(self):
        value = self.sample()
        del value["counter_diagnostic"]["reference_before"]
        result = clock.calibrate_sample(value)
        self.assertEqual(result["status"],"not-calibrated")
        self.assertIsNone(result["gpu_ns_per_tick"])

    def test_unbracketed_or_excessively_wide_references_stay_unknown(self):
        for change in ({"cpu_raw":99999999},{"uptime_ns_after":99999999}):
            value = self.sample()
            value["counter_diagnostic"]["reference_before"].update(change)
            result = clock.calibrate_sample(value)
            self.assertIsNone(result["cpu_unit"])
            self.assertEqual(result["status"],"not-calibrated")

    def test_sentinel_and_bad_raw_types_never_become_seconds(self):
        value = self.sample()
        value["counter_diagnostic"]["samples"][0]["end_raw"] = 2**64-1
        result = clock.calibrate_sample(value)
        self.assertIsNone(result["events"][0]["qualified_duration_seconds"])
        for invalid in (True,-1,.5,float("nan")):
            value = self.sample()
            value["counter_diagnostic"]["reference_before"]["cpu_raw"] = invalid
            with self.subTest(value=invalid),self.assertRaises(ValueError):
                clock.calibrate_sample(value)

    def test_unknown_counter_provenance_and_predicted_command_refused(self):
        value = self.sample()
        value["counter_diagnostic"]["source_kind"]="predicted"
        self.assertEqual(clock.calibrate_sample(value)["status"],"not-calibrated")
        value = self.sample()
        value["counter_diagnostic"]["completed_command_windows"][0]["source_kind"]="synthetic"
        with self.assertRaises(ValueError):
            clock.calibrate_sample(value)

    def test_calibration_interval_and_pair_consistency_checked(self):
        for change in ("outside", "inconsistent"):
            value = self.sample()
            if change=="outside":
                value["counter_diagnostic"]["samples"][0].update(end_raw=5600000,delta_raw=500000)
                self.assertEqual(clock.calibrate_sample(value)["status"],"window-validation-failed")
            else:
                value["counter_diagnostic"]["cpu_pair_before_raw"]+=1
                self.assertEqual(clock.calibrate_sample(value)["status"],"not-calibrated")

    def test_real_corpus_keeps_alias_bug_and_all_scaling_runs(self):
        document = json.loads((ROOT / "review/gpu-clock-evidence-20261016.json").read_text())
        self.assertEqual(document["normalizer_sha256"],hashlib.sha256((ROOT / "scripts/calibrate_metal_timestamps.py").read_bytes()).hexdigest())
        self.assertEqual(len(document["runs"]),12)
        self.assertEqual(sum(run["status"]=="pass" for run in document["runs"]),11)
        self.assertEqual(document["runs"][0]["status"],"incomplete-or-failed")
        accepted = [run for run in document["runs"] if run["status"]=="pass"]
        self.assertEqual(sum(len(sample["events"]) for run in accepted for sample in run["samples"]),716)
        for run in accepted:
            self.assertFalse(run["production_qualified"])
            for sample in run["samples"]:
                self.assertEqual(sample["cpu_unit"],"nanoseconds")
                self.assertEqual(sample["gpu_ns_per_tick"],1.)


class QwenHostProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = json.loads((ROOT / "review/qwen-bf16-profiling-evidence-20261016.json").read_text())

    def document(self):
        return copy.deepcopy(self.corpus["runs"][0]["report"])

    def test_real_points_and_recomputed_summary_preserve_scope(self):
        self.assertEqual(self.corpus["curator_sha256"],hashlib.sha256((ROOT / "scripts/curate_qwen_profile.py").read_bytes()).hexdigest())
        seen=set()
        for run in self.corpus["runs"]:
            summary=profiles.summarize(run["report"])
            self.assertEqual(summary,run["summary"])
            seen.update(summary)
            for value in summary.values():
                self.assertIsNone(value["speedup"])
                self.assertTrue(value["staged_durations_are_not_compiled_kernel_costs"])
        self.assertEqual(seen,{"layer-0-step-1","layer-0-step-10","layer-0-step-19"})

    def test_numeric_pass_rejects_missing_nan_boolean_and_false_finite(self):
        good={"finite":True,"relative_l2":.001}
        self.assertTrue(probe.numeric_pass(good,good))
        for change in ({"relative_l2":None},{"relative_l2":float("nan")},{"relative_l2":True},{"finite":False}):
            bad={**good,**change}
            self.assertFalse(probe.numeric_pass(good,bad))
            value=self.document()
            value["formal_samples"][0]["replica"].update(change)
            with self.assertRaises(ValueError):profiles.summarize(value)

    def test_staged_cannot_become_formal_or_device_timestamps(self):
        for change in ("formal","device"):
            value=self.document()
            if change=="formal":value["diagnostic_samples"][0]["formal_performance_score"]=True
            else:value["physical_device_timestamps"]={"invented":True}
            with self.assertRaises(ValueError):profiles.summarize(value)

    def test_original_dtype_abi_and_fixed_gates_cannot_drift(self):
        for change in ({"source_dtype":"float16"},{"consumer_abi":[1024,4096]},
                       {"gates":{"replica_relative_l2_max":.5,"fp64_sample_relative_l2_max":.5}}):
            value=self.document()
            value.update(change)
            with self.assertRaises(ValueError):profiles.summarize(value)

    def test_order_gaps_duplicates_and_escaped_boundaries_rejected(self):
        for change in ("order","missing","boundary"):
            value=self.document()
            if change=="order":value["diagnostic_samples"][0]["host_events"].reverse()
            elif change=="missing":value["formal_samples"].pop(0)
            else:value["diagnostic_samples"][0]["host_boundary_ns"]=1
            with self.assertRaises(ValueError):profiles.summarize(value)

    def test_host_trace_requires_explicit_point_and_preserves_clock_scope(self):
        value=self.document()
        result=traces.convert_qwen(value,"layer-0-step-1",0)
        self.assertEqual(len(result["traceEvents"]),3)
        self.assertTrue(all(row["cat"]=="cpu-host-diagnostic" for row in result["traceEvents"]))
        self.assertIsNone(result["metadata"]["physical_device_timestamps"])
        self.assertFalse(result["metadata"]["native_counter_calibration_applied"])
        for point in (None,"missing"):
            with self.assertRaises(ValueError):traces.convert_qwen(value,point,0)
        value["diagnostic_samples"][0]["host_events"][0]["clock_domain"]="pretend-gpu-time"
        with self.assertRaises(ValueError):traces.convert_qwen(value,"layer-0-step-1",0)

    def test_curated_trace_works_offline_without_raw_files_or_frameworks(self):
        for run in (None,"missing"):
            with self.assertRaises(ValueError):traces.select_report(self.corpus,run)
        command=[sys.executable,"-I","-S","-B",str(ROOT / "scripts/host_trace.py"),
                 str(ROOT / "review/qwen-bf16-profiling-evidence-20261016.json"),
                 "--run","qwen-bf16-host-profile-v2","--point","layer-0-step-10","--trial","0"]
        child=subprocess.run(command,capture_output=True,text=True,timeout=10)
        self.assertEqual(child.returncode,0,child.stderr)
        value=json.loads(child.stdout)
        self.assertEqual(len(value["traceEvents"]),3)
        self.assertTrue(value["metadata"]["declared_source_hash_not_recomputed"])
        self.assertFalse(value["metadata"]["formal_performance_score"])


if __name__ == "__main__":
    unittest.main()
