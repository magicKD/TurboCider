"""Portable timeline/compile/fixed-runtime gates; no Metal or ANE work."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/profiling/qwen_base"))
import analyze
import collect


def fixture():
    def row(name, begin, end, request, model=0, gpu=(0, 0)):
        return dict(name=name, begin_ns=begin, end_ns=end, request=request, model=model, queue=1,
                    thread=1, ok=True, gpu_begin_ns=gpu[0], gpu_end_ns=gpu[1])
    rows = [row("metal.command_buffer", 10, 40, -2, gpu=(20, 30)),
            row("ane.compile_api", 101, 110, 0, 1), row("ane.load_api", 111, 115, 0, 1)]
    requests = []
    for i, begin in enumerate((100, 300)):
        requests.append(dict(index=i, begin_ns=begin, end_ns=begin+100))
        rows.extend([row("ane.evaluate_api", begin+20, begin+25, i, 1),
                     row("metal.command_buffer", begin+20, begin+90, i, gpu=(begin+30, begin+60)),
                     row("metal.command_buffer", begin+40, begin+95, i, gpu=(begin+50, begin+70))])
    return dict(enabled=True, drained=True, restored_imps=True, pending=0, dropped=0,
                events=rows, counts=dict(Counter(row["name"] for row in rows))), requests


class BaseProfilingTests(unittest.TestCase):
    def test_interval_union_does_not_double_count_overlap(self):
        self.assertEqual(analyze.union_ns([(1, 5), (3, 7), (10, 11)]), 7)
        self.assertEqual(analyze.union_ns([(1, 9), (2, 3), (9, 10)]), 9)
        self.assertEqual(analyze.union_ns([]), 0)
        with self.assertRaises(ValueError):
            analyze.union_ns([(4, 3)])

    def test_actual_compile_first_use_and_no_hot_recompile_are_separate(self):
        value, requests = fixture()
        result = analyze.observer_contract(value, requests, True)
        self.assertEqual(result["runtime_model_count"], 1)
        self.assertEqual(result["requests"][0]["api_calls"]["ane.compile_api"], 1)
        self.assertIs(result["hot_no_compile_observed"], True)
        self.assertAlmostEqual(result["requests"][0]["observed_gpu_window_union_seconds"], 40e-9)
        extra = copy.deepcopy(value["events"][1])
        extra.update(request=1, begin_ns=301, end_ns=310)
        value["events"].append(extra)
        value["counts"]["ane.compile_api"] += 1
        self.assertIs(analyze.observer_contract(value, requests, True)["hot_no_compile_observed"], False)

    def test_one_first_request_cannot_prove_steady_state_no_compile(self):
        value, requests = fixture()
        self.assertIsNone(analyze.observer_contract(value, requests[:1], True)["hot_no_compile_observed"])

    def test_dropped_pending_disabled_and_unrestored_are_rejected(self):
        original, requests = fixture()
        for key, bad in (("enabled", False), ("drained", False), ("restored_imps", False), ("pending", 1), ("dropped", 1)):
            value = {**original, key: bad}
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze.observer_contract(value, requests, True)

    def test_different_runtime_model_or_failed_call_is_rejected(self):
        value, requests = fixture()
        value["events"][-3]["model"] = 2
        with self.assertRaises(ValueError):
            analyze.observer_contract(value, requests, True)
        value, requests = fixture()
        value["events"][-3]["ok"] = False
        with self.assertRaises(ValueError):
            analyze.observer_contract(value, requests, True)

    def test_missing_calibration_wrong_clock_and_forged_counters_are_rejected(self):
        value, requests = fixture()
        value["events"][0]["gpu_end_ns"] = 100000
        with self.assertRaises(ValueError):
            analyze.observer_contract(value, requests, True)
        value, requests = fixture()
        value["events"][0]["gpu_begin_ns"] = 0
        with self.assertRaises(ValueError):
            analyze.observer_contract(value, requests, True)
        value, requests = fixture()
        value["counts"]["ane.compile_api"] += 1
        with self.assertRaises(ValueError):
            analyze.observer_contract(value, requests, True)

    def test_gpu_control_cannot_be_reported_as_runtime(self):
        value, requests = fixture()
        with self.assertRaises(ValueError):
            analyze.observer_contract(value, requests, False)

    def test_cache_admission_and_diagnostic_fences_are_explicit(self):
        flags = collect.environment("cache", "timeline")
        self.assertEqual(int(flags["TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES"]), 1600 << 20)
        self.assertEqual(flags["TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC"], "1")
        self.assertEqual(collect.environment("runtime", "runtime-layers")["TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC"], "0")
        self.assertNotIn("TURBOCIDER_QWEN21_PROFILE_STEPS", collect.environment("gpu", "stock"))
        self.assertFalse(any("LORA" in key or "STUDENT" in key for key in flags))

    def test_fixed_larger_partition_is_not_dynamic_resize_or_cache_override(self):
        self.assertEqual(collect.environment("runtime", "stock", 5120)["TURBOCIDER_PRIVATE_ANE_CHANNELS"], "5120")
        self.assertEqual(collect.environment("runtime", "timeline", 6144)["TURBOCIDER_PRIVATE_ANE_CHANNELS"], "6144")
        for route, channels in (("cache", 6144), ("gpu", 6144), ("cache", 5120), ("gpu", 5120), ("runtime", 7000), ("other", 4096)):
            with self.subTest(route=route, channels=channels), self.assertRaises(ValueError):
                collect.environment(route, "timeline", channels)

    def test_export_uses_numeric_trace_lanes_and_preserves_historical_outputs(self):
        observer, runs = fixture()
        for row in runs:
            row.update(request_wall_including_export=.1, metrics=dict(acceleration_selection="base runtime", timings_seconds={}))
        with tempfile.TemporaryDirectory(prefix="tc-base-profiler-export-") as directory:
            root = Path(directory) / "raw"
            (root / "trial").mkdir(parents=True)
            (root / "stderr.log").write_text("")
            (root / "trial/report.json").write_text(json.dumps(dict(status="complete", runs=runs, mode="timeline", route="runtime", progress=[])))
            (root / "trial/observer.json").write_text(json.dumps(observer))
            output = Path(directory) / "fresh-export"
            analyze.analyze_directory(root, output)
            self.assertFalse((root / "trace.json").exists())
            trace = json.loads((output / "trace.json").read_text())
            self.assertTrue(trace["traceEvents"])
            self.assertTrue(all(type(row["pid"]) is int and type(row["tid"]) is int for row in trace["traceEvents"]))
            with self.assertRaises(FileExistsError):
                analyze.analyze_directory(root, output)

    def test_custom_prompt_changes_only_actual_prompt_not_model_recipe(self):
        args = SimpleNamespace(route="runtime", case="generate", requests=3, output=Path("/synthetic-output"),
                               manifest=Path("/synthetic-manifest"), reference=Path("/synthetic-reference"), prompt=None)
        ordinary = collect.requests(args)
        args.prompt = "玻璃茶壶，中文标签 TEST"
        changed = collect.requests(args)
        for original, custom in zip(ordinary, changed):
            self.assertEqual(custom["prompt"], args.prompt)
            self.assertEqual({k: v for k, v in original.items() if k != "prompt"}, {k: v for k, v in custom.items() if k != "prompt"})
            self.assertNotIn("loras", custom)

    def test_sequence_matrix_reuses_one_recipe_and_restores_generation(self):
        args = SimpleNamespace(route="runtime", case="generate", requests=4, mode="timeline", scenario="sequence-matrix",
                               output=Path("/synthetic-output"), manifest=Path("/synthetic-manifest"), reference=Path("/synthetic-reference"), prompt=None)
        values = collect.requests(args)
        self.assertEqual([v["operation"] for v in values], ["image.generate", "image.edit", "image.edit", "image.generate"])
        self.assertEqual([len(v.get("inputs", [])) for v in values], [0, 1, 2, 0])
        self.assertEqual([v["seed"] for v in values], [73] * 4)
        self.assertEqual({v["ane_manifest"] for v in values}, {str(args.manifest)})
        self.assertEqual({v["hybrid_mlp_mode"] for v in values}, {"runtime"})
        self.assertEqual({v["steps"] for v in values}, {40})
        self.assertEqual({v["qwen21_reference_size"] for v in values if "inputs" in v}, {1024})
        self.assertEqual({k: v for k, v in values[0].items() if k != "output"},
                         {k: v for k, v in values[3].items() if k != "output"})
        for key, bad in (("mode", "stock"), ("route", "gpu"), ("requests", 3)):
            changed = {**vars(args), key: bad}
            with self.assertRaises(ValueError):
                collect.requests(SimpleNamespace(**changed))

    def test_heterogeneous_export_does_not_pool_step_or_request_timings(self):
        observer, runs = fixture()
        for row in runs:
            row.update(request_wall_including_export=.1, metrics=dict(acceleration_selection="runtime", timings_seconds={}))
        with tempfile.TemporaryDirectory(prefix="tc-base-matrix-export-") as directory:
            root = Path(directory)
            (root / "trial").mkdir()
            (root / "trial/report.json").write_text(json.dumps(dict(status="complete", runs=runs, mode="timeline", route="runtime", scenario="sequence-matrix", progress=[])))
            (root / "trial/observer.json").write_text(json.dumps(observer))
            (root / "stderr.log").write_text('\n'.join(json.dumps(dict(qwen21_step=0, phase="prefill", seconds=value)) for value in (1., 10.)))
            result = analyze.analyze_directory(root)
            self.assertIsNone(result["hot_median_seconds"])
            groups = result["diagnostic_groups"]
            self.assertEqual([r["phase_or_layer"] for r in groups], ["request0:prefill", "request1:prefill"])
            self.assertEqual([r["samples"] for r in groups], [1, 1])


if __name__ == "__main__":
    unittest.main()
