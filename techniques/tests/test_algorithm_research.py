"""Offline algorithm witnesses and source/evidence integrity; never GPU work."""
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
sys.path.insert(0, str(ROOT / "labs"))
import algorithm_research as reference


def module(name):
    spec = importlib.util.spec_from_file_location("algorithm_" + name, ROOT / "algorithms" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class AlgorithmResearchTests(unittest.TestCase):
    def test_math_witnesses_across_seeds(self):
        for seed in (1, 17, 29):
            result = reference.algorithm_evidence(seed)
            self.assertFalse(result["production_qualified"])
            self.assertLess(result["details"]["constant_key_partial_tail"]["max_abs"], 1e-12)

    def test_invalid_shapes_nonfinite_and_empty_inputs_reject(self):
        for query, keys, values in (([1], [], []), ([1], [[1]], []), ([1, 2], [[1]], [[2]]),
                                    ([float("nan")], [[1]], [[2]]), ([1], [[1]], [[]])):
            with self.subTest(query=query), self.assertRaises(ValueError):
                reference.attention(query, keys, values)
        for size, selected in ((0, []), (True, []), (2, [1]), (1, [-1])):
            with self.subTest(size=size), self.assertRaises(ValueError):
                reference.piecewise_zero([1], [[1]], [[2]], size, selected)

    def test_forecast_uses_actual_anchor_time_and_refuses_bad_history(self):
        self.assertEqual(reference.linear_forecast(2, 6, 1, 3, 5), 10)
        for previous, current, target in ((1, 1, 2), (2, 1, 3), (0, 2, 1)):
            with self.assertRaises(ValueError):
                reference.linear_forecast(2, 3, previous, current, target)

    def test_local_plan_sink_has_no_duplicate_keys_and_covers_exact_policy(self):
        probe = module("mlx_probe")
        for length in (129, 512, 519):
            for begin, end, ranges, indices in probe.make_plan(length, 128, 64, 16):
                self.assertEqual(len(indices), len(set(indices)))
                self.assertTrue(all(0 <= i < length for i in indices))
                for row in range(begin, end):
                    expected = {i for i in range(length) if abs(i - row) <= 64 or i < 16}
                    self.assertTrue(expected <= set(indices))

    def test_gpu_cli_is_opt_in_even_with_valid_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused.json"
            child = subprocess.run([sys.executable, str(ROOT / "algorithms/mlx_probe.py"), "--output", str(output)],
                                   capture_output=True, text=True)
            self.assertNotEqual(child.returncode, 0)
            self.assertIn("explicit --run", child.stderr)
            self.assertFalse(output.exists())

    def test_sources_structure_and_generated_reading_page_are_current(self):
        curator = module("curate_sources")
        data = curator.validate(json.loads((ROOT / "algorithms/sources.json").read_text()))
        self.assertGreaterEqual(len(data["sources"]), 29)
        self.assertEqual(curator.render(data), (ROOT / "algorithms/sources.md").read_text())
        self.assertTrue(any(s["repository"] == "NVlabs/Sana" for s in data["sources"]))

    def test_source_identity_mutations_reject(self):
        curator = module("curate_sources")
        original = json.loads((ROOT / "algorithms/sources.json").read_text())
        for kind in ("commit", "digest", "url", "duplicate"):
            data = copy.deepcopy(original)
            if kind == "commit":
                data["sources"][0]["commit"] = "main"
            elif kind == "digest":
                data["sources"][0]["files"][0]["sha256"] = "unknown"
            elif kind == "url":
                data["sources"][0]["files"][0]["url"] = "https://example.com/wrong"
            else:
                data["sources"].append(data["sources"][0])
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                curator.validate(data)

    def test_collected_source_is_not_a_model_execution_claim(self):
        probe = module("source_probe")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(probe.collect({"id": "foo", "repository": "author/project"}, root, False)["status"], "not_downloaded")
            self.assertFalse((root / "foo").exists())
            (root / "foo").mkdir()
            with self.assertRaisesRegex(ValueError, "preserved"):
                probe.collect({"id": "foo", "repository": "author/project"}, root, False)
            with self.assertRaises(ValueError):
                probe.collect({"id": "../escape", "repository": "author/project"}, root, False)

    def test_experiment_has_two_complete_seeds_without_media_qualification(self):
        curator = module("curate_experiments")
        data = json.loads((ROOT / "algorithms/experiments-20261016.json").read_text())
        self.assertFalse(data["model_quality_qualified"])
        digest = hashlib.sha256((ROOT / "algorithms/mlx_probe.py").read_bytes()).hexdigest()
        self.assertEqual(len({p["seed"] for p in data["probes"]}), 2)
        for probe in data["probes"]:
            curator.validate_probe(probe, digest)
            self.assertFalse(probe["gpu_exclusivity_proven"])
            self.assertFalse(probe["media_quality_qualified"])
        bad = copy.deepcopy(data["probes"][0])
        bad["workloads"][0]["timings"]["dense"]["samples_ms"].pop()
        with self.assertRaises(ValueError):
            curator.validate_probe(bad, digest)


if __name__ == "__main__":
    unittest.main()
