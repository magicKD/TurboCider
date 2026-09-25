"""CPU-only gates for Qwen21 W8A8 diagnostics; no model or ANE required."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class W8A8DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compare = module("tools/validation/qwen21_a8_boundary_compare.py", "qwen21_a8_compare")
        cls.layer = module("tools/validation/qwen21_w8a8_layer_probe.py", "qwen21_layer_probe")
        cls.layers = module("tools/validation/qwen21_w8a8_layers_probe.py", "qwen21_layers_probe")

    def test_layer_metrics_exact_and_wrong_amplitude(self):
        reference = np.array([1., -2., 3.], dtype=np.float32)
        exact = self.layer.metrics(reference, reference)
        self.assertEqual(exact["relative_rmse"], 0)
        self.assertAlmostEqual(exact["cosine"], 1)
        shrink = self.layer.metrics(reference / 9., reference)
        self.assertGreater(shrink["relative_rmse"], 0.88)
        with self.assertRaises(ValueError):
            self.layer.metrics(reference, np.zeros_like(reference))

    def test_layer_probe_checks_partition_and_row_provenance(self):
        manifest = {
            "export_identity": {"tensor_layout": "qwen21", "activation_precision": "int8",
                                "ane_mlp_end": 6144},
            "shape": {"ane_mlp_start": 0, "ane_mlp_end": 6144, "K": 4096,
                      "mlp_width": 12288, "buckets": [1024]},
            "artifacts": {"0": {"int8_pc": "block0.mlmodelc"}},
        }
        self.assertEqual(self.layer.validate_manifest(manifest), (1024, 6144))
        manifest["export_identity"]["ane_mlp_end"] = 4096
        manifest["shape"]["ane_mlp_end"] = 4096
        manifest["shape"]["buckets"] = [4096]
        self.assertEqual(self.layer.validate_manifest(manifest), (4096, 4096))
        for key, value in (("ane_mlp_end", True), ("ane_mlp_end", 8193),
                           ("ane_mlp_end", 12288)):
            with self.subTest(value=value):
                manifest["export_identity"][key] = value
                with self.assertRaises(ValueError):
                    self.layer.validate_manifest(manifest)
        manifest["export_identity"]["ane_mlp_end"] = 4096
        manifest["shape"]["ane_mlp_end"] = 6144
        with self.assertRaises(ValueError):
            self.layer.validate_manifest(manifest)
        manifest["shape"]["ane_mlp_end"] = 4096
        manifest["export_identity"]["activation_precision"] = "fp16"
        with self.assertRaises(ValueError):
            self.layer.validate_manifest(manifest)
        self.assertEqual(self.layer.validate_manifest(manifest, weight_only_control=True),
                         (4096, 4096))

    def test_policy_output_divergence_cannot_be_hidden_by_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "manifest.json"
            source.write_text(json.dumps({
                "export_identity": {"tensor_layout": "qwen21", "a8_graph": "sq_v1_both"},
                "artifacts": {"0": {"int8_pc": "block0.mlpackage"}},
            }))
            prediction = np.ones((1, 2, 1, 2), np.float16)
            side_effects = [prediction, prediction / 9]

            class MockModel:
                def predict(self, inputs):
                    return {"y": side_effects.pop(0)}

            with patch.object(self.compare.ct.models, "MLModel", return_value=MockModel()):
                record = self.compare.compare(source, prediction)
            self.assertGreater(record["cpu_ne_vs_cpu_rrmse"], 0.88)
            self.assertAlmostEqual(record["cpu_ne_least_squares_gain"], 9, places=2)
            self.assertLess(record["cpu_ne_gain_corrected_rrmse"], 1e-6)
            self.assertEqual(record["boundary"], "sq_v1_both")

    def test_production_session_rejects_fp16_weight_a8_diagnostic(self):
        # No production model is needed to enforce the runtime gate here.
        # The exporter deliberately permits a FP16-weight/A8 research graph.
        pipeline = (ROOT / "native/models/qwen21/pipeline.cpp").read_text()
        self.assertIn('hybrid_->export_variant == "fp16" &&', pipeline)
        self.assertIn('hybrid_->activation_precision == "fp16"', pipeline)
        self.assertIn('hybrid_->export_variant == "int8_pc" &&', pipeline)
        self.assertIn('hybrid_->activation_precision == "int8" &&', pipeline)

    def test_heldout_layer_artifact_must_stay_in_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "block0_mlp_branch.int8_pc.mlpackage"
            artifact.mkdir(parents=True)
            self.assertEqual(self.layers.safe_artifact(root, artifact.name), artifact)
            for name in ("../model.mlmodelc", "/tmp/model.mlmodelc", "other/model.mlmodelc"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.layers.safe_artifact(root, name)
            link = root / "block1_mlp_branch.int8_pc.mlpackage"
            link.symlink_to(artifact)
            with self.assertRaises(ValueError):
                self.layers.safe_artifact(root, link.name)

    def test_generation_comparison_requires_matched_reference_inputs(self):
        torch = ModuleType("torch")
        torch.equal = np.array_equal
        safetensors = ModuleType("safetensors")
        safetensors.__path__ = []
        safetensors.safe_open = lambda *args, **kwargs: None
        safetensors_torch = ModuleType("safetensors.torch")
        safetensors_torch.load_file = lambda _: None
        with patch.dict(sys.modules, {"torch": torch, "safetensors": safetensors,
                                      "safetensors.torch": safetensors_torch}):
            compare = module("tools/validation/qwen21_compare_generations.py", "qwen21_generation_compare")
        baseline = {key: np.array([1]) for key in
                    ("text", "initial", "sigmas", "reference0", "image_slots")}
        candidate = dict(baseline)
        self.assertEqual(set(compare.matched_inputs(baseline, candidate)), set(baseline))
        with self.assertRaisesRegex(ValueError, "reference tensor keys"):
            compare.matched_inputs(baseline, {key: value for key, value in candidate.items()
                                              if key != "reference0"})
        candidate["reference0"] = np.array([2])
        with self.assertRaisesRegex(ValueError, "Unmatched benchmark inputs"):
            compare.matched_inputs(baseline, candidate)
        durations = [2.0, 1.0, 1.0]
        summary = compare.coreml_prediction_summary(
            {"coreml_step_seconds": np.array([0.0, 0.32, 0.64]),
             "coreml_step_calls": np.array([0, 32, 32])}, durations, w8a8=True)
        self.assertAlmostEqual(summary["coreml_prediction_seconds_total"], 0.96)
        self.assertAlmostEqual(summary["coreml_prediction_median_per_ffn_call_ms"], 15)
        partial = compare.coreml_prediction_summary(
            {"coreml_step_seconds": np.array([0.0, 0.29, 0.58]),
             "coreml_step_calls": np.array([0, 29, 29])}, durations, w8a8=True)
        self.assertEqual(partial["w8a8_ffn_layer_coverage"], 29 / 32)
        fp16 = compare.coreml_prediction_summary(
            {"coreml_step_seconds": np.array([0.0, 0.32, 0.64]),
             "coreml_step_calls": np.array([0, 32, 32])}, durations, w8a8=False)
        self.assertNotIn("w8a8_ffn_layer_coverage", fp16)
        self.assertEqual(fp16["hybrid_ffn_layer_coverage"], 1)
        with self.assertRaisesRegex(ValueError, "both per-step"):
            compare.coreml_prediction_summary(
                {"coreml_step_seconds": np.array([0.0, 0.32, 0.64])}, durations, w8a8=True)
        with self.assertRaisesRegex(ValueError, "29/32"):
            compare.coreml_prediction_summary(
                {"coreml_step_seconds": np.array([0.0, 0.29, 0.58]),
                 "coreml_step_calls": np.array([0, 28, 28])}, durations, w8a8=True)
        with self.assertRaisesRegex(ValueError, "no prefill"):
            compare.coreml_prediction_summary(
                {"coreml_step_seconds": np.array([0.1, 0.32, 0.64]),
                 "coreml_step_calls": np.array([0, 32, 32])}, durations, w8a8=True)
        with self.assertRaisesRegex(ValueError, "Invalid per-step"):
            compare.coreml_prediction_summary(
                {"coreml_step_seconds": np.array([0, 1.2, 0.64]),
                 "coreml_step_calls": np.array([0, 32, 32])}, durations, w8a8=True)


if __name__ == "__main__":
    unittest.main()
