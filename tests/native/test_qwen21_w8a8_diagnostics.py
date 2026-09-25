"""CPU-only gates for Qwen21 W8A8 diagnostics; no model or ANE required."""

import importlib.util
import json
from pathlib import Path
import tempfile
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

    def test_layer_metrics_exact_and_wrong_amplitude(self):
        reference = np.array([1., -2., 3.], dtype=np.float32)
        exact = self.layer.metrics(reference, reference)
        self.assertEqual(exact["relative_rmse"], 0)
        self.assertAlmostEqual(exact["cosine"], 1)
        shrink = self.layer.metrics(reference / 9., reference)
        self.assertGreater(shrink["relative_rmse"], 0.88)
        with self.assertRaises(ValueError):
            self.layer.metrics(reference, np.zeros_like(reference))

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
        self.assertIn('hybrid_->activation_precision == "fp16" &&', pipeline)


if __name__ == "__main__":
    unittest.main()
