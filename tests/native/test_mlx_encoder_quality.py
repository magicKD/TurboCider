"""MLX encoder metrics on tiny CPU tensors; no checkpoints or inference.

MLX initialization still requires Metal device visibility on macOS.
"""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
DRIVERS = {
    "h3": "benchmark_h3_encoder_sweep.py",
    "qwen3": "benchmark_qwen3_encoder_sweep.py",
}


class TensorQualityBehavior:
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx

        cls.mx = mx
        cls.previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        script = ROOT / "tools/native" / DRIVERS[cls.driver]
        spec = importlib.util.spec_from_file_location(script.stem, script)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        cls.mx.set_default_device(cls.previous_device)

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="tc-encoder-quality-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.candidate = self.root / "candidate.safetensors"
        self.reference = self.root / "reference.safetensors"

    def save(self, path, values, dtype=None, key=None):
        self.mx.save_safetensors(str(path), {
            key or self.tensor_name: self.mx.array(values, dtype=dtype or self.mx.float32),
        })

    def quality(self, candidate, reference, dtype=None):
        self.save(self.candidate, candidate, dtype)
        self.save(self.reference, reference, dtype)
        return self.module.tensor_quality(self.candidate, self.reference)

    def assert_metrics(self, result, relative_l2, cosine, max_abs, relative_max_abs):
        self.assertEqual(set(result), {"shape_match", "all_finite", "relative_l2",
                                      "cosine", "max_abs", "relative_max_abs"})
        self.assertIs(result["shape_match"], True)
        self.assertIs(result["all_finite"], True)
        for key, expected in (("relative_l2", relative_l2), ("cosine", cosine),
                              ("max_abs", max_abs), ("relative_max_abs", relative_max_abs)):
            self.assertTrue(math.isclose(result[key], expected, rel_tol=2e-6, abs_tol=0),
                            (key, result[key], expected))
        json.dumps(result, allow_nan=False)

    def test_known_orthogonal_metrics_and_source_dtype_conversion(self):
        for dtype in (self.mx.float32, self.mx.float16, self.mx.bfloat16, self.mx.int32):
            with self.subTest(dtype=dtype):
                self.assert_metrics(self.quality([2, 0], [1, 1], dtype),
                                    1, 1 / math.sqrt(2), 1, 1)

    def test_identical_multidimensional_and_zero_tensors(self):
        for values in ([[1., -2.], [3., 4.]], [[0., -0.], [0., 0.]]):
            self.assert_metrics(self.quality(values, values), 0, 1, 0, 0)

    def test_opposite_vectors(self):
        self.assert_metrics(self.quality([-1, 1], [1, -1]), 2, -1, 2, 2)

    def test_zero_candidate_and_zero_reference_remain_distinct(self):
        self.assert_metrics(self.quality([0, 0], [1, -2]), 1, 0, 2, 1)
        self.assert_metrics(self.quality([1, -2], [0, 0]), 1e30, 0, 2, 1e30)

    def test_large_finite_values_keep_scaled_energy_finite(self):
        self.assert_metrics(self.quality([2e30, 0], [1e30, 1e30]),
                            1, 1 / math.sqrt(2), 1e30, 1)

    def test_shape_mismatch_reports_dimensions_before_nonfinite_check(self):
        result = self.quality([float("inf"), 0], [[1, 0]])
        self.assertEqual(result, {
            "shape_match": False, "candidate_shape": [2], "reference_shape": [1, 2],
            "relative_l2": 1e30, "cosine": -1.0,
            "max_abs": 1e30, "relative_max_abs": 1e30,
        })
        json.dumps(result, allow_nan=False)

    def test_nonfinite_values_keep_json_safe_failure_report(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for candidate, reference in (([value], [1]), ([1], [value])):
                with self.subTest(candidate=candidate, reference=reference):
                    result = self.quality(candidate, reference)
                    self.assertEqual(result, {
                        "shape_match": True, "all_finite": False,
                        "relative_l2": 1e30, "cosine": -1.0,
                        "max_abs": 1e30, "relative_max_abs": 1e30,
                    })
                    json.dumps(result, allow_nan=False)

    def test_missing_files_keep_load_error(self):
        with self.assertRaises(RuntimeError):
            self.module.tensor_quality(self.candidate, self.reference)
        self.save(self.candidate, [0])
        with self.assertRaises(RuntimeError):
            self.module.tensor_quality(self.candidate, self.reference)

    def test_missing_selected_tensor_keeps_key_error(self):
        self.save(self.candidate, [0], key="unrelated")
        self.save(self.reference, [0])
        with self.assertRaises(KeyError) as error:
            self.module.tensor_quality(self.candidate, self.reference)
        self.assertEqual(error.exception.args, (self.tensor_name,))


class H3QualityTests(TensorQualityBehavior, unittest.TestCase):
    driver = "h3"
    tensor_name = "hidden_states"


class Qwen3QualityTests(TensorQualityBehavior, unittest.TestCase):
    driver = "qwen3"
    tensor_name = "conditioning"


class CommandLineTests(unittest.TestCase):
    def test_help_without_optional_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in DRIVERS.values():
                with self.subTest(driver=name):
                    result = subprocess.run(
                        [sys.executable, "-I", "-S", "-B", str(ROOT / "tools/native" / name), "--help"],
                        cwd=directory, capture_output=True, text=True, timeout=15,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    for flag in ("--dry-run", "--max-relative-l2", "--min-cosine", "--max-relative-abs"):
                        self.assertIn(flag, result.stdout)

    def test_dry_run_without_optional_packages_preserves_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture.mlmodelc").mkdir()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 2,
                "shape": {"K": 2560, "N": 2560, "mlp_width": 9728, "buckets": [64, 128],
                          "ane_mlp_start": 0, "ane_mlp_end": 4864},
                "artifacts": {str(i): {"64": "fixture.mlmodelc", "128": "fixture.mlmodelc"}
                              for i in range(35)},
            }))
            for driver, name in DRIVERS.items():
                output = root / (driver + "-output")
                args = (["--text-encoder", str(root), "--tokenizer", str(root)] if driver == "h3"
                        else ["--weights", str(root), "--mode", "z_image",
                              "--encoder-ane-manifest", str(manifest)])
                result = subprocess.run(
                    [sys.executable, "-I", "-S", "-B", str(ROOT / "tools/native" / name),
                     "--probe", sys.executable, "--tokens", "64", "128", "--output", str(output),
                     "--dry-run", *args],
                    cwd=root, capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads(result.stdout)
                self.assertEqual(plan["format"], f"turbocider-{driver}-encoder-sweep-plan-v1")
                self.assertEqual([case["tokens"] for case in plan["planned"]], [64, 128])
                variants = ["gpu_reference", "gpu_sdpa"] if driver == "h3" else ["gpu", "hybrid"]
                self.assertEqual(plan["planned"][0]["order"], variants)
                self.assertEqual(plan["planned"][1]["order"], list(reversed(variants)))
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
