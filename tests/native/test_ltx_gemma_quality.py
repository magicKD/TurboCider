"""Characterize Gemma benchmark metrics with tiny BF16 files, no GPU/weights."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TensorQualityBehavior:
    def setUp(self):
        self.script = ROOT / "tools/native" / self.filename
        spec = importlib.util.spec_from_file_location(self.script.stem, self.script)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.candidate = self.root / "candidate.bf16"
        self.reference = self.root / "reference.bf16"

    def words(self, path, values):
        path.write_bytes(struct.pack("<" + "H" * len(values), *values))

    def quality(self, candidate, reference):
        self.words(self.candidate, candidate)
        self.words(self.reference, reference)
        return self.module.tensor_quality(self.candidate, self.reference)

    def assert_metrics(self, result, relative_l2, cosine, max_abs, relative_max_abs):
        self.assertEqual(set(result), {
            "shape_match", "all_finite", "relative_l2", "cosine", "max_abs",
            "relative_max_abs",
        })
        self.assertIs(result["shape_match"], True)
        self.assertIs(result["all_finite"], True)
        for key, expected in (("relative_l2", relative_l2), ("cosine", cosine),
                              ("max_abs", max_abs), ("relative_max_abs", relative_max_abs)):
            self.assertTrue(math.isclose(result[key], expected, rel_tol=1e-14, abs_tol=0),
                            (key, result[key], expected))
        json.dumps(result, allow_nan=False)

    def test_bf16_decodes_little_endian_and_signed_zero(self):
        self.words(self.candidate, [0x3F80, 0xC000, 0x0001, 0x8000])
        values = self.module.bf16(self.candidate)
        self.assertEqual(str(values.dtype), "float32")
        self.assertEqual(values.tolist(), [1.0, -2.0, 2.0 ** -133, -0.0])
        self.assertEqual(math.copysign(1, float(values[3])), -1)

    def test_known_orthogonal_error_metrics(self):
        # Candidate [2, 0], reference [1, 1]: delta [1, -1].
        result = self.quality([0x4000, 0], [0x3F80, 0x3F80])
        self.assert_metrics(result, 1.0, 1 / math.sqrt(2), 1.0, 1.0)

    def test_opposite_vectors_have_negative_cosine(self):
        result = self.quality([0xBF80, 0x3F80], [0x3F80, 0xBF80])
        self.assert_metrics(result, 2.0, -1.0, 2.0, 2.0)

    def test_identical_zero_and_empty_tensors(self):
        for words in ([0x3F80, 0xC000], [0, 0x8000], []):
            with self.subTest(words=words):
                self.assert_metrics(self.quality(words, words), 0.0, 1.0, 0.0, 0.0)

    def test_zero_reference_and_zero_candidate_keep_distinct_errors(self):
        result = self.quality([0, 0], [0x3F80, 0xC000])
        self.assert_metrics(result, 1.0, 0.0, 2.0, 1.0)
        result = self.quality([0x3F80, 0xC000], [0, 0])
        self.assert_metrics(result, 1e30, 0.0, 2.0, 1e30)

    def test_subnormal_values_retain_double_precision_metrics(self):
        result = self.quality([0x0001], [0x0002])
        self.assert_metrics(result, 0.5, 1.0, 2.0 ** -133, 0.5)

    def test_nonfinite_inputs_return_json_safe_failure_metrics(self):
        for word in (0x7F80, 0xFF80, 0x7FC1):
            for candidate, reference in (([word], [0x3F80]), ([0x3F80], [word])):
                with self.subTest(candidate=candidate, reference=reference):
                    result = self.quality(candidate, reference)
                    self.assertEqual(result, {
                        "shape_match": True, "all_finite": False,
                        "relative_l2": 1e30, "cosine": -1.0,
                        "max_abs": 1e30, "relative_max_abs": 1e30,
                    })
                    json.dumps(result, allow_nan=False)

    def test_help_works_outside_repository_without_optional_packages(self):
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(self.script), "--help"],
            cwd=self.root, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--dry-run", "--max-relative-l2", "--min-cosine", "--max-relative-abs"):
            self.assertIn(flag, result.stdout)


class MlpQualityBehavior(TensorQualityBehavior):
    def test_shape_mismatch_preserves_report_fields(self):
        result = self.quality([0x3F80], [0x3F80, 0])
        self.assertEqual(result, {
            "shape_match": False, "relative_l2": 1e30, "cosine": -1.0,
            "max_abs": 1e30, "relative_max_abs": 1e30,
        })
        json.dumps(result, allow_nan=False)

    def test_missing_files_continue_to_raise(self):
        with self.assertRaises(FileNotFoundError):
            self.module.tensor_quality(self.candidate, self.reference)
        self.words(self.candidate, [0])
        with self.assertRaises(FileNotFoundError):
            self.module.tensor_quality(self.candidate, self.reference)

    def test_dry_run_keeps_variant_order_and_does_not_create_output(self):
        checkpoint, shader = self.root / "checkpoint", self.root / "shader"
        checkpoint.touch()
        shader.touch()
        output = self.root / "output"
        command = [sys.executable, "-B", str(self.script), "--checkpoint", str(checkpoint),
                   "--shader", str(shader), "--output", str(output),
                   "--rows", "64", "128", "--runs", "2", "--dry-run"]
        if self.filename == "benchmark_ltx_gemma_mlp_compare.py":
            command += ["--probe", sys.executable]
        else:
            manifest = self.root / "manifest"
            manifest.touch()
            command += ["--gpu-probe", sys.executable, "--hybrid-probe", sys.executable,
                        "--manifest", str(manifest), "--layer", "0"]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["format"], self.plan_format)
        self.assertEqual([case["rows"] for case in payload["planned"]], [64, 128])
        self.assertEqual(payload["planned"][0]["order"], self.variants)
        self.assertEqual(payload["planned"][1]["order"], list(reversed(self.variants)))
        self.assertFalse(output.exists())



class StagedFusedQualityTests(MlpQualityBehavior, unittest.TestCase):
    filename = "benchmark_ltx_gemma_mlp_compare.py"
    variants = ["staged", "fused"]
    plan_format = "turbocider-ltx-gemma-mlp-compare-plan-v1"


class GpuAneQualityTests(MlpQualityBehavior, unittest.TestCase):
    filename = "benchmark_ltx_gemma_ane_mlp.py"
    variants = ["gpu", "hybrid"]
    plan_format = "turbocider-ltx-gemma-ane-mlp-plan-v1"


class EncoderQualityTests(TensorQualityBehavior, unittest.TestCase):
    filename = "benchmark_ltx_gemma_encoder_sweep.py"

    def test_shape_mismatch_preserves_element_counts(self):
        result = self.quality([0x3F80], [0x3F80, 0])
        self.assertEqual(result, {
            "shape_match": False, "candidate_elements": 1, "reference_elements": 2,
            "relative_l2": 1e30, "cosine": -1.0,
            "max_abs": 1e30, "relative_max_abs": 1e30,
        })
        json.dumps(result, allow_nan=False)

    def test_missing_or_directory_inputs_keep_failure_report(self):
        expected = {"shape_match": False, "all_finite": False,
                    "relative_l2": 1e30, "cosine": -1.0,
                    "max_abs": 1e30, "relative_max_abs": 1e30}
        self.assertEqual(self.module.tensor_quality(self.candidate, self.reference), expected)
        self.words(self.candidate, [0])
        self.assertEqual(self.module.tensor_quality(self.candidate, self.reference), expected)
        self.words(self.reference, [0])
        self.assertEqual(self.module.tensor_quality(self.root, self.reference), expected)
        self.assertEqual(self.module.tensor_quality(self.candidate, self.root), expected)

    def test_conditioning_gate_covers_all_streams_and_row_mismatch(self):
        from types import SimpleNamespace

        self.words(self.reference, [0, 0])
        self.words(self.candidate, [0, 0])
        candidate = dict(rows=2, video=self.candidate, audio=self.candidate, mask=self.candidate)
        reference = dict(rows=2, video=self.reference, audio=self.reference, mask=self.reference)
        args = SimpleNamespace(max_relative_l2=0.025, min_cosine=0.999, max_relative_abs=0.1)
        qualities = self.module.conditioning_quality(candidate, reference)
        self.assertEqual(set(qualities), {"video", "audio", "mask"})
        self.assertTrue(self.module.quality_gate(qualities, args))
        changed = self.root / "changed.bf16"
        self.words(changed, [0x3F80, 0])
        for stream in ("video", "audio", "mask"):
            with self.subTest(stream=stream):
                qualities = self.module.conditioning_quality({**candidate, stream: changed}, reference)
                self.assertFalse(self.module.quality_gate(qualities, args))
        qualities = self.module.conditioning_quality({"rows": 1}, {"rows": 2})
        self.assertTrue(all(not item["shape_match"] for item in qualities.values()))
        self.assertFalse(self.module.quality_gate(qualities, args))

    def test_dry_run_keeps_variant_order_and_does_not_create_output(self):
        output = self.root / "output"
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(self.script),
             "--checkpoint", "missing-checkpoint", "--tokenizer", "missing-tokenizer",
             "--prompt", "first", "--prompt", "second", "--warm-runs", "3",
             "--output", str(output), "--dry-run"],
            cwd=self.root, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        variants = ["staged", "fused", "fused_gpu_taps"]
        self.assertEqual(payload["format"], "turbocider-ltx-gemma-encoder-sweep-plan-v2")
        self.assertEqual(payload["warm_runs"], 3)
        self.assertEqual(payload["variants"], variants)
        self.assertEqual(payload["planned"][0]["order"], variants)
        self.assertEqual(payload["planned"][1]["order"], variants[1:] + variants[:1])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
