"""Runtime FFN host contracts; no Core ML SDK, GPU work or model fixtures."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "runtime_ane_export", ROOT / "tools/coreml/export_runtime_ane.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class RuntimeHostTests(unittest.TestCase):
    def test_bounded_allocation_free_overflow_event_prefix(self):
        self.run_host_test("ane_overflow_report_test")
    def test_canonical_gpu_layer_policy_and_invalid_ordinals(self):
        self.run_host_test("ane_gpu_layer_policy_test")
    def run_host_test(self, name, sources=()):
        with tempfile.TemporaryDirectory(prefix="tc-ane-host-") as temporary:
            binary = Path(temporary) / name
            subprocess.run(
                ["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                 str(ROOT / "tests/native" / f"{name}.cpp"),
                 *(str(ROOT / source) for source in sources), "-o", str(binary)],
                check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True)

    def test_row_scheduler_alignment_disable_reprobe_and_isolation(self):
        self.run_host_test("ane_scheduler_test")

    def test_channel_bandwidth_fit_and_memory_constrained_selection(self):
        self.run_host_test("ane_cost_model_test")

    def test_shared_gpu_calibration_order_samples_and_exception_drain(self):
        self.run_host_test("ane_calibration_timing_test")

    def test_native_channel_identity_trial_gate_and_weak_source_cache(self):
        self.run_host_test("ane_channel_selection_test")

    def test_qkv_complete_block_controller_and_periodic_reprobe(self):
        self.run_host_test("ane_qkv_scheduler_test")

    def test_memory_admission_and_host_scratch(self):
        self.run_host_test("ane_memory_test", ("native/backends/ane_memory.cpp",))

    def test_calibration_gpu_memory_layout_and_limits(self):
        self.run_host_test("ane_calibration_memory_test")

    def test_simd_conversion_matches_scalar_including_overflow_and_tails(self):
        self.run_host_test("ane_runtime_convert_test")

    def test_affine_q4_q8_layout_metadata_bounds_and_headroom(self):
        self.run_host_test("ane_runtime_quant_test")

    def test_raw_gguf_convrot_no_dense_temporary_bounds_and_headroom(self):
        self.run_host_test("ane_runtime_packed_test", ("native/core/gguf_decode.cpp",))

    def test_projection_layout_and_validation(self):
        spec = EXPORT.geometry("swiglu", 32, 64, 96, 33, 47)
        self.assertEqual(spec["inputs"], {"x": [32, 64], "wg": [96, 64],
                                          "wu": [96, 64], "wd": [64, 96]})
        self.assertEqual(spec["outputs"], {"y": [32, 64]})
        self.assertEqual(EXPORT.geometry("matmul", 32, 64, 96, 33, 47)["outputs"],
                         {"y": [32, 96]})
        self.assertNotIn("wg", EXPORT.geometry("gelu", 32, 64, 96, 33, 47)["inputs"])
        for rows in (0, -1, True, 32769, 32.5):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                EXPORT.geometry("swiglu", rows, 64, 96, 32, 48)
        with self.assertRaises(ValueError):
            EXPORT.geometry("unknown", 32, 64, 96, 32, 48)


if __name__ == "__main__":
    unittest.main(verbosity=2)
