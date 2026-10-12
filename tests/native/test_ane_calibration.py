"""CPU-only bounded calibration and stdlib S1 algebra contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("s1_sidecar", ROOT / "tools/coreml/create_smoothquant_s1_sidecar.py")
S1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S1)


class AneCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="tc-calibration-cpu-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name).resolve()
        binary = cls.root / "calibration-host"
        subprocess.run(["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                        "tests/native/ane_calibration_test.cpp", "-o", str(binary)],
                       cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
        result = subprocess.run([str(binary), str(cls.root)], check=True, capture_output=True, text=True, timeout=15)
        if "PASS request-local CPU calibration" not in result.stdout:
            raise AssertionError(result.stdout)
        cls.fixture = cls.root / "fixture"

    def setUp(self):
        self.temporary_case = tempfile.TemporaryDirectory(prefix="tc-s1-cpu-")
        self.addCleanup(self.temporary_case.cleanup)
        self.directory = Path(self.temporary_case.name) / "capture"
        shutil.copytree(self.fixture, self.directory)
        self.capture_path = self.directory / "capture.json"
        self.weights_path = self.directory / "weight_stats.json"

    def modify(self, path, function):
        value = json.loads(path.read_text())
        function(value)
        path.write_text(json.dumps(value))

    def build(self):
        return S1.build_sidecar(self.capture_path, self.weights_path)

    def test_cpu_host_fixture_and_unqualified_s1_provenance(self):
        sidecar = self.build()
        expected = struct.unpack("<f", struct.pack("<f", (16 / 2) ** .5))[0]
        self.assertEqual(sidecar["layers"][0]["s1"][3], expected)
        self.assertTrue(sidecar["cpu_replay"]["passed"])
        self.assertIn("synthetic", sidecar["cpu_replay"]["scope"])
        self.assertFalse(sidecar["runtime_applied"])
        self.assertFalse(sidecar["quantization_qualified"])
        self.assertFalse(sidecar["performance_qualified"])
        self.assertEqual(sidecar["binding"]["width"], 512)
        self.assertEqual(sidecar["binding"]["execution_route"], "cpu-test-only")
        self.assertEqual(sidecar["binding"]["actual_execution"], "unknown")
        self.assertEqual(sidecar["binding"]["runtime_recipe"], "unknown")
        self.assertEqual(sidecar["binding"]["configured_backend"], "unknown")
        self.assertEqual(sidecar["binding"]["loras"][0]["strength"], .12345678901234567)
        self.assertLess(len(json.dumps(sidecar)), 16384)

    def test_missing_model_identity_and_provenance_are_rejected(self):
        for key in ("model_fingerprint", "identity_kind", "recipe", "loras", "seed", "width", "height",
                    "total_steps", "reference_size", "reference_count", "execution_route",
                    "runtime_recipe", "configured_backend", "actual_execution"):
            with self.subTest(key=key):
                original = self.capture_path.read_bytes()
                self.modify(self.capture_path, lambda value: value["binding"].pop(key))
                with self.assertRaises(ValueError):
                    self.build()
                self.capture_path.write_bytes(original)

    def test_partial_scan_incomplete_capture_and_missing_region_are_rejected(self):
        mutations = [lambda v: v.update(complete=False),
                     lambda v: v["points"][0].update(rows_observed=128),
                     lambda v: v["points"][0]["regions"][1].update(begin=17),
                     lambda v: v["points"][0].update(sample_file="../outside.f16")]
        for mutation in mutations:
            original = self.capture_path.read_bytes()
            self.modify(self.capture_path, mutation)
            with self.assertRaises(ValueError):
                self.build()
            self.capture_path.write_bytes(original)

    def test_weight_binding_and_incomplete_statistics_are_rejected(self):
        for mutation in (lambda v: v["binding"].update(model_fingerprint="other-model"),
                         lambda v: v.update(layers=[]),
                         lambda v: v["layers"][0].update(gate_up_channel_max=[1]),
                         lambda v: v.update(statistics_scope="sampled_weight_rows")):
            original = self.weights_path.read_bytes()
            self.modify(self.weights_path, mutation)
            with self.assertRaises(ValueError):
                self.build()
            self.weights_path.write_bytes(original)

    def test_nonfinite_and_negative_scales_statistics_are_rejected(self):
        for value in (float("nan"), float("inf"), 1e100, -1, True):
            original = self.weights_path.read_bytes()
            self.modify(self.weights_path, lambda v: v["layers"][0]["gate_up_channel_max"].__setitem__(0, value))
            with self.assertRaises(ValueError):
                self.build()
            self.weights_path.write_bytes(original)
        for alpha in (float("nan"), -1, 2, True):
            with self.assertRaises(ValueError):
                S1.build_sidecar(self.capture_path, self.weights_path, alpha)

    def test_sample_corruption_nonfinite_symlink_and_accounting_are_rejected(self):
        sample = self.directory / "input-0.f16"
        original = sample.read_bytes()
        sample.write_bytes(b"\x00\x7e" + original[2:])
        with self.assertRaises(ValueError):
            self.build()
        sample.write_bytes(original[:-2])
        with self.assertRaises(ValueError):
            self.build()
        sample.unlink()
        external = self.directory.parent / "external.f16"
        external.write_bytes(original)
        sample.symlink_to(external)
        with self.assertRaises(OSError):
            self.build()
        sample.unlink()
        sample.write_bytes(original)
        self.modify(self.capture_path, lambda v: v["used"].update(input_bytes=1))
        with self.assertRaises(ValueError):
            self.build()

    def test_scale_zero_clamping_and_exclusive_output(self):
        self.assertEqual(S1.scales([0, 1e8, 1e-8], [1, 1e-8, 1e8], .5), [1., 16., 1 / 16])
        sidecar = self.build()
        output = self.directory.parent / "s1.json"
        S1.write_sidecar(output, sidecar)
        previous = output.read_bytes()
        with self.assertRaises(FileExistsError):
            S1.write_sidecar(output, sidecar)
        self.assertEqual(output.read_bytes(), previous)
        broken = copy.deepcopy(sidecar)
        broken["layers"][0]["s1"][0] = 0
        with self.assertRaises(ValueError):
            S1.write_sidecar(self.directory.parent / "invalid.json", broken)
        self.assertFalse((self.directory.parent / "invalid.json").exists())
        broken = copy.deepcopy(sidecar)
        del broken["binding"]["recipe"]
        with self.assertRaises(ValueError):
            S1.write_sidecar(self.directory.parent / "incomplete.json", broken)
        self.assertFalse((self.directory.parent / "incomplete.json").exists())

    def test_small_input_content_changes_sidecar_identity(self):
        first = self.build()["calibration_sha256"]
        sample = self.directory / "input-0.f16"
        data = sample.read_bytes()
        sample.write_bytes(b"\x00\x00" + data[2:])
        self.assertNotEqual(first, self.build()["calibration_sha256"])


if __name__ == "__main__":
    unittest.main()
