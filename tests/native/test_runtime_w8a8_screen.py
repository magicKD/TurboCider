"""W8A8 research verdicts cannot grant product/hardware qualification."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
SPEC = importlib.util.spec_from_file_location("runtime_screen", ROOT / "tools/validation/run_runtime_w8a8_proof.py")
SCREEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREEN)


class VerdictTests(unittest.TestCase):
    def test_real_slice_reader_and_explicit_partial_identity(self):
        codes = np.arange(512, dtype=np.int16).astype(np.int8).reshape(2, 256)
        scales = np.array([.01, .02], dtype="<f4")
        header = {"p.weight": {"dtype": "I8", "shape": [2, 256], "data_offsets": [0, 512]},
                  "p.weight_scale": {"dtype": "F32", "shape": [2, 1], "data_offsets": [512, 520]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.safetensors"
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + codes.tobytes() + scales.tobytes())
            w, sw, identity = SCREEN.convrot_slice(path, "p", 1, 256)
            np.testing.assert_array_equal(w, codes[:1])
            np.testing.assert_array_equal(sw, scales[:1])
            self.assertIsNone(identity["checkpoint_content_sha256"])
            self.assertEqual(identity["slice_rows"], 1)
            for outputs, hidden in [(0, 256), (True, 256), (1, 255), (3, 256)]:
                with self.assertRaises(ValueError):
                    SCREEN.convrot_slice(path, "p", outputs, hidden)
            for key, bad_value in [("dtype", "BF16"), ("data_offsets", [False, 512]),
                                   ("data_offsets", [0, 513]), ("shape", [True, 256])]:
                bad = json.loads(json.dumps(header))
                bad["p.weight"][key] = bad_value
                raw = json.dumps(bad).encode()
                path.write_bytes(struct.pack("<Q", len(raw)) + raw + codes.tobytes() + scales.tobytes())
                with self.assertRaises(ValueError):
                    SCREEN.convrot_slice(path, "p", 1, 256)

    def test_slice_reader_rejects_short_duplicate_and_nonfinite(self):
        header = {"p.weight": {"dtype": "I8", "shape": [1, 256], "data_offsets": [0, 256]},
                  "p.weight_scale": {"dtype": "F32", "shape": [1, 1], "data_offsets": [256, 260]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.safetensors"
            raw = json.dumps(header).encode()
            for payload in [b"", bytes(256), bytes(256) + np.array([np.inf], "<f4").tobytes()]:
                path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
                with self.assertRaises(ValueError):
                    SCREEN.convrot_slice(path, "p", 1, 256)
            path.write_bytes(b"short")
            with self.assertRaises(ValueError):
                SCREEN.convrot_slice(path, "p", 1, 256)
            raw = b'{"duplicate":1,"duplicate":2}'
            path.write_bytes(struct.pack("<Q", len(raw)) + raw)
            with self.assertRaises(ValueError):
                SCREEN.convrot_slice(path, "p", 1, 256)

    def test_missing_or_failed_proof_never_passes(self):
        self.assertEqual(SCREEN.classify_screen({}), "numeric_gate_failed")
        good = {"arm": "runtime_qdq", "normalized_numeric_gate": True,
                "graph_files_unchanged": True, "weight_a_repeat_exact": True,
                "weight_b_changes_output": True}
        self.assertEqual(SCREEN.classify_screen(good), "inconclusive_arithmetic_not_observed")
        for key in ("graph_files_unchanged", "weight_a_repeat_exact", "weight_b_changes_output"):
            bad = {**good, key: False}
            self.assertTrue(SCREEN.classify_screen(bad).startswith("unsupported_on_tuple"))
        cpu = {**good, "compute_plan": {"operations": [{"op": "ios18.matmul", "preferred": "MLCPUComputeDevice"}]}}
        self.assertEqual(SCREEN.classify_screen(cpu), "unsupported_on_tuple_cpu_plan")
        # A compute-plan NE preference is only intent, never runtime/INT8 proof.
        ne = {**good, "compute_plan": {"operations": [{"op": "ios18.matmul", "preferred": "MLNeuralEngineComputeDevice"}]}}
        self.assertEqual(SCREEN.classify_screen(ne), "inconclusive_arithmetic_not_observed")
        self.assertEqual(SCREEN.classify_screen({**good, "status": "failed"}), "inconclusive_export_or_execution")

    def test_static_control_does_not_require_dynamic_switch(self):
        entry = {"arm": "frozen_qdq", "normalized_numeric_gate": True,
                 "graph_files_unchanged": True, "weight_a_repeat_exact": True}
        self.assertEqual(SCREEN.classify_screen(entry), "inconclusive_arithmetic_not_observed")

    def test_required_arms_samples_and_reuse_control_exit(self):
        arms = [{"arm": arm, "status": "completed", "samples": [None] * 30,
                 "normalized_numeric_gate": True, "graph_files_unchanged": True,
                 "weight_a_repeat_exact": True, "weight_b_changes_output": True}
                for arm in ("fp16", "frozen_qdq", "runtime_qdq")]
        self.assertEqual(SCREEN.screen_exit_code({"arms": arms}), 0)
        self.assertEqual(SCREEN.screen_exit_code({"arms": arms[:2]}), 2)
        self.assertEqual(SCREEN.screen_exit_code({"arms": [arms[0]] * 3}), 2)
        broken = [{**arm} for arm in arms]
        broken[2]["weight_b_changes_output"] = False
        self.assertEqual(SCREEN.screen_exit_code({"arms": broken}), 1)
        broken[2]["samples"] = [None] * 29
        self.assertEqual(SCREEN.screen_exit_code({"arms": broken}), 2)


if __name__ == "__main__":
    unittest.main()
