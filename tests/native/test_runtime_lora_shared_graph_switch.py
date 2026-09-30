"""Host-only switch-runner contracts; no MLX, Core ML or model fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/validation/runtime_lora_shared_graph_switch.py"
SPEC = importlib.util.spec_from_file_location("runtime_lora_shared_graph_switch", RUNNER)
SWITCH = importlib.util.module_from_spec(SPEC)
with mock.patch.object(sys, "path", [str(RUNNER.parent), *sys.path]):
    SPEC.loader.exec_module(SWITCH)


class SwitchContractTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "model": "z-image-turbo", "operation": "image.generate",
            "prompt": "A red fox in snow.", "seed": 42, "steps": 8,
            "width": 512, "height": 512, "execution": "gpu_ane",
            "residency": "resident", "allow_approximation": True,
            "hybrid_mlp_mode": "runtime", "ane_manifest": "models/base/manifest.json",
            "output": "outputs/base.png",
        }
        self.adapter = copy.deepcopy(self.base)
        self.adapter.update(
            schema_version=1, output="outputs/adapter.png", lora_strategy="inference_time",
            loras=[{"path": "models/adapter.safetensors", "role": "transformer", "strength": 1.0}])

    def test_supported_models_and_graphs_do_not_mutate_requests(self):
        for model, steps in (("z-image-turbo", 8), ("qwen-image-2.1", 6)):
            for mode in ("runtime", "lora_fused"):
                for request in (self.base, self.adapter):
                    request.update(model=model, steps=steps, hybrid_mlp_mode=mode)
                before = copy.deepcopy((self.base, self.adapter))
                self.assertEqual(SWITCH.validate_requests(self.base, self.adapter), (model, mode))
                self.assertEqual((self.base, self.adapter), before)

    def test_workload_differences_cannot_masquerade_as_adapter_changes(self):
        for field, value in (("seed", 43), ("prompt", "Another scene"),
                             ("guidance", 3.5), ("qwen21_reference_size", 512)):
            adapter = {**self.adapter, field: value}
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, "workloads must match"):
                SWITCH.validate_requests(self.base, adapter)

    def test_requires_deterministic_supported_workloads(self):
        for change in ({"seed": None}, {"seed": True}, {"seed": -1}, {"seed": 2147483648},
                       {"prompt": " "}, {"operation": "image.edit"},
                       {"inputs": [{"kind": "image", "path": "reference.png"}]}):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                SWITCH.validate_requests({**self.base, **change}, {**self.adapter, **change})

    def edit_requests(self, count=3):
        change = {"model": "qwen-image-2.1", "steps": 6, "operation": "image.edit",
                  "qwen21_reference_size": 512,
                  "inputs": [{"kind": "image", "role": "reference", "path": f"ref-{i}.png"}
                             for i in range(count)]}
        return {**self.base, **change}, {**self.adapter, **change}

    def test_runtime_editing_accepts_ordered_references_without_mutation(self):
        for count in (1, 2, 3):
            base, adapter = self.edit_requests(count)
            before = copy.deepcopy((base, adapter))
            self.assertEqual(SWITCH.validate_requests(base, adapter), ("qwen-image-2.1", "runtime"))
            self.assertEqual((base, adapter), before)
        adapter["inputs"] = list(reversed(adapter["inputs"]))
        with self.assertRaisesRegex(AssertionError, "workloads must match"):
            SWITCH.validate_requests(base, adapter)

    def test_editing_rejects_unqualified_routes_and_references(self):
        base, adapter = self.edit_requests()
        for change in ({"hybrid_mlp_mode": "lora_fused"}, {"qwen21_reference_size": 1024},
                       {"qwen21_reference_size": 256}, {"inputs": []}, {"inputs": [None]},
                       {"inputs": base["inputs"] + base["inputs"]},
                       {"inputs": [{"kind": "image", "role": "reference", "path": " "}]}):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                SWITCH.validate_requests({**base, **change}, {**adapter, **change})

    def test_reference_identity_detects_reorder_mutation_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"ref-{i}.png" for i in range(2)]
            for i, path in enumerate(paths):
                path.write_bytes(bytes([i]))
            request = {"inputs": [{"path": str(path)} for path in paths]}
            before = SWITCH.reference_identity(request)
            self.assertEqual(SWITCH.reference_identity(request), before)
            request["inputs"].reverse()
            self.assertNotEqual(SWITCH.reference_identity(request), before)
            request["inputs"].reverse()
            paths[0].write_bytes(b"changed")
            self.assertNotEqual(SWITCH.reference_identity(request), before)
            paths[0].unlink()
            with self.assertRaises(FileNotFoundError):
                SWITCH.reference_identity(request)

    def test_edit_receipts_require_consistent_nonzero_reference_tokens(self):
        base, _ = self.edit_requests()
        row = {"operation": "image.edit", "reference_tokens": 3072,
               "plan": {"operation": "image.edit", "qwen21_reference_size": 512}}
        self.assertEqual(SWITCH.validate_edit_receipts([row, row], base), 3072)
        for change in ({"reference_tokens": 0}, {"reference_tokens": True},
                       {"reference_tokens": 1024}, {"operation": "image.generate"},
                       {"plan": {"operation": "image.edit", "qwen21_reference_size": 1024}}):
            with self.subTest(change=change), self.assertRaises((AssertionError, ValueError)):
                SWITCH.validate_edit_receipts([row, {**row, **change}], base)

    def test_runtime_switch_cannot_pass_with_gpu_only_or_reset_counters(self):
        SWITCH.validate_prediction_progress([32, 64, 96, 128, 160])
        for calls in ([], [0, 0, 0], [32, 64, 64, 64, 96], [32, 64, 32], [True], [32.0]):
            with self.subTest(calls=calls), self.assertRaises(AssertionError):
                SWITCH.validate_prediction_progress(calls)

    def test_edit_receipts_delegate_without_mutating_rows(self):
        base, _ = self.edit_requests()
        rows = [{"operation": "image.edit", "reference_tokens": 3072,
                 "plan": {"operation": "image.edit", "qwen21_reference_size": 512}}]
        before = copy.deepcopy(rows)
        with mock.patch.object(SWITCH, "validate_edit_results", return_value=3072) as check:
            self.assertEqual(SWITCH.validate_edit_receipts(rows, base), 3072)
            check.assert_called_once_with(rows, base)
        self.assertEqual(rows, before)
        with mock.patch.object(SWITCH, "validate_edit_results") as check:
            self.assertIsNone(SWITCH.validate_edit_receipts([], self.base))
            check.assert_not_called()

    def test_edit_receipts_reject_empty_or_noninteger_geometry(self):
        base, _ = self.edit_requests()
        with self.assertRaises(AssertionError):
            SWITCH.validate_edit_receipts([], base)
        for size in (True, 512.0, None):
            row = {"operation": "image.edit", "reference_tokens": 3072,
                   "plan": {"operation": "image.edit", "qwen21_reference_size": size}}
            with self.subTest(size=size), self.assertRaises(AssertionError):
                SWITCH.validate_edit_receipts([row], base)

    def test_requires_same_complete_graph_and_runtime_adapter(self):
        for change in ({"hybrid_mlp_mode": "lora_suffix"}, {"hybrid_mlp_mode": "lora_merged"},
                       {"ane_manifest": ""}, {"ane_manifest": None},
                       {"ane_manifest": "another/manifest.json"},
                       {"lora_strategy": "in_memory_merge"}, {"loras": []},
                       {"loras": None}, {"loras": [None]},
                       {"loras": [{"path": "", "role": "transformer"}]}):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                SWITCH.validate_requests(self.base, {**self.adapter, **change})
        with self.assertRaises(AssertionError):
            SWITCH.validate_requests(self.adapter, self.adapter)

    def test_unsupported_shapes_and_nonobjects_fail_before_inference(self):
        for change in ({"width": 1024}, {"steps": 6}, {"execution": "auto"},
                       {"residency": "component_staged"}, {"allow_approximation": False},
                       {"schema_version": 2}, {"model": "unsupported"}):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                SWITCH.validate_requests({**self.base, **change}, {**self.adapter, **change})
        for value in (None, [], "request"):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                SWITCH.validate_requests(value, self.adapter)

    def test_help_does_not_require_optional_python_packages(self):
        result = subprocess.run([sys.executable, "-S", str(RUNNER), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--adapter-request", result.stdout)
        self.assertIn("--qwen-qk-norm-rope", result.stdout)

    def test_qwen_qk_rejects_other_models_before_mlx_or_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, adapter, output = root / "base.json", root / "adapter.json", root / "evidence"
            base.write_text(json.dumps(self.base))
            adapter.write_text(json.dumps(self.adapter))
            result = subprocess.run(
                [sys.executable, "-S", str(RUNNER), "--model", "unused",
                 "--base-request", str(base), "--adapter-request", str(adapter),
                 "--output", str(output), "--qwen-qk-norm-rope"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("Qwen 512px", result.stderr)
            self.assertNotIn("No module named", result.stderr)
            self.assertFalse(output.exists())

    def test_invalid_workload_creates_no_evidence_without_mlx(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, adapter, output = root / "base.json", root / "adapter.json", root / "evidence"
            base.write_text(json.dumps(self.base))
            adapter.write_text(json.dumps({**self.adapter, "seed": 43}))
            result = subprocess.run(
                [sys.executable, "-S", str(RUNNER), "--model", "unused",
                 "--base-request", str(base), "--adapter-request", str(adapter),
                 "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("workloads must match", result.stderr)
            self.assertFalse(output.exists())

    def test_timeout_rejected_before_reading_requests(self):
        result = subprocess.run(
            [sys.executable, "-S", str(RUNNER), "--model", "unused",
             "--base-request", "missing", "--adapter-request", "missing", "--timeout", "0"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("timeout must be positive", result.stderr)

    def test_missing_edit_reference_fails_before_mlx_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, adapter = self.edit_requests(1)
            for data in (base, adapter):
                data["inputs"] = [{"kind": "image", "role": "reference",
                                   "path": str(root / "missing.png")}]
            base_path, adapter_path = root / "base.json", root / "adapter.json"
            base_path.write_text(json.dumps(base))
            adapter_path.write_text(json.dumps(adapter))
            output = root / "evidence"
            result = subprocess.run(
                [sys.executable, "-S", str(RUNNER), "--model", "unused",
                 "--base-request", str(base_path), "--adapter-request", str(adapter_path),
                 "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("missing.png", result.stderr)
            self.assertNotIn("No module named", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
