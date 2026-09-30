"""Host-only plan-inspection contracts; no Core ML import or hardware work."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
with mock.patch.object(sys, "path", [str(ROOT / "tools/validation"), *sys.path]):
    import qwen21_ane_placement as PLACEMENT


def operation(name, device=None, children=(), no_usage=False):
    preferred = type(device, (), {})() if device else None
    return NS(operator_name=name, blocks=list(children),
              usage=None if no_usage else NS(preferred_compute_device=preferred,
                                             supported_compute_devices=[preferred] if preferred else []))


def plan(*ops):
    return NS(model_structure=NS(program=NS(functions={"main": NS(block=NS(operations=list(ops)))})),
              get_compute_device_usage_for_mlprogram_operation=lambda op: op.usage)


class PlacementTests(unittest.TestCase):
    def fixture(self, directory, runtime=True):
        root = Path(directory)
        graph = root / "graph.mlmodelc"
        graph.mkdir()
        (graph / "model.mil").write_text("synthetic graph")
        manifest = ({"backend": "runtime_weight_fp16", "compiled_model": graph.name,
                     "files": {"graph.mlmodelc/model.mil": PLACEMENT.sha256_file(graph / "model.mil")}}
                    if runtime else {"artifacts": {"0": {"int8_pc": graph.name},
                                                   "3": {"int8_pc": graph.name}}})
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path, graph, manifest

    def test_import_and_help_without_sdk_or_process_side_effects(self):
        script = """
import subprocess, sys
sys.path.insert(0, sys.argv[1])
def forbidden(*args, **kwargs):
    raise AssertionError('import started a process')
subprocess.run = subprocess.check_output = subprocess.Popen = forbidden
import qwen21_ane_placement
assert 'coremltools' not in sys.modules
assert 'mlx' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-S", "-B", "-c", script,
                                 str(ROOT / "tools/validation")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-S", "-B", str(ROOT / "tools/validation/qwen21_ane_placement.py"),
                                 "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--blocks", result.stdout)
        self.assertIn("runtime", result.stdout)

    def test_runtime_is_one_graph_not_fictitious_layer_placement(self):
        with tempfile.TemporaryDirectory() as directory:
            path, graph, _ = self.fixture(directory)
            with mock.patch.object(PLACEMENT, "load_plan", return_value=
                                   plan(operation("matmul", "MLNeuralEngineComputeDevice"))) as load:
                report = PLACEMENT.inspect_manifest(path, quiet=True)
            load.assert_called_once_with(graph.resolve())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["observed_ane_residency"], "unknown")
            self.assertIn("not runtime", report["scope"])
            self.assertEqual(list(report["graphs"]), ["runtime"])
            self.assertNotIn("blocks", report)
            self.assertEqual(report["summary"]["blocks"], 0)
            self.assertEqual(report["summary"]["graphs"], 1)
            self.assertEqual(report["summary"]["operator_types"]["matmul"]["neural_engine"], 1)
            self.assertEqual(report["identity"]["manifest_sha256"], PLACEMENT.sha256_file(path))
            self.assertTrue(all(not Path(name).is_absolute() for name in report["identity"]["files"]))

    def test_frozen_blocks_interface_and_counts_remain_available(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = self.fixture(directory, runtime=False)
            for selected, expected in ((None, ["0", "3"]), ([3], ["3"]), ([3, 0], ["3", "0"])):
                with mock.patch.object(PLACEMENT, "load_plan", return_value=plan(operation("const"))):
                    report = PLACEMENT.inspect_manifest(path, selected, quiet=True)
                self.assertEqual(list(report["blocks"]), expected)
                self.assertEqual(report["summary"]["blocks"], len(expected))
                self.assertEqual(report["summary"]["operator_preferences"]["constant"], len(expected))
            for invalid in ([], [0, 0], [8]):
                with self.assertRaises(ValueError), mock.patch.object(PLACEMENT, "load_plan") as load:
                    PLACEMENT.inspect_manifest(path, invalid, quiet=True)
                load.assert_not_called()

    def test_nested_operations_and_missing_preferences_are_not_ane(self):
        nested = NS(operations=[operation("matmul", "MLNeuralEngineComputeDevice"),
                                operation("ios18.constexpr_affine_dequantize", "MLCPUComputeDevice")])
        rows = PLACEMENT.inspect_plan(plan(operation("cond", "MLCPUComputeDevice", [nested]),
                                            operation("reshape"), operation("cast", no_usage=True)))
        self.assertEqual([row["category"] for row in rows],
                         ["other", "neural_engine", "constant", "unassigned", "unassigned"])
        self.assertEqual(rows[1]["location"], "main/0/block0/0")
        self.assertIsNone(rows[-2]["preferred"])
        self.assertEqual(rows[1]["supported"], ["MLNeuralEngineComputeDevice"])

    def test_unsupported_or_empty_program_fails(self):
        for candidate in (plan(), NS(model_structure=NS(program=None)),
                          NS(model_structure=NS(program=NS(functions={})))):
            with self.assertRaisesRegex(ValueError, "nonempty ML Program"):
                PLACEMENT.inspect_plan(candidate)

    def test_runtime_block_selection_and_invalid_receipts_fail_before_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, manifest = self.fixture(directory)
            with mock.patch.object(PLACEMENT, "load_plan") as load:
                for blocks in ([], [0]):
                    with self.assertRaisesRegex(ValueError, "shared graph"):
                        PLACEMENT.inspect_manifest(path, blocks)
                for files in ({}, {"graph.mlmodelc/model.mil": "0" * 64},
                              {**manifest["files"], "missing.bin": "0" * 64},
                              {**manifest["files"], "other": True}):
                    path.write_text(json.dumps({**manifest, "files": files}))
                    with self.assertRaises(ValueError):
                        PLACEMENT.inspect_manifest(path)
                load.assert_not_called()

    def test_unreceipted_compiled_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, graph, _ = self.fixture(directory)
            (graph / "extra.bin").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "incomplete coverage"):
                PLACEMENT.snapshot(path)

    def test_paths_symlinks_uncompiled_and_empty_graphs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, graph, manifest = self.fixture(directory)
            (path.parent / "alias.mlmodelc").symlink_to(graph, target_is_directory=True)
            for name in (None, "", "../escape.mlmodelc", str(graph), "./graph.mlmodelc", "alias.mlmodelc",
                         "graph.mlpackage", "missing.mlmodelc"):
                path.write_text(json.dumps({**manifest, "compiled_model": name}))
                with self.assertRaises(ValueError):
                    PLACEMENT.snapshot(path)
            path.write_text(json.dumps(manifest))
            (graph / "alias").symlink_to(graph / "model.mil")
            with self.assertRaisesRegex(ValueError, "symlink"):
                PLACEMENT.snapshot(path)
        with tempfile.TemporaryDirectory() as directory:
            path, graph, _ = self.fixture(directory, runtime=False)
            (graph / "model.mil").unlink()
            with self.assertRaisesRegex(ValueError, "empty"):
                PLACEMENT.snapshot(path)

    def test_mutation_during_plan_load_does_not_produce_a_complete_report(self):
        for runtime in (False, True):
            with self.subTest(runtime=runtime), tempfile.TemporaryDirectory() as directory:
                path, graph, _ = self.fixture(directory, runtime)
                def mutate(_):
                    (graph / "model.mil").write_text("changed")
                    return plan(operation("matmul", "MLNeuralEngineComputeDevice"))
                with mock.patch.object(PLACEMENT, "load_plan", side_effect=mutate):
                    with self.assertRaises(ValueError):
                        PLACEMENT.inspect_manifest(path, quiet=True)

    def test_cli_preserves_outputs_and_inputs_and_writes_only_complete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path, graph, _ = self.fixture(directory)
            for destination in (path, graph / "report.json", path.parent / "missing/report.json"):
                argv = ["inspect", "--manifest", str(path), "--output", str(destination), "--quiet"]
                before = PLACEMENT.sha256_file(path)
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(PLACEMENT, "load_plan", return_value=plan(operation("const"))), \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        PLACEMENT.main()
                self.assertEqual(PLACEMENT.sha256_file(path), before)
                if destination != path:
                    self.assertFalse(destination.exists())
            output = path.parent / "report.json"
            argv = ["inspect", "--manifest", str(path), "--output", str(output), "--quiet"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(PLACEMENT, "load_plan", return_value=plan(operation("const"))), \
                    contextlib.redirect_stdout(io.StringIO()):
                PLACEMENT.main()
            self.assertEqual(json.loads(output.read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
