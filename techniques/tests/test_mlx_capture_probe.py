"""CPU-only capture contracts; never import real MLX or launch GPU work."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "labs/metal/mlx_capture_probe.py"
SPEC = importlib.util.spec_from_file_location("mlx_capture_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class CaptureProbeTests(unittest.TestCase):
    def run_probe(self, output):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = probe.main(["--output-dir", str(output)])
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(json.loads(stdout.getvalue()), report)
        self.assertFalse(report["production_qualified"])
        self.assertFalse(report["formal_performance_score"])
        self.assertTrue(report["raw_trace_local_only"])
        return status, report

    def test_help_does_not_require_optional_dependencies(self):
        child = subprocess.run([sys.executable, "-I", "-S", "-B", str(SCRIPT), "--help"],
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("--output-dir", child.stdout)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            receipt = output / "report.json"
            receipt.write_text('{"existing": true}\n', encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                probe.main(["--output-dir", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(receipt.read_text(encoding="utf-8"), '{"existing": true}\n')
            self.assertFalse((output / SCRIPT.name).exists())

    def test_unsupported_host_records_failure_before_optional_imports(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(probe.platform, "system", return_value="Linux"), \
                patch.dict(sys.modules, {"numpy": None, "mlx": None, "mlx.core": None}), \
                patch.dict(os.environ, {"MTL_CAPTURE_ENABLED": "unchanged"}):
            output = Path(directory) / "unsupported"
            status, report = self.run_probe(output)
            self.assertEqual(status, 1)
            self.assertEqual(report["status"], "capture-unavailable-or-failed")
            self.assertEqual(report["errors"][0]["type"], "ValueError")
            self.assertEqual(report["samples"], [])
            self.assertEqual(os.environ["MTL_CAPTURE_ENABLED"], "unchanged")
            self.assertEqual((output / SCRIPT.name).read_bytes(), SCRIPT.read_bytes())
            self.assertEqual(report["runner_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())

    def test_low_disk_records_failure_without_loading_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(probe.platform, "system", return_value="Darwin"), \
                patch.object(probe.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), \
                patch.dict(sys.modules, {"numpy": None, "mlx": None, "mlx.core": None}):
            status, report = self.run_probe(Path(directory) / "low-disk")
            self.assertEqual(status, 1)
            self.assertEqual(report["errors"][0]["type"], "ValueError")

    def test_cleanup_failure_preserves_original_error_and_saves_report(self):
        core = ModuleType("mlx.core")
        core.gpu = object()
        core.set_default_device = Mock(side_effect=RuntimeError("device init failed"))
        core.synchronize = Mock(side_effect=RuntimeError("synchronize failed"))
        mlx = ModuleType("mlx")
        mlx.core = core
        modules = {"numpy": ModuleType("numpy"), "mlx": mlx, "mlx.core": core}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(probe.platform, "system", return_value="Darwin"), \
                patch.object(probe.shutil, "disk_usage", return_value=SimpleNamespace(free=1024**3)), \
                patch.dict(sys.modules, modules), patch.dict(os.environ):
            status, report = self.run_probe(Path(directory) / "cleanup-failed")
            self.assertEqual(status, 1)
            self.assertEqual(report["status"], "capture-unavailable-or-failed")
            self.assertEqual([error["message"] for error in report["errors"]],
                             ["device init failed", "synchronize failed"])
            core.synchronize.assert_called_once_with()

    def test_missing_or_empty_artifacts_are_not_successful_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            missing = output / "missing.gputrace"
            empty_file = output / "empty-file.gputrace"
            empty_file.touch()
            empty_bundle = output / "empty-bundle.gputrace"
            empty_bundle.mkdir()
            for capture in (missing, empty_file, empty_bundle):
                with self.subTest(capture=capture.name), self.assertRaises(ValueError):
                    probe.capture_artifacts(capture, output)
            (empty_bundle / "empty.bin").touch()
            with self.assertRaises(ValueError):
                probe.capture_artifacts(empty_bundle, output)

    def test_file_and_bundle_artifacts_have_portable_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            capture = output / "file.gputrace"
            capture.write_bytes(b"tiny synthetic fixture")
            records = probe.capture_artifacts(capture, output)
            self.assertEqual(records, [{
                "path": "file.gputrace", "bytes": 22,
                "sha256": hashlib.sha256(b"tiny synthetic fixture").hexdigest(),
            }])
            bundle = output / "bundle.gputrace"
            bundle.mkdir()
            (bundle / "data.bin").write_bytes(b"synthetic")
            records = probe.capture_artifacts(bundle, output)
            self.assertEqual(records[0]["path"], "bundle.gputrace/data.bin")
            self.assertEqual(records[0]["bytes"], 9)


if __name__ == "__main__":
    unittest.main()
