"""CPU/file characterization of SeedVR2 evidence I/O; no models or servers."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
import seedvr2_framework_screen as framework
import seedvr2_hybrid_screen as hybrid


class EvidenceBehavior:
    def test_file_digests_include_empty_and_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "input.bin"
            for size in (0, 1, (4 << 20) - 1, 4 << 20, (4 << 20) + 1):
                with self.subTest(size=size):
                    payload = (bytes(range(256)) * (size // 256 + 1))[:size]
                    path.write_bytes(payload)
                    self.assertEqual(self.module.sha256(path), hashlib.sha256(payload).hexdigest())

    def test_file_access_errors_and_symlink_input(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaises(FileNotFoundError):
                self.module.sha256(root / "missing")
            with self.assertRaises(IsADirectoryError):
                self.module.sha256(root)
            source = root / "source.bin"
            source.write_bytes(b"original evidence")
            link = root / "link.bin"
            link.symlink_to(source)
            self.assertEqual(self.module.sha256(link), hashlib.sha256(source.read_bytes()).hexdigest())

    def test_json_replaces_destination_and_keeps_format(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "report.metrics"
            path.write_text("old report")
            value = {"label": "图片", "rows": [1, None, True]}
            self.module.write_json(path, value)
            self.assertEqual(path.read_text(), json.dumps(value, indent=2) + "\n")
            self.assertEqual(json.loads(path.read_text()), value)
            self.assertEqual(sorted(p.name for p in Path(raw).iterdir()), [path.name])

    def test_nonfinite_diagnostics_keep_existing_json_policy(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "diagnostic.json"
            self.module.write_json(path, {"value": float("nan"), "limit": float("inf")})
            value = json.loads(path.read_text())
            self.assertTrue(math.isnan(value["value"]))
            self.assertEqual(value["limit"], float("inf"))

    def test_serialization_failure_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "report.json"
            temporary = path.with_suffix(".json.tmp")
            path.write_text("previous report")
            temporary.write_text("previous temporary")
            with self.assertRaises(TypeError):
                self.module.write_json(path, {"unsupported": object()})
            self.assertEqual(path.read_text(), "previous report")
            self.assertEqual(temporary.read_text(), "previous temporary")

    def test_rename_failure_preserves_report_and_written_temporary(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "report.data"
            path.write_text("previous report")
            with patch.object(Path, "replace", side_effect=PermissionError("fixture rename denial")):
                with self.assertRaises(PermissionError):
                    self.module.write_json(path, {"ready": True})
            self.assertEqual(path.read_text(), "previous report")
            self.assertEqual(json.loads(path.with_suffix(".data.tmp").read_text()), {"ready": True})

    def test_missing_parent_is_not_created(self):
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw) / "absent"
            with self.assertRaises(FileNotFoundError):
                self.module.write_json(parent / "report.json", {})
            self.assertFalse(parent.exists())


class FrameworkEvidenceTests(EvidenceBehavior, unittest.TestCase):
    module = framework


class HybridEvidenceTests(EvidenceBehavior, unittest.TestCase):
    module = hybrid


if __name__ == "__main__":
    unittest.main()
