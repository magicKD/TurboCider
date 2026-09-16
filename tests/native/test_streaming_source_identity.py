#!/usr/bin/env python3
"""CPU-only tests for streaming build/source provenance capture."""

from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))

from capture_streaming_source_identity import (  # noqa: E402
    IdentityError,
    capture,
)


class SourceIdentityTests(unittest.TestCase):
    def test_exported_tree_has_stable_manifest(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "native").mkdir()
            (root / "native" / "a.c").write_text("int a;\n")
            (root / "docs").mkdir()
            (root / "docs" / "README.md").write_text("docs\n")
            (root / "build").mkdir()
            (root / "build" / "ignored.o").write_bytes(b"ignored")
            commit = "0123456789abcdef" * 2 + "01234567"
            first = capture(root, commit)
            second = capture(root, commit)
            self.assertEqual(first, second)
            self.assertTrue(first["clean"])
            self.assertEqual(first["tracked_file_count"], 2)
            self.assertNotIn("build/ignored.o", first["untracked_sources"])

    def test_exported_tree_requires_commit_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(IdentityError):
                capture(Path(raw), "not-a-commit")

    def test_git_worktree_records_dirty_and_untracked_state(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Test"],
                check=True,
            )
            tracked = root / "tracked.cpp"
            tracked.write_text("int value = 1;\n")
            subprocess.run(
                ["git", "-C", str(root), "add", "tracked.cpp"], check=True
            )
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "fixture"],
                check=True,
            )
            tracked.write_text("int value = 2;\n")
            (root / "new.cpp").write_text("int new_value;\n")
            (root / "build").mkdir()
            (root / "build" / "ignored.o").write_bytes(b"ignored")

            result = capture(root)
            self.assertEqual(len(result["commit"]), 40)
            self.assertFalse(result["clean"])
            self.assertGreater(result["dirty_diff_bytes"], 0)
            self.assertEqual(result["untracked_sources"], ["new.cpp"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
