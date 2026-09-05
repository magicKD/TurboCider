from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import turbocider.paths as paths


class PathTests(unittest.TestCase):
    def test_workspace_pointer_is_used_for_bundled_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Demo.app" / "Contents" / "Resources" / "TurboCider"
            root.mkdir(parents=True)
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            (root / "workspace.path").write_text(str(workspace) + "\n")
            with mock.patch.object(paths, "PACKAGE_ROOT", root), mock.patch.dict(
                os.environ, {}, clear=True
            ):
                self.assertEqual(paths.workspace_root(), workspace.resolve())
                self.assertEqual(
                    paths.model_root(),
                    (Path.home() / "Library" / "Application Support" / "TurboCider" / "models").resolve(),
                )

    def test_environment_workspace_has_priority(self):
        with tempfile.TemporaryDirectory() as temporary:
            configured = Path(temporary)
            with mock.patch.dict(
                os.environ, {"TURBOCIDER_WORKSPACE": str(configured)}, clear=True
            ):
                self.assertEqual(paths.workspace_root(), configured.resolve())


if __name__ == "__main__":
    unittest.main()
