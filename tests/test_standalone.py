from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import turbocider.paths as paths
from turbocider.bootstrap import bootstrap_engines
from turbocider.registry import ModelRegistry


class StandaloneTests(unittest.TestCase):
    def test_bundled_packs_have_no_monorepo_or_home_paths(self):
        registry = ModelRegistry()
        for model in registry.all():
            raw = model.source.read_text(encoding="utf-8")
            self.assertNotIn("${WORKSPACE}", raw, model.id)
            self.assertNotIn("${HOME}", raw, model.id)
            self.assertNotIn("/gpu_ane/", raw, model.id)

    def test_engine_root_defaults_inside_package_and_respects_env(self):
        with mock.patch.dict(
            os.environ, {"TURBOCIDER_ENGINES_DIR": "/tmp/example-engines"}, clear=True
        ):
            self.assertEqual(
                paths.engine_root(), Path("/tmp/example-engines").resolve()
            )
        self.assertEqual(paths.engine_root(), paths.PACKAGE_ROOT / "engines")

    def test_bootstrap_materializes_expected_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "engines"
            for name in ("h3.c", "ltx-mac"):
                (source / name).mkdir(parents=True)
            gpu = source / "gpu_ane"
            for name in ("flux2-engine", "mac_local_ai", "flux2-runtime", "mflux-runtime"):
                (gpu / name).mkdir(parents=True)
            report = bootstrap_engines(source, target=target)
            self.assertEqual(report["h3"], "linked")
            self.assertEqual(report["ltx-mac"], "linked")
            self.assertEqual(report["flux2/engine"], "linked")
            self.assertTrue((target / "h3").is_symlink())
            self.assertTrue((target / "flux2" / "engine").is_symlink())


if __name__ == "__main__":
    unittest.main()
