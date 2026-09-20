#!/usr/bin/env python3
"""Compatibility keys must track behavioral inputs, not checkout locations."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
import generate_runtime_build_identity as identity


class RuntimeBuildIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "checkout"
        for relative in ("native/model.cpp", "bindings/c/api.h", "apps/cli/main.mm",
                         "services/turbociderd/service.mm", *identity.BUILD_SCRIPTS):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)
        self.mlx = self.root / "dependency"
        for relative in ("include/mlx/mlx.h", "include/mlx/private.h", "lib/libmlx.dylib",
                         "lib/libjaccl.dylib", "lib/mlx.metallib"):
            path = self.mlx / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)
        self.sdk = self.root / "SDK"
        self.sdk.mkdir()
        (self.sdk / "SDKSettings.json").write_text('{"Version":"26.2"}')
        self.tools = self.root / "toolchain"
        self.tools.mkdir()
        for name in ("clang", "clang++", "ld", "ar"):
            (self.tools / name).write_text(name)
        self.args = argparse.Namespace(
            root=self.root, mlx_root=self.mlx, sdk=self.sdk, toolchain=self.tools,
            output=self.root / "build/runtime-build", deployment_target="26.2",
            test_hooks="0", audit_counters="0", experimental_probes="0",
            flags=["--", "-O2", "-isysroot", str(self.sdk), "-isystem", str(self.mlx / "include")])
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {name: "" for name in identity.SEARCH_ENV}).start()
        patch.object(identity.subprocess, "check_output", return_value=
                     "Apple clang version test\nTarget: arm64-apple-darwin\nInstalledDir: /test\n").start()

    def manifest(self):
        return identity.manifest(self.args)

    def test_relocated_identical_checkout_has_same_key(self):
        original = self.manifest()
        relocated = self.root.parent / "moved"
        shutil.copytree(self.root, relocated)
        args = copy.deepcopy(self.args)
        for key in ("root", "mlx_root", "sdk", "toolchain", "output"):
            setattr(args, key, Path(str(getattr(args, key)).replace(str(self.root), str(relocated))))
        args.flags = [flag.replace(str(self.root), str(relocated)) for flag in args.flags]
        self.assertEqual(original, identity.manifest(args))

    def test_source_build_script_and_shader_changes_invalidate(self):
        for relative in ("native/model.cpp", "tools/native/build.sh", "dependency/lib/mlx.metallib"):
            with self.subTest(relative=relative):
                before = self.manifest()
                path = self.root / relative
                path.write_text(path.read_text() + "changed")
                self.assertNotEqual(before["runtime_build_id"], self.manifest()["runtime_build_id"])

    def test_dependency_header_toolchain_and_sdk_changes_invalidate(self):
        for path in (self.mlx / "include/mlx/private.h", self.mlx / "lib/libmlx.dylib",
                     self.tools / "ld", self.sdk / "SDKSettings.json"):
            with self.subTest(path=path):
                before = self.manifest()
                path.write_text(path.read_text() + "change")
                self.assertNotEqual(before["runtime_build_id"], self.manifest()["runtime_build_id"])

    def test_policy_changes_invalidate(self):
        original = self.manifest()
        for name in ("test_hooks", "audit_counters", "experimental_probes", "deployment_target"):
            with self.subTest(name=name):
                args = copy.deepcopy(self.args)
                setattr(args, name, "1")
                self.assertNotEqual(original["runtime_build_id"], identity.manifest(args)["runtime_build_id"])
        self.args.flags.append("-ffast-math")
        self.assertNotEqual(original["runtime_build_id"], self.manifest()["runtime_build_id"])

    def test_documentation_and_build_outputs_are_not_runtime_inputs(self):
        before = self.manifest()
        (self.root / "native/README.md").write_text("documentation")
        catalog = self.root / "native/runtime/streaming/bundled_catalog.json"
        catalog.parent.mkdir(parents=True)
        catalog.write_text('{"revision":"independently-versioned-data","records":[]}')
        self.args.output.mkdir(parents=True)
        (self.args.output / "generated.hpp").write_text("not source")
        self.assertEqual(before, self.manifest())

    def test_build_input_change_and_unknown_header_search_fail(self):
        value = self.manifest()
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(value))
        identity.verify_existing(manifest, value)
        (self.root / "native/model.cpp").write_text("modified while compiling")
        with self.assertRaisesRegex(ValueError, "changed during compilation"):
            identity.verify_existing(manifest, self.manifest())
        with patch.dict(os.environ, {"CPATH": "/untracked/headers"}):
            with self.assertRaisesRegex(ValueError, "CPATH"):
                self.manifest()
        (self.mlx / "lib/mlx.metallib").unlink()
        with self.assertRaisesRegex(ValueError, "missing MLX"):
            self.manifest()


if __name__ == "__main__":
    unittest.main(verbosity=2)
