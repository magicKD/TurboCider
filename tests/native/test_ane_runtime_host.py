"""Runtime FFN host contracts; no Core ML SDK, GPU work or model fixtures."""

import importlib.util
import errno
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "runtime_ane_export", ROOT / "tools/coreml/export_runtime_ane.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class RuntimeHostTests(unittest.TestCase):
    def fake_coreml(self, parent, before_conversion=lambda: None):
        """Only files/receipt control flow; never imports or executes Core ML."""
        compiled = parent / "compiler-result.mlmodelc"

        class Model:
            def save(self, path):
                package = Path(path)
                package.mkdir()
                (package / "source.txt").write_text("synthetic host fixture")

        def convert(*_args, **_kwargs):
            before_conversion()
            return Model()

        def compile_model(_path):
            compiled.mkdir()
            (compiled / "model.mil").write_text("synthetic host fixture")
            return str(compiled)

        fake = SimpleNamespace(
            convert=convert, target=SimpleNamespace(macOS15=15),
            precision=SimpleNamespace(FLOAT16=16), __version__="host-fixture",
            models=SimpleNamespace(utils=SimpleNamespace(compile_model=compile_model)))
        return fake, compiled

    @unittest.skipUnless(sys.platform == "darwin", "macOS exclusive rename contract")
    def test_export_publication_preserves_concurrent_targets_and_cleans_scratch(self):
        for kind in ("empty_directory", "dangling_symlink", "file"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix="tc-ane-export-host-") as temporary:
                parent = Path(temporary)
                destination = parent / "export"
                identity = []

                def competing_owner():
                    if kind == "empty_directory":
                        destination.mkdir()
                    elif kind == "dangling_symlink":
                        destination.symlink_to("missing-user-target", target_is_directory=True)
                    else:
                        destination.write_text("user-owned sentinel")
                    identity.append(destination.lstat().st_ino)

                fake, compiled = self.fake_coreml(parent, competing_owner)
                with mock.patch.dict(sys.modules, {"coremltools": fake}):
                    with self.assertRaises(FileExistsError):
                        EXPORT.export(destination, EXPORT.geometry("matmul", 2, 4, 6, 4, 6),
                                      program_factory=lambda _spec: None)
                self.assertEqual(destination.lstat().st_ino, identity[0])
                if kind == "empty_directory":
                    self.assertEqual(list(destination.iterdir()), [])
                elif kind == "dangling_symlink":
                    self.assertEqual(os.readlink(destination), "missing-user-target")
                else:
                    self.assertEqual(destination.read_text(), "user-owned sentinel")
                self.assertFalse(compiled.exists())
                self.assertEqual(sorted(path.name for path in parent.iterdir()), ["export"])

    @unittest.skipUnless(sys.platform == "darwin", "macOS exclusive rename contract")
    def test_export_success_and_copy_failure_keep_compiler_and_scratch_owned(self):
        for fail_copy in (False, True):
            with self.subTest(fail_copy=fail_copy), tempfile.TemporaryDirectory(prefix="tc-ane-export-host-") as temporary:
                parent = Path(temporary)
                destination = parent / "export"
                fake, compiled = self.fake_coreml(parent)
                with mock.patch.dict(sys.modules, {"coremltools": fake}):
                    if fail_copy:
                        with mock.patch.object(EXPORT.shutil, "copytree", side_effect=OSError("controlled copy failure")):
                            with self.assertRaisesRegex(OSError, "controlled copy failure"):
                                EXPORT.export(destination, EXPORT.geometry("matmul", 2, 4, 6, 4, 6),
                                              program_factory=lambda _spec: None)
                    else:
                        receipt = EXPORT.export(destination, EXPORT.geometry("matmul", 2, 4, 6, 4, 6),
                                                program_factory=lambda _spec: None)
                        self.assertEqual((destination / "graph.mlmodelc/model.mil").read_text(),
                                         "synthetic host fixture")
                        self.assertIn("graph.mlmodelc/model.mil", receipt["files"])
                        self.assertTrue((destination / "manifest.json").is_file())
                self.assertFalse(compiled.exists())
                self.assertEqual(sorted(path.name for path in parent.iterdir()), [] if fail_copy else ["export"])

    @unittest.skipUnless(sys.platform == "darwin", "macOS exclusive rename contract")
    def test_export_missing_exclusive_rename_fails_closed_and_cleans_scratch(self):
        with tempfile.TemporaryDirectory(prefix="tc-ane-export-host-") as temporary:
            parent = Path(temporary)
            fake, compiled = self.fake_coreml(parent)
            with mock.patch.dict(sys.modules, {"coremltools": fake}), \
                    mock.patch.object(EXPORT.ctypes, "CDLL", return_value=SimpleNamespace()):
                with self.assertRaises(OSError) as rejected:
                    EXPORT.export(parent / "export", EXPORT.geometry("matmul", 2, 4, 6, 4, 6),
                                  program_factory=lambda _spec: None)
            self.assertEqual(rejected.exception.errno, errno.ENOTSUP)
            self.assertFalse(compiled.exists())
            self.assertEqual(list(parent.iterdir()), [])

    def run_host_test(self, name, sources=()):
        with tempfile.TemporaryDirectory(prefix="tc-ane-host-") as temporary:
            binary = Path(temporary) / name
            subprocess.run(
                ["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                 str(ROOT / "tests/native" / f"{name}.cpp"),
                 *(str(ROOT / source) for source in sources), "-o", str(binary)],
                check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True)

    def test_row_scheduler_alignment_disable_reprobe_and_isolation(self):
        self.run_host_test("ane_scheduler_test")

    def test_qkv_complete_block_controller_and_periodic_reprobe(self):
        self.run_host_test("ane_qkv_scheduler_test")

    def test_memory_admission_and_host_scratch(self):
        self.run_host_test("ane_memory_test", ("native/backends/ane_memory.cpp",))

    def test_private_artifact_locks_crash_recovery_and_safe_refusal(self):
        self.run_host_test("ane_artifact_lease_test")

    def test_simd_conversion_matches_scalar_including_overflow_and_tails(self):
        self.run_host_test("ane_runtime_convert_test")

    def test_affine_q4_q8_layout_metadata_bounds_and_headroom(self):
        self.run_host_test("ane_runtime_quant_test")

    def test_raw_gguf_convrot_no_dense_temporary_bounds_and_headroom(self):
        self.run_host_test("ane_runtime_packed_test", ("native/core/gguf_decode.cpp",))

    def test_projection_layout_and_validation(self):
        spec = EXPORT.geometry("swiglu", 32, 64, 96, 33, 47)
        self.assertEqual(spec["inputs"], {"x": [32, 64], "wg": [96, 64],
                                          "wu": [96, 64], "wd": [64, 96]})
        self.assertEqual(spec["outputs"], {"y": [32, 64]})
        self.assertEqual(EXPORT.geometry("matmul", 32, 64, 96, 33, 47)["outputs"],
                         {"y": [32, 96]})
        self.assertNotIn("wg", EXPORT.geometry("gelu", 32, 64, 96, 33, 47)["inputs"])
        for rows in (0, -1, True, 32769, 32.5):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                EXPORT.geometry("swiglu", rows, 64, 96, 32, 48)
        with self.assertRaises(ValueError):
            EXPORT.geometry("unknown", 32, 64, 96, 32, 48)


if __name__ == "__main__":
    unittest.main(verbosity=2)
