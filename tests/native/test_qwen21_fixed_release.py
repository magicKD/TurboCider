"""Actual owner-release coverage against the selected native library; small owned fixtures only."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import sysconfig
import tempfile
import unittest

from tests.native.test_qwen21_convrot_streaming import sources, write_fixture

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == "darwin" and os.environ.get("TURBOCIDER_TEST_GPU") == "1",
                     "requires explicit actual Metal opt-in")
class FixedReleaseTests(unittest.TestCase):
    def test_atomic_owner_release_and_stream_lifecycle(self):
        library_dir = (ROOT / os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR", "build/native")).resolve()
        library = library_dir / "libturbocider.dylib"
        manifest = json.loads((library_dir / "runtime-build/runtime-build-manifest.json").read_text())
        explicit = os.environ.get("TURBOCIDER_RELEASE_TEST_OUTPUT")
        if explicit:
            folder = Path(explicit).resolve()
            folder.mkdir(parents=True, exist_ok=False)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="tc-qwen-fixed-release-")
            self.addCleanup(temporary.cleanup)
            folder = Path(temporary.name)
        fixture = folder / "denoiser.safetensors"
        write_fixture(fixture, sources()[1])
        mlx = Path(os.environ.get("MLX_ROOT", str(Path(sysconfig.get_paths()["purelib"]) / "mlx")))
        sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
        binary = folder / "fixed-release"
        command = ["xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-mmacosx-version-min=26.2", "-isysroot", sdk,
            '-DTC_TEST_BOUND_NATIVE_ID="' + manifest["runtime_build_id"] + '"',
            "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core"), "-I", str(ROOT / "bindings/c/include"),
            "-isystem", str(mlx / "include"), str(ROOT / "tests/native/qwen21_fixed_release_test.cpp"),
            "-L", str(library_dir), "-lturbocider", "-L", str(mlx / "lib"), "-lmlx",
            "-Wl,-rpath," + str(library_dir), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(binary)]
        report = {"compile_command": command, "runtime_build_id": manifest["runtime_build_id"],
                  "library": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                  "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()}
        def save():
            (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        def invoke(arguments, stem, timeout, environment=None):
            with (folder / (stem + ".stdout")).open("x") as stdout, (folder / (stem + ".stderr")).open("x") as stderr:
                process = subprocess.Popen(arguments, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                           start_new_session=True)
                report[stem + "_pid"] = process.pid
                save()
                try:
                    code = process.wait(timeout=timeout)
                except BaseException as error:
                    report[stem + "_error"] = repr(error)
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                    except ProcessLookupError:
                        process.wait(timeout=5)
                    report[stem + "_exit_code"] = process.returncode
                    save()
                    raise
                report[stem + "_exit_code"] = code
                save()
                return code
        save()
        self.assertEqual(invoke(command, "compile", 120), 0,
                         (folder / "compile.stdout").read_text() + (folder / "compile.stderr").read_text())
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("TURBOCIDER_") and key not in (
                           "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FORCE_FLAT_NAMESPACE")}
        environment.update(MLX_ENABLE_TF32="0", TURBOCIDER_QWEN21_CONVROT_SHARED_A8="0",
                           TURBOCIDER_QWEN21_SOURCE_REGISTRATION="0")
        report["command"] = [str(binary), str(fixture), str(folder), str(library)]
        report["environment"] = {key: value for key, value in environment.items()
                                  if key.startswith(("TURBOCIDER_", "MLX_"))}
        save()
        code = invoke(report["command"], "probe", 90, environment)
        stdout = (folder / "probe.stdout").read_text()
        stderr = (folder / "probe.stderr").read_text()
        report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        report["library_unchanged"] = hashlib.sha256(library.read_bytes()).hexdigest() == report["library_sha256"]
        save()
        self.assertTrue(report["library_unchanged"])
        self.assertEqual(code, 0, stdout + stderr)
        self.assertIn("PASS Qwen fixed release: prefix1/2", stdout)
        self.assertIn("NATIVE_RUNTIME_ID=" + manifest["runtime_build_id"], stdout)
        print(stdout, end="")


if __name__ == "__main__":
    unittest.main()
