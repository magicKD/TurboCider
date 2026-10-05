"""Private API isolation, host contract and explicit hardware integration.

Default discovery does not compile/load private AppleNeuralEngine code.
Hardware tests require TURBOCIDER_TEST_PRIVATE_ANE=1, and propagate all failures.
"""
import os
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PrivateAneHostTests(unittest.TestCase):
    def test_independent_w8a8_rotation_scale_and_rounding_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "math"
            subprocess.run(["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "tests/native/ane_w8a8_math_test.cpp", "native/core/gguf_decode.cpp", "-o", str(binary)],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True, timeout=10)
    def test_policy_and_native_micrograph(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "policy"
            subprocess.run(["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "tests/native/ane_backend_policy_test.cpp", "native/backends/private/ane_mil.cpp",
                            "-o", str(binary)], cwd=ROOT, check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True, timeout=10)

    def test_default_build_omits_private_source_and_binds_feature_flag(self):
        source = (ROOT / "tools/native/build.sh").read_text()
        self.assertIn('PRIVATE_ANE="${TURBOCIDER_ENABLE_PRIVATE_ANE:-0}"', source)
        private_sources = source.index("SOURCES+=(native/backends/private/")
        self.assertIn('if [[ "$PRIVATE_ANE" == "1" ]]; then', source[private_sources - 50:private_sources])
        self.assertIn("COMMON+=(-DTURBOCIDER_ENABLE_PRIVATE_ANE=1)", source)
        public = (ROOT / "native/backends/ane_runtime.mm").read_text()
        for symbol in ("_ANEClient", "_ANERequest", "_ANESharedEvents"):
            self.assertNotIn(symbol, public)


@unittest.skipUnless(sys.platform == "darwin" and os.environ.get("TURBOCIDER_TEST_PRIVATE_ANE") == "1",
                     "set TURBOCIDER_TEST_PRIVATE_ANE=1 on Apple Silicon for private driver tests")
class PrivateAneHardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="tc-private-ane-")
        cls.addClassCleanup(cls.temporary.cleanup)
        # macOS /var and /tmp are system symlinks. Pass their canonical path
        # so cache security checks can still reject application-created links.
        cls.root = Path(cls.temporary.name).resolve()
        cls.build = cls.root / "build"
        subprocess.run(["bash", "tools/native/build_private_ane_test.sh"], cwd=ROOT,
                       env={**os.environ, "TURBOCIDER_NATIVE_OUT": str(cls.build)}, check=True,
                       capture_output=True, text=True, timeout=120)

    def run_native(self, name, expected):
        cache = self.root / name
        cache.mkdir()
        command = [str(self.build / name), str(cache)]
        for run in range(2):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(expected, result.stdout)
            if name == "private-ane-program-test":
                self.assertIn(f"compiled_now={1 if run == 0 else 0}", result.stdout)
            print(result.stdout.strip())
        # Same-length corruption must not be accepted by a file-size check.
        source = next(cache.glob("*/model.mil"))
        data = source.read_bytes()
        source.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source cache digest mismatch", result.stderr)

    def test_real_shared_event_aba_and_cross_process_cache(self):
        self.run_native("private-ane-program-test", "PASS private ANE GPU-signal/ANE/GPU-wait")

    def test_tiled_tail_multi_chunk_lora_and_failed_staging(self):
        self.run_native("private-ane-executor-test", "PASS private executor kind=1 lora=1")

    def test_real_gpu_transfer_rounding_lora_headroom_and_failure_suppression(self):
        result = subprocess.run([str(self.build / "private-ane-transfer-test"), str(self.root / "gpu-transfer-cache")],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS GPU Executor", result.stdout)
        self.assertIn("PASS GPU tiled I/O", result.stdout)
        self.assertIn("PASS independent GPU I/O", result.stdout)
        print(result.stdout.strip())

    def test_gpu_w8_stager_raw_gguf_affine_and_dense(self):
        result = subprocess.run([str(self.build / "private-ane-w8-stage-test")], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("PASS GPU W8 source="), 9)
        self.assertIn("PASS W8 nonfinite/scale overflow rejection", result.stdout)
        self.assertIn("PASS W8 compact scale cache",result.stdout)
        self.assertIn("PASS W8 immutable sign metadata",result.stdout)
        self.assertIn("PASS W8 pipeline specialization",result.stdout)
        self.assertIn("PASS W8 dense typed loads",result.stdout)
        print(result.stdout.strip())

    def test_convrot_direct_codes_and_comfy_activation_staging(self):
        result = subprocess.run([str(self.build / "private-ane-convrot-stage-test")], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS 37 direct raw/packed Q8 cases", result.stdout)
        self.assertIn("PASS 9 Comfy H256 A8 typed/strided cases", result.stdout)
        print(result.stdout.strip())

    def test_convrot_direct_executor_lora_channels_and_failure_recovery(self):
        result = subprocess.run([str(self.build / "private-ane-convrot-executor-test"),
                                 str(self.root / "convrot-executor-cache")], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("PASS direct ConvRot Executor"), 4)
        print(result.stdout.strip())

    def test_convrot_group256_executor_and_hidden_scale_join(self):
        result = subprocess.run([str(self.build / "private-ane-convrot-executor-test"),
                                 str(self.root / "convrot-group256-cache"), "group256"], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("group=256"), 4)
        print(result.stdout.strip())

    def test_convrot_separate_input_hidden_group_scope(self):
        for scope in ("input", "hidden"):
            result = subprocess.run([str(self.build / "private-ane-convrot-executor-test"),
                                     str(self.root / f"convrot-group-{scope}-cache"), "group256", scope],
                                    cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count(f"scope={scope}"), 4)
            print(result.stdout.strip())

    def test_private_w8a8_normalized_matmul_two_banks_and_gpu_epilogue(self):
        result = subprocess.run([str(self.build / "private-ane-w8-pipeline-test"), str(self.root / "w8-pipeline-cache")],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS private dynamic W8A8", result.stdout)
        print(result.stdout.strip())

    def test_full_w8a8_swiglu_lora_hidden_and_gpu_epilogue(self):
        result = subprocess.run([str(self.build / "private-ane-w8-ffn-test"), str(self.root / "w8-ffn-cache")],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("PASS full private W8A8 SwiGLU"), 2)
        print(result.stdout.strip())

    def test_w8a8_common_executor_multichunk_lora_weight_switch_and_failure_recovery(self):
        for lookahead in ("0", "1"):
            result = subprocess.run([str(self.build / "private-ane-w8-executor-test"),
                                     str(self.root / f"w8-executor-cache-{lookahead}"), lookahead],
                                    cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PASS W8 Executor", result.stdout)
            self.assertIn("PASS full 4224-row W8 bucket", result.stdout)
            self.assertIn(f"bounded A8 lookahead={lookahead}", result.stdout)
            print(result.stdout.strip())

    def test_missing_producer_times_out_without_deadlock_or_reuse(self):
        result = subprocess.run([str(self.build / "private-ane-program-test"),
                                 str(self.root / "timeout-cache"), "timeout"], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS private timeout releases GPU wait", result.stdout)

    @unittest.skipUnless(importlib.util.find_spec("coremltools") is not None, "coremltools required for public template")
    def test_common_factory_build_on_off_and_public_fallback(self):
        spec = importlib.util.spec_from_file_location("private_factory_export", ROOT / "tools/coreml/export_runtime_ane.py")
        export = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export)
        directory = self.root / "factory-template"
        export.export(directory, export.geometry("swiglu", 33, 64, 96, 32, 48, lora_inputs=True))
        common = ["xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                  "-Wno-deprecated-declarations", "-fobjc-arc", "-mmacosx-version-min=15.0",
                  "native/backends/ane_backend.mm", "native/backends/ane_runtime.mm",
                  "native/backends/ane_memory.cpp", "native/core/gguf_decode.cpp",
                  "tests/native/ane_backend_factory_test.cpp", "-framework", "Foundation",
                  "-framework", "CoreML", "-framework", "CoreVideo", "-framework", "IOSurface"]
        for enabled in (False, True):
            binary = self.build / f"factory-{int(enabled)}"
            command = list(common)
            if enabled:
                command += ["-DTURBOCIDER_ENABLE_PRIVATE_ANE=1", "native/backends/private/ane_program.mm",
                            "native/backends/private/ane_mil.cpp", "native/backends/private/ane_executor.mm",
                            "native/backends/private/ane_w8_executor.mm",
                            "-framework", "Metal"]
            command += ["-o", str(binary)]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
            result = subprocess.run([str(binary), str(directory / "manifest.json")], cwd=ROOT,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PASS common geometry", result.stdout)
            if not enabled:
                strings = subprocess.check_output(["strings", str(binary)], text=True)
                for private_symbol in ("_ANEClient", "_ANERequest", "_ANESharedEvents", "_ANEIOSurfaceObject"):
                    self.assertNotIn(private_symbol, strings)
            print(result.stdout.strip())


@unittest.skipUnless(sys.platform == "darwin" and os.environ.get("TURBOCIDER_TEST_PRIVATE_CHANNEL_MLX") == "1",
                     "set TURBOCIDER_TEST_PRIVATE_CHANNEL_MLX=1 and select a private-enabled native library")
class PrivateAneChannelMlxTests(unittest.TestCase):
    def test_all_rows_physical_range_joint_lora_and_late_failure(self):
        spec = importlib.util.spec_from_file_location("private_channel_export", ROOT / "tools/coreml/export_runtime_ane.py")
        export = importlib.util.module_from_spec(spec); spec.loader.exec_module(export)
        library = (ROOT / os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR", "build/native")).resolve()
        self.assertTrue((library / "libturbocider.dylib").is_file(), "build an explicitly private-enabled library")
        with tempfile.TemporaryDirectory(prefix="tc-private-channel-") as directory:
            root = Path(directory).resolve()
            graph = root / "template"
            export.export(graph, export.geometry("swiglu", 33, 128, 1024, 128, 128, lora_inputs=True))
            build, fixtures = root / "build", root / "fixtures"; fixtures.mkdir()
            subprocess.run(["bash", "tools/native/build_ane_channel_test.sh"], cwd=ROOT, check=True,
                           env={**os.environ,"TURBOCIDER_NATIVE_OUT":str(build),"TURBOCIDER_NATIVE_LIBRARY_DIR":str(library)},
                           capture_output=True,text=True,timeout=120)
            result = subprocess.run([str(build / "ane-channel-ffn-test"),str(graph / "manifest.json"),str(fixtures)],
                                    cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("PASS channel MLX/W8 Executor",result.stdout)
            self.assertIn("PASS channel LoRA range callback",result.stdout)
            self.assertIn("PASS fixed async channel",result.stdout)
            self.assertIn("PASS deferred channel join",result.stdout)
            self.assertIn("PASS deferred typed lifetime",result.stdout)
            self.assertIn("PASS channel failure cleanup",result.stdout)
            print(result.stdout.strip())


@unittest.skipUnless(sys.platform == "darwin" and os.environ.get("TURBOCIDER_TEST_PRIVATE_CALIBRATION_MLX") == "1",
                     "set TURBOCIDER_TEST_PRIVATE_CALIBRATION_MLX=1 for prepared W8/MLX calibration tests")
class PrivateAneCalibrationMlxTests(unittest.TestCase):
    def test_prepared_w8_alone_concurrent_lifetime_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="tc-private-calibration-") as directory:
            root = Path(directory).resolve()
            build = root / "build"
            subprocess.run(["bash", "tools/native/build_ane_calibration_test.sh"], cwd=ROOT, check=True,
                           env={**os.environ, "TURBOCIDER_NATIVE_OUT": str(build)},
                           capture_output=True, text=True, timeout=120)
            result = subprocess.run([str(build / "private-ane-calibration-test"), str(root / "cache")],
                                    cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("PASS prepared W8 calibration channels="), 2)
            self.assertIn("PASS prepared calibration ownership", result.stdout)
            self.assertEqual(result.stdout.count("PASS complete GPU calibration:"), 2)
            self.assertIn("not model/E2E calibration or physical overlap proof", result.stdout)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
