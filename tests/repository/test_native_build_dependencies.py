"""Exercise diagnostic build entry points with recorded compiler invocations."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    "build_affine_fp32_projection_test.sh", "build_ane_calibration_test.sh",
    "build_ane_channel_auto_test.sh", "build_ane_ffn_test.sh",
    "build_ane_runtime_probe.sh", "build_convrot_ffn_error_probe.sh",
    "build_convrot_ffn_probe.sh", "build_convrot_rotation_test.sh",
    "build_gpu_weight_consumer_probe.sh", "build_qwen21_encoder_probe.sh",
    "build_seedvr2_frame_history_test.sh", "build_seedvr2_probe.sh",
    "build_tiny_vae_tests.sh", "build_z_image_runtime_ffn_error_probe.sh",
    "build_qwen21_session_probe.sh",
    "build_seedvr2_block_stream_probe.sh",
    "build_seedvr2_hybrid_probes.sh",
    "build_seedvr2_input_lifetime_test.sh",
    "build_seedvr2_media_input_test.sh",
    "build_seedvr2_spatial_probes.sh",
    "build_seedvr2_stream_input_probes.sh",
    "build_seedvr2_vae_ane_decode_probe.sh",
    "build_seedvr2_vae_causal_conv_probe.sh",
    "build_seedvr2_vae_io_probe.sh",
    "build_seedvr2_vae_phase_probe.sh",
    "build_seedvr2_vae_shared_tile_probe.sh",
    "build_seedvr2_vae_upscale_probe.sh",
)
ENTRY_POINTS = ("build.sh", "build_app.sh", "package.sh")
FALLBACKS = {
    "build_seedvr2_block_stream_probe.sh": "26.0",
    "build_seedvr2_hybrid_probes.sh": "",
    "build_seedvr2_spatial_probes.sh": "26.0",
    "build_seedvr2_stream_input_probes.sh": "26.0",
    "build_seedvr2_vae_ane_decode_probe.sh": "26.0",
    "build_seedvr2_vae_causal_conv_probe.sh": "26.0",
    "build_seedvr2_vae_io_probe.sh": "26.0",
    "build_seedvr2_vae_phase_probe.sh": "26.0",
    "build_seedvr2_vae_shared_tile_probe.sh": "26.0",
    "build_seedvr2_vae_upscale_probe.sh": "26.0",
}
OVERRIDABLE = {
    "build_qwen21_session_probe.sh",
    "build_ane_calibration_test.sh", "build_ane_channel_auto_test.sh",
    "build_ane_ffn_test.sh", "build_ane_runtime_probe.sh",
    "build_qwen21_encoder_probe.sh", "build_tiny_vae_tests.sh",
}
BUILD_COMMANDS = """Load command 0
      cmd LC_VERSION_MIN_MACOSX
  version 10.9
Load command 1
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 26.2
      sdk 26.4
Load command 2
      cmd LC_BUILD_VERSION
    minos 99.0
"""


class NativeBuildDependenciesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="tc-build-dependencies-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        scripts = self.root / "tools/native"
        scripts.mkdir(parents=True)
        for name in (*SCRIPTS, *ENTRY_POINTS, "dependencies.sh"):
            shutil.copyfile(ROOT / "tools/native" / name, scripts / name)
        self.mlx = self.root / "MLX with spaces"
        for name in ("include/mlx/mlx.h", "lib/libmlx.dylib"):
            path = self.mlx / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.library = self.root / "native library"
        self.library.mkdir()
        (self.library / "libturbocider.dylib").touch()
        self.license = self.mlx / "LICENSE"
        self.license.write_text("fixture license")
        self.commands = self.root / "commands.jsonl"
        self.otool_calls = self.root / "otool-calls.jsonl"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.write_tool("otool", """import json, os, sys
with open(os.environ['TC_TEST_OTOOL_CALLS'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
sys.stdout.write(os.environ['TC_TEST_OTOOL_OUTPUT'])
sys.exit(int(os.environ.get('TC_TEST_OTOOL_STATUS', '0')))
""")
        self.write_tool("xcrun", """import json, os, sys
if sys.argv[1:] == ['--sdk', 'macosx', '--show-sdk-path']:
    print('/SDK path')
elif sys.argv[1] == 'clang++':
    with open(os.environ['TC_TEST_COMMANDS'], 'a') as stream:
        stream.write(json.dumps(sys.argv[1:]) + '\\n')
else:
    raise SystemExit('unexpected xcrun invocation')
""")
        self.write_tool("build-python", """import sys
assert sys.argv[1] == 'tools/native/check_release_binary.py'
""")
        # Preserve ordinary host tools but discard ambient build overrides.
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith('TURBOCIDER_') and k not in ('MLX_ROOT', 'SDKROOT')}
        self.env.update(
            PATH=str(self.bin) + os.pathsep + os.defpath,
            MLX_ROOT=str(self.mlx), SDKROOT="/SDK path",
            DEVELOPER_DIR=str(self.root / "developer"),
            MLX_LICENSE_PATH=str(self.license),
            TURBOCIDER_BUILD_PYTHON=str(self.bin / "build-python"),
            TURBOCIDER_NATIVE_OUT=str(self.root / "native output"),
            TURBOCIDER_BUILD_OUTPUT_DIR=str(self.root / "build output"),
            TURBOCIDER_NATIVE_LIBRARY_DIR=str(self.library),
            TC_TEST_OTOOL_OUTPUT=BUILD_COMMANDS,
            TC_TEST_OTOOL_CALLS=str(self.otool_calls), TC_TEST_COMMANDS=str(self.commands),
        )

    def write_tool(self, name, body):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o755)

    def run_script(self, name, **changes):
        self.commands.unlink(missing_ok=True)
        self.otool_calls.unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", str(self.root / "tools/native" / name)],
            cwd=self.bin, env={**self.env, **changes},
            capture_output=True, text=True, timeout=15,
        )
        commands = ([json.loads(row) for row in self.commands.read_text().splitlines()]
                    if self.commands.exists() else [])
        return result, commands

    def assert_target(self, name, target, **changes):
        result, commands = self.run_script(name, **changes)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(len(commands), 0)
        self.assertTrue(any(str(self.mlx / 'include') in args for args in commands))
        for args in commands:
            self.assertEqual([arg for arg in args if arg.startswith('-mmacosx-version-min=')],
                             [f'-mmacosx-version-min={target}'])
            self.assertIn('-std=c++20', args)
        self.assertEqual([json.loads(row) for row in self.otool_calls.read_text().splitlines()],
                         [['-l', str(self.mlx / 'lib/libmlx.dylib')]])

    def test_first_build_version_and_paths_with_spaces(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                self.assert_target(name, '26.2')

    def test_missing_build_version_keeps_entry_point_fallback(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                self.assert_target(name, FALLBACKS.get(name, '15.0'), TC_TEST_OTOOL_OUTPUT='cmd LC_UUID\nminos 99.0\n')

    def test_explicit_override_policy_remains_per_entry_point(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                self.assert_target(name, '14.7' if name in OVERRIDABLE else '26.2',
                                   TURBOCIDER_DEPLOYMENT_TARGET='14.7')

    def test_otool_failure_never_invokes_compiler(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                result, commands = self.run_script(name, TC_TEST_OTOOL_STATUS='7')
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertEqual(commands, [])

    def test_missing_mlx_fails_before_metadata_or_compile(self):
        (self.mlx / 'lib/libmlx.dylib').unlink()
        for name in SCRIPTS:
            with self.subTest(script=name):
                result, commands = self.run_script(name)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('MLX C++ headers/libraries missing', result.stderr)
                self.assertEqual(commands, [])
                self.assertFalse(self.otool_calls.exists())

    def test_release_entrypoint_target_selection(self):
        # Execute the original entry-point setup through target selection. The
        # release guard is stubbed; real package builds validate it separately.
        stops = {'build.sh': 'MACOS_FLAGS=(', 'build_app.sh': 'FLAGS=(',
                 'package.sh': 'rm -rf "$APP" "$DIST/cli"'}
        scenarios = (
            ('metadata', {}, '26.2', 0),
            ('missing-metadata', {'TC_TEST_OTOOL_OUTPUT': 'cmd LC_UUID\nminos 99.0\n'}, '15.0', 0),
            ('override', {'TURBOCIDER_DEPLOYMENT_TARGET': '14.7',
                          'TURBOCIDER_PACKAGE_MIN_MACOS': '13.0'}, None, 0),
            ('metadata-failure', {'TC_TEST_OTOOL_STATUS': '7'}, '', 7),
        )
        for name in ENTRY_POINTS:
            source = (self.root / 'tools/native' / name).read_text()
            prefix = source[:source.index('\n' + stops[name])]
            variable = 'PACKAGE_MIN_MACOS' if name == 'package.sh' else 'DEPLOYMENT_TARGET'
            probe = self.root / 'tools/native/target_probe.sh'
            probe.write_text(prefix + f'\nprintf "%s\\n" "${variable}"\n')
            for scenario, changes, expected, status in scenarios:
                with self.subTest(script=name, scenario=scenario):
                    result, commands = self.run_script(probe.name, **changes)
                    self.assertEqual(result.returncode, status, result.stderr)
                    self.assertEqual(commands, [])
                    if expected is None:
                        expected = '13.0' if name == 'package.sh' else '14.7'
                    self.assertEqual(result.stdout, expected + ('\n' if status == 0 else ''))
                    self.assertEqual(
                        [json.loads(row) for row in self.otool_calls.read_text().splitlines()],
                        [['-l', str(self.mlx / 'lib/libmlx.dylib')]],
                    )

    def test_dependency_setup_does_not_eagerly_inspect_macho(self):
        result = subprocess.run(
            ['bash', '-euc', 'source tools/native/dependencies.sh'],
            cwd=self.root, env=self.env, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.otool_calls.exists())
        self.assertFalse(self.commands.exists())


if __name__ == '__main__':
    unittest.main()
