"""Keep the integration entry points portable and host/device work separate."""
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROBE_BUILDERS = (
    "async_step_cancel", "base_swiglu", "lora_swiglu", "qkv_lora_block",
    "qkv_numeric", "qkv_private_combo", "runtime_factor", "static_lora_block",
    "tokenizer_reuse", "vae_fallback", "vae_norm", "vae_precision",
)


class RuntimeAneIntegrationTests(unittest.TestCase):
    def test_portable_shape_descriptors_do_not_require_ignored_exports(self):
        common = dict(schema_version=1, descriptor_version=1, backend="private_runtime_shape",
                      kind="swiglu", layout="out_in", biases=False, lora_inputs=False,
                      hidden=4096, width=12288, tile_k=4096, tile_n=512)
        for rows in (1024, 1536, 2048):
            path = ROOT / f"examples/coreml/qwen21-runtime-shape-r{rows}.json"
            self.assertEqual(json.loads(path.read_text()), dict(common, rows=rows))

    def test_qkv_coordination_is_in_the_main_build(self):
        sources = [f"native/models/qwen21/runtime_private_qkv{suffix}" for suffix in (".cpp", ".mm")]
        present = [source for source in sources if (ROOT / source).is_file()]
        self.assertEqual(len(present), 1)
        self.assertIn(present[0], (ROOT / "tools/native/build.sh").read_text())

    def test_helpers_do_not_silently_link_retired_experiment_builds(self):
        for name in PROBE_BUILDERS:
            with self.subTest(builder=name):
                source = (ROOT / "tools/native" / f"build_qwen21_{name}_probe.sh").read_text()
                self.assertIn('LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native}"', source)

    def test_default_host_suite_excludes_live_native_and_checkpoint_tests(self):
        filename = ROOT / "tests/native/runtime_ane_integration_suite.py"
        spec = importlib.util.spec_from_file_location("runtime_ane_host_suite", filename)
        suite = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(suite)
        modules = suite.HOST_MODULES
        self.assertEqual(len(modules), len(set(modules)))
        self.assertGreater(len(modules), 70)
        for name in modules:
            self.assertTrue((ROOT / "tests/native" / (name + ".py")).is_file(), name)
        self.assertFalse(set(modules) & {
            "test_contract", "test_qwen21_lifecycle", "test_qwen21_async_step_pair",
            "test_qwen21_convrot_runtime", "test_qwen21_qkv_prefix_snapshot",
            "test_qwen21_scheduler", "test_qwen21_tokenizer_reuse_session",
            "test_qwen21_vae_norm", "test_ane_runtime",
        })
        makefile = (ROOT / "Makefile").read_text()
        self.assertIn("test-qwen21-runtime-host:", makefile)
        self.assertIn('tests/native/runtime_ane_integration_suite.py', makefile)


if __name__ == "__main__":
    unittest.main()
