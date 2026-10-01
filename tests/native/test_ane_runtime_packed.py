"""Direct GGUF/ConvRot IOSurface staging; opt-in public Core ML integration."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("packed_runtime_export", ROOT / "tools/coreml/export_runtime_ane.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


@unittest.skipUnless(sys.platform == "darwin" and os.environ.get("TURBOCIDER_TEST_RUNTIME_PACKED") == "1",
                     "set TURBOCIDER_TEST_RUNTIME_PACKED=1 on macOS")
class PackedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="tc-packed-runtime-")
        cls.addClassCleanup(cls.scratch.cleanup)
        cls.directory = Path(cls.scratch.name)
        env = {**os.environ, "TURBOCIDER_NATIVE_OUT": str(cls.directory / "build")}
        subprocess.run(["bash", "tools/native/build_ane_runtime_probe.sh"], cwd=ROOT, env=env,
                       check=True, capture_output=True, text=True, timeout=120)
        EXPORT.export(cls.directory / "graph", EXPORT.geometry("matmul", 32, 256, 512, 256, 512))

    def check_policy(self, policy):
        completed = subprocess.run([str(self.directory / "build/ane-runtime-packed-probe"),
                                    str(self.directory / "graph/manifest.json"), policy],
                                   cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        data = json.loads(completed.stdout)
        self.assertEqual(data["status"], "pass")
        self.assertEqual(data["gguf_types"], 11)
        self.assertEqual(data["weight_swap_cycles"], 10)
        self.assertTrue(data["failed_stage_rejected"])
        self.assertTrue(data["same_graph"])
        self.assertEqual(data["observed_placement"], "unknown")

    def test_cpu_only(self):
        self.check_policy("cpu")

    def test_cpu_and_ne(self):
        self.check_policy("ne")


if __name__ == "__main__":
    unittest.main()
