"""Component receipt observer, not benchmark-quality or GPU-placement tests."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("gpu_component_screen", ROOT / "tools/native/run_gpu_component_screen.py")
SCREEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREEN)


class ComponentObserverTests(unittest.TestCase):
    def test_identity_includes_runtime_library_not_only_thin_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/"probe";library=Path(directory)/"libturbocider.dylib"
            binary.write_bytes(b"thin-cli")
            a=SCREEN.artifact_identity(binary)
            self.assertEqual(a["binary_sha256"],hashlib.sha256(b"thin-cli").hexdigest())
            self.assertIsNone(a["adjacent_library_sha256"])
            library.write_bytes(b"first-runtime")
            b=SCREEN.artifact_identity(binary)
            self.assertNotEqual(a,b)
            library.write_bytes(b"second-runtime")
            c=SCREEN.artifact_identity(binary)
            self.assertEqual(b["binary_sha256"],c["binary_sha256"])
            self.assertNotEqual(b["adjacent_library_sha256"],c["adjacent_library_sha256"])
    def test_comm_only_host_observation_and_duration(self):
        with mock.patch.object(SCREEN.subprocess, "run", return_value=mock.Mock(stdout="12 1 0.4 /bin/probe\n")) as run, \
             mock.patch.object(SCREEN.os, "getloadavg", return_value=(1., 2., 3.)), \
             mock.patch.object(SCREEN.time, "monotonic", side_effect=[10., 10.25]):
            result = SCREEN.observe()
        self.assertEqual(result["processes"], ["12 1 0.4 /bin/probe"])
        self.assertEqual(result["load_average"], [1., 2., 3.])
        self.assertEqual(result["observation_seconds"], .25)
        self.assertEqual(run.call_args.args[0], ["ps", "-axo", "pid=,ppid=,%cpu=,comm="])

    def test_missing_observation_is_not_fabricated(self):
        with mock.patch.object(SCREEN.subprocess, "run", side_effect=subprocess.TimeoutExpired("ps", 5)):
            with self.assertRaises(subprocess.TimeoutExpired):
                SCREEN.observe()

    def test_changed_runtime_preserves_failed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"receipt.json"
            identity=dict(binary_sha256="a"*64,adjacent_library_sha256="b"*64)
            changed=dict(identity,adjacent_library_sha256="c"*64)
            observation=dict(monotonic=1.,load_average=[0.,0.,0.],processes=[],observation_seconds=.01)
            with mock.patch.object(sys,"argv",["screen","--output",str(output),"--","/usr/bin/true"]), \
                 mock.patch.object(SCREEN,"artifact_identity",side_effect=[identity,changed]), \
                 mock.patch.object(SCREEN,"observe",return_value=observation), \
                 mock.patch("builtins.print"):
                with self.assertRaises(SystemExit) as raised:SCREEN.main()
            self.assertEqual(raised.exception.code,1)
            raw=json.loads(output.read_text())
            self.assertEqual(raw["exit_code"],0)
            self.assertFalse(raw["artifacts_unchanged"])
            self.assertEqual(raw["adjacent_library_sha256"],"b"*64)


if __name__ == "__main__":
    unittest.main()
