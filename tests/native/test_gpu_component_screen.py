"""Component receipt observer, not benchmark-quality or GPU-placement tests."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("gpu_component_screen", ROOT / "tools/native/run_gpu_component_screen.py")
SCREEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREEN)


class ComponentObserverTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
