from types import SimpleNamespace
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from mlxserve_local_model_probe import listener_pid, stop_owned, generation_request


class MlxserveProbeTests(unittest.TestCase):
    def test_video_probe_not_mislabeled_as_image_workload(self):
        route, payload = generation_request("video")
        self.assertEqual(route, "/v1/video/generations")
        self.assertEqual((payload["width"], payload["height"], payload["frames"], payload["steps"]), (704, 448, 97, 11))
        self.assertNotIn("size", payload)
        self.assertEqual(generation_request("image")[1]["size"], "512x512")

    def test_exact_listener_pid_set(self):
        with patch("mlxserve_local_model_probe.subprocess.run", return_value=SimpleNamespace(stdout="p42\np43\nf9\n")) as run:
            self.assertEqual(listener_pid(12345), {42, 43})
            self.assertIn("-iTCP:12345", run.call_args.args[0])
            self.assertIn("-sTCP:LISTEN", run.call_args.args[0])

    def test_stop_only_owns_spawned_group_not_process_names(self):
        process = Mock(pid=321)
        process.poll.return_value = None
        with patch("mlxserve_local_model_probe.os.killpg") as signal:
            stop_owned(process)
        self.assertEqual(signal.call_args.args[0], 321)
        process.wait.assert_called_with(timeout=5)

    def test_terminal_server_not_restarted(self):
        process = Mock(pid=321)
        process.poll.return_value = 1
        with patch("mlxserve_local_model_probe.os.killpg") as signal:
            stop_owned(process)
        signal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
