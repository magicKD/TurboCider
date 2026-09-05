from __future__ import annotations

import sys
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from turbocider.adapters.base import CommandSpec
from turbocider.runner import ProcessRunner


class PersistentRunnerTests(unittest.TestCase):
    def test_multiword_progress_phase_is_preserved(self):
        self.assertEqual(
            ProcessRunner._parse_progress("text encoder        17/50"),
            ("text encoder", 17 / 50),
        )

    def test_nonpersistent_worker_result_is_returned_as_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = ProcessRunner().run(
                CommandSpec(
                    argv=[
                        sys.executable,
                        "-c",
                        "import json; print('turbocider_result=' + json.dumps({'wall_seconds': 1.25, 'phase_times': {'decode': 0.5}}))",
                    ],
                    cwd=root,
                ),
                root / "result.log",
                threading.Event(),
            )
            self.assertEqual(result.return_code, 0)
            self.assertEqual(result.metrics["wall_seconds"], 1.25)
            self.assertEqual(result.metrics["phase_times"]["decode"], 0.5)

    def test_worker_is_reused_across_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_script = root / "worker.py"
            worker_script.write_text(
                "import json, os, pathlib, sys\n"
                "count = 0\n"
                "for line in sys.stdin:\n"
                "    count += 1\n"
                "    request = json.loads(line)\n"
                "    pathlib.Path(request['output']).write_text('ok')\n"
                "    response = {'ok': True, 'worker_request_index': count, "
                "'metrics': {'worker_pid': os.getpid()}}\n"
                "    print('turbocider_worker_result=' + json.dumps(response), flush=True)\n",
                encoding="utf-8",
            )
            runner = ProcessRunner()
            try:
                results = []
                for index in (1, 2):
                    output = root / ("output-%d.txt" % index)
                    spec = CommandSpec(
                        argv=[sys.executable, str(worker_script)],
                        cwd=root,
                        output_paths=[output],
                        metadata={
                            "persistent_worker": {
                                "key": "test-worker",
                                "argv": [sys.executable, str(worker_script)],
                                "cwd": str(root),
                                "request": {"output": str(output)},
                            }
                        },
                    )
                    result = runner.run(
                        spec,
                        root / ("run-%d.log" % index),
                        threading.Event(),
                    )
                    self.assertEqual(result.return_code, 0)
                    self.assertTrue(output.is_file())
                    results.append(result.metrics)
                self.assertEqual(
                    results[0]["persistent_worker_pid"],
                    results[1]["persistent_worker_pid"],
                )
                self.assertFalse(results[0]["persistent_worker_reused"])
                self.assertTrue(results[1]["persistent_worker_reused"])
            finally:
                runner.close()

    def test_cancelled_worker_is_restarted_for_next_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_script = root / "worker.py"
            worker_script.write_text(
                "import json, os, pathlib, sys, time\n"
                "count = 0\n"
                "for line in sys.stdin:\n"
                "    count += 1\n"
                "    request = json.loads(line)\n"
                "    pathlib.Path(request['pid_file']).write_text(str(os.getpid()))\n"
                "    print('phase=generate 0/1', file=sys.stderr, flush=True)\n"
                "    if request.get('block'):\n"
                "        time.sleep(60)\n"
                "    pathlib.Path(request['output']).write_text('ok')\n"
                "    response = {'ok': True, 'worker_request_index': count}\n"
                "    print('turbocider_worker_result=' + json.dumps(response), flush=True)\n",
                encoding="utf-8",
            )
            runner = ProcessRunner()
            try:
                first_output = root / "first.txt"
                first_pid_file = root / "first.pid"
                cancel = threading.Event()
                first_spec = CommandSpec(
                    argv=[sys.executable, str(worker_script)],
                    cwd=root,
                    output_paths=[first_output],
                    metadata={
                        "persistent_worker": {
                            "key": "restart-worker",
                            "argv": [sys.executable, str(worker_script)],
                            "cwd": str(root),
                            "request": {
                                "output": str(first_output),
                                "pid_file": str(first_pid_file),
                                "block": True,
                            },
                        }
                    },
                )
                result_holder = []
                thread = threading.Thread(
                    target=lambda: result_holder.append(runner.run(
                        first_spec, root / "first.log", cancel
                    ))
                )
                thread.start()
                deadline = time.monotonic() + 5
                while not first_pid_file.is_file():
                    if time.monotonic() >= deadline:
                        self.fail("persistent worker did not start")
                    time.sleep(0.01)
                first_pid = int(first_pid_file.read_text())
                cancel.set()
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(result_holder), 1)
                self.assertNotEqual(result_holder[0].return_code, 0)
                self.assertFalse(first_output.exists())

                second_output = root / "second.txt"
                second_pid_file = root / "second.pid"
                second_spec = CommandSpec(
                    argv=[sys.executable, str(worker_script)],
                    cwd=root,
                    output_paths=[second_output],
                    metadata={
                        "persistent_worker": {
                            "key": "restart-worker",
                            "argv": [sys.executable, str(worker_script)],
                            "cwd": str(root),
                            "request": {
                                "output": str(second_output),
                                "pid_file": str(second_pid_file),
                                "block": False,
                            },
                        }
                    },
                )
                second = runner.run(
                    second_spec, root / "second.log", threading.Event()
                )
                self.assertEqual(second.return_code, 0)
                self.assertTrue(second_output.is_file())
                self.assertNotEqual(first_pid, second.metrics["persistent_worker_pid"])
                self.assertFalse(second.metrics["persistent_worker_reused"])
            finally:
                runner.close()


if __name__ == "__main__":
    unittest.main()
