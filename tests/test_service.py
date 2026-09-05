from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from turbocider.registry import ModelRegistry
from turbocider.runtime import TurboCiderRuntime
from turbocider.service import TurboCiderHTTPServer


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        model_dir = root / "model"
        model_dir.mkdir()
        executable = root / "fake_h3.py"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
            "out.parent.mkdir(parents=True, exist_ok=True)\n"
            "out.write_bytes(b'test-output')\n"
            "print('phase=generate 1/1')\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        packs = root / "packs"
        packs.mkdir()
        (packs / "test.json").write_text(json.dumps({
            "id": "test-h3",
            "name": "Test H3",
            "engine": "h3",
            "capabilities": {
                "tasks": ["video"],
                "inputs": ["text"],
                "audio_output": True,
                "audio_required": True,
            },
            "config": {"executable_path": str(executable), "model_path": str(model_dir)},
            "plans": [{
                "id": "gpu",
                "execution": "gpu",
                "quality": "exact",
                "profile": "*",
                "production": True,
                "priority": 1,
                "requirements": {"config_paths": ["executable_path", "model_path"]},
            }],
        }), encoding="utf-8")
        self.runtime = TurboCiderRuntime(
            registry=ModelRegistry([packs]),
            state_directory=root / "state",
            output_directory=root / "outputs",
        )
        self.token = "test-token"
        self.server = TurboCiderHTTPServer(
            ("127.0.0.1", 0),
            self.runtime,
            token=self.token,
            allowed_input_roots=(root,),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:%d" % self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.runtime.close()
        self.temporary.cleanup()

    def json_request(self, path, method="GET", body=None, authorized=True):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base + path,
            method=method,
            data=data,
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": "Bearer " + self.token}
                    if authorized else {}
                ),
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)

    def test_submit_poll_and_output_validation(self):
        status, job = self.json_request("/v1/jobs", "POST", {
            "model": "test-h3",
            "prompt": "hello",
            "task": "video",
            "output": {"type": "video", "width": 32, "height": 32, "frames": 22, "fps": 24},
            "policy": {"execution": "gpu", "approximation": "exact"},
        })
        self.assertEqual(status, 202)
        deadline = time.monotonic() + 5.0
        while True:
            _, current = self.json_request("/v1/jobs/" + job["id"])
            if current["state"] in {"succeeded", "failed", "cancelled"}:
                break
            if time.monotonic() >= deadline:
                self.fail("job did not reach a terminal state within 5 seconds")
            time.sleep(0.02)
        self.assertEqual(current["state"], "succeeded")
        self.assertTrue(Path(current["output_paths"][0]).is_file())

    def test_job_events_stream_reaches_terminal_state(self):
        _, job = self.json_request("/v1/jobs", "POST", {
            "model": "test-h3",
            "prompt": "stream this job",
            "task": "video",
            "output": {
                "type": "video", "width": 32, "height": 32,
                "frames": 22, "fps": 24,
            },
            "policy": {"execution": "gpu", "approximation": "exact"},
        })
        request = urllib.request.Request(
            self.base + "/v1/jobs/" + job["id"] + "/events",
            headers={"Authorization": "Bearer " + self.token},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(
                response.headers.get_content_type(), "text/event-stream"
            )
            payload = response.read().decode("utf-8")
        events = [
            json.loads(line[len("data: "):])
            for line in payload.splitlines()
            if line.startswith("data: ")
        ]
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(all(event["id"] == job["id"] for event in events))
        self.assertEqual(events[-1]["state"], "succeeded")
        self.assertEqual(events[-1]["progress"], 1.0)

    def test_task_capability_is_enforced(self):
        request = urllib.request.Request(
            self.base + "/v1/jobs",
            method="POST",
            data=json.dumps({"model": "test-h3", "prompt": "hello", "task": "image"}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.token,
            },
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 400)

    def test_required_native_audio_cannot_be_disabled(self):
        request = urllib.request.Request(
            self.base + "/v1/jobs",
            method="POST",
            data=json.dumps({
                "model": "test-h3",
                "prompt": "hello",
                "task": "video",
                "output": {"type": "video", "audio": False},
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.token,
            },
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 400)
        body = json.loads(caught.exception.read().decode("utf-8"))
        self.assertIn("requires audio output", body["error"])

    def test_bearer_token_is_required(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.json_request("/health", authorized=False)
        self.assertEqual(caught.exception.code, 401)

    def test_models_do_not_expose_local_configuration(self):
        _, response = self.json_request("/v1/models")
        descriptor = response["data"][0]
        self.assertNotIn("config", descriptor)
        self.assertNotIn("source", descriptor)
        self.assertIn("plans", descriptor)

    def test_system_status_does_not_expose_local_paths(self):
        _, response = self.json_request("/v1/system")
        rendered = json.dumps(response)
        self.assertNotIn(self.temporary.name, rendered)
        self.assertEqual(response["models"][0]["available"], True)

    def test_api_rejects_output_outside_runtime_directory(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.json_request("/v1/jobs", "POST", {
                "model": "test-h3",
                "prompt": "hello",
                "task": "video",
                "output": {
                    "type": "video",
                    "path": "/tmp/turbocider-forbidden.mp4",
                    "width": 32,
                    "height": 32,
                    "frames": 22,
                    "fps": 24,
                },
                "policy": {"execution": "gpu", "approximation": "exact"},
            })
        self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
