from __future__ import annotations

import json
import contextlib
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from turbocider.cli import _api_json, _parse_request, build_parser, main
from turbocider.registry import ModelRegistry
from turbocider.runtime import TurboCiderRuntime
from turbocider.service import TurboCiderHTTPServer


class CLITests(unittest.TestCase):
    def test_api_url_environment_is_shared_by_daemon_client_commands(self):
        with mock.patch.dict(
            "os.environ", {"TURBOCIDER_API_URL": "http://127.0.0.1:54321"}
        ):
            parser = build_parser()
            generate = parser.parse_args([
                "generate", "--detach", "--prompt", "hello"
            ])
            jobs = parser.parse_args(["jobs"])
        self.assertEqual(generate.api_url, "http://127.0.0.1:54321")
        self.assertEqual(jobs.api_url, "http://127.0.0.1:54321")

    def test_detach_submits_to_daemon_and_survives_cli_return(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            executable = root / "fake_h3.py"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys, time\n"
                "time.sleep(.1)\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
                "out.write_bytes(b'ok')\n"
                "print('phase=generate 1/1', flush=True)\n",
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
                    "tasks": ["video"], "inputs": ["text"],
                    "audio_output": True,
                },
                "config": {
                    "executable_path": str(executable),
                    "model_path": str(model_dir),
                },
                "plans": [{
                    "id": "gpu", "execution": "gpu", "quality": "exact",
                    "profile": "*", "production": True, "priority": 1,
                    "requirements": {
                        "config_paths": ["executable_path", "model_path"]
                    },
                }],
            }))
            runtime = TurboCiderRuntime(
                registry=ModelRegistry([packs]),
                state_directory=root / "state",
                output_directory=root / "outputs",
            )
            server = TurboCiderHTTPServer(("127.0.0.1", 0), runtime)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = "http://127.0.0.1:%d" % server.server_address[1]
            try:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    main([
                        "generate", "--detach", "--api-url", base,
                        "--model", "test-h3", "--prompt", "hello",
                        "--execution", "gpu", "--approximation", "exact",
                    ])
                submitted = json.loads(output.getvalue())
                deadline = time.monotonic() + 5
                while True:
                    current = _api_json(base, "/v1/jobs/" + submitted["id"])
                    if current["state"] in {"succeeded", "failed", "cancelled"}:
                        break
                    if time.monotonic() >= deadline:
                        self.fail("detached job did not finish")
                    time.sleep(.02)
                self.assertEqual(current["state"], "succeeded")
                self.assertTrue(Path(current["output_paths"][0]).is_file())
            finally:
                server.shutdown()
                server.server_close()
                runtime.close()
    def test_h3_multimodal_and_engine_options_are_preserved(self):
        parser = build_parser()
        args = parser.parse_args([
            "generate",
            "--model", "minimax-h3-turbo",
            "--prompt", "hello",
            "--ref-silent-video", "/tmp/silent.mp4",
            "--ref-video-audio", "/tmp/video.mp4", "/tmp/audio.wav",
            "--engine-options", '{"h3":{"super":true}}',
            "--engine-env", "H3_PRIVATE_ANE_QKV_CHECKPOINT=1",
        ])
        request = _parse_request(args)
        self.assertFalse(request.inputs[0].include_embedded_audio)
        self.assertEqual(request.inputs[1].audio_path, str(Path("/tmp/audio.wav").resolve()))
        self.assertTrue(request.engine_options["h3"]["super"])
        self.assertEqual(
            request.engine_options["h3"]["env"]["H3_PRIVATE_ANE_QKV_CHECKPOINT"],
            "1",
        )

    def test_engine_options_can_be_loaded_from_a_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "options.json"
            path.write_text(json.dumps({
                "flux2": {"dynamic_text_length": False, "ane_blocks": [0, 1]}
            }))
            args = build_parser().parse_args([
                "generate",
                "--model", "flux2-klein-4b",
                "--task", "image",
                "--prompt", "hello",
                "--engine-options", "@" + str(path),
            ])
            request = _parse_request(args)
            self.assertFalse(request.engine_options["flux2"]["dynamic_text_length"])
            self.assertEqual(request.engine_options["flux2"]["ane_blocks"], [0, 1])

    def test_engine_options_must_be_an_object(self):
        args = build_parser().parse_args([
            "generate", "--prompt", "hello", "--engine-options", "[]"
        ])
        with self.assertRaisesRegex(ValueError, "JSON object"):
            _parse_request(args)


if __name__ == "__main__":
    unittest.main()
