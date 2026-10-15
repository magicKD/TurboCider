"""Isolated local API/library inventory regressions; no weights or inference.

Set TURBOCIDER_TEST_NATIVE_DIR to a matching CLI/helper release directory.
Only copies and registries inside each temporary directory are mutated.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native"))
spec = importlib.util.spec_from_file_location("tc_inventory_client", ROOT / "bindings/python/turbocider_local.py")
api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api
spec.loader.exec_module(api)


class InstallationDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tc-installations-", dir="/private/tmp")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bin = self.root / "bin"; self.bin.mkdir()
        self.cli = self.bin / "turbocider"
        self.helper = self.bin / "turbocider-library"
        shutil.copy2(BIN / "turbocider", self.cli)
        shutil.copy2(BIN / "turbocider-library", self.helper)
        for name in ("libturbocider.dylib", "libmlx.dylib", "libjaccl.dylib"):
            (self.bin / name).symlink_to((BIN / name).resolve())
        self.library = self.root / "library"
        self.settings = self.root / "settings.json"
        self.pid_file = self.root / "helper.pid"
        self.socket = "/private/tmp/tc-installations-" + uuid.uuid4().hex[:8] + ".sock"
        self.client = api.Client(self.socket, timeout=15)
        self.environment = {key: value for key, value in os.environ.items()
                            if key not in ("TURBOCIDER_SERVICE_PARENT_PID", "TURBOCIDER_LIBRARY_PARENT_PID",
                                           "TURBOCIDER_MODEL_LIBRARY", "TURBOCIDER_LIBRARY_SETTINGS")}
        self.environment.update(TURBOCIDER_MODEL_LIBRARY=str(self.library),
                                TURBOCIDER_LIBRARY_SETTINGS=str(self.settings),
                                TURBOCIDER_LIBRARY_PARENT_PID="1", TC_INVENTORY_PID_FILE=str(self.pid_file))
        self.process = None
        self.log = (self.root / "service.log").open("w")
        self.addCleanup(self.log.close)
        self.addCleanup(self.stop_service)

    def start(self):
        self.process = subprocess.Popen([str(self.cli), "serve", self.socket, str(self.root / "jobs")],
                                        stdout=self.log, stderr=self.log, env=self.environment)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.fail((self.root / "service.log").read_text())
            if Path(self.socket).exists():
                try:
                    self.pid = self.client.service_status()["pid"]
                    return
                except (api.TransportError, api.APIError):
                    pass
            time.sleep(.025)
        self.fail("isolated service did not become ready")

    def stop_service(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=5)
        Path(self.socket + ".lock").unlink(missing_ok=True)

    def healthy(self):
        value = self.client.service_status()
        self.assertEqual(value["pid"], self.pid)
        self.assertEqual(value["history_count"], 0)
        self.assertFalse(value["session_open"])
        self.assertIsNone(self.process.poll())

    def index(self):
        return {"schemaVersion": 1, "installations": [{"id": "model-1", "modelID": "qwen-image-2.1",
                 "name": "Metadata-only fixture", "path": str(self.root / "absent-model"), "managed": False,
                 "components": {"text_encoder": {"path": str(self.root / "unopened-component"), "compatibility": "fixture"}},
                 "createdAt": 0}],
                "loras": [{"id": "lora-1", "modelID": "qwen-image-2.1", "name": "Ordinary LoRA",
                           "path": str(self.root / "absent-lora.safetensors")}],
                "anePartitions": [{"id": "ane-1", "modelID": "flux2-klein-4b", "path": str(self.root / "absent-manifest.json"),
                    "kind": "compiled", "inputMode": "fixed", "buckets": [512], "checkpoint": str(self.root / "absent-checkpoint"),
                    "loras": [], "partitionCount": 20}]}

    def write_index(self, value):
        self.library.mkdir(exist_ok=True)
        file = self.library / "library.json"
        file.write_text(json.dumps(value))
        return file

    def replace_helper(self, code):
        self.helper.unlink(missing_ok=True)
        self.helper.write_text("#!" + sys.executable + "\nimport json, os, sys, time\n" + code + "\n")
        self.helper.chmod(0o700)

    def success(self):
        return {"ok": True, "result": {"schema_version": 1, "root": str(self.library), "index": {"schemaVersion": 1, "installations": []},
                                      "scope": "registered_metadata", "files_verified": False}}

    def wait_helper_pid(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.pid_file.exists():
                return int(self.pid_file.read_text())
            time.sleep(.025)
        self.fail("helper did not publish its owned PID")

    def assert_child_reaped(self, pid):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(.025)
        self.fail("owned inventory child was not reaped")

    def test_registered_paths_use_service_configuration_and_never_open_weights(self):
        expected = self.index()
        file = self.write_index(expected); before = file.read_bytes()
        os.mkfifo(self.root / "unopened-component")
        self.start()
        with patch.dict(os.environ, {"TURBOCIDER_MODEL_LIBRARY": str(self.root / "client-only-library"),
                                     "TURBOCIDER_LIBRARY_SETTINGS": str(self.root / "client-only-settings.json")}):
            value = self.client.installations()
        self.assertEqual(Path(value["root"]).resolve(), self.library.resolve())
        self.assertEqual(value["index"], expected)
        self.assertEqual(value["scope"], "registered_metadata")
        self.assertIs(value["files_verified"], False)
        self.assertEqual(file.read_bytes(), before)
        self.assertFalse((self.root / "absent-model").exists())
        capabilities = self.client.capabilities()
        self.assertEqual(capabilities["installation_discovery"]["deadline_seconds"], 10)
        action = next(item for item in capabilities["actions"] if item["name"] == "installations")
        self.assertFalse(action["mutates"])
        with self.assertRaises(api.APIError):
            self.client.rpc("installations", root=str(self.root / "override"))
        self.healthy()

    def test_saved_settings_and_missing_registry_do_not_create_directories(self):
        self.environment.pop("TURBOCIDER_MODEL_LIBRARY")
        self.settings.write_text(json.dumps({"modelRoot": str(self.library)}))
        self.start()
        value = self.client.installations()
        self.assertEqual(Path(value["root"]).resolve(), self.library.resolve())
        self.assertEqual(value["index"]["installations"], [])
        self.assertFalse(self.library.exists())
        self.settings.write_text("{broken")
        with self.assertRaisesRegex(api.APIError, "installations:"):
            self.client.installations()
        self.assertEqual(self.settings.read_text(), "{broken")
        self.healthy()

    def test_invalid_or_nonregular_registry_fails_without_rewriting(self):
        self.library.mkdir(); file = self.library / "library.json"
        self.start()
        duplicate = self.index(); duplicate["loras"] *= 2
        relative = self.index(); relative["installations"][0]["path"] = "relative/model"
        malformed = [b"{broken", json.dumps({"schemaVersion": 2, "installations": []}).encode(),
                     json.dumps(duplicate).encode(), json.dumps(relative).encode(), b" " * (4 * 1024 * 1024 + 1)]
        for payload in malformed:
            with self.subTest(size=len(payload)):
                file.write_bytes(payload)
                with self.assertRaisesRegex(api.APIError, "installations:"):
                    self.client.installations()
                self.assertEqual(file.read_bytes(), payload)
                self.healthy()
        file.unlink(); os.mkfifo(file)
        with self.assertRaisesRegex(api.APIError, "regular file"):
            self.client.installations()
        self.healthy()

    def test_fixed_argv_and_current_service_parent_replace_inherited_owner(self):
        response = self.success()
        code = "assert sys.argv[1:] == ['inventory']\nassert int(os.environ['TURBOCIDER_LIBRARY_PARENT_PID']) == os.getppid()\n"
        code += "assert os.environ['TURBOCIDER_MODEL_LIBRARY'] == " + repr(str(self.library)) + "\n"
        code += "assert os.read(0, 1) == b''\nprint(json.dumps(" + repr(response) + "))"
        self.replace_helper(code); self.start()
        self.assertEqual(self.client.installations(), response["result"])
        self.healthy()

    def test_incompatible_failed_and_oversized_helper_preserve_service(self):
        self.helper.unlink(); self.start()
        with self.assertRaisesRegex(api.APIError, "missing or not executable"):
            self.client.installations()
        self.healthy()
        failure_codes = ["print(json.dumps({'ok': False, 'error': 'inventory unavailable'}))",
                         "print(json.dumps({'ok': True, 'result': {'root': '/old-version'}}))",
                         "print(" + repr(json.dumps(self.success())) + "); sys.exit(7)",
                         "os.write(2, b'\\xff'); sys.exit(1)",
                         "print('{\"ok\":true,\"ok\":false,\"result\":{}}')",
                         "os.write(1, b'x' * (8 * 1024 * 1024 + 1))",
                         "os.write(2, b'x' * (64 * 1024 + 1))"]
        for code in failure_codes:
            with self.subTest(code=code[:80]):
                self.replace_helper(code)
                with self.assertRaises(api.APIError) as error:
                    self.client.installations()
                self.assertTrue(str(error.exception).startswith("installations:"))
                self.healthy()

    def test_large_metadata_is_readable_by_cli_rpc(self):
        value = self.index(); value["installations"][0]["name"] = "x" * (1024 * 1024 + 1024)
        self.write_index(value); self.start()
        request = self.root / "request.json"; request.write_text('{"action":"installations"}')
        result = subprocess.run([str(self.cli), "rpc", self.socket, str(request)], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        response = json.loads(result.stdout)
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["index"], value)
        self.healthy()

    def test_deadline_kills_and_reaps_only_owned_helper_and_cli_can_read_error(self):
        self.replace_helper("open(os.environ['TC_INVENTORY_PID_FILE'], 'w').write(str(os.getpid()))\ntime.sleep(30)")
        self.start()
        sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: sentinel.poll() is None and (sentinel.terminate(), sentinel.wait(timeout=3)))
        request = self.root / "request.json"; request.write_text('{"action":"installations"}')
        started = time.monotonic()
        result = subprocess.run([str(self.cli), "rpc", self.socket, str(request)], capture_output=True, text=True, timeout=15)
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["ok"])
        self.assertIn("10 second deadline", json.loads(result.stdout)["error"])
        self.assertGreaterEqual(elapsed, 9.5); self.assertLess(elapsed, 12)
        self.assert_child_reaped(self.wait_helper_pid())
        self.assertIsNone(sentinel.poll())
        self.healthy()

    def test_service_stop_reaps_inflight_inventory_child(self):
        self.replace_helper("open(os.environ['TC_INVENTORY_PID_FILE'], 'w').write(str(os.getpid()))\ntime.sleep(30)")
        self.start()
        replies = []
        def query():
            try:
                replies.append(self.client.installations())
            except (api.APIError, api.TransportError) as error:
                replies.append(error)
        thread = threading.Thread(target=query, daemon=True); thread.start()
        child = self.wait_helper_pid()
        self.process.terminate(); self.process.wait(timeout=5)
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(replies), 1)
        self.assertIsInstance(replies[0], (api.APIError, api.TransportError))
        self.assert_child_reaped(child)


if __name__ == "__main__":
    unittest.main(verbosity=2)
