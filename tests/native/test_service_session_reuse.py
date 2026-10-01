"""Real local-service session receipts with test-only CPU engines; no weights.

The temporary executable links production service.mm and libturbocider. Only
engine create/free/generate/cancel are replaced, so admission, plans, RPC,
durable queue, identity changes and disposable-worker ownership remain real.
Set TURBOCIDER_TEST_NATIVE_DIR to the matching native library directory.
"""

import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()
CLT = Path("/Library/Developer/CommandLineTools")


@unittest.skipUnless(sys.platform == "darwin", "requires the macOS native planner")
class ServiceSessionReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="tc-session-reuse-build-", dir="/private/tmp")
        cls.addClassCleanup(cls.build.cleanup)
        cls.binary = Path(cls.build.name) / "cpu-service-fixture"
        cls.build_environment = {**os.environ, "DEVELOPER_DIR": str(CLT)}
        if not (NATIVE / "libturbocider.dylib").is_file():
            raise AssertionError("build the matching native library first: " + str(NATIVE))
        sdk_path = os.environ.get("SDKROOT")
        if not sdk_path:
            sdk = subprocess.run(
                ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"],
                env=cls.build_environment, capture_output=True, text=True, timeout=10,
            )
            if sdk.returncode:
                raise AssertionError(sdk.stderr + sdk.stdout)
            sdk_path = sdk.stdout.strip()
        module_cache = Path(cls.build.name) / "module-cache"
        module_cache.mkdir()
        command = [
            str(CLT / "usr/bin/clang++"), "-std=c++20", "-O1", "-fobjc-arc",
            "-Wall", "-Wextra", "-Wno-unused-parameter",
            "-isysroot", sdk_path,
            "-mmacosx-version-min=" + platform.mac_ver()[0],
            "-fmodules-cache-path=" + str(module_cache),
            "-I", str(ROOT / "bindings/c/include"),
            str(ROOT / "tests/native/service_session_reuse_fixture.mm"),
            str(ROOT / "services/turbociderd/service.mm"),
            str(ROOT / "native/core/json_keys.cpp"),
            "-L" + str(NATIVE), "-lturbocider", "-framework", "Foundation",
            "-Wl,-rpath," + str(NATIVE), "-o", str(cls.binary),
        ]
        compiled = subprocess.run(command, env=cls.build_environment, cwd=ROOT,
                                  capture_output=True, text=True, timeout=120)
        if compiled.returncode:
            raise AssertionError(compiled.stderr + compiled.stdout)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tc-session-reuse-", dir="/private/tmp")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.socket = "/private/tmp/tc-reuse-" + uuid.uuid4().hex[:12] + ".sock"
        self.journal = self.root / "lifecycle.txt"
        self.log_path = self.root / "service.log"
        self.log = self.log_path.open("w")
        self.addCleanup(self.log.close)
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("TURBOCIDER_", "DYLD_"))}
        environment.update(
            TURBOCIDER_MODEL_LIBRARY=str(self.root / "library"),
            TURBOCIDER_LIBRARY_SETTINGS=str(self.root / "settings.json"),
            TURBOCIDER_SERVICE_PARENT_PID=str(os.getpid()),
            TC_SERVICE_REUSE_JOURNAL=str(self.journal),
            TC_SERVICE_REUSE_HOLD_RELEASE=str(self.root / "hold-release"),
        )
        self.environment = environment
        self.addCleanup(self.stop_service)
        self.start_service()

    def start_service(self):
        # A separate process group lets timeout cleanup reach only this fixture
        # and its own disposable children; it cannot terminate another service.
        self.process = subprocess.Popen(
            [str(self.binary), "serve", self.socket, str(self.root / "jobs")],
            stdout=self.log, stderr=self.log, env=self.environment, start_new_session=True,
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.fail(self.log_path.read_text())
            if Path(self.socket).exists():
                try:
                    self.pid = self.rpc({"action": "service_status"})["pid"]
                    return
                except (OSError, ValueError):
                    pass
            time.sleep(.025)
        self.fail("CPU fixture service did not become ready: " + self.log_path.read_text())

    def stop_service(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # The group was created by this Popen; no unrelated PID or
                # socket owner is discovered or signalled during cleanup.
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
        Path(self.socket + ".lock").unlink(missing_ok=True)

    def rpc_response(self, request):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(3)
            client.connect(self.socket)
            client.sendall(json.dumps(request).encode() + b"\n")
            with client.makefile("rb") as response:
                raw = response.readline(1048577)
        self.assertTrue(raw.endswith(b"\n"), raw)
        return json.loads(raw)

    def rpc(self, request):
        value = self.rpc_response(request)
        self.assertTrue(value["ok"], value)
        return value["result"]

    def request(self, model="qwen-image-2.1", video=False):
        return {
            "model": model,
            "operation": "video.generate" if video else "image.generate",
            "prompt": "CPU lifecycle fixture; no inference",
            "width": 128 if video else 512,
            "height": 128 if video else 512,
            "steps": 11 if video else (25 if model == "qwen-image-2.1" else 4),
            "seed": 42,
            "execution": "gpu",
            "residency": "component_staged",
            "frames": 9 if video else 1,
            "fps": 24,
            "audio": False,
            "output": str(self.root / (uuid.uuid4().hex + (".mp4" if video else ".png"))),
        }

    def submit_and_wait(self, model_path, request, expected_state="succeeded"):
        # The real library validates the exact request that submit will admit.
        plan = self.rpc({"action": "plan", "request": request})
        self.assertEqual(plan["model"], request["model"])
        submitted = self.rpc({"action": "submit", "model_path": str(model_path), "request": request})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = self.rpc({"action": "status", "id": submitted["id"]})
            if job["state"] in ("succeeded", "failed", "cancelled", "interrupted"):
                self.assertEqual(job["state"], expected_state, job)
                if expected_state == "succeeded":
                    self.assertTrue(job["result"]["fixture_cpu_no_inference"])
                self.assertFalse(Path(request["output"]).exists())
                self.assertFalse(Path(model_path).exists())
                return job
            time.sleep(.01)
        self.fail("CPU fixture job did not complete: " + self.log_path.read_text())

    def records(self, event):
        return [int(line.split()[1]) for line in self.journal.read_text().splitlines()
                if line.split()[0] == event]

    def assert_native_receipt(self, job, reused, model, path):
        receipt = job["result"]
        self.assertIs(receipt["service_session_reused"], reused)
        self.assertEqual(receipt["service_execution_path"], "native_session")
        self.assertEqual(receipt["fixture_engine_model"], model)
        self.assertEqual(receipt["fixture_model_path"], str(path))
        self.assertEqual(job["progress"]["phase"], "fixture_cpu")
        return receipt["creation_id"]

    def assert_clean_shutdown(self):
        self.stop_service()
        self.assertEqual(self.process.returncode, 0, self.log_path.read_text())
        self.assertFalse(Path(self.socket).exists())
        # This includes the final idle engine, freed by the real destructor.
        self.assertEqual(self.records("free"), self.records("create"))

    def test_native_session_identity_matches_observed_engine_reuse(self):
        model = "qwen-image-2.1"
        path_a, path_b = self.root / "absent-model-a", self.root / "absent-model-b"
        first = self.submit_and_wait(path_a, self.request())
        first_id = self.assert_native_receipt(first, False, model, path_a)
        second_request = self.request(); second_request["seed"] = 43
        second = self.submit_and_wait(path_a, second_request)
        self.assertEqual(self.assert_native_receipt(second, True, model, path_a), first_id)
        self.assertEqual(self.records("create"), [first_id])
        self.assertEqual(self.records("free"), [])

        switched = self.submit_and_wait(path_b, self.request())
        switched_id = self.assert_native_receipt(switched, False, model, path_b)
        self.assertGreater(switched_id, first_id)
        self.assertEqual(self.records("free"), [first_id])
        repeated = self.submit_and_wait(path_b, self.request())
        self.assertEqual(self.assert_native_receipt(repeated, True, model, path_b), switched_id)

        # Identity is both model and path: a new model at the same fake path
        # creates another engine, then reuses that new engine on repetition.
        flux = "flux2-klein-4b"
        changed_model = self.submit_and_wait(path_b, self.request(flux))
        changed_id = self.assert_native_receipt(changed_model, False, flux, path_b)
        self.assertGreater(changed_id, switched_id)
        repeated_model = self.submit_and_wait(path_b, self.request(flux))
        self.assertEqual(self.assert_native_receipt(repeated_model, True, flux, path_b), changed_id)
        self.assertEqual(self.records("create"), [first_id, switched_id, changed_id])
        self.assertEqual(self.records("generate"),
                         [first_id, first_id, switched_id, switched_id, changed_id, changed_id])
        self.assert_clean_shutdown()

    def test_failed_identity_switch_recreates_previous_native_session(self):
        path_a = self.root / "absent-model-a"
        path_b = self.root / "absent-create-fails"
        first = self.submit_and_wait(path_a, self.request())
        first_id = self.assert_native_receipt(first, False, "qwen-image-2.1", path_a)

        failed = self.submit_and_wait(path_b, self.request(), expected_state="failed")
        self.assertIn("controlled CPU fixture creation failure", failed["error"])
        self.assertEqual(self.records("create_failed"), [0])
        self.assertEqual(self.records("free"), [first_id])
        self.assertEqual(self.records("create"), [first_id])
        self.assertEqual(self.records("generate"), [first_id])
        self.assertFalse(self.rpc({"action": "service_status"})["session_open"])

        # The old identity cannot imply an open engine after B creation failed.
        # A must create a new handle instead of generating through nullptr.
        recovered = self.submit_and_wait(path_a, self.request())
        recovered_id = self.assert_native_receipt(recovered, False, "qwen-image-2.1", path_a)
        self.assertGreater(recovered_id, first_id)
        repeated = self.submit_and_wait(path_a, self.request())
        self.assertEqual(self.assert_native_receipt(repeated, True, "qwen-image-2.1", path_a), recovered_id)
        self.assertEqual(self.records("create"), [first_id, recovered_id])
        self.assertEqual(self.records("generate"), [first_id, recovered_id, recovered_id])
        self.assert_clean_shutdown()

    def test_disposable_worker_reports_no_reuse_and_releases_native_engine(self):
        path = self.root / "absent-model"
        native = self.submit_and_wait(path, self.request())
        first_id = self.assert_native_receipt(native, False, "qwen-image-2.1", path)
        external = self.submit_and_wait(path, self.request("ltx-2.5-distilled", video=True))
        receipt = external["result"]
        self.assertEqual(receipt["service_execution_path"], "disposable_worker")
        self.assertIs(receipt["service_session_reused"], False)
        self.assertTrue(receipt["fixture_external_worker"])
        self.assertNotEqual(receipt["fixture_worker_pid"], self.pid)
        self.assertEqual(self.records("external"), [0])
        self.assertEqual(self.records("free"), [first_id])
        self.assertEqual(self.records("create"), [first_id])
        self.assertFalse(self.rpc({"action": "service_status"})["session_open"])

        after = self.submit_and_wait(path, self.request())
        after_id = self.assert_native_receipt(after, False, "qwen-image-2.1", path)
        self.assertGreater(after_id, first_id)
        warm = self.submit_and_wait(path, self.request())
        self.assertEqual(self.assert_native_receipt(warm, True, "qwen-image-2.1", path), after_id)
        self.assertEqual(self.records("generate"), [first_id, after_id, after_id])
        self.assert_clean_shutdown()

    def test_cancelled_queue_releases_capacity_and_survives_shutdown(self):
        model_path = self.root / "absent-model"
        held_request = self.request()
        held_request["prompt"] = "CPU lifecycle fixture hold"
        self.rpc({"action": "plan", "request": held_request})
        active = self.rpc({"action": "submit", "model_path": str(model_path), "request": held_request})["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = self.rpc({"action": "status", "id": active})["state"]
            if state == "running" and self.journal.exists() and self.records("holding"):
                break
            time.sleep(.01)
        else:
            self.fail("CPU fixture did not hold the active worker: " + self.log_path.read_text())

        queued = []
        for _ in range(32):
            job = self.rpc({"action": "submit", "model_path": str(model_path), "request": self.request()})
            self.assertEqual(job["state"], "queued")
            queued.append(job["id"])
        full = self.rpc_response({"action": "submit", "model_path": str(model_path), "request": self.request()})
        self.assertIs(full["ok"], False)
        self.assertIn("queue is full", full["error"])

        for job_id in queued:
            cancelled = self.rpc({"action": "cancel", "id": job_id})
            self.assertEqual(cancelled["state"], "cancelled")
        replacement = self.rpc({"action": "submit", "model_path": str(model_path), "request": self.request()})
        self.assertEqual(replacement["state"], "queued")
        self.assertEqual(self.rpc({"action": "status", "id": active})["state"], "running")

        # Stop before the held worker could consume any cancelled queue entry.
        # The active fixture exits via cancellation; the remaining queued job
        # becomes interrupted while successful queued cancels remain terminal.
        self.stop_service()
        self.assertEqual(self.process.returncode, 0, self.log_path.read_text())
        for job_id in queued:
            persisted = json.loads((self.root / "jobs" / (job_id + ".json")).read_text())
            self.assertEqual(persisted["state"], "cancelled")
        persisted = json.loads((self.root / "jobs" / (replacement["id"] + ".json")).read_text())
        self.assertEqual(persisted["state"], "interrupted")
        self.assertEqual(self.records("create"), self.records("free"))
        self.assertEqual(len(self.records("generate")), 1)

        self.start_service()
        for job_id in queued:
            self.assertEqual(self.rpc({"action": "status", "id": job_id})["state"], "cancelled")
        self.assertEqual(self.rpc({"action": "status", "id": replacement["id"]})["state"], "interrupted")
        self.assertFalse(self.rpc({"action": "service_status"})["session_open"])
        self.assert_clean_shutdown()


if __name__ == "__main__":
    unittest.main()
