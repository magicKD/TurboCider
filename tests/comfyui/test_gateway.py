"""Real HTTP framing/routing tests with a deterministic native-service fixture."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bindings/python"))
from turbocider_http import HTTPClient, HTTPError
spec = importlib.util.spec_from_file_location("gateway", ROOT / "tools/comfyui/serve.py")
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


class NativeFixture:
    def __init__(self, root):
        self.root = root
        self.records = {}
        self.submits = 0
        (root / "adapter.safetensors").touch()
    def models(self):
        return {"models": [{"id": "flux", "executor": True, "output": "image"}]}
    def installations(self):
        return {"index": {"installations": [{"id": "model1", "modelID": "flux", "name": "Flux", "path": str(self.root)}],
            "loras": [{"id": "wrong", "modelID": "z-image", "path": str(self.root / "adapter.safetensors")},
                      {"id": "adapter", "modelID": "flux", "path": str(self.root / "adapter.safetensors")}]}}
    def workflows(self):
        return {}
    def service_status(self):
        return {"active_job": ""}
    def plan(self, request):
        return {"executable": True, "request": request}
    def submit(self, model_path, request):
        self.submits += 1
        key = f"job-{self.submits}"
        self.records[key] = {"id": key, "state": "queued", "request": copy.deepcopy(request)}
        return key
    def status(self, job_id):
        return self.records[job_id]
    def jobs(self, **kwargs):
        return {"jobs": list(self.records.values())}
    def cancel(self, job_id):
        self.records[job_id]["state"] = "cancelled"
        return self.records[job_id]


class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.native = NativeFixture(cls.root)
        cls.server = gateway.ThreadingHTTPServer(("127.0.0.1", 0), gateway.Handler)
        cls.server.gateway = gateway.Gateway(cls.native, cls.root / "data", "test-secret")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.client = HTTPClient(cls.url, "test-secret")
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def test_catalog_auth_and_health(self):
        self.assertEqual(self.client.catalog()["installations"][0]["id"], "model1")
        self.assertEqual(self.client.exchange("GET", "/v1/health")["active_job"], "")
        with self.assertRaisesRegex(HTTPError, "401"):
            HTTPClient(self.url).catalog()

    def test_browser_origin_and_dns_rebinding_rejected(self):
        for headers in ({"Origin": "https://example.org"}, {"Host": "evil.example"}):
            request = urllib.request.Request(self.url + "/v1/catalog", headers={"Authorization": "Bearer test-secret", **headers})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 403)

    def test_upload_plan_lora_submit_cancel(self):
        data = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 1, 1) + b"\x08\x06\x00\x00\x00" + b"\x00" * 4
        asset = self.client.upload(data)
        request = {"inputs": [{"kind": "image", "role": "reference", "asset_id": asset}],
                   "loras": [{"id": "adapter", "strength": 0.5, "role": "transformer"}]}
        prepared = self.client.plan("model1", request)["request"]
        self.assertEqual(prepared["loras"][0]["path"], str(self.root / "adapter.safetensors"))
        self.assertTrue(Path(prepared["inputs"][0]["path"]).is_file())
        self.assertIn("asset_id", request["inputs"][0])
        job = self.client.submit("model1", request)
        self.assertEqual(self.client.status(job)["state"], "queued")
        self.assertEqual(self.client.cancel(job)["state"], "cancelled")
        with self.assertRaisesRegex(HTTPError, "only after job success"):
            self.client.artifact(job)

    def test_result_containment_and_restart(self):
        job_id = self.client.submit("model1", {})
        job = self.native.records[job_id]
        output = job["request"]["output"]
        Path(output).write_bytes(b"test-image")
        job.update(state="succeeded", result={"output": output})
        self.assertEqual(self.client.artifact(job_id), b"test-image")
        recovered = gateway.Gateway(self.native, self.root / "data", "test-secret")
        self.assertEqual(recovered.artifact(job_id), Path(output))
        job["result"]["output"] = "/etc/passwd"
        with self.assertRaisesRegex(HTTPError, "does not match"):
            self.client.artifact(job_id)

    def test_invalid_submissions_never_enter_native_queue(self):
        before = self.native.submits
        requests = [{"output": "/tmp/x.png"}, {"schema_version": 2}, {"model": "other"},
                    {"inputs": [{"asset_id": "../bad"}]}, {"inputs": [{"asset_id": "absent"}]},
                    {"loras": [{"id": "wrong"}]}, {"loras": [{"id": "absent"}]}, {"inputs": "wrong"}, {"loras": [1]}]
        for request in requests:
            with self.subTest(request=request), self.assertRaises(HTTPError):
                self.client.submit("model1", request)
        self.assertEqual(before, self.native.submits)
        with self.assertRaisesRegex(HTTPError, "Unknown installation"):
            self.client.submit("absent", {})

    def test_duplicate_keys_and_bad_upload_rejected(self):
        with self.assertRaisesRegex(HTTPError, "Duplicate"):
            self.client.exchange("POST", "/v1/plan", b'{"installation_id":"model1","request":{},"request":{}}')
        with self.assertRaisesRegex(HTTPError, "PNG"):
            self.client.upload(b"not a PNG")
        with self.assertRaises(HTTPError):
            self.client.status("../etc/passwd")

    def test_stale_registration(self):
        original = self.native.installations
        def stale():
            result = original()
            result["index"]["installations"][0]["path"] = str(self.root / "absent")
            return result
        self.native.installations = stale
        try:
            with self.assertRaisesRegex(HTTPError, "no longer exists"):
                self.client.submit("model1", {})
        finally:
            self.native.installations = original


if __name__ == "__main__":
    unittest.main()
