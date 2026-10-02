"""No weights: malformed RPC envelopes must not terminate the local service."""

import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[2]
CLI = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")) / "turbocider"
MAX_BYTES = 1048576


class RPCValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tc-rpc-validation-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.socket = "/private/tmp/tc-rpc-" + uuid.uuid4().hex[:8] + ".sock"
        self.log = (self.root / "service.log").open("w")
        self.addCleanup(self.log.close)
        environment = {key: value for key, value in os.environ.items()
                       if key not in ("TURBOCIDER_SERVICE_PARENT_PID", "TURBOCIDER_LIBRARY_PARENT_PID")
                       and not key.startswith("TURBOCIDER_QWEN21_")}
        environment["TURBOCIDER_MODEL_LIBRARY"] = str(self.root / "isolated-library")
        environment["TURBOCIDER_LIBRARY_SETTINGS"] = str(self.root / "isolated-settings.json")
        self.process = subprocess.Popen(
            [str(CLI), "serve", self.socket, str(self.root / "jobs")],
            stdout=self.log, stderr=self.log,
            env=environment,
        )
        self.addCleanup(self.stop_service)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.fail((self.root / "service.log").read_text())
            if Path(self.socket).exists():
                try:
                    response = self.rpc({"action": "service_status"})
                    if response.get("ok"):
                        self.pid = response["result"]["pid"]
                        return
                except (OSError, ValueError):
                    pass
            time.sleep(.05)
        self.fail("service did not become ready")

    def stop_service(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        Path(self.socket + ".lock").unlink(missing_ok=True)

    def rpc(self, request):
        data = request if isinstance(request, bytes) else json.dumps(request).encode()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5)
            client.connect(self.socket)
            client.sendall(data + b"\n")
            with client.makefile("rb") as response:
                return json.loads(response.readline())

    def assert_rejected_without_exit(self, request):
        response = self.rpc(request)
        self.assertIs(response["ok"], False, request)
        self.assertIsInstance(response["error"], str)
        self.assertTrue(response["error"])
        healthy = self.rpc({"action": "service_status"})
        self.assertTrue(healthy["ok"])
        self.assertEqual(healthy["result"]["pid"], self.pid)
        self.assertEqual(healthy["result"]["history_count"], 0)
        self.assertIsNone(self.process.poll())

    def test_shared_workflow_discovery_composition_and_separate_plan(self):
        capabilities = self.rpc({"action": "capabilities"})["result"]
        self.assertEqual(capabilities["workflow"]["catalog_action"], "workflows")
        self.assertEqual(capabilities["workflow"]["construction_action"], "workflow_request")
        catalog = self.rpc({"action": "workflows"})["result"]
        self.assertEqual(catalog["schema_version"], 1)
        self.assertEqual({w["id"] for w in catalog["workflows"]},
                         {"playground." + name for name in ("outfit", "identity", "face", "outpaint", "transparent")})
        request = {"model": "qwen-image-2.1", "width": 512, "height": 512, "steps": 25,
                   "execution": "gpu", "residency": "component_staged", "output": str(self.root / "result.png")}
        value = {"workflow_id": "playground.face", "role_paths": {"target": str(self.root / "absent-target.png"),
                 "person": str(self.root / "absent-person.png")}, "request": request, "instruction": "Keep the target scene."}
        built = self.rpc({"action": "workflow_request", "input": value})
        self.assertTrue(built["ok"], built)
        composed = built["result"]
        self.assertEqual([r["role"] for r in composed["roles"]], ["person", "target"])
        self.assertEqual(composed["request"]["steps"], 25)
        planned = self.rpc({"action": "plan", "request": composed["request"]})
        self.assertTrue(planned["ok"], planned)
        self.assertEqual(planned["result"]["operation"], "image.edit")
        self.assertEqual(self.rpc({"action": "service_status"})["result"]["history_count"], 0)
        self.assertFalse(Path(request["output"]).exists())
        self.assertFalse((self.root / "absent-person.png").exists())
        # Composition is explicitly not execution admission.
        value["request"]["width"] = 513
        invalid = self.rpc({"action": "workflow_request", "input": value})
        self.assertTrue(invalid["ok"], invalid)
        self.assert_rejected_without_exit({"action": "plan", "request": invalid["result"]["request"]})
        for malformed in [
            {"action": "workflows", "unexpected": True},
            {"action": "workflow_request", "request": {}},
            {"action": "workflow_request", "input": []},
            {"action": "workflow_request", "input": {**value, "role_paths": {"unknown": "/file.png"}}},
            b'{"action":"workflow_request","input":{"workflow_id":"playground.transparent","request":{},"request":{}}}',
        ]:
            self.assert_rejected_without_exit(malformed)

    def test_qwen_reference_encoding_discovery_and_plan_opt_in(self):
        models = self.rpc({"action": "models"})
        self.assertTrue(models["ok"])
        qwen = next(model for model in models["result"]["models"]
                    if model["id"] == "qwen-image-2.1")
        self.assertEqual(qwen["reference_encoding"]["default"], 1024)
        self.assertEqual(qwen["reference_encoding"]["schema_v2_field"],
                         "parameters.qwen21_reference_size")
        self.assertEqual(qwen["reference_encoding"]["base_approximation_sizes"], [256, 512])
        self.assertIn("No LoRA", qwen["reference_encoding"]["base_constraints"])
        self.assertIn("six steps", qwen["reference_encoding"]["viggle_r128_gpu_edit_constraints"])
        self.assertEqual(qwen["reference_encoding"]["ordinary_lora_reference_size"], 1024)
        adapter = {"path": "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
                   "role": "transformer", "strength": 1}
        request = {
            "schema_version": 2, "model": "qwen-image-2.1", "operation": "image.edit",
            "inputs": [{"kind": "text", "role": "prompt", "text": "A ceramic teapot"},
                       {"kind": "image", "role": "reference", "path": "not-loaded.png"}],
            "outputs": [{"kind": "image", "path": "edit.png", "width": 512,
                         "height": 512, "frames": 1, "audio": False}],
            "sampling": {"seed": 42, "steps": 6},
            "execution": {"policy": "gpu", "residency": "component_staged",
                          "hybrid_mlp_mode": "auto", "allow_approximation": True,
                          "qwen21_dit_cache": "off"},
            "parameters": {"qwen21_reference_size": 512},
            "lora_strategy": "inference_time", "loras": [adapter],
        }
        planned = self.rpc({"action": "plan", "request": request})
        self.assertTrue(planned["ok"], planned)
        self.assertEqual(planned["result"]["qwen21_reference_size"], 512)
        self.assertIn("qwen21_viggle_r128_reference_resize_512",
                      planned["result"]["algorithm_approximations"])
        # Discovery must describe base resizing as well as the stricter r128
        # shortcut; validate the published sizes against real request admission.
        for size in qwen["reference_encoding"]["base_approximation_sizes"]:
            for count in (1, 3):
                base = {**request, "sampling": {"seed": 42, "steps": 25},
                        "parameters": {"qwen21_reference_size": size}, "loras": [],
                        "lora_strategy": "auto", "inputs": [request["inputs"][0]] + [request["inputs"][1]] * count}
                with self.subTest(base_size=size, references=count):
                    admitted = self.rpc({"action": "plan", "request": base})
                    self.assertTrue(admitted["ok"], admitted)
                    self.assertEqual(admitted["result"]["qwen21_reference_size"], size)
        for changes in (
            {"parameters": {"qwen21_reference_size": 256}},
            {"qwen21_reference_size": 512},
            {"execution": {**request["execution"], "allow_approximation": False}},
            {"execution": {**request["execution"], "qwen21_dit_cache": "balanced"}},
            {"execution": {**request["execution"], "policy": "gpu_ane", "ane_manifest": "not-loaded.json"}},
            {"loras": [{**adapter, "path": adapter["path"].replace("r128", "r256")}]},
            {"loras": [{**adapter, "path": "ordinary.safetensors"}], "sampling": {"steps": 25}},
        ):
            with self.subTest(changes=changes):
                self.assert_rejected_without_exit({"action": "plan", "request": {**request, **changes}})

    def test_offline_cli_capabilities_matches_live_protocol(self):
        result = subprocess.run([str(CLI), "capabilities"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result.stderr)
        self.assertEqual(json.loads(result.stdout), self.rpc({"action": "capabilities"})["result"])

    def test_strict_envelope_types_ranges_and_duplicate_keys(self):
        for action in (None, True, 1, [], {}, "", "unknown", "jobs\0ignored"):
            with self.subTest(action=action):
                self.assert_rejected_without_exit({"action": action})
        for action in ("models", "doctor", "service_status", "capabilities", "installations", "jobs"):
            with self.subTest(extra_field=action):
                self.assert_rejected_without_exit({"action": action, "unexpected": 0})
        for value in (None, True, False, "0", [], {}, -.5, .5, -1, 2147483648, 1e100):
            with self.subTest(offset=value):
                self.assert_rejected_without_exit({"action": "jobs", "offset": value})
        for value in (None, True, False, "20", [], {}, 0, -1, .5, 101, 1e100):
            with self.subTest(limit=value):
                self.assert_rejected_without_exit({"action": "jobs", "limit": value})
        for action in ("status", "cancel"):
            for value in (None, True, 1, [], {}, "", "id\0ignored"):
                with self.subTest(action=action, id=value):
                    self.assert_rejected_without_exit({"action": action, "id": value})
            self.assert_rejected_without_exit({"action": action})
        for action in ("plan", "submit"):
            for value in (None, True, 1, "request", []):
                with self.subTest(action=action, request=value):
                    self.assert_rejected_without_exit({"action": action, "request": value})
            self.assert_rejected_without_exit({"action": action})
        for path in (None, True, 1, [], {}, "", "path\0ignored"):
            with self.subTest(model_path=path):
                self.assert_rejected_without_exit({"action": "submit", "request": {}, "model_path": path})
        duplicates = (
            b'{"action":"jobs","action":"service_status"}',
            b'{"action":"jobs","act\\u0069on":"service_status"}',
            b'{"action":"jobs","offset":0,"offset":1}',
            b'{"action":"plan","request":{"prompt":"first","prompt":"second"}}',
            b'{"action":"plan","request":{"inputs":[{"role":"reference","role":"init"}]}}',
        )
        for request in duplicates:
            with self.subTest(duplicate=request):
                self.assert_rejected_without_exit(request)
        for request in (
            b'{"action":"jobs","offset":1.0000000000000001}',
            b'{"action":"jobs","offset":1e-999999999}',
            b'{"action":"jobs","off\\u0073et":0.999999999999999999}',
            b'{"action":"jobs","limit":99.0000000000000000001}',
        ):
            with self.subTest(exact_fraction=request):
                self.assert_rejected_without_exit(request)
        for request in (b'{"action":', b'[]', b'null', b'42'):
            with self.subTest(invalid_json=request):
                self.assert_rejected_without_exit(request)
        # Extreme exponents must not make Foundation's failure path accept
        # invalid JSON syntax, another action or non-page envelope fields.
        for request in (
            b'{"action":"jobs","offset":00e-999999999}',
            b'{"action":"jobs","offset":+0e999999999}',
            b'{"action":"jobs","offset":.0e999999999}',
            b'{"action":"jobs","offset":0.e999999999}',
            b'{"action":"jobs","offset":0e+}',
            b'{"action":"jobs","offset":0e-999999999,}',
            b'{"action":"jobs","offset":0e-999999999} trailing',
            b'{"action":"models","offset":0e-999999999}',
            b'{"action":"jobs","offset":0e-999999999,"root":"/override"}',
            b'{"action":"jobs","offset":0e-999999999,"off\\u0073et":0}',
            b'{"action":"jobs","offset":0e-999999999,"limit":0e999999999}',
            b'{"action":"jobs","offset":0e-999999999,"limit":99.0000000000000000001}',
            b'{"action":"jobs","offset":0e-999999999,"limit":true}',
        ):
            with self.subTest(extreme_invalid=request):
                self.assert_rejected_without_exit(request)
        for offset, limit in ((0, 1), (2147483647, 100), (0.0, 20.0)):
            with self.subTest(valid_page=(offset, limit)):
                response = self.rpc({"action": "jobs", "offset": offset, "limit": limit})
                self.assertTrue(response["ok"])
                self.assertEqual(response["result"]["jobs"], [])
        self.assertTrue(self.rpc({"action": "jobs"})["ok"])
        for request in (
            b'{"action":"jobs","offset":1e0,"limit":1e2}',
            b'{"action":"jobs","off\\u0073et":100e-2,"limit":20.000}',
            b'{"action":"jobs","offset":0e-999999999}',
            b'{"action":"jobs","offset":0e999999999}',
            b'{"action":"jobs","offset":-0.000e+999999999}',
            b'{"off\\u0073et":-0e-999999999,"action":"j\\u006fbs","limit":100}',
        ):
            with self.subTest(exact_integral=request):
                self.assertTrue(self.rpc(request)["ok"])
        capabilities = self.rpc({"action": "capabilities"})
        self.assertTrue(capabilities["ok"])
        self.assertEqual(capabilities["result"]["transport"]["max_request_bytes"], MAX_BYTES)
        self.assertTrue(self.rpc({"action": "models"})["ok"])
        self.assertTrue(self.rpc({"action": "plan", "request": {
            "model": "flux2-klein-4b", "prompt": "A fox", "steps": 4,
            "width": 512, "height": 512, "frames": 1, "audio": False,
        }})["ok"])

    def test_exact_wire_limit_and_cli_preparse(self):
        base = b'{"action":"service_status"}'
        self.assertTrue(self.rpc(base + b" " * (MAX_BYTES - len(base)))["ok"])
        self.assert_rejected_without_exit(base + b" " * (MAX_BYTES + 1 - len(base)))
        request_file = self.root / "rpc.json"
        for raw in (b'{"action":"jobs","action":"service_status"}',
                    b'{"action":"plan","request":{"steps":4,"steps":8}}',
                    b'{"action":"jobs","offset":{}}',
                    b'{"action":"jobs","offset":00e-999999999}',
                    b'{"action":"jobs","offset":0e-999999999,}',
                    b'{"action":"models","offset":0e-999999999}'):
            with self.subTest(cli_payload=raw):
                request_file.write_bytes(raw)
                rejected = subprocess.run([str(CLI), "rpc", self.socket, str(request_file)],
                                          capture_output=True, text=True, timeout=5)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertTrue(rejected.stderr)
                self.assertTrue(self.rpc({"action": "service_status"})["ok"])
        request_file.write_text(json.dumps({"action": "jobs"}, indent=2))
        accepted = subprocess.run([str(CLI), "rpc", self.socket, str(request_file)],
                                  capture_output=True, text=True, timeout=5)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertTrue(json.loads(accepted.stdout)["ok"])
        for raw in (b'{"action":"jobs","offset":0e-999999999}',
                    b'{"action":"jobs","offset":0e999999999}',
                    b'{"action":"jobs","offset":-0.0e+999999999}'):
            request_file.write_bytes(raw)
            accepted = subprocess.run([str(CLI), "rpc", self.socket, str(request_file)],
                                      capture_output=True, text=True, timeout=5)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertTrue(json.loads(accepted.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
