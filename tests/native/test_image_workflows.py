"""Shared C ABI/CLI image workflow composition; no engine, image reads or GPU."""
import copy
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native"))


@unittest.skipUnless(sys.platform == "darwin", "requires the macOS native library")
class ImageWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = ctypes.CDLL(str(NATIVE / "libturbocider.dylib"))
        cls.library.tc_workflows_json.restype = ctypes.c_void_p
        cls.library.tc_workflow_request_json.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        cls.library.tc_workflow_request_json.restype = ctypes.c_int
        cls.library.tc_plan_json.argtypes = cls.library.tc_workflow_request_json.argtypes
        cls.library.tc_plan_json.restype = ctypes.c_int
        cls.library.tc_string_free.argtypes = [ctypes.c_void_p]
        cls.library.tc_string_free.restype = None

    def consume(self, value):
        if not value:
            return ""
        try:
            return ctypes.string_at(value).decode("utf-8")
        finally:
            self.library.tc_string_free(value)

    def call(self, value, *, expected_error=None):
        output, error = ctypes.c_void_p(), ctypes.c_void_p()
        data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
        status = self.library.tc_workflow_request_json(data, ctypes.byref(output), ctypes.byref(error))
        result, message = self.consume(output.value), self.consume(error.value)
        if expected_error is not None:
            self.assertEqual(status, 1, result)
            self.assertFalse(result)
            self.assertIn(expected_error, message)
            return None
        self.assertEqual(status, 0, message)
        self.assertFalse(message)
        return json.loads(result)

    def input(self, workflow="identity", roles=None, request=None, **fields):
        return {"workflow_id": "playground." + workflow,
                "role_paths": roles if roles is not None else {"person": "/no-such-reference/person.png"},
                "request": request if request is not None else {"model": "qwen-image-2.1"}, **fields}

    def test_catalog_matches_the_single_json_source(self):
        catalog = json.loads(self.consume(self.library.tc_workflows_json()))
        expected = json.loads((ROOT / "native/workflows/image_workflows.json").read_text())
        self.assertEqual(catalog, expected)
        self.assertEqual(catalog["schema_version"], 1)
        self.assertEqual(len(catalog["workflows"]), 5)
        self.assertFalse(catalog["input_schema"]["additionalProperties"])
        ids = {workflow["id"] for workflow in catalog["workflows"]}
        self.assertEqual(ids, set(catalog["input_schema"]["properties"]["workflow_id"]["enum"]))

    def test_role_order_is_declared_not_dictionary_insertion_order(self):
        for workflow, roles in [
            ("outfit", {"clothing": "/clothes.png", "person": "/person.png"}),
            ("identity", {"scene": "/scene.png", "person": "/person.png"}),
            ("face", {"target": "/target.png", "person": "/person.png"}),
        ]:
            with self.subTest(workflow=workflow):
                result = self.call(self.input(workflow, roles))
                self.assertEqual(result["operation"], "image.edit")
                self.assertEqual(result["roles"][0], {"role": "person", "image_number": 1, "path": "/person.png"})
                self.assertEqual(result["roles"][1]["image_number"], 2)
                self.assertEqual([r["path"] for r in result["roles"]], [r["path"] for r in result["request"]["inputs"]])
                self.assertIn("<image1>", result["prompt"])
                self.assertIn("<image2>", result["prompt"])
        identity = self.call(self.input())
        self.assertEqual(len(identity["roles"]), 1)
        self.assertNotIn("<image2>", identity["prompt"])

    def test_transparent_switches_between_generation_and_extraction(self):
        generated = self.call(self.input("transparent", {}))
        self.assertEqual(generated["operation"], "image.generate")
        self.assertEqual(generated["request"]["inputs"], [])
        self.assertIn("卡通狐狸", generated["prompt"])
        extracted = self.call(self.input("transparent", {"source": "/subject.png"}))
        self.assertEqual(extracted["operation"], "image.edit")
        self.assertEqual(extracted["roles"][0]["role"], "source")
        self.assertIn("保留原图主体", extracted["prompt"])
        self.assertNotIn("卡通狐狸", extracted["prompt"])

    def test_composition_replaces_only_owned_fields_and_preserves_normal_lora(self):
        request = {"schema_version": 1, "model": "qwen-image-2.1", "operation": "image.generate",
                   "prompt": "old prompt", "inputs": [{"kind": "image", "path": "/old.png"}],
                   "width": 512, "height": 768, "steps": 25, "seed": 99, "output": "/result.png",
                   "execution": "gpu", "residency": "component_staged", "qwen21_dit_cache": "off",
                   "qwen21_reference_size": 1024, "lora_strategy": "inference_time",
                   "loras": [{"path": "/ordinary.safetensors", "role": "transformer", "strength": .7}]}
        value = self.input(request=request)
        before = copy.deepcopy(value)
        result = self.call(value)
        self.assertEqual(value, before)
        for key, item in request.items():
            if key not in ("prompt", "inputs", "operation"):
                self.assertEqual(result["request"][key], item, key)
        self.assertEqual(result["request"]["steps"], 25)
        self.assertEqual(result["request"]["inputs"][0]["path"], "/no-such-reference/person.png")
        self.assertEqual(result["prompt"], result["request"]["prompt"])

    def test_schema_two_uses_one_text_prompt_and_preserves_settings(self):
        request = {"schema_version": 2, "model": "qwen-image-2.1", "operation": "image.edit",
                   "inputs": [{"kind": "text", "role": "prompt", "text": "old"}],
                   "outputs": [{"kind": "image", "path": "/result.png", "width": 512, "height": 512}],
                   "sampling": {"steps": 40, "seed": 42}, "execution": {"policy": "gpu", "qwen21_dit_cache": "balanced"},
                   "parameters": {"qwen21_reference_size": 1024}}
        result = self.call(self.input("face", {"target": "/target.png", "person": "/person.png"}, request))
        native = result["request"]
        self.assertNotIn("prompt", native)
        self.assertEqual(native["inputs"][0], {"kind": "text", "role": "prompt", "text": result["prompt"]})
        self.assertEqual([r["path"] for r in native["inputs"][1:]], ["/person.png", "/target.png"])
        for key in ("sampling", "execution", "parameters", "outputs"):
            self.assertEqual(native[key], request[key])

    def test_instruction_is_appended_verbatim(self):
        instruction = "  保留文字\nSecond line\t  "
        result = self.call(self.input(instruction=instruction))
        self.assertTrue(result["prompt"].endswith("Additional instruction:\n" + instruction))
        self.assertTrue(self.call(self.input(instruction=""))["prompt"].endswith("Additional instruction:\n"))

    def test_expansion_is_prompt_only_and_has_a_bounded_enum(self):
        for expansion in (1.25, 1.5, 2):
            result = self.call(self.input("outpaint", {"source": "/source.png"},
                                          {"width": 512, "height": 512}, expansion=expansion))
            self.assertIn(str(expansion) + " times", result["prompt"])
            self.assertEqual((result["request"]["width"], result["request"]["height"]), (512, 512))
        default = self.call(self.input("outpaint", {"source": "/source.png"}))
        self.assertIn("1.5 times", default["prompt"])
        for value in (0, 1, 3, True, "1.5", None):
            self.call(self.input("outpaint", {"source": "/source.png"}, expansion=value), expected_error="expansion")
        self.call(self.input(expansion=1.5), expected_error="only")

    def test_invalid_envelope_role_types_and_schema_fail_closed(self):
        cases = [
            (self.input(workflow="unknown"), "unknown workflow_id"),
            (self.input(roles={}), "missing workflow role"),
            (self.input(roles={"scene": "/scene.png"}), "missing workflow role"),
            (self.input(roles={"person": "/person.png", "typo": "/other.png"}), "unknown workflow role"),
            (self.input(roles={"person": "relative.png"}), "absolute"),
            (self.input(roles={"person": ""}), "absolute"),
            (self.input(roles={"person": None}), "must be string"),
            (self.input(roles={"person": "/has\0nul.png"}), "NUL"),
            (self.input(instruction=False), "must be string"),
            (self.input(request=[]), "request must be object"),
            (self.input(extra=True), "unknown workflow field"),
            (self.input(request={"model": "flux2-klein-4b"}), "qwen-image-2.1"),
        ]
        for value, message in cases:
            with self.subTest(value=value):
                self.call(value, expected_error=message)
        for version in (0, 3, True, "1", None, 1.1):
            self.call(self.input(request={"schema_version": version}), expected_error="schema_version")
        self.call({"workflow_id": "playground.identity", "request": {}}, expected_error="missing workflow role")

    def test_composition_does_not_inspect_files_or_perform_runtime_admission(self):
        with tempfile.TemporaryDirectory(prefix="tc-workflow-pure-") as directory:
            missing = str(Path(directory) / "no-image.png")
            # An absent manifest and illegal runtime dimensions are retained.
            # Composition does not inspect them; authoritative plan must reject.
            value = self.input(roles={"person": missing}, request={"model": "qwen-image-2.1", "width": 513,
                               "height": 513, "steps": 25, "ane_manifest": str(Path(directory) / "missing.json")})
            result = self.call(value)
            self.assertEqual(list(Path(directory).iterdir()), [])
            output, error = ctypes.c_void_p(), ctypes.c_void_p()
            status = self.library.tc_plan_json(json.dumps(result["request"]).encode(), ctypes.byref(output), ctypes.byref(error))
            self.assertEqual(status, 1)
            self.consume(output.value)
            self.assertTrue(self.consume(error.value))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_duplicate_keys_and_oversized_inputs_rejected(self):
        self.call(b'{"workflow_id":"playground.transparent","request":{},"request":{}}', expected_error="duplicate")
        self.call(self.input(instruction="x" * 1048576), expected_error="1 MiB")

    def test_cli_catalog_and_composition_match_c_abi(self):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("TURBOCIDER_")}
        catalog = subprocess.run([str(NATIVE / "turbocider"), "workflows"], capture_output=True,
                                 text=True, env=environment, timeout=10)
        self.assertEqual(catalog.returncode, 0, catalog.stderr)
        self.assertEqual(json.loads(catalog.stdout), json.loads(self.consume(self.library.tc_workflows_json())))
        with tempfile.TemporaryDirectory(prefix="tc-workflow-cli-") as directory:
            value = self.input("transparent", {})
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(value))
            result = subprocess.run([str(NATIVE / "turbocider"), "workflow-request", str(path)],
                                    capture_output=True, text=True, env=environment, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), self.call(value))
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["input.json"])


if __name__ == "__main__":
    unittest.main()
