"""Native CLI ANE manifest override contract; no model weights needed."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CLI = Path(os.environ.get("TURBOCIDER_TEST_CLI", ROOT / "build/native/turbocider"))


@unittest.skipUnless(CLI.is_file(), "build the native CLI before this test")
class CLIAneOverrideTests(unittest.TestCase):
    def plan(self, request, manifest):
        return subprocess.run([str(CLI), "plan", str(request), "--ane-manifest", str(manifest)],
                              capture_output=True, text=True, cwd=ROOT)

    def test_schema2_z_image_and_other_ane_model_remain_manifest_driven(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "compiled.json"
            manifest.write_text("{}")
            for model, steps in (("z-image-turbo", 8), ("flux2-klein-4b", 4)):
                request = root / f"{model}.json"
                original = json.dumps({"schema_version": 2, "model": model,
                    "operation": "image.generate", "inputs": [
                        {"kind": "text", "role": "prompt", "text": "A red fox."}],
                    "outputs": [{"kind": "image", "path": str(root / "fox.png"),
                                 "width": 512, "height": 512}],
                    "sampling": {"steps": steps, "seed": 42},
                    "execution": {"policy": "gpu", "residency": "resident"}})
                request.write_text(original)
                result = self.plan(request, manifest)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                self.assertEqual(data["model"], model)
                self.assertEqual(data["requested_execution"], "gpu_ane")
                self.assertEqual(request.read_text(), original)

    def test_schema1_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "compiled.json"
            manifest.write_text("{}")
            request = root / "request.json"
            request.write_text(json.dumps({"model": "z-image-turbo",
                "operation": "image.generate", "prompt": "A red fox.",
                "output": str(root / "out.png"), "width": 512, "height": 512,
                "steps": 8, "audio": False, "execution": "gpu"}))
            result = self.plan(request, manifest)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["requested_execution"], "gpu_ane")

    def test_rejects_missing_or_conflicting_manifests_without_touching_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chosen = root / "chosen.json"
            chosen.write_text("{}")
            request = root / "request.json"
            original = json.dumps({"model": "z-image-turbo", "operation": "image.generate",
                "prompt": "A red fox.", "output": str(root / "out.png"),
                "width": 512, "height": 512, "steps": 8, "audio": False,
                "execution": "gpu", "ane_manifest": str(root / "different.json")})
            request.write_text(original)
            missing = self.plan(request, root / "missing.json")
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("existing manifest JSON file", missing.stderr)
            conflict = self.plan(request, chosen)
            self.assertNotEqual(conflict.returncode, 0)
            self.assertIn("conflicts with request ane_manifest", conflict.stderr)
            self.assertEqual(request.read_text(), original)

    def test_generate_and_batch_validate_the_override_before_loading_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "compiled.json"
            manifest.write_text("{}")
            request = root / "request.json"
            request.write_text(json.dumps({"model": "z-image-turbo",
                "operation": "image.generate", "prompt": "A red fox.",
                "output": str(root / "out.png"), "width": 512, "height": 512,
                "steps": 8, "audio": False, "execution": "gpu",
                "ane_manifest": str(root / "different.json")}))
            for command in (("generate", "model", str(request)),
                            ("batch", "model", str(request), str(request))):
                result = subprocess.run(
                    [str(CLI), *command, "--ane-manifest", str(manifest)],
                    capture_output=True, text=True, cwd=ROOT)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("conflicts with request ane_manifest", result.stderr)
                self.assertFalse((root / "out.png").exists())


if __name__ == "__main__":
    unittest.main()
