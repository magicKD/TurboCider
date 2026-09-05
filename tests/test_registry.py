from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from turbocider.errors import ValidationError
from turbocider.registry import ModelRegistry
from turbocider.runtime import TurboCiderRuntime


class RegistryTests(unittest.TestCase):
    def test_bundled_model_packs_load(self):
        registry = ModelRegistry()
        self.assertEqual(
            [model.id for model in registry.all()],
            [
                "fastmetal-1.3b-qad",
                "flux2-klein-4b",
                "ltx-2.5-distilled",
                "minimax-h3-turbo",
            ],
        )
        self.assertEqual(registry.get("minimax-h3-turbo").engine, "h3")

    def test_custom_engine_id_is_allowed_for_installed_adapters(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "custom.json").write_text(
                json.dumps({
                    "id": "custom-model",
                    "name": "Custom model",
                    "engine": "example.runtime",
                    "capabilities": {"tasks": ["image"]},
                    "plans": [{
                        "id": "custom.gpu",
                        "execution": "gpu",
                        "quality": "exact",
                    }],
                }),
                encoding="utf-8",
            )
            registry = ModelRegistry(search_paths=[directory])
            self.assertEqual(registry.get("custom-model").engine, "example.runtime")

    def test_invalid_custom_engine_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "invalid.json").write_text(
                json.dumps({
                    "id": "invalid-model",
                    "name": "Invalid model",
                    "engine": "Bad Engine",
                    "capabilities": {},
                    "plans": [{
                        "id": "invalid.gpu",
                        "execution": "gpu",
                        "quality": "exact",
                    }],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "invalid engine id"):
                ModelRegistry(search_paths=[directory])

    def test_doctor_reports_uninstalled_custom_adapter_without_crashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packs = root / "packs"
            packs.mkdir()
            (packs / "custom.json").write_text(
                json.dumps({
                    "id": "custom-model",
                    "name": "Custom model",
                    "engine": "definitely.not.installed",
                    "capabilities": {"tasks": ["image"]},
                    "plans": [{
                        "id": "custom.gpu",
                        "execution": "gpu",
                        "quality": "exact",
                    }],
                }),
                encoding="utf-8",
            )
            runtime = TurboCiderRuntime(
                registry=ModelRegistry(search_paths=[packs]),
                state_directory=root / "state",
                output_directory=root / "outputs",
            )
            try:
                report = next(
                    item for item in runtime.doctor()["models"]
                    if item["id"] == "custom-model"
                )
            finally:
                runtime.close()
            self.assertFalse(report["available"])
            self.assertIn("install an adapter entry point", report["error"])


if __name__ == "__main__":
    unittest.main()
