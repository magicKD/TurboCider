from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from turbocider.errors import ModelPreparationError
from turbocider.model_management import ModelPreparer, PreparationOptions
from turbocider.registry import ModelRegistry


class ModelPreparationTests(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry()
        self.preparer = ModelPreparer(registry=self.registry)

    def test_flux_download_dry_run_is_pinned_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "flux"
            report = self.preparer.prepare(
                "flux2-klein-4b",
                PreparationOptions(
                    download=True,
                    dry_run=True,
                    model_directory=destination,
                    minimum_free_gib=0,
                ),
            )
            self.assertEqual(report["status"], "planned")
            self.assertFalse(destination.exists())
            self.assertEqual(
                report["pinned_sources"][0]["revision"],
                "e7b7dc27f91deacad38e78976d1f2b499d76a294",
            )
            self.assertIn(
                "transformer/diffusion_pytorch_model.safetensors",
                report["pinned_sources"][0]["required_paths"],
            )

    def test_ltx_download_selects_only_native_compatible_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self.preparer.prepare(
                "ltx-2.5-distilled",
                PreparationOptions(
                    download=True,
                    dry_run=True,
                    model_directory=Path(temporary) / "ltx",
                    minimum_free_gib=0,
                ),
            )
            patterns = report["pinned_sources"][0]["allow_patterns"]
            self.assertIn(
                "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                patterns,
            )
            self.assertIn(
                "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                patterns,
            )
            self.assertNotIn("diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors", patterns)

    def test_ltx_r256_ane_plan_reports_distinct_kv_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.preparer.prepare(
                "ltx-2.5-distilled",
                PreparationOptions(
                    ane=True,
                    dry_run=True,
                    ane_output=root / "coreml",
                    ltx_text_rows=256,
                    minimum_free_gib=0,
                ),
            )
            self.assertEqual(
                report["artifacts"]["ane_kv_r256_directory"],
                str((root / "coreml" / "text-kv-r256").resolve()),
            )
            self.assertNotIn("ane_kv_directory", report["artifacts"])

    def test_fastmetal_download_and_ane_plan_are_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "FastMetal-1.3B-QAD"
            report = self.preparer.prepare(
                "fastmetal-1.3b-qad",
                PreparationOptions(
                    download=True,
                    ane=True,
                    dry_run=True,
                    model_directory=model,
                    ane_output=root / "coreml",
                    minimum_free_gib=0,
                ),
            )
            self.assertEqual(
                report["pinned_sources"][0]["revision"],
                "2dac0154b217adabf8895d6cde7d6d93e68b7bec",
            )
            self.assertEqual(report["actions"][1]["metadata"]["rows"], 32760)
            self.assertEqual(
                report["actions"][1]["metadata"]["ane_intermediate"], 4096
            )
            self.assertEqual(report["actions"][1]["metadata"]["blocks"], list(range(30)))

    def test_fastmetal_ane_requires_complete_block_coverage(self):
        with self.assertRaisesRegex(ModelPreparationError, "complete block coverage"):
            self.preparer.prepare(
                "fastmetal-1.3b-qad",
                PreparationOptions(
                    ane=True,
                    dry_run=True,
                    blocks="0-28",
                    minimum_free_gib=0,
                ),
            )

    def test_h3_ane_requires_explicit_fixed_rows(self):
        with self.assertRaisesRegex(ModelPreparationError, "--h3-rows"):
            self.preparer.prepare(
                "minimax-h3-turbo",
                PreparationOptions(ane=True, dry_run=True, blocks="0"),
            )

    def test_per_block_estimate_is_scaled(self):
        report = self.preparer.prepare(
            "flux2-klein-4b",
            PreparationOptions(
                ane=True,
                dry_run=True,
                blocks="0",
                minimum_free_gib=0,
            ),
        )
        action = report["actions"][0]
        self.assertEqual(action["estimated_bytes"], 2147483648 // 20)

    def test_public_model_descriptor_does_not_expose_recipe_paths(self):
        descriptor = self.registry.get("flux2-klein-4b").public_dict()
        self.assertNotIn("preparation", descriptor)
        self.assertNotIn("config", descriptor)


if __name__ == "__main__":
    unittest.main()
