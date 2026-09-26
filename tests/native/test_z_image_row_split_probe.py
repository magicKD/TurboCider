"""Fail-closed geometry checks for the export-only whole-FFN row probe."""

import subprocess
import sys
import unittest
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "tools/coreml/export_z_image.py"
SPEC = importlib.util.spec_from_file_location(
    "z_image_row_split_benchmark", ROOT / "tools/coreml/benchmark_z_image_row_split.py")
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class ZImageRowSplitProbeTests(unittest.TestCase):
    def test_benchmark_accepts_long_capture_only_for_matching_split(self):
        shape = {"K": 3840, "N": 3840, "mlp_width": 10240,
                 "ane_mlp_end": 10240, "buckets": [512]}
        identity = {"activation_precision": "int8", "blocks": [18],
                    "row_split_probe": True}
        self.assertEqual(BENCHMARK.validate_split_manifest(
            {"shape": shape, "export_identity": identity}, (1536, 3840), 18, False),
            (512, 10240))
        self.assertEqual(BENCHMARK.validate_split_manifest(
            {"shape": {**shape, "buckets": [544]},
             "export_identity": identity}, (1536, 3840), 18, False),
            (544, 10240))
        channel = {"shape": {**shape, "ane_mlp_end": 4096, "buckets": [1536]},
                   "export_identity": {**identity, "row_split_probe": False,
                                       "region_image_rows": 1024}}
        self.assertEqual(BENCHMARK.validate_split_manifest(
            channel, (1536, 3840), 18, True), (1536, 4096))
        with self.assertRaisesRegex(ValueError, "full captured rows"):
            BENCHMARK.validate_split_manifest(channel, (1056, 3840), 18, True)
        with self.assertRaisesRegex(ValueError, "row probe"):
            BENCHMARK.validate_split_manifest(channel, (1536, 3840), 18, False)
        with self.assertRaisesRegex(ValueError, "row probe"):
            BENCHMARK.validate_split_manifest(
                {"shape": {**shape, "buckets": [576]},
                 "export_identity": identity}, (1536, 3840), 18, False)
        with self.assertRaisesRegex(ValueError, "single-block"):
            BENCHMARK.validate_split_manifest(channel, (1537, 3840), 18, True)

    def test_benchmark_image_only_channel_uses_marked_1024_rows(self):
        source = {"shape": {"K": 3840, "N": 3840, "mlp_width": 10240,
                            "ane_mlp_end": 5120, "buckets": [1024]},
                  "export_identity": {"activation_precision": "int8",
                                      "blocks": list(range(32)),
                                      "image_only_token_rows": 1024}}
        self.assertEqual(BENCHMARK.validate_split_manifest(
            source, (1536, 3840), 18, False, True), (1024, 5120))
        with self.assertRaisesRegex(ValueError, "choose"):
            BENCHMARK.validate_split_manifest(source, (1536, 3840), 18, True, True)
        with self.assertRaisesRegex(ValueError, "single-block"):
            BENCHMARK.validate_split_manifest(
                {**source, "export_identity": {**source["export_identity"],
                                               "blocks": [18]}},
                (1536, 3840), 18, False, True)
        with self.assertRaisesRegex(ValueError, "marked 1024-row"):
            BENCHMARK.validate_split_manifest(
                {**source, "export_identity": {"activation_precision": "int8",
                                               "blocks": list(range(32))}},
                (1536, 3840), 18, False, True)

    def test_two_group_probe_accepts_only_marked_small_image_only_artifacts(self):
        shape = {"K": 3840, "N": 3840, "mlp_width": 10240,
                 "ane_mlp_end": 5120, "buckets": [1024]}
        identity = {"activation_precision": "int8", "blocks": [18],
                    "hidden_a8_channel_groups": 2, "image_only_token_rows": 1024}
        source = {"shape": shape, "export_identity": identity}
        self.assertEqual(BENCHMARK.validate_split_manifest(
            source, (1056, 3840), 18, False, True, True), (1024, 5120))
        with self.assertRaisesRegex(ValueError, "requires image-only"):
            BENCHMARK.validate_split_manifest(source, (1056, 3840), 18,
                                              False, False, True)
        with self.assertRaisesRegex(ValueError, "single-block"):
            BENCHMARK.validate_split_manifest(
                {"shape": shape, "export_identity": {**identity,
                 "hidden_a8_channel_groups": 4}}, (1056, 3840), 18,
                False, True, True)
        with self.assertRaisesRegex(ValueError, "single-block"):
            BENCHMARK.validate_split_manifest(
                {"shape": shape, "export_identity": {**identity,
                 "blocks": list(range(32))}}, (1056, 3840), 18,
                False, True, True)
        self.assertEqual(BENCHMARK.validate_split_manifest(
            {"shape": shape, "export_identity": {**identity, "blocks": [3, 29]}},
            (1056, 3840), 29, False, True, True), (1024, 5120))

    def export_rejection(self, *flags):
        result = subprocess.run(
            [sys.executable, str(EXPORT), "--model", "not-used",
             "--output", "not-created", *flags], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("FileNotFoundError", result.stderr)
        return result.stderr

    def test_full_width_requires_explicit_probe(self):
        self.assertIn("full-width row probe", self.export_rejection(
            "--ane-mlp-width", "10240"))

    def test_row_probe_rejects_missing_real_calibration_and_wrong_width(self):
        basic = ("--row-split-probe", "--blocks", "18", "--bucket", "416",
                 "--calibration-source-rows", "1056", "--activation-precision", "int8",
                 "--hidden-a8-channel-groups", "4")
        self.assertIn("row-split probe", self.export_rejection(*basic))
        self.assertIn("row-split probe", self.export_rejection(
            *basic, "--ane-mlp-width", "10240", "--bucket", "1056"))
        self.assertIn("row-split probe", self.export_rejection(
            *basic, "--ane-mlp-width", "10240", "--blocks", "all"))
        self.assertIn("row-split probe", self.export_rejection(
            *basic, "--ane-mlp-width", "10240", "--bucket", "576"))


if __name__ == "__main__":
    unittest.main()
