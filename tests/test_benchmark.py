from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from turbocider.benchmark import (
    _derived_output,
    _engine_seconds,
    _metric_value,
    _summary,
    compare_media,
    run_benchmark,
)


class BenchmarkTests(unittest.TestCase):
    def test_summary_and_output_naming(self):
        self.assertEqual(_summary([1.0, 3.0])["median"], 2.0)
        self.assertEqual(
            _derived_output(Path("image.png"), "run", 2),
            Path("image-run2.png"),
        )

    def test_identical_decoded_image_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.png"
            second = root / "second.png"
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=16x16", "-frames:v", "1", str(first),
                ],
                check=True,
            )
            second.write_bytes(first.read_bytes())
            report = compare_media(first, second)
            self.assertTrue(report["file_identical"])
            self.assertTrue(report["decoded_visual"]["identical"])
            self.assertEqual(report["decoded_visual"]["mae"], 0.0)

    def test_engine_metric_can_come_from_nested_json_or_stdout(self):
        metrics = {"phase_times": {"native_generation_seconds": 12.5}}
        self.assertEqual(
            _metric_value(metrics, "phase_times.native_generation_seconds"),
            12.5,
        )
        self.assertEqual(
            _engine_seconds(
                {"engine_metric_path": "missing", "engine_metric_regex": r"elapsed=([0-9.]+)"},
                metrics,
                "elapsed=9.25",
            ),
            9.25,
        )

    def test_artifact_only_benchmark_compares_nested_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "target.py"
            script.write_text(
                "import json,pathlib,sys\n"
                "output=pathlib.Path(sys.argv[1]); output.mkdir()\n"
                "(output/'artifact.bin').write_bytes(b'same')\n"
                "pathlib.Path(sys.argv[2]).write_text(json.dumps({'phase_times': {'native_generation_seconds': float(sys.argv[3])}}))\n",
                encoding="utf-8",
            )
            spec = {
                "comparison": {"kind": "artifacts"},
                "direct": {
                    "kind": "command",
                    "output_kind": "directory",
                    "output": str(root / "direct"),
                    "sidecar": "{output}/metrics.json",
                    "engine_metric_path": "phase_times.native_generation_seconds",
                    "command": [sys.executable, str(script), "{output}", "{output}/metrics.json", "2"],
                },
                "turbocider": {
                    "kind": "command",
                    "output_kind": "directory",
                    "output": str(root / "integrated"),
                    "sidecar": "{output}/metrics.json",
                    "engine_metric_path": "phase_times.native_generation_seconds",
                    "command": [sys.executable, str(script), "{output}", "{output}/metrics.json", "2.1"],
                },
                "artifact_pairs": [
                    {
                        "name": "artifact",
                        "direct": "{direct_output}/artifact.bin",
                        "turbocider": "{turbocider_output}/artifact.bin",
                        "comparison": {
                            "command": [
                                sys.executable,
                                "-c",
                                "print('cosine=1.0 rel_l2=0.0')",
                            ],
                            "metric_regexes": {
                                "cosine": "cosine=([0-9.]+)",
                                "rel_l2": "rel_l2=([0-9.]+)",
                            },
                        },
                    }
                ],
                "metric_pairs": [
                    {
                        "name": "native",
                        "direct": "phase_times.native_generation_seconds",
                    }
                ],
            }
            report = run_benchmark(spec)
            self.assertTrue(report["artifacts"][0]["identical"])
            self.assertEqual(
                report["artifacts"][0]["comparison"]["metrics"]["cosine"],
                1.0,
            )
            self.assertTrue(report["quality"]["media_comparison_skipped"])
            self.assertAlmostEqual(
                report["metric_comparisons"][0]["overhead_ratio"], 1.05
            )


if __name__ == "__main__":
    unittest.main()
