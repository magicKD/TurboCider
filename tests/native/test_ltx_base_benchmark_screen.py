from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from ltx_base_benchmark_screen import validate_rows


class LtxBaseBenchmarkTests(unittest.TestCase):
    def test_complete_base_geometry_not_a_short_video_or_adapter_proxy(self):
        row = dict(model="ltx-2.5-distilled", operation="video.generate", width=704, height=448,
                   frames=97, steps=11, seed=42, audio=True)
        validate_rows([row, dict(row)], "gpu")
        for name, value in (("frames", 9), ("steps", 4), ("width", 512), ("lora_applied_projections", 1)):
            bad = dict(row); bad[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_rows([row, bad], "gpu")

    def test_hybrid_route_cannot_report_pure_gpu_as_success(self):
        row = dict(model="ltx-2.5-distilled", operation="video.generate", width=704, height=448,
                   frames=97, steps=11, seed=42, audio=True, execution="gpu", ane_profile=None)
        with self.assertRaises(ValueError):
            validate_rows([row, dict(row)], "int8")


if __name__ == "__main__":
    unittest.main()
