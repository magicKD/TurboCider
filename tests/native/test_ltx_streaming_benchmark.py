#!/usr/bin/env python3
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
import benchmark_ltx_resident as benchmark  # noqa: E402


def bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.pack("<H", bits >> 16)


class LtxStreamingBenchmarkTests(unittest.TestCase):
    def test_bf16_metrics_reports_exact_pair(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = root / "reference.bf16"
            candidate = root / "candidate.bf16"
            payload = b"".join(bf16(value) for value in (1.0, -2.0, 0.5))
            reference.write_bytes(payload)
            candidate.write_bytes(payload)
            metrics = benchmark.bf16_metrics(reference, candidate)
            self.assertTrue(metrics["byte_exact"])
            self.assertTrue(metrics["finite"])
            self.assertEqual(metrics["rel_l2"], 0.0)
            self.assertEqual(metrics["cosine"], 1.0)

    def test_bf16_metrics_detects_difference_and_invalid_shapes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = root / "reference.bf16"
            candidate = root / "candidate.bf16"
            reference.write_bytes(bf16(1.0) + bf16(2.0))
            candidate.write_bytes(bf16(1.0) + bf16(3.0))
            metrics = benchmark.bf16_metrics(reference, candidate)
            self.assertFalse(metrics["byte_exact"])
            self.assertGreater(metrics["rel_l2"], 0.0)
            self.assertLess(metrics["cosine"], 1.0)
            candidate.write_bytes(b"\x00")
            invalid = benchmark.bf16_metrics(reference, candidate)
            self.assertFalse(invalid["size_equal"])
            self.assertFalse(invalid["finite"])
            self.assertTrue(math.isinf(invalid["rel_l2"]))


if __name__ == "__main__":
    unittest.main()
