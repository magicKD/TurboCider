#!/usr/bin/env python3
import importlib.util
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "turbocider_quality_gate", ROOT / "tools/native/quality_gate.py"
)
QUALITY_GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(QUALITY_GATE)


class QualityGateTests(unittest.TestCase):
    def write(self, directory: Path, name: str, width: int, height: int,
              value: bytes) -> Path:
        path = directory / name
        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (struct.pack(">I", len(payload)) + kind + payload +
                    struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))
        rows = b"".join(
            b"\x00" + value[row * width * 3:(row + 1) * width * 3]
            for row in range(height)
        )
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
        )
        return path

    def test_identical_images_pass(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            image = bytes(range(8 * 8 * 3))
            reference = self.write(directory, "reference.png", 8, 8, image)
            candidate = self.write(directory, "candidate.png", 8, 8, image)
            metrics = QUALITY_GATE.image_metrics(reference, candidate)
            self.assertTrue(QUALITY_GATE.passes(
                metrics, min_correlation=0.99, min_cosine=0.995, max_mae_255=5.0
            ))

    def test_shape_or_large_error_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            reference = self.write(directory, "reference.png", 8, 8, bytes(8 * 8 * 3))
            wrong_shape = self.write(directory, "wrong-shape.png", 8, 4, bytes(8 * 4 * 3))
            self.assertFalse(QUALITY_GATE.passes(
                QUALITY_GATE.image_metrics(reference, wrong_shape),
                min_correlation=0.99, min_cosine=0.995, max_mae_255=5.0,
            ))
            changed = bytearray(8 * 8 * 3)
            for row in range(8):
                changed[row * 24:row * 24 + 12] = bytes([255]) * 12
            candidate = self.write(directory, "changed.png", 8, 8, bytes(changed))
            self.assertFalse(QUALITY_GATE.passes(
                QUALITY_GATE.image_metrics(reference, candidate),
                min_correlation=0.99, min_cosine=0.995, max_mae_255=5.0,
            ))

    def test_calibrated_perceptual_mode_does_not_require_aligned_rgb(self):
        metrics = {
            "shape_equal": True,
            "finite": True,
            "correlation": 0.80,
            "cosine": 0.95,
            "mae_255": 20.0,
            "vision_feature_print": {
                "pair_count": 1,
                "maximum_distance": 0.20,
            },
        }
        thresholds = {
            "min_correlation": 0.99,
            "min_cosine": 0.995,
            "max_mae_255": 5.0,
        }
        self.assertFalse(QUALITY_GATE.passes(metrics, **thresholds))
        self.assertTrue(QUALITY_GATE.passes(
            metrics, **thresholds, require_aligned_rgb=False,
            max_vision_distance=0.25,
        ))
        self.assertFalse(QUALITY_GATE.passes(
            metrics, **thresholds, require_aligned_rgb=False,
            max_vision_distance=0.10,
        ))
        self.assertFalse(QUALITY_GATE.passes(
            metrics, **thresholds, require_aligned_rgb=False,
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
