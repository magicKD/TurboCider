#!/usr/bin/env python3
"""Host-only statistics and bounded decoding guards for App export evidence."""
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2] / "tools/validation"))
from seedvr2_app_media_screen import PixelStats, read_frame


class MediaScreenTests(unittest.TestCase):
    def test_exact_and_incremental_statistics(self):
        stats = PixelStats()
        pixels = np.array([0,128,255],dtype=np.uint8)
        stats.add(pixels,pixels)
        stats.add(pixels,pixels)
        result = stats.result()
        self.assertTrue(result["decoded_pixels_bitwise_equal"])
        self.assertEqual(result["decoded_frames"],2)
        self.assertEqual(result["rmse_8bit"],0)
        self.assertIsNone(result["psnr_db"])
        self.assertEqual(result["cosine"],1)

    def test_uint8_difference_does_not_wrap(self):
        stats = PixelStats()
        stats.add(np.array([0,255],dtype=np.uint8),np.array([255,0],dtype=np.uint8))
        self.assertEqual(stats.result()["rmse_8bit"],255)
        self.assertFalse(stats.result()["decoded_pixels_bitwise_equal"])

    def test_rejects_shape_dtype_and_empty_statistics(self):
        stats = PixelStats()
        with self.assertRaises(ValueError): stats.result()
        with self.assertRaises(ValueError): stats.add(np.ones(1),np.ones(1))
        with self.assertRaises(ValueError): stats.add(np.zeros(1,dtype=np.uint8),np.zeros(2,dtype=np.uint8))

    def test_bounded_frame_reader_rejects_truncation(self):
        stream = io.BytesIO(b"123456")
        self.assertEqual(read_frame(stream,3).tobytes(),b"123")
        self.assertEqual(read_frame(stream,3).tobytes(),b"456")
        self.assertIsNone(read_frame(stream,3))
        with self.assertRaises(ValueError): read_frame(io.BytesIO(b"12"),3)


if __name__ == "__main__": unittest.main()
