"""Check endpoint guidance against pixel-frame coordinates, without model weights."""
import ctypes as C
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class EndpointPositionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='tc-ltx-endpoints-')
        library = Path(cls.directory.name) / 'positions.dylib'
        subprocess.run(['xcrun', 'clang', '-std=c11', '-Wall', '-Wextra', '-Werror',
                        '-dynamiclib', str(ROOT/'native/models/ltx_runtime/ltx.c'),
                        '-o', str(library)], check=True)
        cls.library = C.CDLL(str(library))
        cls.positions = cls.library.ltx_compute_endpoint_video_positions
        cls.positions.argtypes = [C.POINTER(C.c_float), C.c_uint64,
                                  C.c_uint32, C.c_uint32, C.c_uint32, C.c_float,
                                  C.c_char_p, C.c_size_t]
        cls.positions.restype = C.c_int

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_guide_uses_last_pixel_frame_and_preserves_causal_grid(self):
        for frames in (2, 3, 5, 13, 16):
            for height, width in ((1, 1), (7, 11), (8, 8), (16, 16)):
                values = (frames + 1) * height * width * 3
                output = (C.c_float * values)()
                error = C.create_string_buffer(256)
                self.assertEqual(self.positions(output, values, frames, height, width, 24,
                                                error, len(error)), 1, error.value)
                times = [(8*(frames-1)+0.5)/24] + [
                    (max(8*f-7, 0)+max(8*f+1, 0))/48 for f in range(frames)]
                expected = [v for time in times for y in range(height) for x in range(width)
                            for v in (time, y*32+16, x*32+16)]
                for actual, wanted in zip(output, expected):
                    self.assertAlmostEqual(actual, wanted, places=5)
                self.assertGreater(output[0], output[-3], 'Tail guide used the last latent midpoint')

    def test_invalid_geometry_and_short_buffer_leave_output_untouched(self):
        for frames, height, width, fps, size in (
            (0, 1, 1, 24, 12), (1, 1, 1, 24, 12), (2, 0, 1, 24, 12),
            (2, 1, 0, 24, 12), (2, 1, 1, 0, 12), (2, 1, 1, math.nan, 12),
            (2, 1, 1, math.inf, 12), (2, 1, 1, 24, 8),
            (0xffffffff, 0xffffffff, 0xffffffff, 24, 12),
        ):
            output = (C.c_float * 12)(*([123.0]*12))
            error = C.create_string_buffer(256)
            self.assertEqual(self.positions(output, size, frames, height, width, fps,
                                            error, len(error)), 0)
            self.assertEqual(list(output), [123.0]*12)
            self.assertTrue(error.value)


if __name__ == '__main__':
    unittest.main()
