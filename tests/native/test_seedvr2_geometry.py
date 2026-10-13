#!/usr/bin/env python3
"""Compare native geometry to a small independent definition, without GPU work."""
import json
import math
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "build/seedvr2/seedvr2-probe"


def windows(t, h, w, shifted):
    scale = math.sqrt(3600 / (h * w))
    wt, wh, ww = math.ceil(min(t, 30) / 4), math.ceil(round(h * scale) / 3), math.ceil(round(w * scale) / 3)
    dims, sizes = (t, h, w), (wt, wh, ww)
    shifts = tuple(.5 if shifted and size < dim else 0 for size, dim in zip(sizes, dims))
    counts = tuple(math.ceil((dim-shift)/size) + 1 if shift else
                   (1 if shifted else math.ceil(dim/size)) for dim, size, shift in zip(dims, sizes, shifts))
    result = []
    for iw in range(counts[2]):
        for ih in range(counts[1]):
            for it in range(counts[0]):
                bounds = [[max(int((index-shift)*size), 0), min(int((index-shift+1)*size), dim)]
                          for index, shift, size, dim in zip((it, ih, iw), shifts, sizes, dims)]
                if all(stop > start for start, stop in bounds):
                    result.append(bounds)
    return result


class GeometryTests(unittest.TestCase):
    def test_adaptive_and_shifted_window_partitions(self):
        self.assertTrue(PROBE.is_file(), "build with bash tools/native/build_seedvr2_probe.sh")
        for shape in ((1, 4, 4), (1, 32, 32), (1, 64, 64), (3, 32, 32),
                      (5, 64, 64), (9, 32, 32), (25, 80, 45), (33, 47, 83), (61, 31, 27)):
            for shifted in (False, True):
                with self.subTest(shape=shape, shifted=shifted):
                    output = subprocess.check_output([str(PROBE), "windows", *map(str, shape), str(int(shifted))], text=True)
                    actual = json.loads(output)
                    self.assertEqual(actual, windows(*shape, shifted))
                    counts = [0] * math.prod(shape)
                    for (t0, t1), (h0, h1), (w0, w1) in actual:
                        for it in range(t0, t1):
                            for ih in range(h0, h1):
                                for iw in range(w0, w1):
                                    counts[(it*shape[1]+ih)*shape[2]+iw] += 1
                    self.assertTrue(all(count == 1 for count in counts))


if __name__ == "__main__":
    unittest.main()
