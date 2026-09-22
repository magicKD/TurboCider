import unittest

import numpy as np

from tools.validation.qwen21_mask_edit_report import measure


class MaskEditReportTests(unittest.TestCase):
    def test_only_edit_region_changed(self):
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        output = source.copy()
        mask = np.array([[True, False], [False, False]])
        output[mask] = 255
        report = measure(source, output, mask)
        self.assertEqual(report['regions']['edit']['rgb_rmse'], 1)
        self.assertEqual(report['regions']['preserve']['rgb_rmse'], 0)
        self.assertFalse(report['quality_accepted'])

    def test_outside_change_is_reported(self):
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        output = source.copy()
        mask = np.array([[True, False], [False, False]])
        output[~mask, 0] = 255
        report = measure(source, output, mask)
        self.assertEqual(report['regions']['edit']['rgb_mae'], 0)
        self.assertEqual(report['regions']['preserve']['rgb_mean_change'], [1, 0, 0])
        self.assertEqual(report['regions']['preserve']['fraction_pixels_changed'], 1)

    def test_invalid_geometry_and_empty_regions(self):
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        for mask in (np.zeros((2, 2), dtype=bool), np.ones((2, 2), dtype=bool),
                     np.zeros((3, 2), dtype=bool), np.zeros((2, 2), dtype=np.uint8)):
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with self.assertRaises(ValueError):
                    measure(source, source, mask)


if __name__ == '__main__':
    unittest.main()
