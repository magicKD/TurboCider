from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_vae_coreml_conv_screen import reference_3d,check_reference,quality


class ConvolutionScreenTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.x = rng.normal(size=(1,3,3,7,7)).astype(np.float16)
        self.w = rng.normal(size=(4,3,3,3,3)).astype(np.float16)

    def test_original_temporal_convolution_order_and_fp64_validation(self):
        reference = reference_3d(self.x,self.w)
        self.assertEqual(reference.shape,(1,4,5,5))
        self.assertTrue(check_reference(self.x,self.w,reference)["finite"])
        expected = float(np.sum(self.x[0,:,:,2:5,2:5].astype(np.float64)*self.w[1].astype(np.float64)))
        self.assertAlmostEqual(float(reference[0,1,2,2]),expected,places=5)

    def test_wrong_and_nonfinite_references_are_rejected(self):
        reference = reference_3d(self.x,self.w)
        for broken in (reference+1.,np.full_like(reference,np.nan)):
            with self.assertRaises(ValueError): check_reference(self.x,self.w,broken)

    def test_unchanged_quality_thresholds_and_nonfinite_rejection(self):
        reference = reference_3d(self.x,self.w)
        self.assertTrue(quality(reference,reference.astype(np.float16))["passed"])
        self.assertFalse(quality(reference,-reference)["passed"])
        self.assertFalse(quality(reference,np.full_like(reference,np.inf))["passed"])


if __name__ == "__main__": unittest.main()
