from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/coreml"))
from export_seedvr2_vae_conv import temporal_conv2d_weights


class VAEConvolutionExportTests(unittest.TestCase):
    def test_temporal_taps_and_output_prefix_preserve_every_original_coefficient(self):
        value = np.arange(4*3*3*3*3,dtype=np.float16).reshape(4,3,3,3,3)
        converted = temporal_conv2d_weights(value,2)
        self.assertEqual(converted.shape,(2,9,3,3))
        self.assertTrue(converted.flags.c_contiguous)
        for tap in range(3):
            self.assertEqual(converted[:,tap*3:(tap+1)*3].tobytes(),value[:2,:,tap].copy().tobytes())

    def test_stacked_conv2d_matches_original_causal_dot_product(self):
        rng = np.random.default_rng(12)
        value = rng.normal(size=(4,3,3,3,3)).astype(np.float16)
        pixels = rng.normal(size=(3,3,5,5)).astype(np.float16)  # C,T,H,W
        converted = temporal_conv2d_weights(value,4)
        stacked = pixels.transpose(1,0,2,3).reshape(9,5,5)
        for y in range(3):
            for x in range(3):
                a = (value.astype(np.float64)*pixels[:,:,y:y+3,x:x+3].astype(np.float64)[None]).sum(axis=(1,2,3,4))
                b = (converted.astype(np.float64)*stacked[:,y:y+3,x:x+3].astype(np.float64)[None]).sum(axis=(1,2,3))
                np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)

    def test_invalid_precision_geometry_and_prefix_are_rejected(self):
        for shape,dtype,channels in (((4,3,3,3,3),np.float32,2),((4,3,3,3),np.float16,2),
                                     ((4,3,2,3,3),np.float16,2),((4,3,3,3,3),np.float16,0),
                                     ((4,3,3,3,3),np.float16,5)):
            with self.subTest(shape=shape,dtype=dtype,channels=channels),self.assertRaises(ValueError):
                temporal_conv2d_weights(np.zeros(shape,dtype=dtype),channels)


if __name__ == "__main__": unittest.main()
