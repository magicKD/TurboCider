#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/coreml"))
from export_seedvr2_vae_upscale import projection_weights, spatial_shuffle

class UpscaleTests(unittest.TestCase):
    def test_original_coefficients_preserve_channel_order_and_precision(self):
        value=((np.arange(1024*256,dtype=np.int32)%2048)-1024).astype(np.float16).reshape(1024,256,1,1,1)
        weight=projection_weights(value)
        self.assertEqual(weight.shape,(1024,256,1,1))
        self.assertEqual(weight.dtype,np.float16)
        np.testing.assert_array_equal(weight[:,:,0,0],value[:,:,0,0,0])
        with self.assertRaises(ValueError):projection_weights(value.astype(np.float32))

    def test_shuffle_matches_independent_scalar_phase_index(self):
        value=np.arange(2*12*3*5,dtype=np.int32).reshape(2,12,3,5)
        result=spatial_shuffle(value)
        for b in range(2):
            for c in range(3):
                for y in range(6):
                    for x in range(10):
                        self.assertEqual(result[b,c,y,x],value[b,((y%2)*2+x%2)*3+c,y//2,x//2])
        self.assertEqual(result.dtype,value.dtype)

    def test_shuffle_rejects_incomplete_phase_channels(self):
        with self.assertRaises(ValueError):spatial_shuffle(np.zeros((1,3,2,2)))
        with self.assertRaises(ValueError):spatial_shuffle(np.zeros((4,2,2)))

if __name__=="__main__":unittest.main()
