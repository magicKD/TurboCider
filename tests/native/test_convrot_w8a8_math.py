"""CPU-only independent ConvRot/A8/normalized INT8 mathematical contracts."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('convrot_math',ROOT/'tools/validation/convrot_w8a8_math.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class ConvrotMathTests(unittest.TestCase):
    def test_comfy_rotation_orthogonal_ordering_and_inverse(self):
        h=M.hadamard256();np.testing.assert_array_equal(np.einsum('ij,kj->ik',h,h,optimize=False),np.eye(256,dtype=np.float32))
        basis=np.zeros((1,256),np.float32);basis[0,0]=1
        np.testing.assert_array_equal(M.rotate(basis)[0],h[0])
        x=np.random.default_rng(4).normal(size=(3,768)).astype(np.float32)
        np.testing.assert_allclose(M.rotate(M.rotate(x)),x,rtol=1e-5,atol=2e-6)
        with self.assertRaises(ValueError):M.rotate(x[:,:767])
        x[0,0]=np.inf
        with self.assertRaises(ValueError):M.rotate(x)

    def test_row_scale_ties_even_zero_and_signed_limits(self):
        x=np.array([[127,.5,1.5,-.5,-1.5,-127],[0,0,0,0,0,0]],np.float32)
        codes,scales=M.quantize_rows(x)
        np.testing.assert_array_equal(codes[0],[127,0,2,0,-2,-127])
        np.testing.assert_array_equal(scales,[1,1]);np.testing.assert_array_equal(codes[1],0)
        with self.assertRaises(ValueError):M.quantize_rows(np.array([[np.nan]],np.float32))

    def test_integer_dot_original_minus128_and_fast_oracle(self):
        rng=np.random.default_rng(43)
        x=rng.integers(-127,128,size=(5,256),dtype=np.int8)
        w=rng.integers(-128,128,size=(7,256),dtype=np.int8);w[0,0]=-128
        np.testing.assert_array_equal(M.integer_dot(x,w),M.integer_dot(x,w,fast=True))
        x[0,0]=-128
        with self.assertRaises(ValueError):M.integer_dot(x,w)
        with self.assertRaises(OverflowError):M.integer_dot(np.zeros((1,132105),np.int8),np.zeros((1,132105),np.int8))

    def test_normalized_codes_exact_all_i8_and_output_rounding(self):
        w=np.arange(-128,128,dtype=np.int16).astype(np.int8)
        normalized=M.normalize(w)
        np.testing.assert_array_equal(normalized.astype(np.float32)*128,w.astype(np.float32))
        dot=np.array([[1,1025,-1025,2**24+1]],np.int32)
        np.testing.assert_array_equal(M.normalized_output(dot),(dot.astype(np.float32)/16384).astype(np.float16))
        with self.assertRaises(OverflowError):M.normalized_output(np.array([[np.iinfo(np.int32).max]],np.int32))

    def test_scale_restore_nonuniform_geometry_and_ulp(self):
        dot=np.array([[0,1024],[-1024,2048]],np.int32)
        output=M.normalized_output(dot);sx=np.array([.02,.5],np.float32);sw=np.array([.01,.04],np.float32)
        expected=output.astype(np.float32)*((sx[:,None]*sw[None,:])*16384)+np.array([1,2],np.float32)
        np.testing.assert_array_equal(M.restore_normalized(output,sx,sw,[1,2]),expected)
        zero=np.array([0.,-0.],np.float16);np.testing.assert_array_equal(M.fp16_ulp_distance(zero,zero[::-1]),[0,0])
        a=np.array([1.,-1.],np.float16);b=np.nextafter(a,np.array([2.,-2.],np.float16))
        np.testing.assert_array_equal(M.fp16_ulp_distance(a,b),[1,1])
        with self.assertRaises(ValueError):M.restore_normalized(output,[.1],sw)

    def test_integer_epilogue_order_bias_and_overflow(self):
        dot = np.array([[2**24 + 1, -12345], [71, 987654]], np.int32)
        sx = np.array([.013, .037], np.float32)
        sw = np.array([-.029, .011], np.float32)
        bias = np.array([.5, -1], np.float32)
        expected = (dot.astype(np.float32) * sx[:, None]) * sw[None, :] + bias
        np.testing.assert_array_equal(M.integer_epilogue(dot, sx, sw, bias), expected)
        with self.assertRaises(ValueError):
            M.integer_epilogue(dot.astype(np.int64), sx, sw)
        with self.assertRaises(OverflowError):
            M.integer_epilogue(dot, [1e38, 1], sw)

    def test_group_scale_is_applied_before_sum(self):
        x = np.array([[1, 2, 3, 4]], np.int8)
        w = np.array([[4, 3, 2, -128]], np.int8)
        sx = np.array([[.01, 1]], np.float32)
        sw = np.array([.02], np.float32)
        first = x[:, :2].astype(np.int64) @ w[:, :2].astype(np.int64).T
        second = x[:, 2:].astype(np.int64) @ w[:, 2:].astype(np.int64).T
        expected = (first.astype(np.float32) * sx[:, :1]) * sw
        expected += (second.astype(np.float32) * sx[:, 1:]) * sw
        actual = M.group_integer_projection(x, w, sx, sw, 2)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(actual, M.integer_epilogue(M.integer_dot(x, w), sx[:, 0], sw)))
        with self.assertRaises(ValueError):
            M.group_integer_projection(x, w, sx, sw, 3)

if __name__=='__main__':unittest.main()
