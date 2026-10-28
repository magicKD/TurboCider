#!/usr/bin/env python3
"""CPU checks for immutable FP16/legacy-BF16 Core ML operand preparation."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/coreml"))
SPEC = importlib.util.spec_from_file_location("seedvr2_ffn_export",ROOT/"tools/coreml/export_seedvr2_ffn.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class FFNExportTests(unittest.TestCase):
    def test_original_fp16_preserves_every_operand_bit(self):
        source = np.array([0x0000,0x8000,0x0001,0x0400,0x2E66,0x33E7,0x3555,0x7BFF],dtype=np.uint16).view(np.float16)
        output = exporter.prepare_operand(source,"original-fp16")
        np.testing.assert_array_equal(output.view(np.uint16),source.view(np.uint16))
        self.assertTrue(output.flags.c_contiguous)

    def test_noncontiguous_down_columns_are_packed_without_changing_values(self):
        source = np.arange(48,dtype=np.float16).reshape(6,8)[:,::2]
        output = exporter.prepare_operand(source,"original-fp16")
        self.assertTrue(output.flags.c_contiguous)
        np.testing.assert_array_equal(output,source)

    def test_legacy_rounding_is_explicit_and_does_not_modify_source(self):
        source = np.array([.1234,-.2345,.4567],dtype=np.float16)
        old = source.copy()
        output = exporter.prepare_operand(source,"bf16-rounded")
        self.assertTrue(np.any(output!=source))
        np.testing.assert_array_equal(source,old)
        self.assertNotEqual(exporter.OPERANDS["original-fp16"],exporter.OPERANDS["bf16-rounded"])

    def test_unknown_operand_recipe_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"recipe"):
            exporter.prepare_operand(np.ones(3,dtype=np.float16),"automatic")

    def test_block_ranges_are_bounded_and_ordered(self):
        self.assertEqual(exporter.parse_blocks("0,2,10-12,31"),[0,2,10,11,12,31])
        for value in ("3-2","0-999999999","-1","32"):
            with self.subTest(value=value),self.assertRaises(ValueError):
                exporter.parse_blocks(value)


if __name__ == "__main__":
    unittest.main()
