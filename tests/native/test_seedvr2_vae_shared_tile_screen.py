from copy import deepcopy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_vae_shared_tile_screen import validate_pair
import test_seedvr2_vae_native_io_screen as io_guards


class SharedTileGuards(unittest.TestCase):
    def bank(self):
        bank=io_guards.NativeIOFixtureGuards().fixture();bank.update(output_channels=128,rows=64,cols=64,input_rows=66,input_cols=66,layer="decoder.hotconv")
        return bank
    def test_complete_and_disjoint_banks(self):
        full=self.bank();prefix=deepcopy(full);prefix["output_channels"]=64
        validate_pair(full,prefix,128);validate_pair(full,None,256)
    def test_incompatible_bank_or_incomplete_output_rejected(self):
        for key,bad in (("layer","decoder.other"),("output_channels",128),("rows",128),("original_shape",[128,128,3,3,3])):
            full=self.bank();prefix=deepcopy(full);prefix["output_channels"]=64;prefix[key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):validate_pair(full,prefix,128)
        full=self.bank();full["output_channels"]=64
        with self.assertRaises(ValueError):validate_pair(full,None,128)
    def test_nondivisible_or_unbounded_geometry_rejected(self):
        for size in (True,96,2048):
            with self.subTest(size=size),self.assertRaises(ValueError):validate_pair(self.bank(),None,size)
        full=self.bank();full.update(original_shape=[4096,512,3,3,3],output_channels=4096,total_output_channels=4096,input_channels=512)
        with self.assertRaises(ValueError):validate_pair(full,None,1024)


if __name__=="__main__":unittest.main()
