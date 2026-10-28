from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_vae_native_io_screen import CHECKPOINT_SHA, SCHEMA, validate_manifest


class NativeIOFixtureGuards(unittest.TestCase):
    def fixture(self):
        return dict(schema=SCHEMA,checkpoint_sha256=CHECKPOINT_SHA,bias_applied=False,precision="fp16",
            source_operand_recipe="original-fp16-temporal-channel-stack",original_shape=[128,256,3,3,3],
            output_channels=64,total_output_channels=128,rows=128,cols=128,input_rows=130,input_cols=130,
            input_layout="N(T*C)HW",output_layout="NOHW",temporal_taps=3,input_channels=256)

    def test_valid_temporal_prefix_and_full_pointwise(self):
        validate_manifest(self.fixture())
        value=self.fixture()
        value.update(original_shape=[4096,512,1,1,1],total_output_channels=4096,output_channels=4096,
            rows=32,cols=32,input_rows=32,input_cols=32,temporal_taps=1,input_channels=512)
        validate_manifest(value)

    def test_recipe_and_geometry_drift_rejected(self):
        for key,bad in (("bias_applied",True),("bias_applied",0),("precision","fp32"),("input_rows",129),
            ("input_cols",128),("rows",True),("output_channels",129),("temporal_taps",1),
            ("source_operand_recipe","rounded-bf16"),("input_layout","NCTHW")):
            value=deepcopy(self.fixture());value[key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):validate_manifest(value)

    def test_large_fixture_rejected_before_allocation(self):
        value=self.fixture()
        value.update(original_shape=[4096,512,3,3,3],total_output_channels=4096,output_channels=4096,
            rows=256,cols=256,input_rows=258,input_cols=258,input_channels=512)
        with self.assertRaises(ValueError):validate_manifest(value)


if __name__=="__main__":unittest.main()
