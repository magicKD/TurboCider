from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_spatial_fp32_compare import validate_oracle


class SpatialOracleGuards(unittest.TestCase):
    def fixtures(self):
        screen=dict(status="complete",checkpoint_sha256="a"*64,prefix_fixture_sha256="b"*64)
        oracle=dict(schema="seedvr2-vae-fp32-decoder-oracle-v1",status="complete",checkpoint_sha256="a"*64,input_sha256="b"*64,
            parameter_dtypes=["torch.float32"],output_dtype="torch.float32",floating_dispatch_values_checked=12)
        return screen,oracle
    def test_same_input_explicit_fp32_vae_is_accepted(self):validate_oracle(*self.fixtures())
    def test_old_dit_report_fp16_vae_or_mismatched_input_is_not_relabelled(self):
        for key,bad in (("schema","seedvr2-independent-component-oracle-v1"),("parameter_dtypes",["torch.float16"]),
            ("output_dtype","torch.float16"),("input_sha256","c"*64),("status","running"),("floating_dispatch_values_checked",0)):
            screen,oracle=self.fixtures();oracle[key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):validate_oracle(screen,oracle)
    def test_unfinished_spatial_screen_is_rejected(self):
        screen,oracle=self.fixtures();screen["status"]="failed"
        with self.assertRaises(ValueError):validate_oracle(screen,oracle)


if __name__=="__main__":unittest.main()
