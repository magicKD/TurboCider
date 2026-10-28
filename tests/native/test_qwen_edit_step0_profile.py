"""Host-only explicit edit first-step environment and unchanged default."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
import collect


class EditStep0ProfileTests(unittest.TestCase):
    def test_only_explicit_joint_edit_prefix_changes_one_flag(self):
        kwargs=dict(runtime_prefix=True,private_qkv=True,fixed_program=True,ffn_activation='sigmoid-complement')
        old=collect.environment('runtime','stock',6144,**kwargs)
        ane=collect.environment('runtime','stock',6144,**kwargs,edit_prefix_qkv_step0='ane')
        gpu=collect.environment('runtime','stock',6144,**kwargs,edit_prefix_qkv_step0='gpu')
        flag='TURBOCIDER_QWEN21_RUNTIME_EDIT_PREFIX_QKV_STEP0'
        self.assertNotIn(flag,old)
        self.assertEqual(set(gpu)-set(old),{flag})
        self.assertEqual({k for k in ane if ane[k]!=gpu[k]},{flag})
        for key in ('runtime_prefix','private_qkv','fixed_program'):
            bad=dict(kwargs);bad[key]=False
            with self.assertRaises(ValueError):collect.environment('runtime','stock',6144,**bad,edit_prefix_qkv_step0='gpu')
        with self.assertRaises(ValueError):collect.environment('runtime','stock',6144,**kwargs,edit_prefix_qkv_step0='auto')


if __name__=='__main__':unittest.main()
