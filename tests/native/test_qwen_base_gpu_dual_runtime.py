"""CPU-only explicit full40 GPU-dual environment and actual route receipts."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/profiling/qwen_base"))
import collect
from test_qwen_base_text_prefix import fixture


def receipts(enabled,matrix=False):
    rows=fixture(matrix)
    for row in rows:
        hit=row["qwen_runtime_prefix_snapshot"]["hit"]
        n=(1280 if hit else 1248) if enabled else 0
        row["qwen_gpu_dual_swiglu"]=dict(requested=enabled,enabled=enabled,validation_passed=enabled,disabled=False,
            failure_reason="",recipe="gpu-dual-swiglu-register-m32n64-target1024-v1",extra_owned_weight_bytes=0,
            attempts_this_request=n,completed_blocks_this_request=n,ordinary_blocks_this_request=1280-n,fallback_blocks_this_request=0,
            validation_calls_session_total=1 if enabled else 0,validation_seconds_session_total=.02 if enabled else 0.)
    return rows


class RuntimeGpuDualTests(unittest.TestCase):
    def test_legacy_dictionary_unchanged_and_off_on_only_new_key(self):
        kwargs=dict(channels=6144,private_qkv=True,fixed_program=True,ffn_activation="sigmoid-complement",private_qkv_prefetch=True)
        legacy=collect.environment("runtime","stock",**kwargs)
        self.assertNotIn("TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU",legacy)
        off=collect.environment("runtime","stock",**kwargs,gpu_dual_swiglu=False)
        on=collect.environment("runtime","stock",**kwargs,gpu_dual_swiglu=True)
        self.assertEqual(set(off)-set(legacy),{"TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU"})
        self.assertEqual({k for k in off if off[k]!=on[k]},{"TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU"})

    def test_only_explicit_fixed_joint6144_sigmoid_recipe_admitted(self):
        original=dict(channels=6144,private_qkv=True,fixed_program=True,ffn_activation="sigmoid-complement")
        for change in (dict(channels=5120),dict(private_qkv=False),dict(fixed_program=False),dict(ffn_activation="exp"),dict(resident_prefix=True),dict(ffn_projection="single")):
            with self.assertRaises(ValueError):collect.environment("runtime","stock",**{**original,**change},gpu_dual_swiglu=True)
        with self.assertRaises(ValueError):collect.environment("runtime","runtime-layers",**original,gpu_dual_swiglu=True)
        with self.assertRaises(ValueError):collect.environment("gpu","stock",gpu_dual_swiglu=True)
        with self.assertRaises(ValueError):collect.environment("runtime","stock",**original,gpu_dual_swiglu=1)

    def test_actual_miss_hit_and_prompt_matrix_counts(self):
        for enabled in (False,True):
            collect.validate_gpu_dual_swiglu(receipts(enabled),enabled)
            collect.validate_gpu_dual_swiglu(receipts(enabled,True),enabled)
        self.assertEqual(receipts(True)[0]["qwen_gpu_dual_swiglu"]["ordinary_blocks_this_request"],32)
        self.assertEqual(receipts(True)[1]["qwen_gpu_dual_swiglu"]["completed_blocks_this_request"],1280)

    def test_no_fake_credit_fallback_wrong_blocks_or_validation_retries(self):
        changes=(lambda r:r[1]["qwen_gpu_dual_swiglu"].update(completed_blocks_this_request=1248),
            lambda r:r[0]["qwen_gpu_dual_swiglu"].update(attempts_this_request=0),
            lambda r:r[0]["qwen_gpu_dual_swiglu"].update(fallback_blocks_this_request=1),
            lambda r:r[0]["qwen_gpu_dual_swiglu"].update(validation_passed=False),
            lambda r:r[1]["qwen_gpu_dual_swiglu"].update(validation_calls_session_total=2),
            lambda r:r[1]["qwen_gpu_dual_swiglu"].update(validation_seconds_session_total=.001),
            lambda r:r[0]["qwen_gpu_dual_swiglu"].update(disabled=True),
            lambda r:r[0]["qwen_gpu_dual_swiglu"].update(failure_reason="source changed"),
            lambda r:r[0]["qwen_gpu_dual_swiglu"].update(extra_owned_weight_bytes=1024),
            lambda r:r[0]["qwen_ffn_phases"]["decode"].update(steps_this_request=38))
        for change in changes:
            bad=receipts(True);change(bad)
            with self.assertRaises(ValueError):collect.validate_gpu_dual_swiglu(bad,True)
        bad=receipts(False);bad[0]["qwen_gpu_dual_swiglu"].update(completed_blocks_this_request=1)
        with self.assertRaises(ValueError):collect.validate_gpu_dual_swiglu(bad,False)

    def test_native_port_preserves_qualified_gpu_source_body(self):
        old=(ROOT/"tools/profiling/qwen_base/dual_gup.hpp").read_text()
        new=(ROOT/"native/models/qwen21/metal/base_dual_swiglu.hpp").read_text()
        new=new.replace("namespace tc::qwen21::metal {","namespace tc::dual_gup_lab {").replace("prepare_base_dual_swiglu(","prepare(")
        self.assertEqual(old[old.index("#include <mlx"):],new[new.index("#include <mlx"):])


if __name__=="__main__":unittest.main()
