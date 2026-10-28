"""Host-only complete-request explicit-W compiled complement receipts."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
import collect
from test_qwen_base_text_prefix import fixture


def metrics(matrix=False):
    rows=fixture(matrix)
    for row in rows:
        hit=row['qwen_runtime_prefix_snapshot']['hit'];n=1280 if hit else 1248
        row['acceleration_selection']=row.get('acceleration_selection','')+'; experimental shared compiled GPU complement with explicit runtime W'
        row['qwen_compiled_complement']=dict(requested=True,enabled=True,validation_passed=True,disabled=False,
            attempts_this_request=n,completed_blocks_this_request=n,ordinary_blocks_this_request=1280-n,fallback_blocks_this_request=0,
            validation_calls_session_total=1,validation_seconds_session_total=.02,trace_count_session_total=1,
            failure_reason='',recipe='gpu-compiled-complement-explicit-w-target1024-v1',extra_owned_weight_bytes=0)
    return rows


class CompiledComplementRequestTests(unittest.TestCase):
    def test_original_environment_unchanged_and_only_one_explicit_difference(self):
        kwargs=dict(channels=6144,private_qkv=True,fixed_program=True,ffn_activation='sigmoid-complement',private_qkv_prefetch=True)
        old=collect.environment('runtime','stock',**kwargs)
        off=collect.environment('runtime','stock',**kwargs,compiled_complement=False)
        on=collect.environment('runtime','stock',**kwargs,compiled_complement=True)
        flag='TURBOCIDER_QWEN21_COMPILED_COMPLEMENT'
        self.assertNotIn(flag,old);self.assertEqual(set(off)-set(old),{flag})
        self.assertEqual({k for k in off if off[k]!=on[k]},{flag})

    def test_reject_unsupported_runtime_shape_and_other_arithmetic(self):
        kwargs=dict(channels=6144,private_qkv=True,fixed_program=True,ffn_activation='sigmoid-complement')
        for change in (dict(channels=5120),dict(private_qkv=False),dict(fixed_program=False),dict(ffn_activation='exp'),
            dict(gpu_dual_swiglu=True),dict(function_rows='512,1024'),dict(resident_prefix=True),dict(ffn_projection='single')):
            with self.assertRaises(ValueError):collect.environment('runtime','stock',**{**kwargs,**change},compiled_complement=True)
        for mode in ('gpu-ops','runtime-layers'):
            with self.assertRaises(ValueError):collect.environment('runtime',mode,**kwargs,compiled_complement=True)
        with self.assertRaises(ValueError):collect.environment('gpu','stock',compiled_complement=True)
        with self.assertRaises(ValueError):collect.environment('runtime','stock',**kwargs,compiled_complement=1)

    def test_all_prefill_decode_and_prompt_recovery_counts(self):
        for matrix in (False,True):collect.validate_compiled_complement(metrics(matrix),True)
        collect.validate_compiled_complement(fixture(),False)
        self.assertEqual(metrics()[0]['qwen_compiled_complement']['ordinary_blocks_this_request'],32)
        self.assertEqual(metrics()[1]['qwen_compiled_complement']['completed_blocks_this_request'],1280)

    def test_reject_missing_actual_work_retrace_and_revalidation(self):
        changes=(lambda r:r[0].pop('qwen_compiled_complement'),lambda r:r[0].update(acceleration_selection=''),
            lambda r:r[1]['qwen_compiled_complement'].update(completed_blocks_this_request=1248),
            lambda r:r[0]['qwen_compiled_complement'].update(attempts_this_request=0),
            lambda r:r[0]['qwen_compiled_complement'].update(fallback_blocks_this_request=1),
            lambda r:r[1]['qwen_compiled_complement'].update(trace_count_session_total=2),
            lambda r:r[1]['qwen_compiled_complement'].update(validation_calls_session_total=2),
            lambda r:r[1]['qwen_compiled_complement'].update(validation_seconds_session_total=.021),
            lambda r:r[0]['qwen_compiled_complement'].update(extra_owned_weight_bytes=1024),
            lambda r:r[0]['qwen_compiled_complement'].update(disabled=True),
            lambda r:r[0]['qwen_ffn_phases']['decode'].update(steps_this_request=38))
        for change in changes:
            bad=copy.deepcopy(metrics());change(bad)
            with self.assertRaises(ValueError):collect.validate_compiled_complement(bad,True)
        with self.assertRaises(ValueError):collect.validate_compiled_complement(metrics(),False)


if __name__=='__main__':unittest.main()
