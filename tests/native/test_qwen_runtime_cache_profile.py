"""Host-only cache-tier policy/actual-work accounting; no inference."""
import argparse
import copy
from contextlib import ExitStack
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
import runtime_cache_collect as screen


def rows(case='generate',cache='conservative'):
    values=[];ffn=blocks=qkv=gpu=claims=discards=0
    for i in range(3):
        hit=i>0;s=15 if cache=='conservative' else 0;saved=24*s;n=1280-saved
        ffn+=n if hit else (1440 if case=='edit' else 1312)-saved;blocks+=n
        nq=(1280 if case=='edit' and hit else 1248)-saved;qkv+=nq;gpu+=0 if case=='edit' and hit else 32
        claims+=nq-(1 if case=='edit' and hit else 0);discards+=s;sub=claims+discards
        row=dict(model='qwen-image-2.1',operation='image.edit' if case=='edit' else 'image.generate',width=512,height=512,steps=40,actual_denoise_steps=40,
            qwen_runtime_prefix_snapshot=dict(admitted=True,hit=hit,published=not hit,decline_reason='',recipe_sha256='a'*64),
            qwen_ffn_phases=dict(prefill=dict(steps_this_request=0 if hit else 1,actual_rows=0 if hit else 5155 if case=='edit' else 1048,
                completed_channel_blocks_this_request=0 if hit else 32),decode=dict(steps_this_request=40 if hit else 39,actual_rows=1024,
                completed_channel_blocks_this_request=(1280 if hit else 1248)-saved)),
            hybrid=dict(bucket=1024,runtime_calls_session_total=ffn,runtime_weight=dict(hybrid_blocks_session_total=blocks,device_io_calls_session_total=ffn,
                ane_channels=6144,gpu_channels=6144,function_rows=[],capability_calls_session_total=3)),
            qwen_private_qkv=dict(failed=False,failure_reason='',validation_passed=True,validation_calls_session_total=1,validation_relative_l2=.003,
                predictions_session_total=qkv,completed_blocks_session_total=qkv,weight_stages_session_total=qkv,fallback_blocks_session_total=0,
                gpu_blocks_session_total=gpu,program_key='fixed'),qwen_private_qkv_prefetch=dict(requested=True,enabled=True,future_pending=False))
        for key,value in dict(attempts=sub,submissions=sub,producer_submissions=3*sub,claims=claims,weight_completions=claims,
            completed_hits=claims,discards=discards,retired_claims=0,failures=0).items():row['qwen_private_qkv_prefetch'][key+'_session_total']=value
        if cache=='conservative':
            row['qwen21_dbcache']=dict(mode='conservative',front_blocks=8,back_blocks=0,warmup_steps=8,max_consecutive=1,threshold=.15,cached_steps=s,saved_middle_blocks=saved)
            row['qwen_runtime_conservative_cache']=dict(requested=True,recipe=screen.RECIPE,actual_sampler_steps=40,skipped_middle_blocks=saved,
                skip_callbacks_this_request=s,discarded_qkv_futures_this_request=s,restaged_step_heads_this_request=s,skipped_steps_this_request=list(range(9,39,2)))
        values.append(row)
    return values


class RuntimeCacheProfileTests(unittest.TestCase):
    def test_new_seed_pairs_and_legacy_defaults_are_exact(self):
        self.assertEqual(screen.seed_sequence(173,3),(173,174,173))
        self.assertEqual(screen.seed_sequence(211,4),(211,211,212,211))
        self.assertEqual(screen.seed_sequence(0,1),(0,))
        for seed,count in ((True,3),(-1,3),(1<<32,3),((1<<32)-1,3),(73,True),(73,2),(73,3.)):
            with self.assertRaises(ValueError):screen.seed_sequence(seed,count)
    def validate(self,data,case,cache):
        # The shared backend health predicates have their own tests and real
        # device windows. These mocks isolate this new layer's accounting.
        with ExitStack() as stack:
            for module,name in ((screen,'validate_results'),(screen,'validate_fp32_channel_join'),(screen,'validate_fixed_async'),
                (screen.full,'validate_fixed_program'),(screen.full,'validate_activation_receipt'),(screen.full,'validate_projection_receipt'),
                (screen.full,'validate_weight_code_cache')):stack.enter_context(patch.object(module,name))
            return screen.validate(data,'runtime',case,cache)

    def test_environment_one_explicit_runtime_flag_and_gpu_policy_unchanged(self):
        for case in ('generate','edit'):
            off=screen.environment('runtime','stock',case,'off');on=screen.environment('runtime','stock',case,'conservative')
            self.assertEqual({k for k in off if off[k]!=on[k]},{'TURBOCIDER_QWEN21_RUNTIME_CONSERVATIVE_CACHE'})
            self.assertEqual(screen.environment('gpu','stock',case,'off'),screen.environment('gpu','stock',case,'conservative'))
        for bad in ('balanced','fast'):
            with self.assertRaises(ValueError):screen.environment('runtime','stock','generate',bad)

    def test_conservative_budget_actual_phase_calls_and_healthy_retire_restaging(self):
        for case in ('generate','edit'):
            for cache in ('off','conservative'):
                coverage=self.validate(rows(case,cache),case,cache)
                self.assertEqual([r['actual_ffn_blocks'] for r in coverage],[1280 if cache=='off' else 920]*3)
        data=rows();self.assertEqual(data[0]['hybrid']['runtime_calls_session_total'],952)
        self.assertEqual(rows('edit')[0]['hybrid']['runtime_calls_session_total'],1080)

    def test_reject_false_full_coverage_and_missing_future_retirement(self):
        for mutate in (lambda r:r[0]['hybrid']['runtime_weight'].update(hybrid_blocks_session_total=1280),
            lambda r:r[1]['qwen_private_qkv_prefetch'].update(discards_session_total=0),
            lambda r:r[1]['qwen_runtime_conservative_cache'].update(restaged_step_heads_this_request=0),
            lambda r:r[1]['qwen_runtime_conservative_cache'].update(skipped_steps_this_request=list(range(8,23))),
            lambda r:r[0]['qwen_runtime_conservative_cache'].update(skipped_steps_this_request=[39]*15),
            lambda r:r[0]['qwen_runtime_prefix_snapshot'].update(published=False),
            lambda r:r[1]['qwen_runtime_prefix_snapshot'].update(published=True),
            lambda r:r[1]['qwen_private_qkv'].update(validation_passed=False),
            lambda r:r[1]['qwen_private_qkv_prefetch'].update(future_pending=True),
            lambda r:r[1]['qwen21_dbcache'].update(threshold=float('nan'))):
            data=copy.deepcopy(rows());mutate(data)
            with self.assertRaises(ValueError):self.validate(data,'generate','conservative')
        with self.assertRaises(ValueError):self.validate(rows(),'generate','off')

    def test_real_requests_keep_forty_steps_and_original_reference(self):
        a=argparse.Namespace(requests=3,case='edit',route='runtime',cache='conservative',prompt=None,output=Path('/synthetic'),
            manifest=Path('/fixed.json'),reference=Path('/original.png'))
        values=screen.requests(a)
        self.assertEqual([r['seed'] for r in values],[73,74,73]);self.assertTrue(all(r['steps']==40 for r in values))
        self.assertTrue(all(r['inputs']==[dict(kind='image',role='reference',path='/original.png')] and r['qwen21_reference_size']==1024 for r in values))
        a.requests=4
        self.assertEqual([r['seed'] for r in screen.requests(a)],[73,73,74,73])
        a.seed=307
        self.assertEqual([r['seed'] for r in screen.requests(a)],[307,307,308,307])


if __name__=='__main__':unittest.main()
