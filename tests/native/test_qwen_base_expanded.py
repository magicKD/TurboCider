"""CPU-only complete Turbo/shape request accounting and source/route guards."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import tempfile

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
import expanded_collect as screen
import expanded_audit as audit


def args():return SimpleNamespace(variant='turbo',format='bf16',route='gpu',case='generate',width=512,height=512,steps=8,scenario='repeat',requests=3,
    channels=6144,mode='stock',encoder_ane=False,retain_encoder_weights=False,fresh_prompts=False,output=Path('/tmp/tc-expanded-fixture'),
    runtime_prefix_snapshot=False,runtime_text_prefix_snapshot=False,
    manifest=Path('/tmp/descriptor.json'),reference=Path('/tmp/reference.png'),encoder_manifest=None)


def report(a):
    expected=screen.requests(a);r=dict(runs=[],progress=[])
    for i,q in enumerate(expected):
        m={k:q[k] for k in ('model','model_variant','operation','width','height','steps','seed')}
        m.update(actual_denoise_steps=8,checkpoint=screen.LEAVES[a.variant,a.format],qwen21_sampling_schedule='qwen21-turbo-authored-fixed8-cfg1-v1')
        r['runs'].append(dict(index=i,metrics=m,request_wall_including_export=10.))
        r['progress'] += [dict(request=i,event=dict(phase='denoise',completed=n,total=8)) for n in range(9)]
    return r,expected


class ExpandedProfileTests(unittest.TestCase):
    def test_rank_scheduling_is_runtime_only_with_actual_adapter(self):
        a=args();a.lora_ranks='both'
        with self.assertRaises(ValueError):screen.environment(a)
        a.variant='base';a.steps=6;a.lora=Path(next(iter(screen.LORA_PINS)))
        with self.assertRaises(ValueError):screen.environment(a)
        a.route='runtime';e=screen.environment(a)
        self.assertEqual(e['TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS'],'1');self.assertEqual(e['TURBOCIDER_QWEN21_RUNTIME_SPLIT_DOWN_RANKS'],'1')
        m=dict(lora_strategy='inference_time',lora_applied_projections=227,shared_lora_ranks=dict(enabled=True,prepared_sets_this_request=192,completed_hybrid_blocks_this_request=192,completed_adapter_rank_arrays_this_request=384))
        screen.validate_lora(m,a)
        m['shared_lora_ranks']['completed_hybrid_blocks_this_request']=191
        with self.assertRaises(ValueError):screen.validate_lora(m,a)

    def test_pinned_lora_uses_same_authored6_and_exp_not_base_or_turbo8(self):
        a=args();a.variant='base';a.steps=6;a.lora=Path(next(iter(screen.LORA_PINS)))
        self.assertEqual(screen.lora_pin(a)[0],128)
        e=screen.environment(a);self.assertEqual(e['TURBOCIDER_QWEN21_LORA_SWIGLU'],'1');self.assertEqual(e['TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT'],'1')
        q=screen.requests(a);self.assertEqual(q[0]['loras'][0]['strength'],1.);self.assertEqual(q[0]['lora_strategy'],'inference_time')
        a.route='runtime';e=screen.environment(a);self.assertEqual(e['TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION'],'exp');self.assertEqual(e['TURBOCIDER_QWEN21_PRIVATE_QKV'],'0')
        a.mode='timeline';self.assertEqual(screen.environment(a)['TURBOCIDER_SOURCE_REGISTRATION_CACHE'],'')
        a.mode='stock';self.assertNotIn('TURBOCIDER_SOURCE_REGISTRATION_CACHE',screen.environment(a))
        a.steps=40
        with self.assertRaises(ValueError):screen.environment(a)
        a.steps=6;a.variant='turbo'
        with self.assertRaises(ValueError):screen.environment(a)

    def test_lora_actual_coverage_and_prefix_not_conditioning_hit(self):
        a=args();a.variant='base';a.steps=6;a.lora=Path(next(iter(screen.LORA_PINS)));a.case='edit'
        f=dict(enabled=True,completed_sampler_steps=6,completed_decode_blocks=192,complete_down_lora=True,online_independent_ranks=True,rank_math='f32',prefix_snapshot_enabled=True,prefix_snapshot_hit=True)
        m=dict(lora_strategy='inference_time',lora_applied_projections=227,qwen_lora_swiglu=f,prompt_cache_hit=False)
        m['qwen_corrected_qkv']=dict(enabled=True,recipe='decode-only-corrected-qkv-v3-original-buffers',weight_storage='original-buffers',bank_bytes=0,
            online_lora=True,completed_decode_blocks=192,completed_sampler_steps=6,rank_math='f32',prefill_changed=False)
        screen.validate_lora(m,a,1)
        m['prompt_cache_hit']=True;f['prefix_snapshot_hit']=False
        with self.assertRaises(ValueError):screen.validate_lora(m,a,1)
        f['prefix_snapshot_hit']=True;m['lora_applied_projections']=226
        with self.assertRaises(ValueError):screen.validate_lora(m,a,1)

    def test_actual_fixed_bucket_not_merely_requested_descriptor(self):
        with tempfile.TemporaryDirectory(prefix='tc-qwen-profile-geometry-') as root:
            a=args();a.manifest=Path(root)/'shape.json';a.manifest.write_text(json.dumps(dict(rows=1536)))
            h=dict(bucket=1536,hidden=4096,mlp_width=12288,output_channels=4096);screen.validate_runtime_geometry(h,a)
            for key,value in (('bucket',1024),('hidden',1024),('mlp_width',6144),('output_channels',128)):
                with self.assertRaises(ValueError):screen.validate_runtime_geometry({**h,key:value},a)

    def test_experiment_deadline_is_bounded_and_cannot_be_nan(self):
        self.assertEqual(screen.timeout_seconds(180.),180.)
        for value in (0,-1,3601,float('nan'),float('inf')):
            with self.assertRaises(ValueError):screen.timeout_seconds(value)

    def test_convrot_quantization_is_explicit_and_scope_isolation_is_actual(self):
        a=args();a.variant='base';a.format='convrot';a.route='runtime'
        for mode,(g,h,recipe) in screen.CONVROT_QUANT.items():
            a.convrot_quant=mode;flags=screen.environment(a)
            self.assertEqual(flags.get('TURBOCIDER_PRIVATE_ANE_A8_GROUP_SIZE'), '256' if g or h else None)
            self.assertEqual(flags.get('TURBOCIDER_PRIVATE_ANE_CONVROT_BF16_BOUNDARIES'), '1' if mode=='bf16-row' else None)
            w=dict(a8_group_size=g,hidden_a8_group_size=h,source_recipe=recipe+'+fp32-partial-join-v1',fp32_channel_join_enabled=True)
            screen.validate_convrot_quant(w,a)
            for key,value in (('a8_group_size',g+1),('hidden_a8_group_size',h+1),('source_recipe','wrong'),('fp32_channel_join_enabled',False)):
                bad={**w,key:value}
                with self.assertRaises(ValueError):screen.validate_convrot_quant(bad,a)
        self.assertEqual(screen.environment(a).get('TURBOCIDER_PRIVATE_ANE_A8_GROUP_SCOPE'),None)

    def test_non_convrot_and_gpu_cannot_silently_select_alternate_quantization(self):
        a=args();a.convrot_quant='both256'
        with self.assertRaises(ValueError):screen.environment(a)
        a.format='convrot'
        with self.assertRaises(ValueError):screen.environment(a)
        a.route='runtime';a.encoder_ane=True
        with self.assertRaises(ValueError):screen.environment(a)
        a.encoder_ane=False;a.convrot_quant='unsupported'
        with self.assertRaises(ValueError):screen.environment(a)

    def test_cross_mode_pair_is_explicitly_quality_only_never_speed(self):
        g=dict(variant='base',format='convrot',case='generate',scenario='repeat',requests=3,mode='stock',analysis=dict(hot_median_seconds=50.))
        a=copy.deepcopy(g);a['mode']='timeline';a['analysis']['hot_median_seconds']=1.
        with self.assertRaises(ValueError):audit.pair_fields(g,a)
        audit.pair_fields(g,a,True);self.assertIsNone(audit.pair_speed(g,a,True));self.assertIsNone(audit.pair_speed(g,a))
        a['mode']='stock';self.assertEqual(audit.pair_speed(g,a)['ratio'],50.)
        self.assertIsNone(audit.pair_speed(g,a,True))
        a['format']='bf16'
        with self.assertRaises(ValueError):audit.pair_fields(g,a,True)

    def test_turbo_is_actual8_not_base40_or_six_step_adapter(self):
        a=args();r,expected=report(a);self.assertEqual(screen.validate(r,a,expected)['completed_denoise_steps'],24)
        for change in (lambda x:x['runs'][0]['metrics'].update(actual_denoise_steps=7),
            lambda x:x['runs'][1]['metrics'].update(model_variant='qwen-image-2.1'),
            lambda x:x['runs'][1]['metrics'].update(checkpoint='qwen_image_2.1_bf16.safetensors'),
            lambda x:x['runs'][0]['metrics'].pop('qwen21_sampling_schedule')):
            bad=copy.deepcopy(r);change(bad)
            with self.assertRaises(ValueError):screen.validate(bad,a,expected)

    def test_full_denoise_completion_progress_is_not_request_echo(self):
        a=args();r,expected=report(a)
        for change in (lambda x:x['progress'].pop(),lambda x:x['progress'][1]['event'].update(completed=0),
            lambda x:x['progress'][1]['event'].update(total=40)):
            bad=copy.deepcopy(r);change(bad)
            with self.assertRaises(ValueError):screen.validate(bad,a,expected)

    def test_shape_restoration_one_session_keeps_authored_steps_and_seed(self):
        a=args();a.scenario='shapes';a.requests=4;a.mode='timeline';q=screen.requests(a)
        self.assertEqual([(x['width'],x['height']) for x in q],[(512,512),(768,512),(1024,1024),(512,512)])
        self.assertEqual({x['seed'] for x in q},{73});self.assertEqual({x['steps'] for x in q},{8})

    def test_explicit_private_fixed_flags_not_silent_gpu_or_gguf_f32_join(self):
        a=args();e=screen.environment(a);self.assertEqual(e['TURBOCIDER_QWEN21_PRIVATE_QKV'],'0')
        a.route='runtime';e=screen.environment(a);self.assertEqual(e['TURBOCIDER_PRIVATE_ANE_CHANNELS'],'6144')
        self.assertEqual(e['TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM'],'1');self.assertEqual(e['TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION'],'sigmoid-complement')
        a.variant='base';a.format='gguf';self.assertEqual(screen.environment(a)['TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN'],'0')
        a.format='convrot';e=screen.environment(a)
        self.assertEqual(e['TURBOCIDER_PRIVATE_ANE_DATA_PATH'],'convrot_w8a8')
        self.assertEqual(e['TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION'],'exp')
        self.assertEqual(e['TURBOCIDER_QWEN21_CONVROT_PACKED'],'1')

    def test_encoder_retention_and_fresh_conditioning_are_actual_not_requested(self):
        a=args();a.retain_encoder_weights=True;a.fresh_prompts=True;r,expected=report(a)
        for i,x in enumerate(r['runs']):x['metrics'].update(prompt_cache_hit=False,encoder_weight_residency=dict(enabled=True,weights_retained=True,
            loads_session_total=1,decline_reason='',weights_reused=i>0))
        screen.validate(r,a,expected)
        for change in (lambda x:x['runs'][0]['metrics']['encoder_weight_residency'].update(weights_retained=False),
            lambda x:x['runs'][1]['metrics'].update(prompt_cache_hit=True),lambda x:x['runs'][1]['metrics']['encoder_weight_residency'].update(loads_session_total=2)):
            bad=copy.deepcopy(r);change(bad)
            with self.assertRaises(ValueError):screen.validate(bad,a,expected)

    def test_request_local_fresh_encoder_must_release_and_reload_exact_source(self):
        a=args();a.fresh_prompts=True;r,expected=report(a)
        for i,x in enumerate(r['runs']):x['metrics'].update(prompt_cache_hit=False,encoder_weight_residency=dict(enabled=False,weights_retained=False,
            weights_reused=False,retained_bytes=0,loads_session_total=i+1,decline_reason='',source_bytes=17534247392))
        screen.validate(r,a,expected)
        bad=copy.deepcopy(r);bad['runs'][1]['metrics']['encoder_weight_residency']['weights_retained']=True
        with self.assertRaises(ValueError):screen.validate(bad,a,expected)


if __name__=='__main__':unittest.main()
