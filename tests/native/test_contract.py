import ctypes as C
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
lib = C.CDLL(str(ROOT / 'build/native/libturbocider.dylib'))
lib.tc_plan_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
lib.tc_string_free.argtypes = [C.c_void_p]
lib.tc_engine_create.argtypes = [C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
lib.tc_engine_create_model.argtypes = [C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
lib.tc_engine_free.argtypes = [C.c_void_p]
lib.tc_ltx_gemma_tokenize_json.argtypes = [
    C.c_char_p, C.c_char_p, C.c_uint32, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
]
lib.tc_ltx_gemma_inspect_json.argtypes = [
    C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
]
lib.tc_ltx_lora_preflight_json.argtypes = [
    C.c_char_p, C.c_char_p, C.c_float,
    C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
]
lib.tc_fastmetal_lora_preflight_json.argtypes = [
    C.c_char_p, C.c_char_p, C.c_float,
    C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
]
lib.tc_ltx_audio_preflight_json.argtypes = [
    C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
]
lib.tc_models_json.restype = C.c_void_p

def consume(p):
    if not p.value:return None
    value=C.string_at(p).decode();lib.tc_string_free(p);return value

def plan(r):
    out,err=C.c_void_p(),C.c_void_p()
    status=lib.tc_plan_json(json.dumps(r).encode(),C.byref(out),C.byref(err))
    a,b=consume(out),consume(err)
    return status,json.loads(a) if a else None,b

class ContractTests(unittest.TestCase):
    def test_flux(self):
        code,p,error=plan({'width':512,'height':512})
        self.assertEqual(code,0,error);self.assertTrue(p['executable'])
        self.assertEqual(p['decoded_shape'],[512,512,1])
        self.assertEqual([x['id'] for x in p['stages']],['text_encode','denoise','vae_decode','export'])

    def test_flux_text_taps_are_config_guarded_and_dead_tail_is_elided(self):
        source=(ROOT/'native/models/flux.mm').read_text()
        encoder=(ROOT/'native/models/flux_text.mm').read_text()
        self.assertIn('text_layers_=[q[@"num_hidden_layers"] intValue]',source)
        self.assertIn('text_layers_==36',source)
        self.assertIn('constexpr int output_layer=27',encoder)
        self.assertIn('for(int i=0;i<output_layer;++i)',encoder)
        self.assertIn('i==8||i==17||i==26',encoder)

    def test_flux_lora_identity_is_content_bound_and_not_ane_reused(self):
        source=(ROOT/'native/models/flux.mm').read_text()
        module=(ROOT/'native/models/flux_module.mm').read_text()
        for token in ['canonical(', 'file_size(', 'last_write_time(',
                      'sha256_file(', 'lora_hash_cache_', 'LoRA changed while it was being hashed']:
            self.assertIn(token,source)
        self.assertIn('FLUX LoRA currently requires GPU execution',module)
        code,p,error=plan({'model':'flux2-klein-4b','execution':'gpu_ane',
                           'ane_manifest':'/tmp/flux.json',
                           'loras':[{'path':'/tmp/style.safetensors','strength':0.8}]})
        self.assertNotEqual(code,0)
    def test_ltx_schedule_and_shape(self):
        code,p,error=plan({'model':'ltx-2.5-distilled','width':704,'height':448,'frames':97,'steps':11})
        self.assertEqual(code,0,error);self.assertTrue(p['executable'])
        self.assertEqual(p['decoded_shape'],[704,448,97]);self.assertEqual(p['validation'],'native_video_executor')
        self.assertEqual(p['residency'],'component_staged')
        self.assertEqual(p['weight_validation'],'checkpoint-validated-at-load')
        stages={s['id']:s for s in p['stages']}
        self.assertEqual(stages['av_stage1']['iterations'],8);self.assertEqual(stages['av_stage2']['iterations'],3)
        self.assertNotIn('audio_vae_vocoder',stages)
        self.assertEqual(stages['export']['dependencies'],['video_vae'])
        self.assertFalse(p['audio'])
        code,p,error=plan({'model':'ltx-2.5-distilled','width':704,'height':448,
                           'frames':97,'steps':11,'audio':True})
        self.assertEqual(code,0,error)
        self.assertFalse(p['executable'])
        self.assertTrue(p['audio'])
        self.assertEqual(p['audio_capability'],'latent_to_48khz_aac_candidate')
        self.assertIn('audio_vae_vocoder',{stage['id'] for stage in p['stages']})
        self.assertIn('mux',{stage['id'] for stage in p['stages']})
        code,p,error=plan({'model':'ltx-2.5-distilled','width':704,'height':448,'frames':97,'steps':11,'loras':[{'path':'/tmp/ltx.safetensors','role':'refiner','strength':0.8}]})
        self.assertEqual(code,0,error)
        self.assertEqual(p['lora_fusion'],'runtime_bake_cache')
        self.assertEqual(p['weight_validation'],'runtime-cache-or-sidecar-verified-at-execution')
        self.assertNotEqual(plan({'model':'ltx-2.5-distilled','width':704,'height':448,'frames':97,'steps':11,'loras':[{'path':'/tmp/ltx.safetensors','role':'text_encoder','strength':0.8}]})[0],0)
        self.assertNotEqual(plan({'model':'ltx-2.5-distilled','width':704,'height':448,'frames':97,'steps':11,'loras':[{'path':'/tmp/ltx.safetensors','strength':-0.8}]})[0],0)

    def test_ltx_image_to_video_contract(self):
        request={
            'model':'ltx-2.5-distilled',
            'operation':'video.image',
            'prompt':'A cinematic red fox running through a snowy forest',
            'inputs':[{'kind':'image','role':'first_frame',
                       'path':'/tmp/first-frame.png','strength':0.65}],
            'width':704,'height':448,'frames':97,'steps':11,
        }
        code,p,error=plan(request)
        self.assertEqual(code,0,error)
        self.assertEqual(p['operation'],'video.image')
        self.assertEqual(p['decoded_shape'],[704,448,97])
        self.assertEqual(p['lora_fusion'],'none')
        stages={stage['id']:stage for stage in p['stages']}
        self.assertIn('first_frame_vae_encode',stages)
        self.assertIn('first_frame_vae_encode',
                      stages['av_stage1']['dependencies'])
        for invalid in [
            {**request,'inputs':[]},
            {**request,'inputs':[{'kind':'image','role':'reference',
                                  'path':'/tmp/first-frame.png'}]},
            {**request,'inputs':[{'kind':'image','role':'first_frame',
                                  'path':'/tmp/first-frame.png','strength':1.1}]},
            {**request,'operation':'video.generate'},
        ]:
            with self.subTest(invalid=invalid):
                self.assertNotEqual(plan(invalid)[0],0)

    def test_ltx_i2v_native_path_uses_stage_specific_clean_prefixes(self):
        source=(ROOT/'native/models/ltx_session.mm').read_text()
        self.assertIn('ltx_mlx_video_vae_create_encoder',source)
        self.assertIn('ltx_mlx_video_vae_encode_pixels_bf16',source)
        self.assertIn('stage1_clean_prefix',source)
        self.assertIn('stage2_clean_prefix',source)
        self.assertIn('first_frame_strength',source)
        self.assertIn('workload.stage1_latent_width * 32u',source)
        self.assertIn('workload.stage2_latent_width * 32u',source)

    def test_ltx_resident_denoiser_cache_is_seed_independent(self):
        source=(ROOT/'native/models/ltx_session.mm').read_text()
        self.assertIn('root_(std::filesystem::absolute(root))',source)
        start=source.index('std::string denoiser_key =')
        end=source.index('if (denoiser_key != denoiser_key_)', start)
        self.assertNotIn('request.seed', source[start:end])
        self.assertIn('selected_identity', source[start:end])
        self.assertNotIn('options.seed = request.seed',source)
        self.assertIn('ltx_native_run(denoiser_.get(), 1, request.seed',source)
        self.assertIn('ltx_native_run(denoiser_.get(), 2, request.seed',source)

    def test_ltx_gpu_ane_requires_explicit_profile_and_binds_all_stages(self):
        source=(ROOT/'native/models/ltx_session.mm').read_text()
        runtime=(ROOT/'native/models/ltx_runtime/ltx_blocks.c').read_text()
        header=(ROOT/'native/models/ltx_runtime/ltx_native.h').read_text()
        self.assertIn('turbocider-ltx-ane-v1',source)
        self.assertIn('complete_ane_directory',source)
        self.assertIn('options.mlp_directories[0]',source)
        self.assertIn('options.mlp_directories[1]',source)
        self.assertIn('options.kv_directory',source)
        self.assertIn('options.qkv_directories[0]',source)
        self.assertIn('options.qkv_directories[1]',source)
        self.assertIn('request.execution == "gpu_ane"',source)
        for field in ['preload_ane_stage2', 'release_full_gpu_mlp',
                      'detach_ane_stage1', 'detach_ane_stage2',
                      'release_blocks_final_step', 'ane_kv_stage_mask',
                      'ane_mlp_fused_residual',
                      'ane_mlp_fused_adaln_pack']:
            self.assertIn(field, header)
            self.assertIn(f'options.{field}', source)
        self.assertIn('ltx_native_release_full_gpu_mlp',runtime)
        self.assertIn('detach_ane_stage(ctx->weights',runtime)
        self.assertIn('stage==2&&ctx->options.release_blocks_final_step',runtime)
        self.assertIn('loaded_v2a != 48u',runtime)
        self.assertIn('loaded_qkv != 48u',runtime)

    def test_ltx_denoiser_cache_key_binds_residency_and_final_step_lifecycle(self):
        source=(ROOT/'native/models/ltx_session.mm').read_text()
        self.assertIn(':residency=" + request.residency',source)
        self.assertIn(':release_blocks=" +',source)

    def test_ltx_component_staged_uses_isolated_video_vae_helper(self):
        session=(ROOT/'native/models/ltx_session.mm').read_text()
        build=(ROOT/'tools/native/build.sh').read_text()
        helper=ROOT/'tools/native/ltx_video_vae_decode.c'
        self.assertTrue(helper.is_file())
        self.assertIn('decode_ltx_video_isolated',session)
        self.assertIn('video_vae_isolation = "process"',session)
        self.assertIn('posix_spawn',session)
        self.assertIn('ltx-video-vae-decode',build)
        self.assertIn('ltx_mlx_video_vae_decode_tokens_bf16',
                      helper.read_text())

    def test_ltx_service_conditioning_cache_is_bound_and_observable(self):
        session=(ROOT/'native/models/ltx_session.mm').read_text()
        service=(ROOT/'services/turbociderd/service.mm').read_text()
        finalizer=(ROOT/'tools/native/ltx_video_finalizer.mm').read_text()
        for token in [
            'TURBOCIDER_LTX_CONDITIONING_CACHE_DIR',
            'ltx_conditioning_cache_location',
            'selected_checkpoint, request.prompt',
            'video_sha256', 'audio_sha256', 'mask_sha256',
            'conditioning_cache_hit', 'conditioning_mode',
        ]:
            self.assertIn(token, session)
        self.assertIn('ltx-conditioning-cache', service)
        self.assertIn('TURBOCIDER_LTX_CONDITIONING_CACHE_DIR=', service)
        self.assertIn('TURBOCIDER_LTX_PRE_FINALIZER_SECONDS', session)
        self.assertIn('request_wall_estimate', finalizer)

    def test_service_routes_ltx_external_worker_from_original_request(self):
        source=(ROOT/'services/turbociderd/service.mm').read_text()
        self.assertIn('external_ltx_request(inference)',source)
        self.assertNotIn('external_ltx_request(plan_value)',source)

    def test_service_reuses_resident_ltx_candidate_session(self):
        source=(ROOT/'services/turbociderd/service.mm').read_text()
        for token in [
            'resident_ltx_candidate_request',
            'TURBOCIDER_LTX_RESIDENT_CANDIDATE',
            'resident_candidate',
            'service_execution_path',
            'service_session_reused',
            'resident_session',
        ]:
            self.assertIn(token,source)
        self.assertIn('check([plan_value[@"executable"] boolValue] || external_worker ||',
                      source)
        self.assertIn('loaded_ == identity',source)

    def test_ltx_service_worker_execs_shared_cpp_video_finalizer(self):
        cli=(ROOT/'apps/cli/main.mm').read_text()
        service=(ROOT/'services/turbociderd/service.mm').read_text()
        session=(ROOT/'native/models/ltx_session.mm').read_text()
        build=(ROOT/'tools/native/build.sh').read_text()
        finalizer=ROOT/'tools/native/ltx_video_finalizer.mm'
        converter=ROOT/'native/models/ltx_runtime/ltx_video_convert.cpp'
        self.assertIn('TURBOCIDER_LTX_EXEC_FINALIZER',cli)
        self.assertIn('ltx_exec_finalizer_plan',cli)
        self.assertIn('TURBOCIDER_LTX_CONDITIONING_CACHE_DIR',cli)
        self.assertNotIn('TURBOCIDER_LTX_SMOKE_EXECUTOR',cli)
        self.assertIn('run_ltx_worker',service)
        self.assertIn('active_child_',service)
        self.assertIn('exec_ltx_video_finalizer',session)
        self.assertIn('ltx-video-finalizer',build)
        self.assertTrue(finalizer.is_file())
        self.assertTrue(converter.is_file())
        self.assertIn('ltx_mlx_video_vae_decode_tokens_bf16',
                      finalizer.read_text())
        self.assertIn('ltx_video_bf16_planar_to_rgb24',
                      finalizer.read_text())

    def test_ltx_ane_lifecycle_profile_is_typed_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); stage1=root/'stage1'; stage2=root/'stage2'
            for stage, rows in ((stage1,1001),(stage2,4004)):
                stage.mkdir()
                for block in range(48):
                    block_dir=stage/f'block-{block}'
                    block_dir.mkdir()
                    (block_dir/'manifest.json').write_text(json.dumps({
                        'schema':'ltx-ane-mlp-v1',
                        'block_index':block,
                        'shape':{'rows':rows},
                    }))
            profile=root/'profile.json'
            base={
                'schema':'turbocider-ltx-ane-v1',
                'variant':'int8_pc',
                'mlp_stage1':'stage1','mlp_stage2':'stage2',
                'parallel_av':True,'preload_stage2':True,
                'release_full_gpu_mlp':True,'detach_stage1':True,
                'detach_stage2':False,'release_blocks_final_step':True,
                'fused_mlp_residual':False,
                'fused_mlp_adaln_pack':False,'kv_stage_mask':1,
            }
            request={'model':'ltx-2.5-distilled','execution':'gpu_ane',
                     'ane_manifest':str(profile),'width':704,'height':448,
                     'frames':97,'steps':11}
            profile.write_text(json.dumps(base))
            code,p,error=plan(request)
            self.assertEqual(code,0,error)
            profile.write_text(json.dumps({**base,'preload_stage2':False}))
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('requires Stage-2 preload',error)
            profile.write_text(json.dumps({**base,
                'fused_mlp_residual':True,
                'fused_mlp_adaln_pack':True}))
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('cannot both be enabled',error)
            profile.write_text(json.dumps({**base,'kv_stage_mask':4}))
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('must be 0..3',error)

    def test_ltx_candidate_dumps_stage_boundary_latents(self):
        source=(ROOT/'native/models/ltx_session.mm').read_text()
        for name in ['stage1_video', 'stage1_audio', 'stage2_input_video',
                     'stage2_video', 'stage2_audio', 'metadata.json']:
            self.assertIn(name, source)
        self.assertIn('request.dump', source)

    def test_ltx_gpu_ane_plan_rejects_incomplete_artifact_tree(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); stage1=root/'stage1'; stage2=root/'stage2'
            for stage, rows in ((stage1,1001),(stage2,4004)):
                stage.mkdir()
                for block in range(48):
                    block_dir=stage/f'block-{block}'
                    block_dir.mkdir()
                    (block_dir/'manifest.json').write_text(json.dumps({
                        'schema':'ltx-ane-mlp-v1',
                        'block_index':block,
                        'shape':{'rows':rows},
                    }))
            request={'model':'ltx-2.5-distilled','execution':'gpu_ane',
                     'ane_manifest':str(root),'width':704,'height':448,
                     'frames':97,'steps':11}
            code,p,error=plan(request)
            self.assertEqual(code,0,error)
            block0=stage1/'block-0'/'manifest.json'
            block0.write_text(json.dumps({
                'schema':'ltx-ane-mlp-v1',
                'block_index':0,
                'shape':{'rows':154},
            }))
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('do not match request rows',error)
            block0.write_text(json.dumps({
                'schema':'ltx-ane-mlp-v1',
                'block_index':0,
                'shape':{'rows':1001},
            }))
            (stage2/'block-47'/'manifest.json').unlink()
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('48 block manifests',error)

    def test_ltx_conditioning_cache_is_prompt_bound(self):
        source=(ROOT/'native/models/ltx_session.mm').read_text()
        self.assertIn('conditioning_prompt',source)
        self.assertIn('root_, selected_checkpoint, request.prompt',source)
        self.assertIn('if (!bound_prompt || *bound_prompt != requested_prompt)',source)

    def test_ltx_manifest_preflight_verifies_artifact_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); lora_dir=root/'models'/'loras'; diff_dir=root/'models'/'diffusion_models'
            lora_dir.mkdir(parents=True); diff_dir.mkdir(parents=True)
            base=diff_dir/'base.safetensors'; lora=lora_dir/'adapter.safetensors'; output=diff_dir/'merged.safetensors'
            base.write_bytes(b'base checkpoint'); lora.write_bytes(b'lora adapter'); output.write_bytes(b'merged checkpoint')
            digest=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest={
                'schema':'h3-super-merged-refiner-v1',
                'algorithm':'comfy-runtime-equivalent-fp16-lora-fp32-mm-convrot-int8-v1',
                'repository':'Lightricks/LTX-2.5',
                'revision':'bf86adedf518142442575d1ce2e767b7d01c8c76',
                'base':{'filename':base.name,'bytes':base.stat().st_size,'sha256':digest(base)},
                'lora':{'filename':lora.name,'bytes':lora.stat().st_size,'sha256':digest(lora),'strength':0.8},
                'output':{'filename':output.name,'bytes':output.stat().st_size,'sha256':digest(output)},
                'mapping':{'total':2,'missing':0,'int8_convrot':1,'bf16':1},
            }
            (diff_dir/'merged.safetensors.manifest.json').write_text(json.dumps(manifest))
            out,err=C.c_void_p(),C.c_void_p()
            code=lib.tc_ltx_lora_preflight_json(str(root/'models').encode(),str(lora).encode(),C.c_float(0.8),C.byref(out),C.byref(err))
            value,failure=consume(out),consume(err)
            self.assertEqual(code,0,failure); payload=json.loads(value)
            self.assertEqual(payload['validation'],'premerged_manifest_verified')
            self.assertEqual(payload['checkpoint_sha256'],digest(output))
            out,err=C.c_void_p(),C.c_void_p()
            code=lib.tc_ltx_lora_preflight_json(str(root/'models').encode(),str(lora).encode(),C.c_float(0.7),C.byref(out),C.byref(err))
            self.assertNotEqual(code,0); self.assertTrue(consume(err)); self.assertIsNone(consume(out))
            lora.write_bytes(b'lora adaptER')
            out,err=C.c_void_p(),C.c_void_p()
            code=lib.tc_ltx_lora_preflight_json(str(root/'models').encode(),str(lora).encode(),C.c_float(0.8),C.byref(out),C.byref(err))
            self.assertNotEqual(code,0); self.assertIn('SHA-256',consume(err)); self.assertIsNone(consume(out))
    def test_h3_execution_and_validation_are_separate(self):
        code,p,error=plan({'model':'minimax-h3-turbo','frames':22,'width':512,'height':512,'steps':4})
        self.assertEqual(code,0,error);self.assertTrue(p['executable'])
        self.assertEqual(p['backend'],'h3-metal-mps')
        self.assertEqual(p['validation'],'manifest_verified_native')
        self.assertEqual(p['weight_validation'],'manifest-verified')
        code,p,error=plan({'model':'minimax-h3-turbo','frames':22,'width':512,'height':512,'steps':4,'loras':[{'path':'/tmp/h3.safetensors','strength':0.0625}]})
        self.assertEqual(code,0,error)
        self.assertEqual(p['lora_fusion'],'runtime_bake_cache')
        self.assertNotEqual(plan({'model':'minimax-h3-turbo','frames':22,'width':512,'height':512,'steps':4,'loras':[{'path':'/tmp/h3.safetensors','role':'text_encoder'}]})[0],0)

    def test_fastmetal_is_an_executable_persistent_runtime_candidate(self):
        request={'model':'fastmetal-1.3b-qad','width':832,'height':480,
                 'frames':81,'fps':16,'steps':3}
        code,p,error=plan(request)
        self.assertEqual(code,0,error)
        self.assertTrue(p['executable'])
        self.assertEqual(p['backend'],'fastmetal-mlx')
        self.assertEqual(p['validation'],'manifest_verified_python_runtime')
        self.assertEqual([s['id'] for s in p['stages']],
                         ['text_encode','dit','video_vae','export'])
        for invalid in [
            {**request,'frames':80}, {**request,'steps':4},
            {**request,'fps':24},
        ]:
            with self.subTest(invalid=invalid):
                self.assertNotEqual(plan(invalid)[0],0)
        code,p,error=plan({**request,'loras':[{'path':'/tmp/fastmetal.safetensors',
                                               'role':'transformer','strength':0.8}]})
        self.assertEqual(code,0,error)
        self.assertEqual(p['lora_fusion'],'premerged_manifest_verified')
        self.assertEqual(p['weight_validation'],'premerged-manifest-verified-at-execution')
        self.assertNotEqual(plan({**request,'loras':[{'path':'/tmp/fastmetal.safetensors',
                                                      'role':'text_encoder','strength':0.8}]})[0],0)
        self.assertNotEqual(plan({**request,'execution':'gpu_ane',
                                  'ane_manifest':'/tmp/fastmetal-ane.json',
                                  'loras':[{'path':'/tmp/fastmetal.safetensors',
                                            'role':'transformer','strength':0.8}]})[0],0)
        source=(ROOT/'native/models/fastmetal_module.mm').read_text()
        self.assertIn('PersistentWorker',source)
        self.assertIn('fastmetal-python-persistent',source)
        self.assertIn('uses_parent_mlx() const override { return false; }',source)
        self.assertIn('turbocider-fastmetal-premerged-lora-v1',source)
        self.assertIn('--mlx-checkpoint', (ROOT/'tools/native/fastmetal_worker.py').read_text())
        self.assertIn('turbocider-fastmetal-v1', (ROOT/'profiles/fastmetal.example.json').read_text())
        models=json.loads(consume(C.c_void_p(lib.tc_models_json())))['models']
        descriptor=next(value for value in models if value['id']=='fastmetal-1.3b-qad')
        self.assertTrue(descriptor['supports_lora'])
        self.assertFalse(descriptor['runtime_lora'])
        self.assertEqual(descriptor['lora_mode'],'premerged-manifest')

    def test_fastmetal_gpu_ane_manifest_requires_complete_fixed_shape_tree(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for block in range(30):
                (root/f'block{block}.mlmodelc').mkdir()
            manifest={
                'schema':'turbocider-fastmetal-ane-mlp-v1',
                'checkpoint':'/missing/fastmetal/mlx_dit.safetensors',
                'checkpoint_sha256':'a48f7370cab9664ebc71afefa6cbc2ea2ab1970b06aa9ad77712179cf52213ff',
                'mlx_dit_json_sha256':'db5603223f17a03051d36e4a3a218477dd48a7723117743d5d916d1db3f87691',
                'shape':{'rows':32760,'hidden':1536,'intermediate':8960,
                         'ane_intermediate':4096,'gpu_intermediate':4864,
                         'blocks':30,'attention_heads':12,'attention_head_dim':128},
                'variant':'int8_pc','blocks':list(range(30)),
                'artifacts':{str(i):f'block{i}.mlmodelc' for i in range(30)},
            }
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            request={'model':'fastmetal-1.3b-qad','execution':'gpu_ane',
                     'ane_manifest':str(path),'width':832,'height':480,
                     'frames':81,'fps':16,'steps':3}
            code,p,error=plan(request)
            self.assertEqual(code,0,error)
            self.assertEqual(p['backend'],'fastmetal-mlx+ane_parallel')
            (root/'block29.mlmodelc').rmdir()
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('ANE block artifact',error)

    def test_fastmetal_lora_preflight_binds_pinned_base_and_merged_checkpoint(self):
        configured=os.environ.get('TURBOCIDER_FASTMETAL_TEST_MODEL')
        fixture=(Path(configured).resolve() if configured else
                 (ROOT.parent/'gpu_ane/fastmetal-runtime/models/FastMetal-1.3B-QAD').resolve())
        base_weights=fixture/'mlx_dit.safetensors'
        base_config=fixture/'mlx_dit.json'
        if not base_weights.is_file() or not base_config.is_file():
            self.skipTest('validated FastMetal base fixture is unavailable')

        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            model=root/'model';model.mkdir()
            (model/'mlx_dit.safetensors').symlink_to(base_weights)
            (model/'mlx_dit.json').symlink_to(base_config)
            lora=root/'style.safetensors';lora.write_bytes(b'fastmetal lora fixture')
            merged=root/'merged';merged.mkdir()
            output_weights=merged/'mlx_dit.safetensors'
            output_weights.write_bytes(b'premerged FastMetal checkpoint fixture')
            output_config=merged/'mlx_dit.json'
            output_config.write_bytes(base_config.read_bytes())
            manifest={
                'schema':'turbocider-fastmetal-premerged-lora-v1',
                'algorithm':'fastvideo-mlx-runtime-equivalent-transformer-lora-premerge-v1',
                'repository':'FastVideo/FastMetal-1.3B-QAD',
                'revision':'2dac0154b217adabf8895d6cde7d6d93e68b7bec',
                'base':{
                    'weights_filename':'mlx_dit.safetensors',
                    'weights_bytes':base_weights.stat().st_size,
                    'weights_sha256':'a48f7370cab9664ebc71afefa6cbc2ea2ab1970b06aa9ad77712179cf52213ff',
                    'config_filename':'mlx_dit.json',
                    'config_bytes':base_config.stat().st_size,
                    'config_sha256':'db5603223f17a03051d36e4a3a218477dd48a7723117743d5d916d1db3f87691',
                },
                'lora':{
                    'filename':lora.name,'bytes':lora.stat().st_size,
                    'sha256':digest(lora),'role':'transformer','strength':0.8,
                    'adapter_format':'safetensors',
                },
                'output':{
                    'directory':merged.name,
                    'weights_filename':'mlx_dit.safetensors',
                    'weights_bytes':output_weights.stat().st_size,
                    'weights_sha256':digest(output_weights),
                    'config_filename':'mlx_dit.json',
                    'config_bytes':output_config.stat().st_size,
                    'config_sha256':digest(output_config),
                },
                'mapping':{'total':2,'matched':2,'missing':0},
                'shape':{'rows':32760,'hidden':1536,'intermediate':8960,
                         'ane_intermediate':4096,'gpu_intermediate':4864,
                         'blocks':30,'attention_heads':12,'attention_head_dim':128},
                'model_config':{
                    'format':'mlx_dit-v1','format_version':1,
                    'config_sha256':'db5603223f17a03051d36e4a3a218477dd48a7723117743d5d916d1db3f87691',
                    'num_blocks':30,'in_channels':16,'out_channels':16,
                    'attention_head_dim':128,'num_attention_heads':12,
                    'ffn_dim':8960,'text_dim':4096,'patch_size':[1,2,2],
                    'quantization_mode':'affine','quantization_bits':8,
                    'quantization_group_size':64,
                },
            }
            manifest_path=Path(str(lora)+'.manifest.json')
            manifest_path.write_text(json.dumps(manifest))

            def preflight(strength=0.8):
                out,err=C.c_void_p(),C.c_void_p()
                code=lib.tc_fastmetal_lora_preflight_json(
                    str(model).encode(),str(lora).encode(),C.c_float(strength),
                    C.byref(out),C.byref(err))
                value,failure=consume(out),consume(err)
                return code,json.loads(value) if value else None,failure

            code,payload,error=preflight()
            self.assertEqual(code,0,error)
            self.assertEqual(payload['validation'],'premerged_manifest_verified')
            self.assertEqual(payload['checkpoint_sha256'],digest(output_weights))
            self.assertEqual(payload['lora_sha256'],digest(lora))
            self.assertEqual(payload['lora_strength'],0.8)
            self.assertEqual(payload['execution'],'gpu')
            self.assertFalse(payload['runtime_lora'])

            code,payload,error=preflight(0.7)
            self.assertNotEqual(code,0)
            self.assertIn('strength',error)
            self.assertIsNone(payload)
            lora.write_bytes(b'tampered adapter')
            code,payload,error=preflight()
            self.assertNotEqual(code,0)
            self.assertIn('size',error)
            self.assertIsNone(payload)
            lora.write_bytes(b'fastmetal lora fixture')
            manifest['mapping']['missing']=1
            manifest_path.write_text(json.dumps(manifest))
            code,payload,error=preflight()
            self.assertNotEqual(code,0)
            self.assertIn('mapping is incomplete',error)
            self.assertIsNone(payload)
    def test_flux9_and_lora_contract(self):
        code,p,error=plan({'model':'flux2-klein-9b','width':512,'height':512,'loras':[{'path':'/tmp/style.safetensors','strength':0.8}]})
        self.assertEqual(code,0,error);self.assertTrue(p['executable']);self.assertEqual(p['lora_count'],1)
        self.assertEqual(p['lora_fusion'],'load_time_baked');self.assertGreater(p['memory_estimate_bytes'],24<<30)
        self.assertNotEqual(plan({'model':'flux2-klein-9b','execution':'gpu_ane','ane_manifest':'/tmp/a'})[0],0)

    def test_flux_single_projection_has_runtime_layout_guard(self):
        source=(ROOT/'native/models/flux_transformer.mm').read_text()
        self.assertIn('parts.size()==5',source)
        self.assertIn('parts[4].shape(-1)==hidden_*3',source)

    def test_ltx_gemma_tokenizer_matches_reference_vectors(self):
        out,err=C.c_void_p(),C.c_void_p()
        tokenizer=ROOT/'models/LTX-2.5/gemma4-12b-ltx-v1/tokenizer.json'
        if not tokenizer.is_file():
            self.skipTest('local Gemma4 tokenizer fixture is unavailable')
        code=lib.tc_ltx_gemma_tokenize_json(
            str(tokenizer).encode(),
            b'A cinematic red fox running through a snowy forest',
            0, C.byref(out), C.byref(err))
        value,failure=consume(out),consume(err)
        self.assertEqual(code,0,failure)
        self.assertEqual(json.loads(value)['ids'],
                         [2,236776,60420,2604,37423,4710,1343,496,54530,6426])

        out,err=C.c_void_p(),C.c_void_p()
        code=lib.tc_ltx_gemma_tokenize_json(
            str(tokenizer).encode(), b'hello world', 8,
            C.byref(out), C.byref(err))
        value,failure=consume(out),consume(err)
        self.assertEqual(code,0,failure)
        payload=json.loads(value)
        self.assertEqual(payload['ids'],[0,0,0,0,0,2,23391,1902])
        self.assertEqual(payload['mask'],[0,0,0,0,0,1,1,1])

    def test_ltx_gemma_checkpoint_structure(self):
        checkpoint=ROOT/'models/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors'
        if not checkpoint.is_file():
            self.skipTest('local Gemma4 ConvRot checkpoint is unavailable')
        out,err=C.c_void_p(),C.c_void_p()
        code=lib.tc_ltx_gemma_inspect_json(
            str(checkpoint).encode(),C.byref(out),C.byref(err))
        value,failure=consume(out),consume(err)
        self.assertEqual(code,0,failure)
        payload=json.loads(value)
        self.assertTrue(payload['validated'])
        self.assertEqual(payload['layers'],48)
        self.assertEqual(payload['hidden_size'],3840)
        self.assertEqual(payload['projection_input_dim'],188160)
        self.assertEqual(payload['quantization'],'int8_convrot_256')

    def test_ltx_video_executor_and_gated_capabilities(self):
        pointer=lib.tc_models_json()
        payload=json.loads(consume(C.c_void_p(pointer)))
        model=next(value for value in payload['models']
                   if value['id']=='ltx-2.5-distilled')
        self.assertTrue(model['executor'])
        self.assertEqual(model['executor_operations'],['video.generate'])
        self.assertEqual(model['default_residency'],'component_staged')
        self.assertTrue(model['native_gemma4_candidate'])
        self.assertTrue(model['native_conditioning_connector'])
        self.assertTrue(model['native_i2v_clean_prefix'])
        self.assertTrue(model['native_gpu_ane_profile'])
        self.assertTrue(model['supports_lora'])
        self.assertTrue(model['runtime_lora'])
        self.assertEqual(model['lora_mode'],'runtime-bake-cache')
        self.assertTrue(any('broader prompt-suite qualification remains pending'
                            in value for value in model['candidate_limitations']))
        self.assertTrue(any('end-to-end native Session parity still pending' in value
                            for value in model['candidate_limitations']))
        self.assertFalse(model['audio_output'])
        self.assertTrue(model['native_audio_output_candidate'])
        self.assertTrue(model['native_audio_vae_candidate'])
        self.assertTrue(model['native_base_vocoder_candidate'])
        self.assertEqual(model['audio_capability'],'latent_to_48khz_aac_candidate')
        self.assertFalse(model['default_audio'])

    def test_ltx_shared_hot_path_stays_synced_with_ltx_mac(self):
        upstream=ROOT.parent/'ltx-mac'
        if not upstream.is_dir():
            self.skipTest('sibling ltx-mac checkout is unavailable')
        shared=[
            'ltx.c','ltx_conditioning.c','ltx_connector.c',
            'ltx_transformer_io.c','ltx_latent_stats.c','ltx_rng.c',
            'ltx_shaders.metal','ltx_ane_mlp.m','ltx_ane_v2a.m',
            'ltx_ane_kv.m','ltx_ane_qkv.m','ltx_mlx_upsampler.cpp',
            'ltx_upsampler.m',
        ]
        vendored=ROOT/'native/models/ltx_runtime'
        for name in shared:
            with self.subTest(source=name):
                self.assertEqual((vendored/name).read_bytes(),
                                 (upstream/name).read_bytes())

    def test_ltx_audio_preflight_is_fail_closed_and_read_only(self):
        def preflight(root):
            out,err=C.c_void_p(),C.c_void_p()
            code=lib.tc_ltx_audio_preflight_json(str(root).encode(),
                                                 C.byref(out),C.byref(err))
            return code, json.loads(consume(out)) if out.value else None, consume(err)

        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            code,payload,error=preflight(root)
            self.assertEqual(code,0,error)
            self.assertEqual(payload['status'],'missing')
            self.assertFalse(payload['executor_ready'])
            artifact=root/'ltx-2.5-audio-vae-bf16.safetensors'
            artifact.write_bytes(b'validated audio bundle')
            code,payload,error=preflight(root)
            self.assertEqual(code,0,error)
            self.assertEqual(payload['status'],'unverified')
            self.assertFalse(payload['assets_verified'])

            digest=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest={
                'schema':'turbocider-ltx-audio-assets-v1',
                'repository':'Lightricks/LTX-2.5',
                'revision':'bf86adedf518142442575d1ce2e767b7d01c8c76',
                'artifact':{'filename':artifact.name,'bytes':artifact.stat().st_size,
                            'sha256':digest(artifact)},
                'components':{
                    'audio_vae_decoder':'audio_vae_decoder',
                    'base_vocoder':'base_vocoder',
                    'bwe_vocoder':'bwe_vocoder',
                    'mel_stft':'mel_stft',
                },
                'sample_rates':[16000,48000],
                'channels':2,
            }
            (Path(str(artifact)+'.manifest.json')).write_text(json.dumps(manifest))
            code,payload,error=preflight(root)
            self.assertEqual(code,0,error)
            self.assertEqual(payload['status'],'verified_native_candidate_session_parity_pending')
            self.assertTrue(payload['assets_verified'])
            self.assertTrue(payload['native_runtime_available'])
            self.assertTrue(payload['native_audio_candidate'])
            self.assertFalse(payload['native_audio_supported'])
            self.assertFalse(payload['executor_ready'])
            self.assertIn('end_to_end_session_audio_parity',payload['blocking_components'])
            artifact.write_bytes(b'validated audio bundlE')
            code,payload,error=preflight(root)
            self.assertEqual(code,0,error)
            self.assertEqual(payload['status'],'invalid')
            self.assertIn('SHA-256',payload['reason'])
    def test_invalid_requests(self):
        requests=[{'width':0},{'width':513},{'height':4096},{'width':'512'},{'width':True},{'steps':0},{'steps':51},{'seed':-1},{'frames':2},{'execution':'gpu_ane'},{'model':'flux2-9b'},{'dynamic_text':'yes'},{'schema_version':2},{'engine_env':{}},{'mode':'image_edit'},{'model':'ltx-2.5-distilled','steps':4},{'model':'ltx-2.5-distilled','steps':11,'frames':96}]
        for r in requests:
            with self.subTest(r=r):
                code,p,error=plan(r);self.assertNotEqual(code,0);self.assertIsNone(p);self.assertTrue(error)
    def test_invalid_json(self):
        for raw in [b'[]',b'null',b'nope',None]:
            out,err=C.c_void_p(),C.c_void_p();self.assertNotEqual(lib.tc_plan_json(raw,C.byref(out),C.byref(err)),0)
            self.assertIsNone(consume(out));self.assertTrue(consume(err))
    def test_model_missing(self):
        e,err=C.c_void_p(),C.c_void_p()
        with tempfile.TemporaryDirectory() as d:
            self.assertNotEqual(lib.tc_engine_create(d.encode(),C.byref(e),C.byref(err)),0)
            self.assertIsNone(e.value);self.assertTrue(consume(err))
    def test_ltx_video_executor_is_publicly_creatable(self):
        e,err=C.c_void_p(),C.c_void_p()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for relative in [
                'diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors',
                'latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors',
                'vae/ltx-2.5-video-vae-conv-bf16.safetensors',
            ]:
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'fixture')
            self.assertEqual(lib.tc_engine_create_model(
                b'ltx-2.5-distilled',str(root).encode(),C.byref(e),C.byref(err)),0,
                consume(err))
            self.assertTrue(e.value)
            lib.tc_engine_free(e)
    def test_multimodal_schema2(self):
        base={'schema_version':2,'model':'flux2-klein-4b','operation':'image.edit','inputs':[{'kind':'text','role':'prompt','text':'编辑图像'},{'kind':'image','role':'reference','path':'/tmp/reference.png'}],'outputs':[{'kind':'image','path':'/tmp/out.png','width':256,'height':256}]}
        code,p,error=plan(base);self.assertEqual(code,0,error)
        stages={s['id']:s for s in p['stages']};self.assertIn('image_encode',stages['denoise']['dependencies'])
        for change in [{'operation':'image.generate'},{'outputs':[{'kind':'video'}]},{'inputs':base['inputs']+[base['inputs'][0]]},{'execution':{'residency':'streamed'}},{'inputs':[{'kind':'image','role':'reference','path':'/tmp/ref','strength':1.1}]}]:
            self.assertNotEqual(plan({**base,**change})[0],0)
    def test_disabled_profile_and_hardware_guard(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'profile.json'
            path.write_text(json.dumps({'schema_version':1,'enabled':False}))
            self.assertEqual(plan({'profile':str(path)})[0],0)
            self.assertNotEqual(plan({'profile':str(path),'execution':'gpu_ane'})[0],0)
            path.write_text(json.dumps({'schema_version':1,'enabled':True,'match':{'gpu_name':'Wrong GPU','memory_bytes':1},'models':{}}))
            self.assertNotEqual(plan({'profile':str(path)})[0],0)
    def test_abi(self): self.assertEqual(lib.tc_abi_version(),1)

if __name__=='__main__':unittest.main(verbosity=2)
