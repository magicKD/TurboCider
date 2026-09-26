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
lib.tc_wan_lora_preflight_json.argtypes = [
    C.c_char_p, C.c_char_p, C.c_float,
    C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
]
lib.tc_ltx_audio_preflight_json.argtypes = [
    C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
]
lib.tc_models_json.restype = C.c_void_p
lib.tc_system_json.restype = C.c_void_p

def consume(p):
    if not p.value:return None
    value=C.string_at(p).decode();lib.tc_string_free(p);return value

def plan(r):
    out,err=C.c_void_p(),C.c_void_p()
    status=lib.tc_plan_json(json.dumps(r).encode(),C.byref(out),C.byref(err))
    a,b=consume(out),consume(err)
    return status,json.loads(a) if a else None,b

class ContractTests(unittest.TestCase):
    def test_qwen21_canvas_memory_estimate_covers_recorded_peak(self):
        # Planning only; does not perform high-resolution inference.
        request = dict(model='qwen-image-2.1', operation='image.generate', prompt='A teapot',
                       width=2048, height=2048, steps=1, audio=False, frames=1,
                       execution='gpu', residency='component_staged')
        code, result, error = plan(request)
        self.assertEqual(code, 0, error)
        self.assertGreater(result['memory_estimate_bytes'], 62259019410)
        self.assertEqual(result['memory_estimate_kind'], 'conservative_heuristic_not_hard_limit')

    def test_qwen21_reference_memory_estimate_includes_prefix_kv(self):
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A teapot',
                    width=512, height=512, steps=1, audio=False, frames=1, execution='gpu')
        per_reference = 2 * 32 * 4096 * 4096 * 2 + (512 << 20)
        for residency in ('component_staged', 'resident'):
            request = {**base, 'residency': residency}
            code, empty, error = plan(request)
            self.assertEqual(code, 0, error)
            for count in (1, 3, 10):
                with self.subTest(residency=residency, references=count):
                    refs = [dict(kind='image', role='reference', path=f'/tmp/ref-{i}.png')
                            for i in range(count)]
                    code, result, error = plan({**request, 'operation':'image.edit', 'inputs':refs})
                    self.assertEqual(code, 0, error)
                    self.assertEqual(result['memory_estimate_bytes'],
                                     empty['memory_estimate_bytes'] + count * per_reference)
                    self.assertEqual(result['memory_estimate_kind'], 'conservative_heuristic_not_hard_limit')
                    if count == 10:
                        # Actual 512-square ten-reference MLX peak; not total
                        # process memory and not a general upper-bound proof.
                        self.assertGreater(result['memory_estimate_bytes'], 44596678034)

    def test_qwen21_experimental_pe_edit_contract(self):
        refs = [dict(kind='image', role='reference', path=f'/tmp/pe-ref-{i}.png') for i in range(10)]
        request = dict(model='qwen-image-2.1', operation='image.edit', prompt='Make it matte',
                       inputs=refs, width=512, height=512, steps=40, audio=False, frames=1,
                       prompt_enhance=True, prompt_enhancer_path='/tmp/pe-i2i-not-loaded',
                       prompt_enhance_edit_experimental=True)
        code, result, error = plan(request)
        self.assertEqual(code, 0, error)
        self.assertTrue(result['prompt_enhance_edit_experimental'])
        self.assertIn('prompt_enhance', next(stage for stage in result['stages']
                                           if stage['id'] == 'text_encode')['dependencies'])
        version2 = dict(schema_version=2, model=request['model'], operation='image.edit',
                        inputs=[dict(kind='text', role='prompt', text=request['prompt']), *refs],
                        outputs=[dict(kind='image', path='/tmp/pe-edit-unused.png', width=512,
                                      height=512, frames=1, audio=False)],
                        sampling=dict(seed=42, steps=40), parameters=dict(prompt_enhance=True,
                            prompt_enhancer_path=request['prompt_enhancer_path'],
                            prompt_enhance_edit_experimental=True))
        code, result2, error = plan(version2)
        self.assertEqual(code, 0, error)
        self.assertTrue(result2['prompt_enhance_edit_experimental'])
        self.assertEqual(result2['stages'], result['stages'])
        for invalid in [dict(prompt_enhance=False), dict(prompt_enhancer_path=''),
                        dict(prompt_enhance_edit_experimental=False),
                        dict(prompt_enhance_edit_experimental='true'), dict(inputs=[]),
                        dict(inputs=refs+[refs[0]]), dict(model='z-image-turbo'),
                        dict(operation='image.generate', inputs=[]),
                        dict(execution='gpu_ane', allow_approximation=True, ane_manifest='/tmp/no.json')]:
            with self.subTest(invalid=invalid):
                self.assertNotEqual(plan({**request, **invalid})[0], 0)
        version2['parameters']['prompt_enhance_edit_experimental'] = 'true'
        self.assertNotEqual(plan(version2)[0], 0)

    def test_qwen21_prompt_enhancer_contract(self):
        request = dict(model='qwen-image-2.1', operation='image.generate', prompt='A teapot',
                       width=512, height=512, steps=1, audio=False, frames=1,
                       prompt_enhance=True, prompt_enhancer_path='/tmp/pe-t2i-test-not-loaded')
        code, result, error = plan(request)
        self.assertEqual(code, 0, error)
        self.assertTrue(result['prompt_enhance'])
        self.assertEqual(result['prompt_enhancer_path'], request['prompt_enhancer_path'])
        self.assertEqual(result['stages'][0]['id'], 'prompt_enhance')
        self.assertIn('prompt_enhance', next(stage for stage in result['stages']
                                           if stage['id'] == 'text_encode')['dependencies'])
        version2 = dict(schema_version=2, model=request['model'], operation='image.generate',
                        inputs=[dict(kind='text', role='prompt', text=request['prompt'])],
                        outputs=[dict(kind='image', path='/tmp/pe-unused.png', width=512,
                                      height=512, frames=1, audio=False)],
                        sampling=dict(seed=42, steps=1),
                        parameters=dict(prompt_enhance=True,
                                        prompt_enhancer_path=request['prompt_enhancer_path']))
        code2, result2, error2 = plan(version2)
        self.assertEqual(code2, 0, error2)
        self.assertTrue(result2['prompt_enhance'])
        self.assertEqual(result2['stages'], result['stages'])
        for invalid in [dict(prompt_enhancer_path=''), dict(model='z-image-turbo'),
                        dict(operation='image.edit', inputs=[dict(kind='image', role='reference', path='/tmp/ref.png')]),
                        dict(prompt_enhance='true')]:
            self.assertNotEqual(plan({**request, **invalid})[0], 0, invalid)
        disabled = {**request, 'prompt_enhance':False, 'prompt_enhancer_path':''}
        self.assertEqual(plan(disabled)[0], 0)

    def test_qwen21_contract(self):
        request = dict(model='qwen-image-2.1', operation='image.generate', prompt='A teapot',
                       width=512, height=512, steps=40, audio=False, frames=1)
        code, result, error = plan(request)
        self.assertEqual(code, 0, error)
        self.assertEqual(result['residency'], 'component_staged')
        for width, height in [(2048,2048),(2400,1792),(1792,2400),(2528,1696),
                              (1696,2528),(2752,1536),(1536,2752)]:
            self.assertEqual(plan({**request, 'width':width, 'height':height})[0], 0)
        refs = [dict(kind='image', role='reference', path=f'/tmp/qwen-ref-{i}.png') for i in range(10)]
        # Planning validates role/count, not media existence (decoding is a run-time check).
        edit = {**request, 'operation':'image.edit', 'inputs':refs}
        code, result, error = plan(edit)
        self.assertEqual(code, 0, error)
        hybrid = {**request, 'execution':'gpu_ane', 'allow_approximation':True,
                  'ane_manifest':'/tmp/qwen-ane-not-loaded-during-planning.json'}
        code, result, error = plan(hybrid)
        self.assertEqual(code, 0, error)
        self.assertEqual(result['precision'], 'bf16_gpu+fp16_mlp_fp16_io')
        self.assertEqual(result['gpu_graph'], 'qwen21_decode_mlp_complement')
        self.assertEqual(result['algorithm_approximations'], ['qwen21_decode_mlp_fp16_partition'])
        for invalid in [dict(width=1024), dict(operation='image.edit', inputs=refs[:1]),
                        dict(encoder_ane_manifest='/tmp/encoder.json'), dict(allow_approximation=False),
                        dict(ane_manifest='')]:
            self.assertNotEqual(plan({**hybrid, **invalid})[0], 0, invalid)
        for invalid in [dict(width=528), dict(width=4096, height=4096), dict(frames=2), dict(audio=True),
                        dict(residency='streamed'), dict(operation='image.transform'), dict(inputs=refs),
                        dict(operation='image.edit',inputs=[]), dict(operation='image.edit',inputs=refs+[refs[0]]),
                        dict(operation='image.edit',inputs=[{**refs[0],'role':'init_image'}]),
                        # Visual masks remain ordered references, not alpha injection.
                        dict(operation='image.edit',inputs=[refs[0],{**refs[1],'role':'mask'}]),
                        dict(execution='gpu',ane_manifest='/tmp/no-qwen-ane.json')]:
            self.assertNotEqual(plan({**request, **invalid})[0], 0, invalid)

    def test_device_optimization_profile(self):
        system = json.loads(consume(C.c_void_p(lib.tc_system_json())))
        expected = system['gpu'] == 'Apple M5 Pro' and system['physical_memory_bytes'] == 24 << 30
        profile = system['optimization_profile']
        self.assertEqual(profile['id'], 'm5pro24-v1' if expected else 'legacy')
        for flag in ['z_image_suffix_streaming', 'z_image_hybrid_segments',
                     'z_image_memory_lifecycle', 'z_image_smallest_partition',
                     'external_automatic_partitions', 'coreml_output_copy', 'z_image_int8_streaming']:
            self.assertIs(profile[flag], expected)

    def test_z_image_streaming_contract(self):
        request = dict(model='z-image-turbo', operation='image.generate',
                       prompt='A red fox', width=512, height=512, frames=1,
                       steps=8, audio=False, execution='gpu', residency='streamed',
                       memory_budget_bytes=10 << 30)
        code, result, error = plan(request)
        self.assertEqual(code, 0, error)
        self.assertEqual(result['residency'], 'streamed')
        self.assertEqual(result['memory_budget_bytes'], 10 << 30)
        hybrid = {**request, 'execution': 'gpu_ane', 'allow_approximation': True,
                  'ane_manifest': '/tmp/z-image-compiled.json'}
        code, result, error = plan(hybrid)
        system = json.loads(consume(C.c_void_p(lib.tc_system_json())))
        if system['optimization_profile']['z_image_suffix_streaming']:
            self.assertEqual(code, 0, error)
            self.assertEqual(result['execution'], 'gpu_ane_experimental')
        else:
            self.assertNotEqual(code, 0)
            self.assertIn('M5 Pro 24 GiB', error)
        # A matching user profile selects execution parameters but cannot
        # enable device optimizations that the native whitelist rejects.
        with tempfile.TemporaryDirectory() as folder:
            profile = Path(folder) / 'profile.json'
            profile.write_text(json.dumps({
                'schema_version': 1, 'enabled': True,
                'match': {'gpu_name': system['gpu'], 'memory_bytes': system['physical_memory_bytes']},
                'models': {'z-image-turbo': {
                    'policy': 'gpu_ane', 'residency': 'streamed', 'allow_approximation': True,
                    'ane_manifest': '/tmp/z-image-compiled.json', 'memory_budget_bytes': 10 << 30}}
            }))
            profile_code, _, profile_error = plan({**request, 'profile': str(profile)})
            self.assertEqual(profile_code == 0,
                             system['optimization_profile']['z_image_suffix_streaming'], profile_error)
            config = json.loads(profile.read_text())
            for field in ('z_image_int8_streaming', 'z_image_suffix_streaming'):
                injected = json.loads(json.dumps(config))
                injected['models']['z-image-turbo'][field] = True
                profile.write_text(json.dumps(injected))
                code, _, error = plan({**request, 'profile': str(profile)})
                self.assertNotEqual(code, 0)
                self.assertIn('unknown profile field', error)
        self.assertNotEqual(plan({**hybrid, 'allow_approximation': False})[0], 0)
        self.assertNotEqual(plan({**hybrid, 'loras': [dict(
            path='/tmp/style.safetensors', strength=1, role='transformer')]})[0], 0)
        for change in [dict(execution='auto'), dict(memory_budget_bytes=1 << 30),
                       dict(streaming_offload=True), dict(residency='component_staged'),
                       dict(loras=[dict(path='/tmp/style.safetensors', strength=1, role='transformer')])]:
            self.assertNotEqual(plan({**request, **change})[0], 0, change)
        schema2 = dict(schema_version=2, model='z-image-turbo', operation='image.generate',
                       inputs=[dict(kind='text', role='prompt', text='A red fox')],
                       outputs=[dict(kind='image', path='/tmp/z-stream.png', width=512, height=512)],
                       sampling=dict(steps=8, seed=42),
                       execution=dict(policy='gpu', residency='streamed', memory_budget_bytes=8 << 30))
        code, result, error = plan(schema2)
        self.assertEqual(code, 0, error)
        self.assertEqual(result['residency'], 'streamed')
        self.assertEqual(result['memory_budget_bytes'], 8 << 30)

    def test_ltx_sparse_patterns_are_explicit_stage2_only(self):
        request = {
            'model': 'ltx-2.5-distilled', 'width': 768, 'height': 448,
            'frames': 121, 'steps': 11, 'execution': 'gpu',
            'ltx_backend': 'c_metal', 'ltx_sol_stage2': True,
            'allow_approximation': True, 'ltx_sparse_mode': 2,
            'ltx_sparse_radius': 1, 'ltx_sparse_anchor_stride': 16,
            'ltx_sparse_tokens_per_frame': 336,
        }
        status, result, error = plan(request)
        self.assertEqual(status, 0, error)
        for key in ('ltx_sparse_mode', 'ltx_sparse_radius',
                    'ltx_sparse_anchor_stride', 'ltx_sparse_tokens_per_frame'):
            self.assertEqual(result[key], request[key])
        topk = {**request, 'ltx_sparse_mode': 4, 'ltx_sparse_keep_blocks': 32}
        status, result, error = plan(topk)
        self.assertEqual(status, 0, error)
        self.assertEqual(result['ltx_sparse_keep_blocks'], 32)
        status, result, error = plan({**topk, 'ltx_sparse_mode': 5})
        self.assertEqual(status, 0, error)
        self.assertEqual(result['ltx_sparse_mode'], 5)
        schema2 = {
            'schema_version': 2,
            'model': 'ltx-2.5-distilled',
            'operation': 'video.generate',
            'inputs': [{'kind': 'text', 'role': 'prompt', 'text': 'A red fox'}],
            'outputs': [{'kind': 'video', 'path': '/tmp/ltx-sparse.mp4',
                         'width': 768, 'height': 448, 'frames': 121,
                         'fps': 24, 'audio': False}],
            'sampling': {'seed': 42, 'steps': 11},
            'execution': {
                'policy': 'gpu', 'residency': 'component_staged',
                'allow_approximation': True, 'ltx_backend': 'c_metal',
                'ltx_sol_stage2': True, 'ltx_sparse_mode': 5,
                'ltx_sparse_radius': 0, 'ltx_sparse_keep_blocks': 32,
                'ltx_sparse_tokens_per_frame': 336,
            },
        }
        status, result, error = plan(schema2)
        self.assertEqual(status, 0, error)
        self.assertEqual(result['ltx_sparse_mode'], 5)
        self.assertEqual(result['ltx_sparse_keep_blocks'], 32)
        self.assertNotEqual(plan({**topk, 'ltx_sparse_mode': 5,
                                 'ltx_sparse_keep_blocks': 0})[0], 0)
        self.assertNotEqual(plan({**topk, 'ltx_sparse_keep_blocks': 257})[0], 0)
        self.assertNotEqual(plan({**request, 'ltx_sparse_keep_blocks': 32})[0], 0)
        for change in (
            {'allow_approximation': False}, {'ltx_sol_stage2': False},
            {'ltx_sol_stage1': True}, {'ltx_sparse_mode': 4},
            {'ltx_sparse_radius': -1}, {'ltx_sparse_anchor_stride': 257},
            {'ltx_sparse_tokens_per_frame': 335}, {'ltx_backend': 'cpp_mlx'},
            {'ltx_sparse_mode': 0},
        ):
            with self.subTest(change=change):
                self.assertNotEqual(plan({**request, **change})[0], 0)

    def test_z_image_step_range_and_defaults(self):
        for model in ['z-image-turbo', 'z-image-turbo-gguf']:
            request = {'model': model, 'width': 512, 'height': 512,
                       'frames': 1, 'audio': False, 'execution': 'gpu'}
            code, configured, error = plan(request)
            self.assertEqual(code, 0, error)
            self.assertEqual(configured['stages'][1]['iterations'], 9)
            for steps in [1, 8, 9, 20, 50]:
                code, configured, error = plan({**request, 'steps': steps})
                self.assertEqual(code, 0, error)
                self.assertEqual(configured['stages'][1]['iterations'], steps)
            for steps in [0, 51]:
                self.assertNotEqual(plan({**request, 'steps': steps})[0], 0)

    def test_flux(self):
        code,p,error=plan({'width':512,'height':512})
        self.assertEqual(code,0,error);self.assertTrue(p['executable'])
        self.assertEqual(p['decoded_shape'],[512,512,1])
        self.assertEqual([x['id'] for x in p['stages']],['text_encode','denoise','vae_decode','export'])

    def test_z_image_native_contract_and_separate_lora(self):
        request={
            'model':'z-image-turbo', 'operation':'image.generate',
            'prompt':'A red fox in snow', 'width':1024, 'height':1024,
            'frames':1, 'steps':9, 'audio':False, 'execution':'auto',
            'noise_path':'/tmp/z-image-shared-noise.safetensors',
            'loras':[{'path':'/tmp/z-image-style.safetensors',
                      'role':'transformer','strength':0.7}],
        }
        code,p,error=plan(request)
        self.assertEqual(code,0,error)
        self.assertTrue(p['executable'])
        self.assertEqual(p['decoded_shape'],[1024,1024,1])
        self.assertEqual(p['validation'],'native_candidate')
        self.assertEqual(p['backend'],'mlx_cpp_metal')
        self.assertEqual(p['lora_strategy'],'in_memory_merge')
        self.assertEqual(p['lora_fusion'],'in_memory_delta')
        self.assertEqual(p['weight_validation'],
                         'in-memory-lora; comfy-oracle-validated')
        self.assertEqual([stage['id'] for stage in p['stages']],
                         ['text_encode','denoise','vae_decode','export'])
        self.assertEqual(p['stages'][1]['iterations'],9)
        for steps in [1, 8, 9, 20, 50]:
            code, configured, error = plan({**request, 'steps': steps})
            self.assertEqual(code, 0, error)
            self.assertEqual(configured['stages'][1]['iterations'], steps)
        for steps in [0, 51]:
            self.assertNotEqual(plan({**request, 'steps': steps})[0], 0)
        self.assertNotEqual(plan({**request,'operation':'image.edit'})[0],0)
        code,_,error=plan({**request,'execution':'gpu_ane',
                           'allow_approximation':True,
                           'ane_manifest':'/tmp/z-image.json'})
        self.assertEqual(code,0,error)
        self.assertNotEqual(plan({**request,'loras':[{
            'path':'/tmp/z-image-style.safetensors',
            'role':'text_encoder','strength':0.7}]})[0],0)
        schema2={
            'schema_version':2,
            'model':'z-image-turbo',
            'operation':'image.generate',
            'inputs':[{'kind':'text','role':'prompt','text':'A red fox in snow'}],
            'outputs':[{'kind':'image','path':'/tmp/z-image.png',
                        'width':1024,'height':1024,'frames':1,'audio':False}],
            'sampling':{'seed':42,'steps':9},
            'execution':{'policy':'gpu','residency':'resident'},
            'parameters':{'noise_path':'/tmp/z-image-shared-noise.safetensors'},
        }
        code,p,error=plan(schema2)
        self.assertEqual(code,0,error)
        self.assertTrue(p['executable'])

    def test_z_image_gguf_resident_gpu_and_request_time_lora_contract(self):
        request={
            'model':'z-image-turbo-gguf', 'operation':'image.generate',
            'prompt':'A red fox in snow', 'output':'/tmp/z-image-gguf.png',
            'width':1024, 'height':1024, 'frames':1, 'steps':9,
            'audio':False, 'execution':'gpu', 'model_variant':'Q8_0',
            'loras':[{'path':'/tmp/z-image-style.safetensors',
                      'role':'transformer','strength':0.7}],
        }
        code,p,error=plan(request)
        self.assertEqual(code,0,error)
        self.assertTrue(p['executable'])
        self.assertEqual(p['backend'],'mlx_cpp_metal_gguf')
        self.assertEqual(p['gpu_graph'],'native_quantized_blocks')
        self.assertEqual(p['precision'],'checkpoint_defined_gguf')
        self.assertEqual(p['lora_strategy'],'inference_time')
        self.assertEqual(p['lora_fusion'],'inference_time_low_rank')
        self.assertEqual(p['validation'],'native_gguf_candidate')
        self.assertEqual([stage['id'] for stage in p['stages']],
                         ['text_encode','denoise','vae_decode','export'])
        self.assertEqual(p['stages'][1]['iterations'],9)
        self.assertEqual(plan({**request,'steps':8})[0],0)
        hybrid={**request,'model_variant':'Q8_0','execution':'gpu_ane',
                'allow_approximation':True,
                'ane_manifest':'/tmp/z-image-gguf.json',
                'lora_strategy':'in_memory_merge'}
        code,p,error=plan(hybrid)
        self.assertEqual(code,0,error)
        self.assertEqual(p['execution'],'gpu_ane_experimental')
        self.assertEqual(p['lora_strategy'],'in_memory_merge')
        self.assertEqual(p['lora_fusion'],'in_memory_delta')
        self.assertNotEqual(plan({**hybrid,'allow_approximation':False})[0],0)
        self.assertNotEqual(plan({**request,'loras':[{
            'path':'/tmp/z-image-style.safetensors',
            'role':'text_encoder','strength':0.7}]})[0],0)
        session=(ROOT/'native/models/z_image/gguf.cpp').read_text()
        self.assertNotIn('NSTask',session)
        self.assertNotIn('getenv',session)
        self.assertIn('validate_native_gguf',session)
        streaming={**request, 'residency':'streaming', 'streaming_offload':True,
                   'memory_budget_bytes':8*(1<<30), 'loras':[]}
        self.assertNotEqual(plan(streaming)[0],0)
        self.assertNotEqual(plan({**request,'streaming_offload':True})[0],0)
        models=json.loads(consume(C.c_void_p(lib.tc_models_json())))
        descriptor=next(d for d in models['models'] if d['id']=='z-image-turbo-gguf')
        self.assertEqual(descriptor['backend'],'mlx_cpp_metal_gguf')
        self.assertEqual(descriptor['runtime_dependency'],'bundled-native-mlx-cpp')

    def test_z_image_comfy_sampler_precision_contract(self):
        source=(ROOT/'native/models/z_image/z_image.cpp').read_text()
        self.assertIn('training_steps - (i * training_steps / steps)',source)
        self.assertIn('return mx::astype(noise, mx::float32);',source)
        self.assertIn('auto model_input = mx::astype(latent, mx::bfloat16);',source)
        self.assertIn('z_vae_decode(mx::astype(latent, mx::bfloat16)',source)
        self.assertIn('result.lora_applied_projections = lora_applied_projections_;',source)
        self.assertIn('mlx_cpp_metal_convrot_packed_q8',source)
        results=(ROOT/'native/platform/apple/results.mm').read_text()
        self.assertIn('@"lora_applied_projections"',results)

    def test_in_memory_lora_failure_discards_partially_modified_weights(self):
        source=(ROOT/'native/backends/mlx.cpp').read_text()
        begin=source.index('size_t Weights::apply_loras(')
        end=source.index('Tensor linear(',begin)
        implementation=source[begin:end]
        self.assertIn('std::atomic<bool> &cancelled,\n                            bool inference_time) try {',implementation)
        self.assertIn('catch (...) {',implementation)
        self.assertIn('clear();',implementation)
        self.assertIn('throw;',implementation)

    def test_coreml_ffn_bridge_reinterprets_fp16_bits(self):
        header=(ROOT/'bindings/c/include/turbocider/turbocider.h').read_text()
        source=(ROOT/'native/api/c_api.mm').read_text()
        benchmark=(ROOT/'tools/native/benchmark_coreml_ffn_bridge.py').read_text()
        self.assertIn('tc_coreml_ffn_create',header)
        self.assertIn('tc_coreml_ffn_predict',header)
        self.assertIn('const_cast<uint16_t *>(input)',source)
        self.assertIn('tc::mx::float16, [](void *) {}',source)
        self.assertNotIn('tc::Tensor(input, {1, rows, metrics.hidden}',source)
        compile(benchmark,str(ROOT/'tools/native/benchmark_coreml_ffn_bridge.py'),'exec')

    def test_coreml_overhead_telemetry_separates_prepare_and_runtime_phases(self):
        metrics=(ROOT/'native/runtime/session.hpp').read_text()
        coreml=(ROOT/'native/backends/coreml.mm').read_text()
        results=(ROOT/'native/platform/apple/results.mm').read_text()
        for token in [
            'manifest_validation_seconds',
            'output_backing_setup_seconds',
            'model_load_seconds',
            'model_interface_setup_seconds',
            'zero_input_warmup_seconds',
            'first_runtime_prediction_seconds',
            'subsequent_runtime_prediction_seconds',
        ]:
            self.assertIn(token,metrics)
            self.assertIn(token,coreml)
        for key in [
            '@"manifest_validation_seconds"',
            '@"output_backing_setup_seconds"',
            '@"model_load_seconds"',
            '@"model_interface_setup_seconds"',
            '@"zero_input_warmup_seconds"',
            '@"first_runtime_prediction_seconds_session_total"',
            '@"subsequent_runtime_prediction_seconds_session_total"',
        ]:
            self.assertIn(key,results)
        self.assertIn('predict(input, rows, true)',coreml)
        code,p,error=plan({
            'model':'z-image-turbo','width':256,'height':256,'steps':9,
            'audio':False,'execution':'gpu_ane','allow_approximation':True,
            'ane_manifest':'/tmp/z.json','warmup_iterations':1,
        })
        self.assertEqual(code,0,error)
        code,p,error=plan({
            'schema_version':2,'model':'z-image-turbo',
            'operation':'image.generate',
            'inputs':[{'kind':'text','role':'prompt','text':'fox'}],
            'outputs':[{'kind':'image','path':'/tmp/z.png','width':256,
                        'height':256,'frames':1,'audio':False}],
            'sampling':{'seed':42,'steps':9},
            'execution':{'policy':'gpu_ane','allow_approximation':True,
                         'ane_manifest':'/tmp/z.json','warmup_iterations':1},
        })
        self.assertEqual(code,0,error)
        self.assertNotEqual(plan({
            'model':'z-image-turbo','width':256,'height':256,'steps':9,
            'audio':False,'warmup_iterations':9,
        })[0],0)

    def test_z_image_and_llada_prepare_have_load_only_paths(self):
        z_header=(ROOT/'native/models/z_image/z_image.hpp').read_text()
        z_source=(ROOT/'native/models/z_image/z_image.cpp').read_text()
        llada=(ROOT/'native/models/llada/llada.cpp').read_text()
        llada_session=(ROOT/'native/platform/apple/llada_session.mm').read_text()
        self.assertIn('bool load_only = false',z_header)
        self.assertIn('run(requested, event, cancelled, warmup, !warmup)',z_source)
        self.assertIn('if (load_only) {',z_source)
        self.assertIn('result.prepared = true;',z_source)
        self.assertIn('bool load_only = false',llada)
        self.assertIn('run(request, event, cancelled, warmup, !warmup)',llada)
        self.assertIn('if (load_only) {',llada)
        self.assertIn('return native().prepare(request, warmup, event, cancelled);',
                      llada_session)

    def test_z_image_sharded_hybrid_provenance_contract(self):
        source=(ROOT/'native/models/z_image/z_image.cpp').read_text()
        coreml=(ROOT/'native/backends/coreml.mm').read_text()
        resources=(ROOT/'native/backends/coreml_resources.mm').read_text()
        exporter=(ROOT/'tools/coreml/export_z_image.py').read_text()
        mlx=(ROOT/'native/backends/mlx.cpp').read_text()
        self.assertIn('diffusion_pytorch_model.safetensors.index.json',source)
        self.assertIn('transformer_checkpoint_ = std::move(diffusers_index)',source)
        self.assertIn('checkpoint_shards',coreml)
        self.assertIn('std::filesystem::file_size(checkpoint)',coreml)
        self.assertIn('sha256_file(checkpoint)',coreml)
        self.assertIn('diffusion_pytorch_model.safetensors.index.json',exporter)
        self.assertIn('class SafetensorsSource',exporter)
        self.assertIn('self.weight_map.get(name)',exporter)
        self.assertIn('checkpoint_shards',exporter)
        self.assertIn('!f.is_symlink()',mlx)
        self.assertIn('f.is_regular_file()',mlx)

    def test_coreml_export_is_offline_only(self):
        lib.tc_coreml_resources_json.argtypes = [C.c_char_p, C.c_void_p,
            C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        out, err = C.c_void_p(), C.c_void_p()
        status = lib.tc_coreml_resources_json(b'{"action":"export"}',
            None, None, C.byref(out), C.byref(err))
        result, error = consume(out), consume(err)
        self.assertNotEqual(status, 0)
        self.assertIsNone(result)
        self.assertIn('offline-only', error)
        resources=(ROOT/'native/backends/coreml_resources.mm').read_text()
        self.assertIn('Core ML export is offline-only',resources)
        self.assertNotIn('NSTask',resources)
        self.assertNotIn('export_arguments',resources)
        for name in ['export_flux2.py', 'export_z_image.py']:
            exporter=(ROOT/'tools/coreml'/name).read_text()
            self.assertIn('lora_records',exporter)
            self.assertIn('export identity changed',exporter)

    def test_flux_text_taps_are_config_guarded_and_dead_tail_is_elided(self):
        platform=(ROOT/'native/platform/apple/device.mm').read_text()
        encoder=(ROOT/'native/models/flux2/flux_text.cpp').read_text()
        shared=(ROOT/'native/components/text/qwen3.cpp').read_text()
        header=(ROOT/'native/components/text/qwen3.hpp').read_text()
        self.assertIn('[q[@"num_hidden_layers"] intValue] == 36',platform)
        self.assertIn('[q[@"num_attention_heads"] intValue] == 32',platform)
        self.assertIn('[q[@"num_key_value_heads"] intValue] == 8',platform)
        # FLUX and Z-Image share the Qwen3 implementation; the consumer only
        # selects the tap policy.  The loop is bounded by the final requested
        # tap, so Klein does not execute the dead tail of the checkpoint.
        self.assertIn('Qwen3Conditioning::flux_klein()', encoder)
        self.assertIn('config.output_layers = {8, 17, 26}', shared)
        self.assertIn('const int layers = config.output_layers.back() + 1', shared)
        self.assertIn('std::vector<int> output_layers', header)

    def test_flux_prepare_returns_resolved_execution_plan(self):
        source=(ROOT/'native/models/flux2/pipeline.cpp').read_text()
        prepare=source[source.index('RunResult Flux::prepare('):
                       source.index('RunResult Flux::generate(', source.index('RunResult Flux::prepare('))]
        self.assertIn('plan = make_plan(r);', prepare)
        self.assertIn('result.plan = std::move(plan);', prepare)

    def test_flux_lora_identity_is_content_bound_and_requires_bound_ane_artifact(self):
        source=(ROOT/'native/models/flux2/pipeline.cpp').read_text()
        module=(ROOT/'native/models/flux_module.cpp').read_text()
        coreml=(ROOT/'native/backends/coreml.mm').read_text()
        for token in ['canonical(', 'file_size(', 'last_write_time(',
                      'sha256_file(', 'lora_hash_cache_', 'LoRA changed while it was being hashed']:
            self.assertIn(token,source)
        self.assertNotIn('FLUX LoRA currently requires GPU execution',module)
        self.assertIn('active_loras_', source)
        self.assertNotIn('automatic GPU+ANE remains disabled until a LoRA-bound artifact passes the performance gate', source)
        self.assertIn('Core ML artifact does not match the active LoRA set', coreml)
        self.assertIn('Core ML LoRA SHA-256 mismatch', coreml)
        code,p,error=plan({'model':'flux2-klein-4b','execution':'gpu_ane',
                           'allow_approximation':True,'ane_manifest':'/tmp/flux.json',
                           'loras':[{'path':'/tmp/style.safetensors','strength':0.8}]})
        self.assertEqual(code,0,error)
        self.assertEqual(p['execution'],'gpu_ane_experimental')
        self.assertEqual(p['lora_strategy'],'in_memory_merge')
        self.assertEqual(p['lora_fusion'],'load_time_baked')

    def test_z_image_lora_can_use_identity_bound_ane_artifact(self):
        source=(ROOT/'native/models/z_image/z_image.cpp').read_text()
        module=(ROOT/'native/models/z_image_module.cpp').read_text()
        results=(ROOT/'native/platform/apple/results.mm').read_text()
        self.assertNotIn('Z-Image LoRA currently requires GPU execution',module)
        self.assertIn('transformer_checkpoint_, active_loras_',source)
        self.assertIn('compiled_fused_blocks',results)
        self.assertIn('compiled_mlp_complement',results)
        code,p,error=plan({'model':'z-image-turbo','width':1024,'height':1024,
                           'steps':9,'execution':'gpu_ane','allow_approximation':True,
                           'ane_manifest':'/tmp/z-image.json',
                           'loras':[{'path':'/tmp/z-style.safetensors','strength':0.7}]})
        self.assertEqual(code,0,error)
        self.assertEqual(p['execution'],'gpu_ane_experimental')
        self.assertEqual(p['lora_strategy'],'in_memory_merge')
        self.assertEqual(p['lora_fusion'],'in_memory_delta')

    def test_unified_lora_strategy_contract_is_explicit_and_fail_closed(self):
        adapter=[{'path':'/tmp/style.safetensors','role':'transformer','strength':0.8}]
        code,p,error=plan({'model':'flux2-klein-4b','loras':adapter})
        self.assertEqual(code,0,error)
        self.assertEqual(p['lora_strategy'],'in_memory_merge')
        code,p,error=plan({'model':'flux2-klein-4b','loras':adapter,
                           'lora_strategy':'in_memory_merge'})
        self.assertEqual(code,0,error)
        self.assertEqual(p['lora_strategy'],'in_memory_merge')
        code,p,error=plan({'model':'z-image-turbo','width':1024,'height':1024,
                           'steps':9,'audio':False,'loras':adapter,
                           'lora_strategy':'inference_time'})
        self.assertEqual(code,0,error)
        self.assertEqual(p['lora_strategy'],'inference_time')
        self.assertEqual(p['lora_fusion'],'inference_time_low_rank')
        self.assertNotEqual(plan({
            'model':'z-image-turbo','width':1024,'height':1024,'steps':9,
            'audio':False,'execution':'gpu_ane','allow_approximation':True,
            'ane_manifest':'/tmp/z.json','loras':adapter,
            'lora_strategy':'inference_time'})[0],0)
        for request in [
            {'model':'flux2-klein-4b','loras':adapter,
             'lora_strategy':'disk_premerge'},
            {'model':'flux2-klein-4b','loras':adapter,
             'lora_strategy':'unknown'},
            {'model':'flux2-klein-4b','lora_strategy':'in_memory_merge'},
        ]:
            with self.subTest(request=request):
                self.assertNotEqual(plan(request)[0],0)
        schema2={
            'schema_version':2,'model':'z-image-turbo',
            'operation':'image.generate',
            'inputs':[{'kind':'text','role':'prompt','text':'fox'}],
            'outputs':[{'kind':'image','path':'/tmp/z.png','width':1024,
                        'height':1024,'frames':1,'audio':False}],
            'sampling':{'seed':42,'steps':9},
            'execution':{'policy':'gpu','residency':'resident'},
            'loras':adapter,'lora_strategy':'in_memory_merge',
        }
        code,p,error=plan(schema2)
        self.assertEqual(code,0,error)
        self.assertEqual(p['lora_strategy'],'in_memory_merge')
        models=json.loads(consume(C.c_void_p(lib.tc_models_json())))['models']
        expected={
            'flux2-klein-4b':(['in_memory_merge'],'in_memory_merge'),
            'flux2-klein-9b':(['in_memory_merge'],'in_memory_merge'),
            'z-image-turbo':(['in_memory_merge','inference_time'],'in_memory_merge'),
            'z-image-turbo-gguf':(['inference_time','in_memory_merge'],'inference_time'),
            'minimax-h3-turbo':(['disk_premerge'],'disk_premerge'),
            'ltx-2.5-distilled':(['disk_premerge'],'disk_premerge'),
            'wan2.1-1.3b-qad':(['disk_premerge'],'disk_premerge'),
            'llada-image-turbo':([],None),
        }
        for model_id,(strategies,default) in expected.items():
            descriptor=next(value for value in models if value['id']==model_id)
            self.assertEqual(descriptor['lora_strategies'],strategies)
            if default is None:
                self.assertNotIn('default_lora_strategy',descriptor)
            else:
                self.assertEqual(descriptor['default_lora_strategy'],default)

    def test_native_inference_time_lora_keeps_packed_weights(self):
        header=(ROOT/'native/backends/mlx.hpp').read_text()
        source=(ROOT/'native/backends/mlx.cpp').read_text()
        z_image=(ROOT/'native/models/z_image/z_image.cpp').read_text()
        self.assertIn('runtime_loras_',header)
        self.assertIn('bool inference_time = false',header)
        self.assertIn('if (inference_time) {',source)
        self.assertIn('auto input = mx::astype(x, mx::float32);',source)
        self.assertIn('auto low = mx::matmul(input, mx::transpose(down));',source)
        self.assertIn('Tensor(adapter.scale, mx::float32)',source)
        self.assertIn('if (!values_.count(key) && target.ends_with(".attention.to_out.0"))',source)
        self.assertIn('target_key_counts[resolve_target_key(target)]',source)
        self.assertIn('const bool shared_projection = target_key_counts[key] > 1;',source)
        self.assertIn('{down, up, scale, begin,',source)
        self.assertIn('!w.has_runtime_loras()',z_image)
        self.assertIn('active_lora_strategy_ == "inference_time"',z_image)
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
        self.assertTrue(p['executable'])
        self.assertTrue(p['audio'])
        self.assertEqual(p['audio_capability'],'latent_to_48khz_aac_candidate')
        self.assertIn('audio_vae_vocoder',{stage['id'] for stage in p['stages']})
        self.assertIn('mux',{stage['id'] for stage in p['stages']})
        code,p,error=plan({'model':'ltx-2.5-distilled','width':704,'height':448,'frames':97,'steps':11,'loras':[{'path':'/tmp/ltx.safetensors','role':'refiner','strength':0.8}]})
        self.assertEqual(code,0,error)
        self.assertEqual(p['lora_fusion'],'premerged_manifest_verified')
        self.assertEqual(p['lora_strategy'],'disk_premerge')
        self.assertEqual(p['weight_validation'],'premerged-sidecar-verified-at-execution')
        self.assertNotEqual(plan({'model':'ltx-2.5-distilled','width':704,'height':448,'frames':97,'steps':11,'loras':[{'path':'/tmp/ltx.safetensors','role':'text_encoder','strength':0.8}]})[0],0)
        self.assertNotEqual(plan({'model':'ltx-2.5-distilled','width':704,'height':448,'frames':97,'steps':11,'loras':[{'path':'/tmp/ltx.safetensors','strength':-0.8}]})[0],0)

        streamed = {
            'model':'ltx-2.5-distilled', 'operation':'video.generate',
            'width':704, 'height':448, 'frames':97, 'steps':11,
            'audio':False, 'execution':'gpu', 'residency':'streamed',
            'memory_budget_bytes':12*(1<<30),
        }
        code,p,error = plan(streamed)
        self.assertEqual(code,0,error)
        self.assertEqual(p['residency'],'streamed')
        self.assertEqual(p['memory_budget_scope'],
                         'ltx_denoiser_working_set_target_not_process_cap')
        self.assertGreater(p['memory_estimate_bytes'],
                           p['memory_budget_bytes'])
        self.assertNotIn('streamed', p['algorithm_approximations'])
        self.assertNotEqual(plan({**streamed, 'audio':True})[0], 0)
        self.assertNotEqual(plan({**streamed, 'operation':'video.image'})[0], 0)
        self.assertNotEqual(plan({**streamed, 'execution':'gpu_ane',
                                 'allow_approximation':True,
                                 'ane_manifest':'/tmp/ltx.json'})[0], 0)
        self.assertNotEqual(plan({**streamed,
                                 'memory_budget_bytes':7*(1<<30)})[0], 0)

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

    def test_ltx_approximate_fast_path_is_explicit_and_shape_gated(self):
        request={
            'model':'ltx-2.5-distilled', 'operation':'video.generate',
            'prompt':'A cinematic red fox running through a snowy forest',
            'output':'/tmp/ltx-fast-approx.mp4',
            'width':704, 'height':448, 'frames':97, 'steps':11,
            'fps':24, 'audio':True, 'execution':'gpu',
            'ltx_backend':'c_metal', 'ltx_fast_av':True,
            'ltx_sol_stage2':True, 'ltx_sol_tau':1.0,
            'ltx_sol_dense_edge_blocks':1,
            'ltx_sol_dense_edge_steps':0,
            'ltx_stage2_text_rows':256,
        }
        code,_,error=plan(request)
        self.assertNotEqual(code,0)
        self.assertIn('allow_approximation=true',error)
        code,p,error=plan({**request,'allow_approximation':True})
        self.assertEqual(code,0,error)
        self.assertEqual(p['gpu_graph'],'ltx_c_metal_fast_av_sol_text_pruned')
        self.assertTrue(p['ltx_sol_stage2'])
        self.assertEqual(p['ltx_sol_tau'],1.0)
        self.assertEqual(p['ltx_stage2_text_rows'],256)
        self.assertIn('ltx_sol_stage2',p['algorithm_approximations'])
        self.assertIn('ltx_stage2_text_context_pruning',
                      p['algorithm_approximations'])
        self.assertNotEqual(plan({**request,'allow_approximation':True,
                                  'ltx_backend':'cpp_mlx'})[0],0)
        code,p,error=plan({**request,'allow_approximation':True,
                           'width':1280,'height':704,'frames':121})
        self.assertEqual(code,0,error)
        self.assertEqual(p['requested_shape'],[1280,704,121])
        self.assertTrue(p['ltx_sol_stage2'])
        self.assertNotEqual(plan({**request,'allow_approximation':True,
                                  'width':1536,'height':960,
                                  'frames':121})[0],0)
        self.assertNotEqual(plan({**request,'allow_approximation':True,
                                  'ltx_stage2_text_rows':64})[0],0)
        self.assertNotEqual(plan({**request,'allow_approximation':True,
                                  'ltx_video_attention_batch':True})[0],0)

        exact={key:value for key,value in request.items()
               if not key.startswith('ltx_sol_') and
                  key != 'ltx_stage2_text_rows'}
        exact['ltx_video_attention_batch']=True
        code,p,error=plan(exact)
        self.assertEqual(code,0,error)
        self.assertTrue(p['ltx_video_attention_batch'])
        self.assertEqual(p['algorithm_approximations'],[])
        self.assertNotEqual(plan({**exact,'execution':'gpu_ane',
                                  'allow_approximation':True,
                                  'ane_manifest':'/tmp/missing'})[0],0)

    def test_ltx_i2v_native_path_uses_stage_specific_clean_prefixes(self):
        source=(ROOT/'native/platform/apple/ltx_session.mm').read_text()
        self.assertIn('ltx_mlx_video_vae_create_encoder',source)
        self.assertIn('ltx_mlx_video_vae_encode_pixels_bf16',source)
        self.assertIn('stage1_clean_prefix',source)
        self.assertIn('stage2_clean_prefix',source)
        self.assertIn('first_frame_strength',source)
        self.assertIn('workload.stage1_latent_width * 32u',source)
        self.assertIn('workload.stage2_latent_width * 32u',source)

    def test_ltx_resident_denoiser_cache_is_seed_independent(self):
        source=(ROOT/'native/platform/apple/ltx_session.mm').read_text()
        self.assertIn('root_(std::filesystem::absolute(root))',source)
        start=source.index('std::string denoiser_key =')
        end=source.index('if (!denoiser_cache_hit)', start)
        self.assertNotIn('request.seed', source[start:end])
        self.assertIn('selected_identity', source[start:end])
        self.assertNotIn('options.seed = request.seed',source)
        self.assertIn('ltx_native_run(denoiser_.get(), 1, request.seed',source)
        self.assertIn('ltx_native_run(denoiser_.get(), 2, request.seed',source)

    def test_ltx_gpu_ane_requires_explicit_profile_and_binds_all_stages(self):
        source=(ROOT/'native/platform/apple/ltx_session.mm').read_text()
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
                      'ane_mlp_fused_adaln_pack',
                      'ane_mlp_first_block', 'ane_mlp_block_count',
                      'ane_mlp_stage_mask', 'ane_variant']:
            self.assertIn(field, header)
            self.assertIn(f'options.{field}', source)
        self.assertIn('ctx->options.ane_variant', runtime)
        self.assertIn('ltx_native_release_full_gpu_mlp',runtime)
        self.assertIn('detach_ane_stage(ctx->weights',runtime)
        self.assertIn('stage==2&&ctx->options.release_blocks_final_step',runtime)
        self.assertIn('loaded_v2a != 48u',runtime)
        self.assertIn('loaded_qkv != 48u',runtime)

    def test_ltx_denoiser_cache_key_binds_residency_and_final_step_lifecycle(self):
        source=(ROOT/'native/platform/apple/ltx_session.mm').read_text()
        self.assertIn(':residency=" + request.residency',source)
        self.assertIn(':release_blocks=" +',source)

    def test_ltx_component_staged_uses_isolated_video_vae_helper(self):
        session=(ROOT/'native/platform/apple/ltx_session.mm').read_text()
        build=(ROOT/'tools/native/build.sh').read_text()
        helper=ROOT/'tools/native/ltx_video_vae_decode.c'
        self.assertTrue(helper.is_file())
        self.assertIn('decode_ltx_video_isolated',session)
        self.assertIn('video_vae_isolation = "process"',session)
        self.assertIn('posix_spawn',session)
        self.assertIn('ltx-video-vae-decode',build)
        self.assertIn('ltx_mlx_video_vae_decode_tokens_bf16',
                      helper.read_text())

    def test_ltx_quality_defaults_keep_approximation_and_ane_off(self):
        contracts=(ROOT/'native/core/contracts.hpp').read_text()
        self.assertIn('allow_approximation = false',contracts)
        self.assertIn('bool ltx_video_attention_batch = false',contracts)
        self.assertIn('bool ltx_sol_stage1 = false',contracts)
        self.assertIn('bool ltx_sol_stage2 = false',contracts)
        request=(ROOT/'native/platform/apple/request.mm').read_text()
        self.assertIn('boolean(d, @"allow_approximation", false)',request)
        self.assertIn('supports_encoder_gpu_ane', request)
        self.assertIn('request.execution == "gpu_ane"',
                      (ROOT/'native/platform/apple/ltx_session.mm').read_text())

    def test_gpu_ane_capabilities_are_optional_and_encoder_scoped(self):
        pointer=lib.tc_models_json()
        payload=json.loads(consume(C.c_void_p(pointer)))
        models={value['id']:value for value in payload['models']}
        for model in models.values():
            self.assertEqual(model['default_execution'], 'gpu')
            expected = ('optional_manifest_gated'
                        if model['supports_gpu_ane'] else 'unsupported')
            self.assertEqual(model['gpu_ane_policy'], expected)
        encoder_models = {
            'flux2-klein-4b', 'flux2-klein-9b',
            'z-image-turbo', 'z-image-turbo-gguf',
            'ltx-2.5-distilled',
            'minimax-h3-fasth3-mlx-int6',
            'minimax-h3-fasth3-mlx-int6-vsa',
            'minimax-h3-vdn',
        }
        for model_id, model in models.items():
            expected = model_id in encoder_models
            self.assertEqual(model['supports_encoder_gpu_ane'], expected)
            self.assertEqual(
                model['encoder_gpu_ane_policy'],
                'optional_explicit_manifest' if expected else 'unsupported',
            )

    def test_ltx_service_conditioning_cache_is_bound_and_observable(self):
        session=(ROOT/'native/platform/apple/ltx_session.mm').read_text()
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
        route=source[source.index('bool external_ltx_request'):
                     source.index('bool resident_ltx_candidate_request')]
        self.assertIn('value.residency == "component_staged"',route)
        self.assertNotIn('!value.audio',route)

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
        session=(ROOT/'native/platform/apple/ltx_session.mm').read_text()
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
        self.assertIn('ltx_mlx_audio_vae_decode_bf16',
                      finalizer.read_text())
        self.assertIn('ltx_mlx_vocoder_decode_base_bf16',
                      finalizer.read_text())
        self.assertIn('ltx_mlx_bwe_extend_f32',finalizer.read_text())
        self.assertIn('mux_video_with_audio',finalizer.read_text())
        self.assertIn('video_vae_weight_load',finalizer.read_text())
        self.assertIn('video_vae_compute',finalizer.read_text())
        self.assertIn('remove_managed_staging_directory',finalizer.read_text())
        self.assertIn('turbocider-ltx-exec-finalizer-',finalizer.read_text())
        self.assertIn('"$OUT/native_media_audio.o" "$OUT/native_media_video.o"',build)
        self.assertNotIn('"$OUT/audio.o"',build)
        self.assertNotIn('"$OUT/video.o"',build)

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
                'mlp_block_start':0,'mlp_block_count':48,
                'mlp_stage_mask':3,
            }
            request={'model':'ltx-2.5-distilled','execution':'gpu_ane',
                     'allow_approximation':True,
                     'ane_manifest':str(profile),'width':704,'height':448,
                     'frames':97,'steps':11}
            profile.write_text(json.dumps(base))
            code,p,error=plan(request)
            self.assertEqual(code,0,error)
            profile.write_text(json.dumps({**base,'variant':'fp16'}))
            code,p,error=plan(request)
            self.assertEqual(code,0,error)
            profile.write_text(json.dumps({**base,'variant':'q4'}))
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('unsupported LTX ANE artifact variant',error)
            stage2_only = {**base,
                'release_full_gpu_mlp':False,
                'mlp_block_start':36,'mlp_block_count':12,
                'mlp_stage_mask':2}
            stage2_only.pop('mlp_stage1')
            profile.write_text(json.dumps(stage2_only))
            code,p,error=plan(request)
            self.assertEqual(code,0,error)
            stage1_only = {**base,
                'release_full_gpu_mlp':False,
                'mlp_block_start':36,'mlp_block_count':12,
                'mlp_stage_mask':1}
            stage1_only.pop('mlp_stage2')
            profile.write_text(json.dumps(stage1_only))
            code,p,error=plan(request)
            self.assertEqual(code,0,error)
            missing_enabled_stage = dict(stage2_only)
            missing_enabled_stage.pop('mlp_stage2')
            profile.write_text(json.dumps(missing_enabled_stage))
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('missing stage2 MLP',error)
            self.assertIn('mlp_window=',
                          (ROOT/'native/platform/apple/ltx_session.mm').read_text())
            profile.write_text(json.dumps({**base,
                'mlp_block_start':36,'mlp_block_count':12}))
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('complete Stage-1/Stage-2 coverage',error)
            profile.write_text(json.dumps({**base,
                'release_full_gpu_mlp':False,
                'mlp_block_start':40,'mlp_block_count':12}))
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('fit within 48 blocks',error)
            profile.write_text(json.dumps({**base,
                'release_full_gpu_mlp':False,'mlp_stage_mask':0}))
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('mlp_stage_mask must be 1..3',error)
            profile.write_text(json.dumps(base))
            code,p,error=plan({**request,'allow_approximation':False})
            self.assertNotEqual(code,0)
            self.assertIn('allow_approximation=true',error)
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
        source=(ROOT/'native/platform/apple/ltx_session.mm').read_text()
        for name in ['stage1_video', 'stage1_audio', 'stage2_input_video',
                     'stage2_video', 'stage2_audio', 'metadata.json']:
            self.assertIn(name, source)
        self.assertIn('request.dump', source)

    def test_ltx_streaming_reuses_refill_slots_and_direct_file_reads(self):
        blocks=(ROOT/'native/models/ltx_runtime/ltx_blocks.c').read_text()
        safetensors=(ROOT/'native/models/ltx_runtime/ltx_safetensors.m').read_text()
        benchmark=(ROOT/'tools/native/benchmark_ltx_resident.py').read_text()
        self.assertIn('refill_block_weights', blocks)
        self.assertIn('LTX_MAX_REFILL_SLOTS = 3', blocks)
        self.assertIn('streaming_slots[LTX_MAX_REFILL_SLOTS]', blocks)
        self.assertIn('tc_block_residency_plan_build', blocks)
        policy=(ROOT/'native/runtime/block_residency.c').read_text()
        self.assertIn('capacity >= active_blocks', policy)
        self.assertIn('return finish_plan(plan, active_blocks, 0u, 1)', policy)
        self.assertIn('request_slot_refills',
                      (ROOT/'native/platform/apple/ltx_session.mm').read_text())
        self.assertIn('ltx_st_read_mapped_data', blocks)
        self.assertIn('ltx_gpu_buffer_contents', blocks)
        self.assertIn('ltx_pread_exact(mapping->descriptor', safetensors)
        self.assertIn('stage2_video.bf16', benchmark)
        self.assertIn('compare_video', benchmark)
        self.assertIn('require_speedup', benchmark)
        self.assertIn('--ltx-fast-mode', benchmark)
        self.assertIn('--ltx-video-attention-batch', benchmark)
        self.assertIn('ltx_stage2_text_rows', benchmark)
        self.assertIn('ltx_sol_dense_edge_steps', benchmark)

    def test_ltx_mlx_convrot_group_tuning_is_explicit_and_bounded(self):
        session = (ROOT / 'native/platform/apple/ltx_session.mm').read_text()
        native = (ROOT / 'native/models/ltx_mlx/native.cpp').read_text()
        options = (ROOT / 'native/models/ltx_mlx/native.hpp').read_text()
        sweep = (ROOT / 'tools/native/benchmark_ltx_qmm_groups.py').read_text()
        self.assertIn('TURBOCIDER_LTX_MLX_CONVROT_GROUP_SIZE', session)
        self.assertIn('options.convrot_group_size', session)
        self.assertIn('options->convrot_group_size == 32u', native)
        self.assertIn('options->convrot_group_size == 128u', native)
        self.assertIn('uint32_t convrot_group_size', options)
        self.assertIn('stage2_video_ffn_in', sweep)
        self.assertIn('fastest_group', sweep)

    def test_ltx_rejected_attention_batch_stays_opt_in_and_excludes_sol(self):
        blocks = (ROOT / 'native/models/ltx_runtime/ltx_blocks.c').read_text()
        self.assertIn('TURBOCIDER_LTX_VIDEO_ATTENTION_BATCH', blocks)
        self.assertNotIn('video_attention_batch=1;', blocks)
        self.assertIn(
            '(ctx->options.sol_stage1 || ctx->options.sol_stage2)', blocks)

    def test_ltx_mlx_periodic_eval_is_explicit_and_bounded(self):
        model = (ROOT / 'native/models/ltx_mlx/model.cpp').read_text()
        self.assertIn('TURBOCIDER_LTX_MLX_EVAL_EVERY', model)
        self.assertIn('parsed < 1 || parsed > 48', model)
        self.assertIn('periodic_flush', model)

    def test_ltx_mlx_rope_cache_is_identity_bound_and_opt_in(self):
        model = (ROOT / 'native/models/ltx_mlx/model.cpp').read_text()
        header = (ROOT / 'native/models/ltx_mlx/model.hpp').read_text()
        self.assertIn('TURBOCIDER_LTX_MLX_CACHE_ROPE', model)
        self.assertIn('video_positions.id()', model)
        self.assertIn('audio_positions.id()', model)
        self.assertIn('rope_video_positions_id_', header)
        self.assertIn('cached_video_cross_rope_', header)

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
                     'allow_approximation':True,
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
        source=(ROOT/'native/platform/apple/ltx_session.mm').read_text()
        self.assertIn('conditioning_prompt',source)
        self.assertIn('root_, selected_checkpoint, request.prompt',source)
        self.assertIn('if (!bound_prompt || *bound_prompt != requested_prompt)',source)

    def test_ltx_manifest_preflight_verifies_artifact_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); lora_dir=root/'models'/'loras'; diff_dir=root/'models'/'diffusion_models'
            lora_dir.mkdir(parents=True); diff_dir.mkdir(parents=True)
            base=diff_dir/'base.safetensors'; lora=lora_dir/'adapter.safetensors'; output=diff_dir/'ltx-2.5-22b-runtime-refiner-comfy-int8-convrot.safetensors'
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
            (diff_dir/(output.name + '.manifest.json')).write_text(json.dumps(manifest))
            out,err=C.c_void_p(),C.c_void_p()
            code=lib.tc_ltx_lora_preflight_json(str(root/'models').encode(),str(lora).encode(),C.c_float(0.8),C.byref(out),C.byref(err))
            value,failure=consume(out),consume(err)
            self.assertEqual(code,0,failure); payload=json.loads(value)
            self.assertEqual(payload['validation'],'premerged_manifest_verified')
            self.assertEqual(payload['checkpoint'],str(output))
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
        self.assertEqual(p['lora_fusion'],'premerged_manifest_verified')
        self.assertEqual(p['lora_strategy'],'disk_premerge')
        self.assertNotEqual(plan({'model':'minimax-h3-turbo','frames':22,'width':512,'height':512,'steps':4,'loras':[{'path':'/tmp/h3.safetensors','role':'text_encoder'}]})[0],0)
        hybrid={'model':'minimax-h3-turbo','frames':22,'width':512,
                'height':512,'steps':4,'execution':'gpu_ane',
                'ane_manifest':'/tmp/h3-coreml'}
        code,p,error=plan({**hybrid,'allow_approximation':True})
        self.assertEqual(code,0,error)
        code,p,error=plan(hybrid)
        self.assertNotEqual(code,0)
        self.assertIn('allow_approximation=true',error)

    def test_fasth3_mlx_int6_is_explicit_and_does_not_replace_legacy_h3(self):
        request={
            'model':'minimax-h3-fasth3-mlx-int6',
            'operation':'video.generate',
            'prompt':'A red fox runs through fresh snow.',
            'output':'/tmp/h3-mlx.mp4',
            'frames':22,'width':832,'height':480,'fps':24,'steps':4,
            'audio':True,'execution':'gpu','residency':'component_staged',
        }
        code,p,error=plan(request)
        self.assertEqual(code,0,error)
        self.assertTrue(p['executable'])
        self.assertEqual(p['backend'],'mlx_cpp_metal')
        self.assertEqual(p['gpu_graph'],'fasth3_int6_qmm')
        self.assertEqual(p['precision'],'int6_g64_bf16_activation')
        self.assertEqual(p['validation'],'modelscope_int6_parity_candidate')
        self.assertEqual(p['audio_capability'],'full_h3_audio_vae_32khz_stereo')
        self.assertEqual([stage['id'] for stage in p['stages']],
                         ['text_encode','av_denoise','video_decode',
                          'audio_decode','mux'])
        self.assertEqual(p['stages'][1]['iterations'],4)
        for invalid in [
            {**request,'steps':3},
            {**request,'fps':25},
            {**request,'frames':23},
            {**request,'width':816},
            {**request,'execution':'gpu_ane'},
            {**request,'loras':[{'path':'/tmp/h3.safetensors'}]},
        ]:
            self.assertNotEqual(plan(invalid)[0],0)
        legacy={
            'model':'minimax-h3-turbo','frames':22,'width':512,
            'height':512,'fps':24,'steps':4,
        }
        code,p,error=plan(legacy)
        self.assertEqual(code,0,error)
        self.assertEqual(p['backend'],'h3-metal-mps')

    def test_fasth3_mlx_vsa_is_separate_validated_profile(self):
        request={
            'model':'minimax-h3-fasth3-mlx-int6-vsa',
            'operation':'video.generate',
            'prompt':'A red fox runs through fresh snow.',
            'output':'/tmp/h3-mlx-vsa.mp4',
            'frames':22,'width':832,'height':480,'fps':24,'steps':4,
            'audio':True,'execution':'gpu','residency':'component_staged',
            'vsa':True,'vsa_sparsity':0.9,'vsa_tile_size':64,
            'vsa_prefix_mode':'exempt','vsa_dense_first_n_steps':0,
            'vsa_dense_layers':[],'vsa_impl':'reference',
        }
        code,p,error=plan(request)
        self.assertEqual(code,0,error)
        self.assertTrue(p['executable'])
        self.assertEqual(p['backend'],'mlx_cpp_metal')
        self.assertEqual(p['gpu_graph'],'fasth3_int6_vsa')

        self.assertEqual(p['precision'],'int6_g64_bf16_activation')
        self.assertEqual(p['validation'],'modelscope_int6_parity_candidate')
        self.assertEqual([stage['id'] for stage in p['stages']],
                         ['text_encode','av_denoise','video_decode',
                          'audio_decode','mux'])
        self.assertEqual(p['stages'][1]['iterations'],4)
        for change in [
            {'vsa':False},
            {'vsa_sparsity':-0.01},
            {'vsa_sparsity':1.0},
            {'vsa_tile_size':128},
            {'vsa_prefix_mode':'dense'},
            {'vsa_dense_first_n_steps':5},
            {'vsa_dense_layers':[50]},
            {'vsa_dense_layers':[3,3]},
            {'vsa_impl':'metal'},
        ]:
            with self.subTest(change=change):
                self.assertNotEqual(plan({**request,**change})[0],0)

        dense={**request,'model':'minimax-h3-fasth3-mlx-int6',
               'output':'/tmp/h3-mlx-dense.mp4'}
        self.assertNotEqual(plan(dense)[0],0)
        dense_defaults={
            'model':'minimax-h3-fasth3-mlx-int6',
            'operation':'video.generate','prompt':request['prompt'],
            'output':'/tmp/h3-mlx-dense.mp4','frames':22,
            'width':832,'height':480,'fps':24,'steps':4,
            'audio':True,'execution':'gpu','residency':'component_staged',
        }
        self.assertEqual(plan(dense_defaults)[0],0)

        schema2={
            'schema_version':2,
            'model':'minimax-h3-fasth3-mlx-int6-vsa',
            'operation':'video.generate',
            'inputs':[{'kind':'text','role':'prompt','text':request['prompt']}],
            'outputs':[{'kind':'video','path':'/tmp/h3-mlx-vsa.mp4',
                        'width':832,'height':480,'frames':22,
                        'fps':24,'audio':True}],
            'sampling':{'seed':2026,'steps':4},
            'execution':{'policy':'gpu','residency':'component_staged'},
            'parameters':{
                'vsa':True,'vsa_sparsity':0.9,'vsa_tile_size':256,
                'vsa_prefix_mode':'compete','vsa_dense_first_n_steps':1,
                'vsa_dense_layers':[3,7],'vsa_impl':'auto',
            },
        }
        code,p,error=plan(schema2)
        self.assertEqual(code,0,error)
        self.assertEqual(p['gpu_graph'],'fasth3_int6_vsa')

    def test_vdn_h3_is_a_separate_six_step_profile(self):
        request={
            'model':'minimax-h3-vdn',
            'operation':'video.generate',
            'prompt':'A red fox runs through fresh snow.',
            'output':'/tmp/h3-vdn.mp4',
            'frames':124,'width':960,'height':544,'fps':24,'steps':6,
            'audio':True,'execution':'gpu','residency':'component_staged',
        }
        code,p,error=plan(request)
        self.assertEqual(code,0,error)
        self.assertTrue(p['executable'])
        self.assertEqual(p['backend'],'mlx_cpp_metal')
        self.assertEqual(p['gpu_graph'],'h3_vdn_int6_window_delta')
        self.assertEqual(p['precision'],'int6_g64_base+bf16_vdn+fp32_solve')
        self.assertEqual(p['validation'],'modelscope_vdn_stage_dmd_candidate')
        self.assertEqual(p['audio_capability'],'full_h3_audio_vae_32khz_stereo')
        self.assertEqual([stage['id'] for stage in p['stages']],
                         ['text_encode','av_denoise','video_decode',
                          'audio_decode','mux'])
        self.assertEqual(p['stages'][1]['iterations'],6)
        for invalid in [
            {**request,'steps':4},
            {**request,'fps':25},
            {**request,'frames':123},
            {**request,'width':944},
            {**request,'execution':'gpu_ane','allow_approximation':True,
             'ane_manifest':'/tmp/vdn-ane'},
            {**request,'loras':[{'path':'/tmp/other.safetensors'}]},
            {**request,'vsa':True},
            {**request,'memory_budget_bytes':32 << 30},
        ]:
            with self.subTest(invalid=invalid):
                self.assertNotEqual(plan(invalid)[0],0)
        streamed={**request,'residency':'streamed',
                  'memory_budget_bytes':32 << 30}
        code,p,error=plan(streamed)
        self.assertEqual(code,0,error)
        self.assertTrue(p['streaming_offload'])
        self.assertEqual(p['memory_budget_scope'],
                         'vdn_base_branch_working_set_target_not_process_cap')

        fast={**request,'model':'minimax-h3-fasth3-mlx-int6','steps':4,
              'output':'/tmp/h3-fast.mp4','frames':22,'width':832,'height':480}
        code,p,error=plan(fast)
        self.assertEqual(code,0,error)
        self.assertEqual(p['gpu_graph'],'fasth3_int6_qmm')
        self.assertEqual(p['precision'],'int6_g64_bf16_activation')
        self.assertEqual(p['validation'],'modelscope_int6_parity_candidate')
        self.assertEqual([stage['id'] for stage in p['stages']],
                         ['text_encode','av_denoise','video_decode',
                          'audio_decode','mux'])
        self.assertEqual(p['stages'][1]['iterations'],4)

    def test_h3_streaming_budget_drives_fail_closed_pinned_prefix(self):
        request={'model':'minimax-h3-turbo','frames':22,'width':512,
                 'height':512,'steps':4,'residency':'streamed',
                 'memory_budget_bytes':16 << 30}
        code,p,error=plan(request)
        self.assertEqual(code,0,error)
        self.assertTrue(p['streaming_offload'])
        self.assertEqual(p['memory_budget_scope'],
                         'h3_dit_working_set_target_not_process_cap')
        minimum=(4 << 30)+2*770725376
        code,_,error=plan({**request,'memory_budget_bytes':minimum-1})
        self.assertNotEqual(code,0)
        self.assertIn('two-slot minimum',error)
        code,_,error=plan({**request,'residency':'resident',
                           'memory_budget_bytes':64 << 30})
        self.assertNotEqual(code,0)
        self.assertIn('requires streamed residency',error)
        session=(ROOT/'native/platform/apple/h3_session.mm').read_text()
        runtime=(ROOT/'native/models/h3_runtime/h3_dit.c').read_text()
        public=(ROOT/'native/models/h3_runtime/h3.h').read_text()
        core=(ROOT/'native/models/h3_runtime/h3.c').read_text()
        self.assertIn('parameters.ssd_memory_budget_bytes',session)
        self.assertIn('r.residency == "resident" ||',session)
        self.assertIn('r.residency == "streamed"',session)
        self.assertIn('h3_cache_set_decoder_enabled(context_.get()',session)
        self.assertIn('@"ssd_pinned_blocks"',session)
        self.assertIn('@"ssd_request_bytes_read"',session)
        self.assertIn('@"cache_prepared_dit"',session)
        self.assertIn('@"cache_video_decoder"',session)
        self.assertIn('stream_block_pinned(dit, index)',runtime)
        self.assertIn('first_streamed_block(dit)',runtime)
        self.assertIn('next_streamed_block(dit, block)',runtime)
        self.assertIn('h3_stream_plan_build(',runtime)
        self.assertIn('int ssd_pinned_prefix;',public)
        self.assertIn('uint64_t ssd_memory_budget_bytes;',public)
        self.assertIn('uint64_t ssd_request_bytes_read;',public)
        self.assertIn('void h3_cache_set_decoder_enabled(',public)
        self.assertIn('|ssd-pinned=%d|ssd-budget=%llu',core)
        self.assertIn('h3_decoder_cache_enabled(ctx)',core)

    def test_h3_quantized_streaming_is_explicit_and_reported(self):
        request={'model':'minimax-h3-turbo','frames':22,'width':512,
                 'height':512,'steps':4,'residency':'streamed',
                 'quantized_cache':'/tmp/h3-quantized-cache',
                 'allow_approximation':True}
        code,p,error=plan(request)
        self.assertEqual(code,0,error)
        self.assertEqual(p['precision'],
                         'int8_weight_bf16_activation_streamed')
        self.assertEqual(p['algorithm_approximations'],
                         ['row_symmetric_int8_weight_quantization'])
        self.assertEqual(p['quantized_cache'],
                         '/tmp/h3-quantized-cache')
        self.assertTrue(p['streaming_offload'])
        code,_,error=plan({**request,'residency':'resident'})
        self.assertNotEqual(code,0)
        self.assertIn('requires streamed residency',error)
        code,_,error=plan({**request,'allow_approximation':False})
        self.assertNotEqual(code,0)
        self.assertIn('allow_approximation=true',error)
        minimum=(4 << 30)+2*385617408
        code,_,error=plan({**request,'memory_budget_bytes':minimum})
        self.assertEqual(code,0,error)
        code,_,error=plan({**request,'memory_budget_bytes':minimum-1})
        self.assertNotEqual(code,0)
        self.assertIn('two-slot minimum',error)

    def test_wan_is_an_executable_persistent_runtime_candidate(self):
        request={'model':'wan2.1-1.3b-qad','width':832,'height':480,
                 'frames':81,'fps':16,'steps':3}
        code,p,error=plan(request)
        self.assertEqual(code,0,error)
        self.assertTrue(p['executable'])
        self.assertEqual(p['backend'],'wan-mlx')
        self.assertEqual(p['validation'],'manifest_verified_native')
        self.assertEqual([s['id'] for s in p['stages']],
                         ['text_encode','dit','video_vae','export'])
        for invalid in [
            {**request,'frames':80}, {**request,'steps':4},
            {**request,'fps':24},
        ]:
            with self.subTest(invalid=invalid):
                self.assertNotEqual(plan(invalid)[0],0)
        code,p,error=plan({**request,'loras':[{'path':'/tmp/wan.safetensors',
                                               'role':'transformer','strength':0.8}]})
        self.assertEqual(code,0,error)
        self.assertEqual(p['lora_fusion'],'premerged_manifest_verified')
        self.assertEqual(p['lora_strategy'],'disk_premerge')
        self.assertEqual(p['weight_validation'],'premerged-manifest-verified-at-execution')
        self.assertNotEqual(plan({**request,'loras':[{'path':'/tmp/wan.safetensors',
                                                      'role':'text_encoder','strength':0.8}]})[0],0)
        self.assertNotEqual(plan({**request,'execution':'gpu_ane',
                                  'ane_manifest':'/tmp/wan-ane.json',
                                  'loras':[{'path':'/tmp/wan.safetensors',
                                            'role':'transformer','strength':0.8}]})[0],0)
        source=(ROOT/'native/platform/apple/wan_session.mm').read_text()
        self.assertNotIn('PersistentWorker',source)
        self.assertNotIn('posix_spawn',source)
        self.assertNotIn('getenv(',source)
        self.assertIn('uses_parent_mlx() const override { return true; }',source)
        self.assertIn('wan::Pipeline',source)
        self.assertIn('turbocider-wan-premerged-lora-v1',source)
        self.assertNotIn('fastmetal_worker.py', (ROOT/'tools/native/package.sh').read_text())
        models=json.loads(consume(C.c_void_p(lib.tc_models_json())))['models']
        descriptor=next(value for value in models if value['id']=='wan2.1-1.3b-qad')
        self.assertEqual(descriptor['runtime_dependency'], 'native')
        self.assertTrue(descriptor['supports_lora'])
        self.assertFalse(descriptor['runtime_lora'])
        self.assertEqual(descriptor['lora_mode'],'premerged-manifest')
        self.assertEqual(descriptor['lora_strategies'],['disk_premerge'])
        self.assertEqual(descriptor['default_lora_strategy'],'disk_premerge')

    def test_wan_gpu_ane_manifest_requires_complete_fixed_shape_tree(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for block in range(30):
                (root/f'block{block}.mlmodelc').mkdir()
            manifest={
                'schema':'turbocider-wan-ane-mlp-v1',
                'checkpoint':'/missing/wan/mlx_dit.safetensors',
                'checkpoint_sha256':'a48f7370cab9664ebc71afefa6cbc2ea2ab1970b06aa9ad77712179cf52213ff',
                'mlx_dit_json_sha256':'db5603223f17a03051d36e4a3a218477dd48a7723117743d5d916d1db3f87691',
                'shape':{'rows':32760,'hidden':1536,'intermediate':8960,
                         'ane_intermediate':4096,'gpu_intermediate':4864,
                         'blocks':30,'attention_heads':12,'attention_head_dim':128},
                'variant':'int8_pc','blocks':list(range(30)),
                'artifacts':{str(i):f'block{i}.mlmodelc' for i in range(30)},
            }
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            request={'model':'wan2.1-1.3b-qad','execution':'gpu_ane',
                     'allow_approximation':True,
                     'ane_manifest':str(path),'width':832,'height':480,
                     'frames':81,'fps':16,'steps':3}
            code,p,error=plan(request)
            self.assertEqual(code,0,error)
            self.assertEqual(p['backend'],'wan-mlx+coreml')
            manifest['schema']='turbocider-fastmetal-ane-mlp-v1'
            path.write_text(json.dumps(manifest))
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('schema',error)
            manifest['schema']='turbocider-wan-ane-mlp-v1'
            path.write_text(json.dumps(manifest))
            code,p,error=plan({**request,'allow_approximation':False})
            self.assertNotEqual(code,0)
            self.assertIn('allow_approximation=true',error)
            (root/'block29.mlmodelc').rmdir()
            code,p,error=plan(request)
            self.assertNotEqual(code,0)
            self.assertIn('ANE block artifact',error)

    def test_wan_lora_preflight_binds_pinned_base_and_merged_checkpoint(self):
        configured=os.environ.get('TURBOCIDER_WAN_TEST_MODEL')
        fixture=(Path(configured).resolve() if configured else
                 (ROOT/'models/Wan2.1-1.3B-QAD').resolve())
        base_weights=fixture/'mlx_dit.safetensors'
        base_config=fixture/'mlx_dit.json'
        if not base_weights.is_file() or not base_config.is_file():
            self.skipTest('validated Wan base fixture is unavailable')

        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            model=root/'model';model.mkdir()
            (model/'mlx_dit.safetensors').symlink_to(base_weights)
            (model/'mlx_dit.json').symlink_to(base_config)
            lora=root/'style.safetensors';lora.write_bytes(b'wan lora fixture')
            merged=root/'merged';merged.mkdir()
            output_weights=merged/'mlx_dit.safetensors'
            output_weights.write_bytes(b'premerged Wan checkpoint fixture')
            output_config=merged/'mlx_dit.json'
            output_config.write_bytes(base_config.read_bytes())
            manifest={
                'schema':'turbocider-wan-premerged-lora-v1',
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
                code=lib.tc_wan_lora_preflight_json(
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
            lora.write_bytes(b'tampered adapter with different length')
            code,payload,error=preflight()
            self.assertNotEqual(code,0)
            self.assertIn('size',error)
            self.assertIsNone(payload)
            lora.write_bytes(b'wan lora fixture')
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
        self.assertEqual(p['lora_strategy'],'in_memory_merge')
        encoder_only = {
            'model': 'flux2-klein-9b', 'width': 512, 'height': 512,
            'execution': 'gpu', 'allow_approximation': True,
            'encoder_ane_manifest': '/tmp/qwen3-encoder.json',
        }
        code, configured, error = plan(encoder_only)
        self.assertEqual(code, 0, error)
        self.assertEqual(configured['execution'], 'gpu')
        self.assertEqual(configured['gpu_graph'], 'eager_blocks')
        self.assertEqual(configured['encoder_execution'], 'gpu_ane_experimental')
        self.assertEqual(configured['encoder_backend'], 'mlx_cpp_metal+coreml')
        self.assertNotEqual(plan({
            **encoder_only, 'execution': 'gpu_ane',
            'ane_manifest': '/tmp/flux9-denoiser.json',
        })[0], 0)

    def test_qwen3_encoder_manifest_is_separate_from_dit_hybrid(self):
        base = {
            'model': 'flux2-klein-4b', 'operation': 'image.generate',
            'prompt': 'A red fox', 'width': 512, 'height': 512,
            'steps': 4, 'execution': 'gpu',
            'encoder_ane_manifest': '/tmp/qwen3-encoder.json',
        }
        self.assertNotEqual(plan(base)[0], 0)
        code, configured, error = plan({**base, 'allow_approximation': True})
        self.assertEqual(code, 0, error)
        self.assertEqual(configured['backend'], 'mlx_cpp_metal')
        self.assertEqual(configured['gpu_graph'], 'eager_blocks')
        self.assertEqual(configured['encoder_execution'], 'gpu_ane_experimental')
        self.assertEqual(configured['encoder_backend'], 'mlx_cpp_metal+coreml')
        self.assertEqual(configured['encoder_gpu_graph'],
                         'qwen3_encoder_mlp_complement')
        self.assertEqual(configured['encoder_precision'],
                         'bf16_gpu+int8_mlp_fp16_io')
        self.assertEqual(configured['precision'], 'bf16')
        self.assertIn('qwen3_encoder_mlp_int8_per_channel',
                      configured['algorithm_approximations'])

        both = {**base, 'execution': 'gpu_ane', 'ane_manifest': '/tmp/dit.json',
                'allow_approximation': True}
        code, configured, error = plan(both)
        self.assertEqual(code, 0, error)
        self.assertEqual(configured['backend'], 'mlx_cpp_metal+coreml')
        self.assertEqual(configured['gpu_graph'], 'compiled_hybrid_complement')
        self.assertEqual(configured['execution'], 'gpu_ane_experimental')
        self.assertEqual(configured['encoder_execution'], 'gpu_ane_experimental')
        self.assertEqual(configured['encoder_backend'], 'mlx_cpp_metal+coreml')
        self.assertEqual(configured['precision'], 'bf16_gpu+int8_mlp_fp16_io')

        gguf = {**base, 'model': 'z-image-turbo-gguf',
                'model_variant': 'Q8_0', 'width': 512, 'height': 512,
                'allow_approximation': True}
        code, configured, error = plan(gguf)
        self.assertEqual(code, 0, error)
        self.assertEqual(configured['backend'], 'mlx_cpp_metal_gguf')
        self.assertEqual(configured['gpu_graph'], 'native_quantized_blocks')
        self.assertEqual(configured['encoder_backend'], 'mlx_cpp_metal+coreml')
        self.assertEqual(configured['encoder_gpu_graph'],
                         'qwen3_encoder_mlp_complement')
        self.assertEqual(configured['precision'], 'checkpoint_defined_gguf')

        schema2 = {
            'schema_version': 2, 'model': 'z-image-turbo',
            'operation': 'image.generate',
            'inputs': [{'kind': 'text', 'role': 'prompt', 'text': 'A red fox'}],
            'outputs': [{'kind': 'image', 'path': '/tmp/z.png',
                         'width': 512, 'height': 512, 'frames': 1,
                         'audio': False}],
            'sampling': {'seed': 42, 'steps': 9},
            'execution': {'policy': 'gpu', 'allow_approximation': True,
                          'encoder_ane_manifest': '/tmp/qwen3-encoder.json'},
        }
        code, configured, error = plan(schema2)
        self.assertEqual(code, 0, error)
        self.assertEqual(configured['encoder_execution'], 'gpu_ane_experimental')
        self.assertEqual(configured['backend'], 'mlx_cpp_metal')
        self.assertEqual(configured['encoder_backend'], 'mlx_cpp_metal+coreml')

        for model in ['minimax-h3-turbo', 'llada-image-turbo']:
            self.assertNotEqual(plan({**base, 'model': model,
                                      'encoder_ane_manifest': '/tmp/q.json',
                                      'allow_approximation': True})[0], 0)

        h3_base = {
            'model': 'minimax-h3-fasth3-mlx-int6',
            'operation': 'video.generate', 'prompt': 'A red fox',
            'output': '/tmp/h3.mp4', 'width': 512, 'height': 512,
            'frames': 22, 'steps': 4, 'audio': False, 'execution': 'gpu',
            'allow_approximation': True,
            'encoder_ane_manifest': '/tmp/h3-qwen3-vl.json',
        }
        code, configured, error = plan(h3_base)
        self.assertEqual(code, 0, error)
        self.assertEqual(configured['backend'], 'mlx_cpp_metal')
        self.assertEqual(configured['encoder_backend'], 'mlx_cpp_metal+coreml')
        self.assertEqual(configured['encoder_gpu_graph'],
                         'qwen3_vl_encoder_mlp_complement')
        self.assertIn('qwen3_vl_encoder_mlp_int8_per_channel',
                      configured['algorithm_approximations'])

        h3_vsa = {**h3_base, 'model': 'minimax-h3-fasth3-mlx-int6-vsa',
                  'vsa': True}
        code, configured, error = plan(h3_vsa)
        self.assertEqual(code, 0, error)
        self.assertEqual(configured['encoder_gpu_graph'],
                         'qwen3_vl_encoder_mlp_complement')

        h3_vdn = {**h3_base, 'model': 'minimax-h3-vdn', 'steps': 6}
        code, configured, error = plan(h3_vdn)
        self.assertEqual(code, 0, error)
        self.assertEqual(configured['encoder_backend'], 'mlx_cpp_metal+coreml')
        self.assertNotEqual(plan({**h3_base, 'allow_approximation': False})[0], 0)

        ltx = {
            'model': 'ltx-2.5-distilled', 'operation': 'video.generate',
            'prompt': 'A red fox', 'output': '/tmp/ltx.mp4',
            'width': 704, 'height': 448, 'frames': 97, 'steps': 11,
            'fps': 24, 'audio': False, 'execution': 'gpu',
            'allow_approximation': True,
            'encoder_ane_manifest': '/tmp/ltx-gemma-bank',
        }
        code, configured, error = plan(ltx)
        self.assertEqual(code, 0, error)
        self.assertEqual(configured['encoder_backend'], 'metal_mps+coreml')
        self.assertEqual(configured['encoder_gpu_graph'],
                         'gemma4_encoder_mlp_complement')
        self.assertIn('gemma4_encoder_mlp_int8_per_channel',
                      configured['algorithm_approximations'])
        self.assertNotIn('qwen3_encoder_mlp_int8_per_channel',
                         configured['algorithm_approximations'])

        profile_source = (ROOT/'native/platform/apple/profile.mm').read_text()
        self.assertIn('@"encoder_ane_manifest"', profile_source)
        self.assertIn('r.encoder_ane_manifest', profile_source)

    def test_flux_single_projection_has_runtime_layout_guard(self):
        source=(ROOT/'native/models/flux2/flux_transformer.cpp').read_text()
        self.assertIn('parts.size() == 5',source)
        self.assertIn('parts[4].shape(-1) == hidden_ * 3',source)

    def test_flux_gpu_uses_fused_rope_sdpa_and_compiled_step_boundary(self):
        transformer=(ROOT/'native/models/flux2/flux_transformer.cpp').read_text()
        mlx=(ROOT/'native/backends/mlx.cpp').read_text()
        pipeline=(ROOT/'native/models/flux2/pipeline.cpp').read_text()
        results=(ROOT/'native/platform/apple/results.mm').read_text()
        self.assertIn('rope_pairs_pair', transformer)
        self.assertIn('TURBOCIDER_FLUX_EAGER_ROPE', transformer)
        self.assertIn('TURBOCIDER_FLUX_FORCE_FUSED_SDPA', transformer)
        self.assertIn('TURBOCIDER_FLUX_SYNC_BLOCKS', transformer)
        self.assertIn('tc_rope_qk', mlx)
        self.assertIn('r.compile_gpu = true', pipeline)
        self.assertIn('@"compiled_hybrid_complement"', results)

    def test_flux_hybrid_prefix_computes_gpu_mlp_complement(self):
        transformer=(ROOT/'native/models/flux2/flux_transformer.cpp').read_text()
        coreml=(ROOT/'native/backends/coreml.mm').read_text()
        exporter=(ROOT/'tools/coreml/export_flux2.py').read_text()
        resources=(ROOT/'native/backends/coreml_resources.mm').read_text()
        self.assertIn('make_hybrid_gpu_graph',transformer)
        self.assertIn('projection_offset + gpu_mlp_start',transformer)
        self.assertIn('hidden + gpu_mlp_start',transformer)
        self.assertIn('hybrid_->ane_mlp_end',transformer)
        self.assertIn('@"ane_mlp_end"',coreml)
        self.assertIn('ane_mlp_end <= mlp_width',coreml)
        self.assertIn("--ane-mlp-width",exporter)
        self.assertIn("'ane_mlp_end':a.ane_mlp_width",exporter)
        self.assertIn('ane_mlp_width',resources)

    def test_qwen3_hybrid_quality_gate_is_explicit_and_telemetry_is_recorded(self):
        qwen=(ROOT/'native/components/text/qwen3.cpp').read_text()
        coreml=(ROOT/'native/backends/coreml.mm').read_text()
        session=(ROOT/'native/runtime/session.hpp').read_text()
        results=(ROOT/'native/platform/apple/results.mm').read_text()
        probe=(ROOT/'tools/native/qwen3_quant_probe.cpp').read_text()
        plan_probe=(ROOT/'tools/native/qwen3_prefill_plan_probe.cpp').read_text()
        exporter=(ROOT/'tools/coreml/export_qwen3.py').read_text()
        for token in [
            'TURBOCIDER_QWEN3_MAX_RELATIVE_L2',
            'TURBOCIDER_QWEN3_MIN_COSINE',
            'TURBOCIDER_QWEN3_MAX_RELATIVE_ABS',
            'TURBOCIDER_QWEN3_FUSED_SDPA',
            'TURBOCIDER_QWEN3_FUSED_SDPA_MIN_TOKENS',
            'TURBOCIDER_QWEN3_MLP_MAX_PADDING_PERCENT',
            'TURBOCIDER_QWEN3_DISABLE_MLP_TAIL_PADDING',
            'TURBOCIDER_QWEN3_DISABLE_HYBRID_OVERLAP',
            'qwen3_prefill_plan',
            'record_quality(relative_l2, cosine, absolute,',
            'Qwen3 hybrid MLP quality gate failed',
        ]:
            self.assertIn(token, qwen)
        self.assertIn('void HybridSession::record_quality', coreml)
        for token in [
            'quality_validation_calls', 'quality_max_relative_l2',
            'quality_min_cosine', 'quality_max_abs',
            'quality_max_relative_abs', 'quality_validation_passed',
            'prefill_actual_tokens', 'prefill_compute_tokens',
            'prefill_padding_tokens', 'prefill_fixed_shape',
            'prefill_plan_reason',
            'runtime_failures', 'runtime_failed', 'runtime_failure_block',
        ]:
            self.assertIn(token, session)
            self.assertIn(token, coreml)

        sweep=(ROOT/'tools/native/benchmark_qwen3_encoder_sweep.py').read_text()
        mlx_adapter=(ROOT/'tools/native/qwen3_mlx_encoder_probe.py').read_text()
        h3_sweep=(ROOT/'tools/native/benchmark_h3_encoder_sweep.py').read_text()
        for token in [
            'turbocider-qwen3-encoder-sweep-v1',
            'must point to a compiled-cache',
            'warm_median_seconds',
            'warm_cv',
            'max_warm_cv',
            'stability_passed',
            'min_warm_samples',
            'min_warm_speedup',
            'relative_l2',
            'output_copy_bytes_session_total',
            'hybrid_executed',
            'TURBOCIDER_QWEN3_ANE_MIN_TOKENS',
            'TURBOCIDER_QWEN3_MLP_MAX_PADDING_PERCENT',
            'mlx_reference_quality_vs_gpu',
            'hybrid_quality_vs_mlx_reference',
            'min_mlx_reference_speedup',
            'mlx_reference_stability_passed',
            'runtime_failure_block',
            'qualified_flexible_backing',
        ]:
            self.assertIn(token, sweep)
        for token in [
            'mlx_lm.models.qwen3', 'deterministic token IDs',
            'OUTPUT_LAYERS', 'scaled_dot_product_attention',
            'model.load_weights', 'mlx_peak_bytes',
            'safetensors directory',
        ]:
            self.assertIn(token, mlx_adapter)
        for token in [
            'turbocider-h3-encoder-sweep-v1',
            'gpu_reference', 'gpu_sdpa',
            'gpu_ane_sdpa', 'warm_median_seconds',
            'warm_speedup_reference',
            'warm_speedup_vs_gpu_reference',
            '"gpu_sdpa" if variant == "gpu_ane_sdpa"',
            'max_warm_cv', 'min_warm_samples', 'warmup_discard',
            'discarded_warm_seconds',
            'warm_cv', 'stability_passed', 'reference_stability_passed',
            'conservative_warm_speedup',
            'min(results[speed_reference]["warm_seconds"])',
            'max(results[variant]["warm_seconds"])',
            'final 50-layer hidden states compared after timing',
        ]:
            self.assertIn(token, h3_sweep)
        for token in [
            'ane_first_runtime_seconds_session_total',
            'ane_subsequent_runtime_seconds_session_total',
            'output_copy_bytes_session_total',
            'coreml_output_backing_setup_seconds',
            'coreml_block_count',
            'qualified_flexible_backing',
        ]:
            self.assertIn(token, probe)
        for token in [
            'first_runtime_prediction_seconds_session_total',
            'subsequent_runtime_prediction_seconds_session_total',
            'output_copy_bytes_session_total',
            'output_backing_setup_seconds',
        ]:
            self.assertIn(token, results)
        self.assertIn('Qwen3Conditioning::flux_klein()', probe)
        self.assertIn('turbocider-qwen3-prefill-plan-v1', plan_probe)
        self.assertIn('qwen3_prefill_plan(', plan_probe)
        self.assertIn('qwen3_prefill_plan',
                      (ROOT/'tools/validation/build_quantization_probes.sh').read_text())
        self.assertIn('hybrid_bucket_plan', coreml)
        self.assertIn('qwen3_prefill_plan(',
                      (ROOT/'native/models/flux2/pipeline.cpp').read_text())
        self.assertIn('qwen3_prefill_plan(',
                      (ROOT/'native/models/z_image/z_image.cpp').read_text())
        self.assertIn('qwen3_checkpoint_path(weight_path)', probe)
        self.assertIn('.safetensors.index.json', probe)
        self.assertIn('weight_path.parent_path()', probe)
        self.assertIn('"output_scale": args.output_scale', exporter)
        self.assertIn('np.float16(1.0 / args.output_scale)', exporter)
        self.assertIn('ane = ane * Tensor(active_hybrid->output_scale', qwen)
        self.assertIn('bool used_hybrid_output = false', qwen)
        self.assertIn('if (used_hybrid_output ||', qwen)
        self.assertIn('allow_flexible_backing', coreml)
        self.assertIn('output_ = (!flexible_ || allow_flexible_backing_) ? output : nil',
                      coreml)
        self.assertIn('impl_->allow_flexible_backing = impl_->flexible && qualified_flexible_backing',
                      coreml)
        self.assertIn('if (output_)', coreml)
        self.assertIn('if (!output_ || actual_output.dataPointer != output_.dataPointer)',
                      coreml)
        self.assertIn('required_blocks ? required_blocks : manifest_blocks', coreml)
        self.assertIn('std::vector<LoRAAsset>{}, 0, 27, true',
                      (ROOT/'native/models/flux2/pipeline.cpp').read_text())
        self.assertIn('std::vector<LoRAAsset>{}, 0, 35, true',
                      (ROOT/'native/models/z_image/z_image.cpp').read_text())
        self.assertIn('mode == "flux_klein" ? 27 : 35, true', probe)

    def test_qwen3_sweep_and_gemma_mlp_probe_are_fail_closed(self):
        sweep=(ROOT/'tools/native/benchmark_qwen3_encoder_sweep.py').read_text()
        gemma_header=(ROOT/'native/models/ltx_runtime/ltx_gemma_encoder.h').read_text()
        gemma_runtime=(ROOT/'native/models/ltx_runtime/ltx_gemma_encoder.m').read_text()
        gemma_probe=(ROOT/'tools/native/ltx_gemma_mlp_probe.c').read_text()
        build=(ROOT/'tools/native/build.sh').read_text()
        for token in [
            'MODE_GEOMETRY', 'required_blocks', 'INVALID_METRIC',
            'encoder manifest geometry does not match',
            'encoder manifest needs blocks',
            'compiled-cache manifest',
        ]:
            self.assertIn(token, sweep)
        for token in [
            'LTX_GEMMA_MLP_PROBE_MAX_RUNS',
            'ltx_gemma_mlp_probe_result',
            'ltx_gemma_mlp_gpu_probe',
            'resident_weight_bytes',
            'workspace_bytes',
        ]:
            self.assertIn(token, gemma_header)
            self.assertIn(token, gemma_runtime)
        for token in [
            'turbocider-ltx-gemma-mlp-gpu-probe-v1',
            'synthetic_input', 'batch_commands', 'input.bf16', 'output.bf16',
            'output_all_finite',
        ]:
            self.assertIn(token, gemma_probe)
        compare=(ROOT/'tools/native/benchmark_ltx_gemma_mlp_compare.py').read_text()
        for token in [
            'turbocider-ltx-gemma-mlp-compare-v1',
            'staged', 'fused', 'warm_speedup',
            'relative_l2', 'min_warm_speedup',
        ]:
            self.assertIn(token, compare)
        sweep_path=ROOT/'tools/native/benchmark_ltx_gemma_mlp_sweep.py'
        sweep_source=sweep_path.read_text()
        for token in [
            'DEFAULT_ROWS = (64, 128, 256, 512, 1024)',
            'turbocider-ltx-gemma-mlp-gpu-sweep-v1',
            'quality": "not_applicable_isolated_gpu_baseline"',
            'prior evidence is preserved',
        ]:
            self.assertIn(token, sweep_source)
        self.assertIn('ltx_gemma_mlp_probe_tool.o', build)
        self.assertIn('ltx-gemma-mlp-probe', build)
        self.assertIn('TURBOCIDER_BUILD_EXPERIMENTAL_PROBES', build)
        self.assertIn('if [[ "$EXPERIMENTAL_PROBES" == "1" ]]', build)

    def test_qwen3_exporter_rejects_symlinked_or_escaping_sources(self):
        import importlib.util
        exporter_path = ROOT/'tools/coreml/export_qwen3.py'
        spec = importlib.util.spec_from_file_location('export_qwen3_contract', exporter_path)
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/'model'; model.mkdir()
            (model/'model.safetensors').write_bytes(b'fixture')
            source = exporter.QwenSource(model)
            self.assertEqual(source.checkpoint.name, 'model.safetensors')
            self.assertIsNone(source.weight_map)
            (model/'model.safetensors').unlink()
            (model/'model.safetensors').symlink_to(root/'outside.safetensors')
            (root/'outside.safetensors').write_bytes(b'outside')
            with self.assertRaises(ValueError):
                exporter.QwenSource(model)

            index = model/'model.safetensors.index.json'
            (model/'model.safetensors').unlink()
            index.write_text(json.dumps({'weight_map': {
                'model.layers.0.mlp.gate_proj.weight': '../outside.safetensors'}}))
            with self.assertRaises(ValueError):
                exporter.QwenSource(model)

            artifact = root/'block.mlpackage'; artifact.mkdir()
            payload = artifact/'weights.bin'; payload.write_bytes(b'weights')
            receipt = root/'block.json'
            receipt.write_text(json.dumps({'../outside.safetensors':
                                           hashlib.sha256(b'outside').hexdigest()}))
            with self.assertRaises(ValueError):
                exporter.artifact_receipt(artifact, receipt)
            receipt.write_text(json.dumps({'weights.bin':
                                           hashlib.sha256(b'weights').hexdigest()}))
            self.assertEqual(exporter.artifact_receipt(artifact, receipt),
                             json.loads(receipt.read_text()))

    def test_flux_m4_max_profile_is_fail_closed_to_validated_prefix(self):
        profile=json.loads((ROOT/'profiles/apple-m4-max-64gb.example.json').read_text())
        model=profile['models']['flux2-klein-4b']
        self.assertEqual(profile['match'],{'gpu_name':'Apple M4 Max','memory_bytes':64<<30})
        self.assertEqual(model['coreml_export']['ane_mlp_width'],6144)
        self.assertIn('a6144',model['ane_manifest'])
        discovery=(ROOT/'apps/macos/AccelerationDiscovery.swift').read_text()
        self.assertIn('Apple M4 Max',discovery)
        self.assertIn('end == 6144',discovery)

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
        runtime=(ROOT/'native/models/ltx_runtime/ltx_gemma_encoder.m').read_text()
        gpu=(ROOT/'native/models/ltx_runtime/ltx_gpu.h').read_text()
        shader=(ROOT/'native/models/ltx_runtime/ltx_shaders.metal').read_text()
        self.assertIn('TURBOCIDER_LTX_GEMMA_FUSED_MLP', runtime)
        self.assertIn('TURBOCIDER_LTX_GEMMA_GPU_TAPS', runtime)
        self.assertIn('ltx_gpu_gemma_projection_tap_bf16', runtime)
        self.assertIn('ltx_gpu_gated_mlp_int8_convrot_mps_bf16', gpu)
        self.assertIn('ltx_gpu_gemma_projection_tap_bf16', gpu)
        self.assertIn('kernel void ltx_gemma_projection_tap_bf16', shader)

    def test_ltx_gemma_weight_streaming_avoids_mmap_page_residency(self):
        runtime=(ROOT/'native/models/ltx_runtime/ltx_gemma_encoder.m').read_text()
        gpu=(ROOT/'native/models/ltx_runtime/ltx_gpu.m').read_text()
        start=runtime.index('static ltx_gpu_buffer *gemma_load_tensor(')
        tensor_loader=runtime[start:runtime.index(
            'static int gemma_load_linear(', start)]
        self.assertIn('ltx_st_read_mapped_data', tensor_loader)
        self.assertIn('ltx_gpu_buffer_contents(buffer)', tensor_loader)
        self.assertNotIn('gemma_tensor_bytes', tensor_loader)
        self.assertIn('ltx_st_map_discard(&encoder->mapping', runtime)
        self.assertIn('TURBOCIDER_LTX_GEMMA_RESIDENT_WEIGHTS', runtime)
        self.assertIn('TURBOCIDER_LTX_GEMMA_ANE_LOAD_WORKERS', runtime)
        self.assertIn('gemma_ane_preload_worker', runtime)
        self.assertIn('ltx_gemma_encoder_prepare_prompt', runtime)
        self.assertIn('strcmp(resident_weights, "1") == 0', runtime)
        self.assertIn('ltx_gpu_buffer_retain(cached)', tensor_loader)
        self.assertIn('resident_weight_cache_hits', runtime)
        self.assertIn('resident_weight_cache_misses', runtime)
        self.assertLess(
            runtime.index('free(encoder->resident_weight_cache)'),
            runtime.index('ltx_gpu_free(encoder->gpu)'),
        )
        self.assertIn('uint32_t references;', gpu)
        self.assertIn('ltx_gpu_buffer *ltx_gpu_buffer_retain', gpu)

    def test_ltx_gemma_ane_exporter_is_shape_and_convrot_gated(self):
        exporter=(ROOT/'tools/coreml/export_ltx_gemma_mlp.py').read_text()
        for token in [
            'ltx-gemma-ane-mlp-v1', 'HIDDEN = 3840',
            'INTERMEDIATE = 15360', 'LAYERS = 48',
            'convrot_last_axis', 'gelu-tanh-gated',
            'gpu_gate.weight.i8', 'gpu_up.weight.i8',
            'gpu_down.weight.i8', 'ane_intermediate',
            'caller-owned-row-major-fp16-partial',
        ]:
            self.assertIn(token, exporter)
        runtime=(ROOT/'native/models/ltx_runtime/ltx_gemma_ane_mlp.m').read_text()
        encoder=(ROOT/'native/models/ltx_runtime/ltx_gemma_encoder.m').read_text()
        gpu=(ROOT/'native/models/ltx_runtime/ltx_gpu.m').read_text()
        header=(ROOT/'native/models/ltx_runtime/ltx_gemma_ane_mlp.h').read_text()
        build=(ROOT/'tools/native/build.sh').read_text()
        probe=(ROOT/'tools/native/ltx_gemma_ane_mlp_probe.c').read_text()
        benchmark=(ROOT/'tools/native/benchmark_ltx_gemma_ane_mlp.py').read_text()
        for token in [
            'ltx-gemma-ane-mlp-v1', 'CPUAndNeuralEngine',
            'caller-owned output backing', 'ltx_gpu_gated_mlp_int8_convrot_mps_bf16',
            'gm_constraint_supports_shape', 'sha256',
            'gm_persistent_worker', 'gm_worker_initialize',
            'gm_prediction_start', 'gm_prediction_wait',
            'pthread_cond_wait', 'pthread_cond_broadcast',
        ]:
            self.assertIn(token, runtime)
        evaluate = runtime[runtime.index('int ltx_gemma_ane_mlp_eval('):]
        self.assertNotIn('pthread_create', evaluate)
        free = runtime[runtime.index('void ltx_gemma_ane_mlp_free('):
                       runtime.index(
                           'const ltx_gemma_ane_mlp_shape *',
                           runtime.index('void ltx_gemma_ane_mlp_free('),
                       )]
        self.assertLess(free.index('gm_prediction_wait'),
                        free.index('pthread_join'))
        self.assertLess(free.index('pthread_join'), free.index('mlp->model = nil'))
        for token in [
            'ltx_gemma_ane_mlp_create', 'ltx_gemma_ane_mlp_eval',
            'ltx_gemma_ane_mlp_workspace_bytes',
        ]:
            self.assertIn(token, header)
        self.assertIn('ltx_gemma_ane_mlp', build)
        self.assertIn('ltx-gemma-ane-mlp-probe', build)
        self.assertIn('TURBOCIDER_BUILD_EXPERIMENTAL_PROBES', build)
        self.assertIn('turbocider-ltx-gemma-ane-mlp-probe-v1', probe)
        self.assertIn('turbocider-ltx-gemma-ane-mlp-qualification-v1', benchmark)
        self.assertIn('TURBOCIDER_LTX_GEMMA_ANE_MAX_PADDING_PERCENT', encoder)
        self.assertIn('maximum_padding_percent = 0u', encoder)
        self.assertIn('ltx_gpu_buffer_copy(encoder->gpu, output', encoder)
        self.assertNotIn('ltx_gpu_slice_rows_bf16_f16(\n            encoder->gpu, output',
                         encoder)
        self.assertIn('copyFromBuffer:ltx_buffer(input)', gpu)

    def test_ltx_gemma_full_encoder_sweep_is_resident_and_quality_gated(self):
        tool = (ROOT/'tools/native/ltx_gemma_encode.c').read_text()
        sweep = (ROOT/'tools/native/benchmark_ltx_gemma_encoder_sweep.py').read_text()
        encoder = (
            ROOT/'native/models/ltx_runtime/ltx_gemma_encoder.m'
        ).read_text()
        for token in [
            'WARM_RUNS', 'warm_runs', 'create_seconds', 'prepare_seconds',
            'first_seconds',
            'warm_seconds', 'fused_mlp', 'gpu_taps',
            'raw_video_context.bf16',
            'raw_audio_context.bf16', 'raw_text_mask.bf16',
        ]:
            self.assertIn(token, tool)
        for token in [
            'turbocider-ltx-gemma-encoder-sweep-v2',
            'separate staged/fused GPU processes',
            'full raw video/audio/mask outputs compared',
            'warm_median_seconds', 'process_peak_rss_bytes',
            'relative_l2', 'relative_max_abs', 'min_warm_speedup',
            'not_claimed_single_layer_qualification_only',
            'fused_gpu_taps', 'TURBOCIDER_LTX_GEMMA_GPU_TAPS',
            'quality_by_variant', '--variants', 'warm_cv',
            'MIN_QUALIFYING_WARM_RUNS', 'stability_passed',
            'gpu_taps_ab_vs_fused',
            'resident_weight_telemetry',
            'resident_weight_cache_hits',
            'resident_weight_cache_misses',
            'ane_preload_models_session_total',
            'ane_preload_seconds_session_total',
        ]:
            self.assertIn(token, sweep)
        self.assertIn(
            'encoder->ane_preload_workers = started + 1u', encoder,
        )
        session = (ROOT/'native/platform/apple/ltx_session.mm').read_text()
        for token in [
            'encoder_resident_weights_enabled',
            'encoder_resident_weight_cache_hits',
            'encoder_resident_weight_cache_misses',
            'encoder_resident_weight_bytes',
            'encoder_ane_preload_models_session_total',
            'encoder_ane_preload_workers',
            'encoder_ane_preload_seconds_session_total',
        ]:
            self.assertIn(token, session)

    def test_ltx_video_executor_and_gated_capabilities(self):
        pointer=lib.tc_models_json()
        payload=json.loads(consume(C.c_void_p(pointer)))
        model=next(value for value in payload['models']
                   if value['id']=='ltx-2.5-distilled')
        self.assertTrue(model['executor'])
        self.assertEqual(model['executor_operations'],['video.generate','video.image'])
        self.assertEqual(model['default_residency'],'component_staged')
        self.assertTrue(model['native_gemma4_candidate'])
        self.assertTrue(model['native_conditioning_connector'])
        self.assertTrue(model['native_i2v_clean_prefix'])
        self.assertTrue(model['native_gpu_ane_profile'])
        self.assertTrue(model['supports_lora'])
        self.assertFalse(model['runtime_lora'])
        self.assertEqual(model['lora_mode'],'premerged-manifest')
        self.assertEqual(model['lora_strategies'],['disk_premerge'])
        self.assertEqual(model['default_lora_strategy'],'disk_premerge')
        self.assertTrue(any('broader prompt-suite qualification remains pending'
                            in value for value in model['candidate_limitations']))
        self.assertTrue(any('5-second multi-prompt quality suite remains pending' in value
                            for value in model['candidate_limitations']))
        self.assertTrue(model['audio_output'])
        self.assertTrue(model['native_audio_output_candidate'])
        self.assertTrue(model['native_audio_vae_candidate'])
        self.assertTrue(model['native_base_vocoder_candidate'])
        self.assertEqual(model['audio_capability'],'latent_to_48khz_aac_candidate')
        self.assertFalse(model['default_audio'])

    def test_ltx_shared_hot_path_matches_recorded_source(self):
        shared=[
            'ltx.c','ltx_conditioning.c','ltx_connector.c',
            'ltx_transformer_io.c','ltx_latent_stats.c','ltx_rng.c',
            'ltx_shaders.metal','ltx_ane_mlp.m','ltx_ane_v2a.m',
            'ltx_ane_kv.m','ltx_ane_qkv.m','ltx_mlx_upsampler.cpp',
            'ltx_upsampler.m',
        ]
        vendored=ROOT/'native/models/ltx_runtime'
        manifest=json.loads((vendored/'SOURCE_MANIFEST.json').read_text())
        self.assertEqual(manifest['schema_version'],1)
        self.assertEqual(set(manifest['files']),set(shared))
        for name in shared:
            with self.subTest(source=name):
                self.assertEqual(hashlib.sha256((vendored/name).read_bytes()).hexdigest(),manifest['files'][name])
        # Explicit development gate; unrelated sibling edits must not change the
        # result of a standalone checkout's default test suite.
        reference=os.environ.get('TURBOCIDER_LTX_REFERENCE')
        if reference:
            upstream=Path(reference)
            self.assertTrue(upstream.is_dir(),'Configured LTX reference is missing')
            for name in shared:
                with self.subTest(reference_source=name):
                    self.assertEqual((vendored/name).read_bytes(),(upstream/name).read_bytes())

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
            self.assertIn('pinned_distribution_identity',payload['blocking_components'])
            artifact.write_bytes(b'validated audio bundlE')
            code,payload,error=preflight(root)
            self.assertEqual(code,0,error)
            self.assertEqual(payload['status'],'invalid')
            self.assertIn('SHA-256',payload['reason'])
    def test_llada_native_text_contract_and_edit_is_not_advertised(self):
        text={
            'model':'llada-image-turbo','operation':'image.generate',
            'prompt':'A red fox in snow','width':256,'height':256,
            'steps':4,'frames':1,'audio':False,'execution':'gpu',
        }
        code,p,error=plan(text)
        self.assertEqual(code,0,error)
        self.assertTrue(p['executable'])
        self.assertEqual(p['backend'],'mlx_cpp_metal')
        self.assertEqual(p['validation'],'native_llada_candidate')
        self.assertEqual(p['decoded_shape'],[256,256,1])
        self.assertEqual([stage['id'] for stage in p['stages']],
                         ['text_encode','denoise','vae_decode','export'])
        edit={**text,'operation':'image.edit','inputs':[
            {'kind':'image','role':'reference','path':'/tmp/reference.png'}
        ]}
        self.assertNotEqual(plan(edit)[0],0)
        models=json.loads(consume(C.c_void_p(lib.tc_models_json())))['models']
        descriptor=next(value for value in models
                        if value['id']=='llada-image-turbo')
        self.assertEqual(descriptor['operations'],['image.generate'])
        self.assertEqual(descriptor['inputs'],['text'])
        self.assertNotIn('Python',descriptor['runtime_dependency'])
        for change in [
            {'steps':5}, {'execution':'gpu_ane','ane_manifest':'/tmp/llada.json'},
            {'operation':'image.edit','inputs':[]},
            {'operation':'image.edit','width':272,'inputs':edit['inputs']},
            {'loras':[{'path':'/tmp/adapter.safetensors','role':'transformer'}]},
        ]:
            with self.subTest(change=change):
                self.assertNotEqual(plan({**text,**change})[0],0)

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
    def test_resource_api_rejects_missing_engine(self):
        out, err = C.c_void_p(), C.c_void_p()
        self.assertNotEqual(lib.tc_engine_load(None, None, None, C.byref(out), C.byref(err)), 0)
        self.assertIsNone(consume(out)); self.assertIn('missing engine', consume(err))
        self.assertNotEqual(lib.tc_engine_unload(None, C.byref(out), C.byref(err)), 0)
        self.assertIsNone(consume(out)); self.assertIn('missing engine', consume(err))
    def test_abi(self): self.assertEqual(lib.tc_abi_version(),1)

if __name__=='__main__':unittest.main(verbosity=2)
