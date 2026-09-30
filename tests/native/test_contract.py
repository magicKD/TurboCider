import ctypes as C
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get('TURBOCIDER_TEST_NATIVE_DIR', ROOT / 'build/native')).resolve()
lib = C.CDLL(str(NATIVE / 'libturbocider.dylib'))
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
    def test_qwen21_runtime_qkv_is_base_only_and_distinct_from_ffn(self):
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A fox',
                    width=1024, height=1024, steps=5, audio=False, frames=1,
                    execution='gpu_ane', allow_approximation=True, residency='resident',
                    hybrid_mlp_mode='runtime_qkv', ane_manifest='matmul-checked-at-execution.json')
        code, result, error = plan(base)
        self.assertEqual(code, 0, error)
        self.assertEqual(result['gpu_graph'], 'runtime_weight_token_row_qkv')
        self.assertIn('runtime_weight_fp16_token_row_qkv', result['algorithm_approximations'])
        self.assertNotIn('runtime_weight_fp16_token_row_ffn', result['algorithm_approximations'])
        for invalid in (dict(execution='gpu'), dict(allow_approximation=False),
                        dict(ane_manifest=''), dict(residency='component_staged'),
                        dict(width=512, height=512),
                        dict(operation='image.edit', inputs=[dict(kind='image', role='reference',
                            path='ref.png')]), dict(qwen21_w8a8=True),
                        dict(loras=[dict(path='a.safetensors',
                            role='transformer', strength=1.)], lora_strategy='inference_time')):
            with self.subTest(invalid=invalid):
                self.assertNotEqual(plan({**base, **invalid})[0], 0)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC': '1'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC': '1'}):
            self.assertNotEqual(plan(base)[0], 0)
        for flag in ('TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN',
                     'TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC',
                     'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC'):
            with patch.dict(os.environ, {flag: '1'}):
                self.assertNotEqual(plan(base)[0], 0)

    def test_request_local_qwen21_hybrid_mlp_modes(self):
        adapter = dict(path='Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors',
                       strength=1, role='transformer')
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A fox',
                    width=512, height=512, steps=6, audio=False, frames=1,
                    execution='gpu_ane', allow_approximation=True, qwen21_w8a8=True,
                    ane_manifest='manifest-checked-at-execution.json', lora_strategy='inference_time',
                    loras=[adapter])
        with patch.dict(os.environ, {
            'TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC': '0',
            'TURBOCIDER_QWEN21_LORA_GATE_UP_DIAGNOSTIC': '0'
        }):
            for mode, expected in (
                ('lora_suffix', 'qwen21_runtime_lora_base_ane_suffix_only_diagnostic'),
                ('lora_gate_up', 'qwen21_w8a8_gate_up_gpu_silu_down_diagnostic'),
                ('lora_fused', 'qwen21_w8a8_fused_lora_pre_silu_diagnostic')
            ):
                with self.subTest(mode=mode):
                    code, result, error = plan({**base, 'hybrid_mlp_mode': mode})
                    self.assertEqual(code, 0, error)
                    self.assertEqual(result['hybrid_mlp_mode'], mode)
                    self.assertIn(expected, result['algorithm_approximations'])
            for bad in ('base_fused', 'unknown'):
                self.assertNotEqual(plan({**base, 'hybrid_mlp_mode': bad})[0], 0)
            self.assertNotEqual(plan({**base, 'execution': 'gpu', 'hybrid_mlp_mode': 'lora_suffix'})[0], 0)
            code, result, error = plan({**base, 'loras': [], 'steps': 5,
                                        'lora_strategy': 'auto', 'hybrid_mlp_mode': 'base_fused'})
            self.assertEqual(code, 0, error)
            self.assertEqual(result['hybrid_mlp_mode'], 'base_fused')
            for steps in (5, 6, 20, 40):
                code, result, error = plan({**base, 'loras': [], 'steps': steps,
                                            'lora_strategy': 'auto',
                                            'hybrid_mlp_mode': 'lora_fused'})
                self.assertEqual(code, 0, error)
                self.assertEqual(result['hybrid_mlp_mode'], 'lora_fused')
            other = dict(path='alternate-six-step-adapter.safetensors', strength=0.75,
                         role='transformer')
            code, result, error = plan({**base, 'loras': [other],
                                        'hybrid_mlp_mode': 'lora_fused'})
            self.assertEqual(code, 0, error)
            self.assertEqual(result['hybrid_mlp_mode'], 'lora_fused')
            self.assertNotEqual(plan({**base, 'loras': [other],
                                      'hybrid_mlp_mode': 'lora_suffix'})[0], 0)
            self.assertNotEqual(plan({**base, 'loras': [], 'steps': 7,
                                      'hybrid_mlp_mode': 'lora_fused'})[0], 0)

    def test_z_image_request_local_base_and_runtime_lora_suffix(self):
        base = dict(model='z-image-turbo', operation='image.generate', prompt='A fox',
                    width=512, height=512, steps=8, audio=False, frames=1,
                    execution='gpu_ane', residency='resident', allow_approximation=True,
                    ane_manifest='base-manifest-checked-at-execution.json')
        adapter = dict(path='z_image_turbo_distill_patch_lora_bf16.safetensors',
                       strength=1, role='transformer')
        code, result, error = plan({**base, 'hybrid_mlp_mode': 'base_fused'})
        self.assertEqual(code, 0, error)
        self.assertEqual(result['hybrid_mlp_mode'], 'base_fused')
        runtime = {**base, 'hybrid_mlp_mode': 'lora_suffix', 'loras': [adapter],
                   'lora_strategy': 'inference_time'}
        code, result, error = plan(runtime)
        self.assertEqual(code, 0, error)
        self.assertEqual(result['hybrid_mlp_mode'], 'lora_suffix')
        self.assertIn('z_image_runtime_lora_base_ane_suffix_only_diagnostic',
                      result['algorithm_approximations'])
        code, result, error = plan({**runtime, 'hybrid_mlp_mode': 'lora_fused'})
        self.assertEqual(code, 0, error)
        self.assertIn('z_image_runtime_lora_base_fused_pre_silu_experimental',
                      result['algorithm_approximations'])
        code, result, error = plan({**base, 'hybrid_mlp_mode': 'lora_fused'})
        self.assertEqual(code, 0, error)
        self.assertEqual(result['hybrid_mlp_mode'], 'lora_fused')
        # The same explicit base-only FFN mode is request-local: it must not
        # lock the artifact choice to a particular runtime adapter identity.
        other_adapter = dict(path='different-transformer-lora.safetensors',
                             strength=0.75, role='transformer')
        code, result, error = plan({**runtime, 'loras': [other_adapter],
                                    'hybrid_mlp_mode': 'lora_fused'})
        self.assertEqual(code, 0, error)
        self.assertEqual(result['hybrid_mlp_mode'], 'lora_fused')
        code, result, error = plan({**runtime, 'hybrid_mlp_mode': 'lora_merged',
                                    'lora_strategy': 'in_memory_merge'})
        self.assertEqual(code, 0, error)
        self.assertEqual(result['hybrid_mlp_mode'], 'lora_merged')
        for invalid in (dict(lora_strategy='in_memory_merge'), dict(steps=9),
                        dict(width=1024), dict(residency='streamed'),
                        dict(hybrid_mlp_mode='lora_gate_up'),
                        dict(loras=[], hybrid_mlp_mode='lora_suffix')):
            with self.subTest(invalid=invalid):
                self.assertNotEqual(plan({**runtime, **invalid})[0], 0)

    def test_qwen21_dbcache_diagnostic_gate(self):
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A teapot',
                    width=512, height=512, steps=20, audio=False, frames=1,
                    execution='gpu', allow_approximation=True)
        flag = 'TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC'
        threshold = 'TURBOCIDER_QWEN21_DBCACHE_THRESHOLD'
        max_consecutive = 'TURBOCIDER_QWEN21_DBCACHE_MAX_CONSECUTIVE'
        with patch.dict(os.environ, {flag: '1', threshold: '0.08'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertIn('qwen21_decode_dbcache_diagnostic', result['algorithm_approximations'])
            for count in (1, 2, 3):
                refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                        for i in range(count)]
                self.assertEqual(plan({**base, 'operation': 'image.edit', 'inputs': refs,
                                       'steps': 40})[0], 0)
            hybrid = {**base, 'execution': 'gpu_ane', 'qwen21_w8a8': True,
                      'ane_manifest': 'not-loaded-in-planning.json'}
            self.assertEqual(plan(hybrid)[0], 0)
            rectangle = {**base, 'width': 768, 'height': 512}
            portrait = {**base, 'width': 512, 'height': 768}
            self.assertEqual(plan(rectangle)[0], 0)
            self.assertEqual(plan(portrait)[0], 0)
            for shape in (rectangle, portrait):
                candidate = {**shape, **{key: hybrid[key] for key in (
                    'execution', 'qwen21_w8a8', 'ane_manifest')}}
                self.assertNotEqual(plan(candidate)[0], 0)
                with patch.dict(os.environ, {'TURBOCIDER_QWEN21_RECT_W8A8_DIAGNOSTIC': '1'}):
                    code, combined, error = plan(candidate)
                    self.assertEqual(code, 0, error)
                    self.assertIn('qwen21_decode_dbcache_diagnostic',
                                  combined['algorithm_approximations'])
                    self.assertIn('qwen21_rectangular_decode_w8a8_tiled_diagnostic',
                                  combined['algorithm_approximations'])
                    self.assertNotEqual(plan({**candidate, 'qwen21_reference_size': 512})[0], 0)
            for invalid in [dict(steps=6), dict(steps=41), dict(width=640),
                            dict(allow_approximation=False),
                            dict(operation='image.edit', inputs=[dict(kind='image', role='reference',
                                                                   path='ref.png')] * 4),
                            dict(execution='gpu_ane', ane_manifest='')]:
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)
        for value in ('1', '2', '4', '8'):
            with patch.dict(os.environ, {flag: '1', max_consecutive: value}):
                self.assertEqual(plan(base)[0], 0, value)
        for value in ('', '0', '9', '2bad', '-1'):
            with patch.dict(os.environ, {flag: '1', max_consecutive: value}):
                self.assertNotEqual(plan(base)[0], 0, value)
        with patch.dict(os.environ, {flag: '0', max_consecutive: '4'}):
            self.assertNotEqual(plan(base)[0], 0)
        for value in ('', 'nan', 'inf', '-1', '0', '0.6', '0.08bad'):
            with patch.dict(os.environ, {flag: '1', threshold: value}):
                self.assertNotEqual(plan(base)[0], 0, value)
        with patch.dict(os.environ, {flag: '2'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_memory_constrained_contract_is_explicit_and_fail_closed(self):
        request = {
            'model': 'flux2-klein-4b', 'width': 512, 'height': 512,
            'frames': 1, 'audio': False, 'execution': 'gpu',
        }
        status, ordinary, error = plan(request)
        self.assertEqual(status, 0, error)
        self.assertNotIn('memory_policy', ordinary)

        constrained = {
            **request,
            'memory_constrained': {
                'enabled': True,
                'limit_bytes': 16 * (1 << 30),
                'buffer_percent': 15,
                'min_free_bytes': 1 << 30,
                'max_refill_slots': 2,
                'allow_quality_preserving_tiling': True,
            },
        }
        status, configured, error = plan(constrained)
        self.assertEqual(status, 0, error)
        policy = configured['memory_policy']
        self.assertTrue(policy['enabled'])
        self.assertEqual(policy['user_limit_bytes'], 16 * (1 << 30))
        self.assertEqual(policy['effective_budget_bytes'], 14602888806)
        self.assertEqual(policy['buffer_percent'], 15)
        self.assertEqual(policy['max_refill_slots'], 2)
        self.assertEqual(policy['admission_state'], 'rejected')
        self.assertFalse(policy['execution_supported'])
        self.assertFalse(policy['estimate_fits'])
        self.assertEqual(policy['capability_level'], 'declared')
        self.assertEqual(policy['certification_state'], 'unsupported')
        self.assertFalse(policy['release_stable'])
        self.assertTrue(policy['digest'])

        disabled = {**request, 'memory_constrained': {'enabled': False}}
        status, configured, error = plan(disabled)
        self.assertEqual(status, 0, error)
        self.assertNotIn('memory_policy', configured)

        for memory in (
            {'enabled': True},
            {'enabled': True, 'limit_bytes': 0},
            {'enabled': True, 'limit_bytes': 16 * (1 << 30),
             'buffer_percent': 4},
            {'enabled': True, 'limit_bytes': 16 * (1 << 30),
             'buffer_percent': 31},
            {'enabled': True, 'limit_bytes': 16 * (1 << 30),
             'max_refill_slots': 0},
            {'enabled': True, 'limit_bytes': 16 * (1 << 30),
             'max_refill_slots': 4},
            {'enabled': True, 'limit_bytes': True},
            {'enabled': True, 'limit_bytes': 1.5},
            {'enabled': True, 'limit_bytes': 16 * (1 << 30),
             'unknown': 1},
        ):
            with self.subTest(memory=memory):
                self.assertNotEqual(plan({**request,
                                          'memory_constrained': memory})[0], 0)

        self.assertNotEqual(plan({
            **constrained, 'memory_budget_bytes': 8 * (1 << 30)
        })[0], 0)
        self.assertNotEqual(plan({
            **constrained, 'execution': 'gpu_ane',
            'allow_approximation': True, 'ane_manifest': '/tmp/fake.json'
        })[0], 0)

        schema2 = {
            'schema_version': 2,
            'model': 'flux2-klein-4b',
            'operation': 'image.generate',
            'inputs': [{'kind': 'text', 'role': 'prompt', 'text': 'fox'}],
            'outputs': [{'kind': 'image', 'path': '/tmp/flux.png',
                         'width': 512, 'height': 512, 'frames': 1,
                         'fps': 24, 'audio': False}],
            'sampling': {'seed': 42, 'steps': 4},
            'execution': {
                'policy': 'gpu', 'residency': 'resident',
                'memory_constrained': constrained['memory_constrained'],
            },
        }
        status, configured, error = plan(schema2)
        self.assertEqual(status, 0, error)
        self.assertEqual(configured['memory_policy']['digest'], policy['digest'])

    def test_memory_constrained_routes_remain_plan_only_without_manifest(self):
        physical = json.loads(consume(C.c_void_p(lib.tc_system_json())))["physical_memory_bytes"]
        common = {
            'memory_constrained': {
                'enabled': True,
                'limit_bytes': min(48 * (1 << 30), physical),
                'buffer_percent': 15,
                'min_free_bytes': 1 << 30,
            },
            'execution': 'auto',
            'audio': False,
        }
        h3 = {
            **common,
            'model': 'minimax-h3-turbo',
            'operation': 'video.generate',
            'width': 512,
            'height': 512,
            'frames': 22,
            'fps': 24,
            'steps': 4,
            'memory_constrained': {
                **common['memory_constrained'], 'max_refill_slots': 2,
            },
        }
        status, configured, error = plan(h3)
        self.assertEqual(status, 0, error)
        policy = configured['memory_policy']
        self.assertTrue(policy['route_available'])
        self.assertFalse(policy['execution_supported'])
        self.assertEqual(policy['capability_level'], 'hook_bridged')
        self.assertEqual(policy['certification_state'], 'plan_only')
        self.assertEqual(policy['admission_state'],
                         'plan_only' if policy['estimate_fits'] else 'rejected')
        self.assertFalse(policy['release_stable'])
        self.assertEqual(policy['manifest_digest'], '')
        self.assertEqual(policy['evidence_digest'], '')
        self.assertIn('no verified capability manifest' if policy['estimate_fits']
                      else 'exceeds effective budget', policy['reason'])
        self.assertEqual(policy['adapter_candidate'],
                         'h3_c_metal_streamed_v1')
        self.assertEqual(policy['effective_residency'], 'streamed')
        self.assertEqual(policy['refill_slots'], 2)
        self.assertEqual(configured['execution'], 'gpu')
        self.assertEqual(configured['residency'], 'streamed')
        self.assertEqual(policy['denoiser_budget_bytes'],
                         policy['effective_budget_bytes'] - 4 * (1 << 30))

        # The native API must reject an oversized user limit on every machine.
        status, _, error = plan({**h3, 'memory_constrained': {
            **h3['memory_constrained'], 'limit_bytes': physical + (1 << 30)}})
        self.assertNotEqual(status, 0)
        self.assertIn('exceeds physical memory', error)

        h3_unsupported = (
            {**h3, 'audio': True},
            {**h3, 'operation': 'video.reference',
             'inputs': [{'kind': 'image', 'role': 'reference',
                         'path': '/tmp/reference.png'}]},
            {**h3, 'loras': [{'role': 'transformer',
                              'path': '/tmp/h3.safetensors',
                              'strength': 1.0}]},
            {**h3, 'quantized_cache': '/tmp/h3-q8',
             'allow_approximation': True, 'residency': 'streamed'},
        )
        for unsupported_h3 in h3_unsupported:
            with self.subTest(unsupported_h3=unsupported_h3):
                status, configured, error = plan(unsupported_h3)
                self.assertEqual(status, 0, error)
                self.assertFalse(
                    configured['memory_policy']['route_available'])

        ltx = {
            **common,
            'model': 'ltx-2.5-distilled',
            'operation': 'video.generate',
            'width': 704,
            'height': 448,
            'frames': 97,
            'fps': 24,
            'steps': 11,
            'ltx_backend': 'auto',
        }
        status, configured, error = plan(ltx)
        self.assertEqual(status, 0, error)
        policy = configured['memory_policy']
        self.assertTrue(policy['route_available'])
        self.assertFalse(policy['execution_supported'])
        self.assertEqual(policy['capability_level'], 'hook_bridged')
        self.assertEqual(policy['certification_state'], 'plan_only')
        self.assertEqual(policy['admission_state'],
                         'plan_only' if policy['estimate_fits'] else 'rejected')
        self.assertFalse(policy['release_stable'])
        self.assertIn('no verified capability manifest' if policy['estimate_fits']
                      else 'exceeds effective budget', policy['reason'])
        self.assertEqual(policy['adapter_candidate'],
                         'ltx_c_metal_streamed_video_v1')
        self.assertEqual(configured['ltx_backend'], 'c_metal')
        self.assertEqual(configured['residency'], 'streamed')

        unsupported = {**ltx, 'audio': True}
        status, configured, error = plan(unsupported)
        self.assertEqual(status, 0, error)
        self.assertFalse(configured['memory_policy']['route_available'])
        self.assertEqual(configured['residency'], 'component_staged')

        one_slot_h3 = {
            **h3,
            'memory_constrained': {
                **h3['memory_constrained'], 'max_refill_slots': 1,
            },
        }
        status, configured, error = plan(one_slot_h3)
        self.assertNotEqual(status, 0)
        self.assertIn('two certified refill slots', error)

    def test_memory_constrained_execution_entry_is_normalized_and_observed(self):
        source = (ROOT / 'native/api/c_api.mm').read_text()
        results = (ROOT / 'native/platform/apple/results.mm').read_text()
        execution = (
            ROOT / 'native/runtime/memory_execution.cpp'
        ).read_text()
        execution_header = (
            ROOT / 'native/runtime/memory_execution.hpp'
        ).read_text()
        schedule_header = (
            ROOT / 'native/core/memory_schedule_c.h'
        ).read_text()
        ltx = (ROOT / 'native/platform/apple/ltx_session.mm').read_text()
        h3 = (ROOT / 'native/platform/apple/h3_session.mm').read_text()
        h3_runtime = (ROOT / 'native/models/h3_runtime/h3.c').read_text()
        h3_header = (ROOT / 'native/models/h3_runtime/h3.h').read_text()
        h3_vae_header = (
            ROOT / 'native/models/h3_runtime/h3_video_vae.h'
        ).read_text()
        h3_vae = (
            ROOT / 'native/models/h3_runtime/h3_video_vae.c'
        ).read_text()
        build = (ROOT / 'tools/native/build.sh').read_text()
        makefile = (ROOT / 'Makefile').read_text()
        probe_header = (ROOT / 'native/platform/apple/memory_probe.hpp').read_text()
        probe = (ROOT / 'native/platform/apple/memory_probe.mm').read_text()
        self.assertIn('if (request.memory_constrained.enabled)', source)
        self.assertIn('request = request_plan->request;', source)
        self.assertIn('parsed_request->memory_constrained.enabled', source)
        self.assertNotIn('auto request_plan = tc::make_plan(request);', source)
        self.assertIn('prepare_memory_execution(', source)
        self.assertIn('memory_execution->checkpoint(phase)', source)
        self.assertIn('!memory_execution->uses_explicit_schedule()', source)
        self.assertIn('memory_execution->emit_terminal_schedule_event()',
                      source)
        self.assertIn('memory_execution->begin_running()', source)
        self.assertIn('memory_execution->begin_draining()', source)
        self.assertIn('memory_execution->finish_success()', source)
        self.assertIn('drain_memory_execution(', source)
        self.assertIn('memory_quarantined', source)
        self.assertIn('MemoryFailureDisposition::Clean', source)
        self.assertIn('session.unload();', source)
        self.assertIn('runtime_denoiser_budget', source)
        self.assertIn('MemoryAdmissionMetrics', results)
        self.assertIn('@"execution_state"', results)
        self.assertIn('@"watchdog_sample_count"', results)
        self.assertIn('@"watchdog_critical"', results)
        self.assertIn('@"trace_event_count"', results)
        self.assertIn('@"trace_overflowed"', results)
        self.assertIn('@"schedule_event_attempted_count"', results)
        self.assertIn('@"schedule_event_rejected_count"', results)
        self.assertIn('@"schedule_cursor_state"', results)
        self.assertIn('@"schedule_first_failure"', results)
        self.assertIn('@"memory_trace"', results)
        self.assertIn('tc_memory_schedule_hooks_v1', schedule_header)
        self.assertIn('TC_MEMORY_EVENT_TERMINAL', schedule_header)
        self.assertIn('MemoryScheduleCursorState::Poisoned', execution)
        self.assertIn('emit_terminal_schedule_event()', execution_header)
        self.assertIn('compile_options.require_explicit_schedule', execution)
        self.assertIn('compile_options.schedule = resolution.record.schedule',
                      execution)
        self.assertIn('component_staged || streamed', ltx)
        self.assertIn('!request.memory_constrained.enabled &&', ltx)
        self.assertIn('H3_MEMORY_CONSTRAINED', h3)
        self.assertIn('H3_MEMORY_CONSTRAINED', h3_runtime)
        self.assertIn('h3_host_memory_hooks', h3_header)
        self.assertIn('const tc_memory_schedule_hooks_v1 *schedule_hooks',
                      h3_header)
        self.assertIn('parameters.host_memory_hooks', h3)
        self.assertIn('parameters.schedule_hooks = &memory_schedule_hooks_',
                      h3)
        self.assertIn(
            'memory_schedule_hooks_ = context->make_schedule_hooks()', h3
        )
        self.assertIn('h3_accounted_host_malloc(', h3_runtime)
        self.assertIn('h3.vae.decoded_host_envelope_v1', h3_runtime)
        self.assertIn('h3_video_vae_decoder_load_with_options',
                      h3_vae_header)
        self.assertIn('h3_video_vae_decode_with_options', h3_vae)
        self.assertIn('h3_gpu_create_with_options(', h3_vae)
        self.assertIn('h3_gpu_tensor_from_f32_classified', h3_vae)
        self.assertNotIn('h3_gpu_tensor_from_f32(vae->gpu', h3_vae)
        self.assertNotIn('h3_gpu_tensor_new_f32(vae->gpu', h3_vae)
        self.assertIn('uses_parent_mlx(const Request& request)', ltx)
        self.assertIn('request.memory_constrained.enabled) return false',
                      ltx)
        self.assertIn('native/runtime/memory_accounting.cpp', build)
        self.assertIn('native/platform/apple/memory_probe.mm', build)
        self.assertIn('tests/native/test_memory_accounting.py', makefile)
        self.assertIn('tests/native/test_memory_probe.py', makefile)
        self.assertIn('probe_memory_capability', h3)
        self.assertIn('probe_memory_capability', ltx)
        self.assertIn('MemoryCheckpointHashCache', probe_header)
        self.assertIn('resolve_probe_sidecar', probe)
        # The probe intentionally uses an fd-level read with a before/after
        # fstat snapshot.  This keeps sidecar validation independent of
        # Foundation's mmap policy and makes the TOCTOU check explicit.
        self.assertIn('read_probe_sidecar', probe)
        self.assertIn('O_CLOEXEC', probe)
        self.assertIn('fstat', probe)

    def test_memory_constrained_ltx_allocator_is_admission_guarded(self):
        gpu_header = (
            ROOT / 'native/models/ltx_runtime/ltx_gpu.h'
        ).read_text()
        gpu = (ROOT / 'native/models/ltx_runtime/ltx_gpu.m').read_text()
        native_header = (
            ROOT / 'native/models/ltx_runtime/ltx_native.h'
        ).read_text()
        native = (
            ROOT / 'native/models/ltx_runtime/ltx_blocks.c'
        ).read_text()
        session = (
            ROOT / 'native/platform/apple/ltx_session.mm'
        ).read_text()
        api = (ROOT / 'native/api/c_api.mm').read_text()
        accounting = (
            ROOT / 'native/runtime/memory_accounting.hpp'
        ).read_text()
        execution = (
            ROOT / 'native/runtime/memory_execution.cpp'
        ).read_text()
        results = (
            ROOT / 'native/platform/apple/results.mm'
        ).read_text()
        public_header = (
            ROOT / 'bindings/c/include/turbocider/turbocider.h'
        ).read_text()
        planner = (ROOT / 'native/runtime/plan.cpp').read_text()
        makefile = (ROOT / 'Makefile').read_text()
        for symbol in (
            'ltx_gpu_memory_hooks', 'ltx_gpu_set_memory_hooks',
            'ltx_gpu_set_memory_hooks_for_queue',
            'ltx_gpu_buffer_new_classified',
            'ltx_gpu_buffer_new_copy_classified',
        ):
            self.assertIn(symbol, gpu_header)
        self.assertIn('gpu->memory_hooks.reserve(', gpu)
        self.assertIn('gpu->memory_hooks.commit(', gpu)
        self.assertIn('gpu->memory_hooks.complete(', gpu)
        self.assertIn('buffer->memory_hooks.release(', gpu)
        self.assertIn('ltx_release_deferred_memory_tokens', gpu)
        self.assertIn('const ltx_gpu_memory_hooks *memory_hooks', native_header)
        self.assertIn('const tc_memory_schedule_hooks_v1 *schedule_hooks',
                      native_header)
        self.assertIn('ltx_gpu_set_memory_hooks_for_queue(', native)
        self.assertIn('ltx_native_schedule_emit(', native)
        self.assertIn('TC_MEMORY_STAGE_DENOISER_LOAD', native)
        self.assertIn('schedule_step_begin', native)
        self.assertIn('set_memory_admission(MemoryAdmission* admission)', session)
        self.assertIn('bind_memory_context(MemoryExecutionContext* context)', session)
        self.assertIn('options.memory_hooks = &memory_hooks_', session)
        self.assertIn('options.schedule_hooks = &memory_schedule_hooks_',
                      session)
        self.assertIn(
            'memory_schedule_hooks_ = context->make_schedule_hooks()',
            session
        )
        self.assertIn('memory_hooks_.version = 2u', session)
        self.assertIn('memory_hooks_.retire = ltx_memory_retire', session)
        self.assertIn('memory_hooks_.complete = ltx_memory_complete', session)
        self.assertIn('ScopedMemoryExecutionBinding', api)
        self.assertIn('tc_engine_take_last_memory_report_json', api)
        self.assertIn('tc_engine_take_last_memory_report_json', public_header)
        self.assertIn('last_memory_report', api)
        self.assertIn('SwapActivityObservation', accounting)
        self.assertIn('observe_swap_activity()', api)
        self.assertIn('memory_swap_activity_detected', execution)
        self.assertIn('explicit epoch transition is required', execution)
        self.assertIn('require_explicit_epoch', execution)
        self.assertIn('@"swapouts_delta"', results)
        self.assertIn('@"swap_counter_invalid"', results)
        self.assertIn('@"automatic_epoch_transition_count"', results)
        self.assertNotIn(
            'route_available && memory_policy->estimate_fits', planner
        )
        self.assertNotIn('execution_adapter_ready', planner)
        self.assertIn('tests/native/test_ltx_gpu_memory_hooks.py', makefile)
        self.assertIn('tests/native/test_memory_schedule_adapter.py', makefile)

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

    def test_qwen21_1024_w8a8_diagnostic_gate(self):
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A teapot',
                    width=1024, height=1024, steps=5, audio=False, frames=1,
                    execution='gpu_ane', allow_approximation=True, qwen21_w8a8=True,
                    ane_manifest='diagnostic-manifest-not-loaded-during-planning.json')
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_1024_W8A8_DIAGNOSTIC': '0'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_1024_W8A8_DIAGNOSTIC': '1'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertEqual(result['execution'], 'gpu_ane_experimental')
            for invalid in [dict(width=512, height=1024), dict(qwen21_w8a8=False),
                            dict(operation='image.edit', inputs=[dict(kind='image', role='reference', path='ref.png')]),
                            dict(qwen21_gpu_full_ffn_blocks=[3, 5, 7]),
                            dict(allow_approximation=False), dict(ane_manifest='')]:
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)

    def test_qwen21_rectangle_reuses_1024_row_w8a8_graph_only_by_opt_in(self):
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                for i in range(3)]
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A teapot',
                    width=768, height=512, steps=20, audio=False, frames=1,
                    execution='gpu_ane', allow_approximation=True, qwen21_w8a8=True,
                    ane_manifest='512-row-manifest-not-loaded-during-planning.json')
        flag = 'TURBOCIDER_QWEN21_RECT_W8A8_DIAGNOSTIC'
        with patch.dict(os.environ, {flag: '0'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag: '1'}):
            for width, height in ((768, 512), (512, 768)):
                code, result, error = plan({**base, 'width':width, 'height':height})
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_rectangular_decode_w8a8_tiled_diagnostic',
                              result['algorithm_approximations'])
            for count in (1, 2, 3):
                edit = {**base, 'operation':'image.edit', 'inputs':refs[:count]}
                self.assertEqual(plan(edit)[0], 0)
            for invalid in (dict(width=512), dict(height=1024), dict(steps=5),
                            dict(steps=41), dict(qwen21_w8a8=False),
                            dict(qwen21_gpu_w8a16=True),
                            dict(qwen21_gpu_full_ffn_blocks=[3, 5, 7]),
                            dict(allow_approximation=False), dict(execution='gpu', ane_manifest=''),
                            dict(operation='image.edit', inputs=refs * 2),
                            dict(loras=[dict(path='adapter.safetensors', scale=1)])):
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)
        with patch.dict(os.environ, {flag: 'invalid'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_runtime_lora_base_ane_is_suffix_only_diagnostic(self):
        adapter = dict(path='Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors',
                       role='transformer', strength=1)
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A fox',
                    width=512, height=512, steps=6, audio=False, frames=1,
                    execution='gpu_ane', allow_approximation=True, qwen21_w8a8=True,
                    qwen21_reference_size=1024, loras=[adapter],
                    ane_manifest='base-manifest-not-loaded-during-planning.json')
        flag = 'TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC'
        with patch.dict(os.environ, {flag: '0'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag: '1'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertIn('qwen21_runtime_lora_base_ane_suffix_only_diagnostic',
                          result['algorithm_approximations'])
            for invalid in (dict(steps=20), dict(width=768), dict(qwen21_w8a8=False),
                            dict(qwen21_gpu_w8a16=True), dict(loras=[]),
                            dict(allow_approximation=False), dict(execution='gpu', ane_manifest=''),
                            dict(qwen21_gpu_full_ffn_blocks=[3, 5, 7])):
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)
        with patch.dict(os.environ, {flag: 'bad'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_lora_compatible_gate_up_requires_explicit_manifest_route(self):
        adapter = dict(path='Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors',
                       role='transformer', strength=1)
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A fox',
                    width=512, height=512, steps=6, audio=False, frames=1,
                    execution='gpu_ane', allow_approximation=True, qwen21_w8a8=True,
                    qwen21_reference_size=1024, loras=[adapter],
                    ane_manifest='gate-up-manifest-checked-at-execution.json')
        gate = 'TURBOCIDER_QWEN21_LORA_GATE_UP_DIAGNOSTIC'
        lora = 'TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC'
        with patch.dict(os.environ, {gate: '1', lora: '0'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {gate: '1', lora: '1'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertIn('qwen21_w8a8_gate_up_gpu_silu_down_diagnostic',
                          result['algorithm_approximations'])
            self.assertNotIn('qwen21_runtime_lora_base_ane_suffix_only_diagnostic',
                             result['algorithm_approximations'])
            plain = {**base, 'loras': [], 'steps': 20}
            self.assertNotEqual(plan(plain)[0], 0)  # LoRA-only flag still requires an adapter
            for invalid in (dict(width=768), dict(qwen21_gpu_w8a16=True),
                            dict(allow_approximation=False), dict(steps=20),
                            dict(qwen21_gpu_full_ffn_blocks=[3, 5, 7])):
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)
            with patch.dict(os.environ, {
                'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC': '16'
            }):
                self.assertNotEqual(plan(base)[0], 0)
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC': '1'}):
                self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {gate: '1', lora: '0'}):
            for steps in (5, 20, 40):
                code, result, error = plan({**base, 'steps': steps, 'loras': []})
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_w8a8_gate_up_gpu_silu_down_diagnostic',
                              result['algorithm_approximations'])
        with patch.dict(os.environ, {gate: '2', lora: '1'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_viggle_local_prefill_requires_explicit_hybrid_lora(self):
        adapter = dict(path='Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors',
                       role='transformer', strength=1)
        request = dict(model='qwen-image-2.1', operation='image.edit', prompt='Two teapots',
                       width=512, height=512, steps=6, audio=False, frames=1,
                       execution='gpu_ane', allow_approximation=True, qwen21_w8a8=True,
                       qwen21_reference_size=1024, loras=[adapter],
                       ane_manifest='base-manifest-not-loaded-during-planning.json')
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png') for i in range(3)]
        local = 'TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION'
        lora_hybrid = 'TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC'
        with patch.dict(os.environ, {local: '3', lora_hybrid: '1'}):
            for count in (2, 3):
                code, result, error = plan({**request, 'inputs': refs[:count]})
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_last16_reference_local_attention_diagnostic',
                              result['algorithm_approximations'])
            self.assertNotEqual(plan({**request, 'inputs': refs[:1]})[0], 0)
        with patch.dict(os.environ, {local: '3', lora_hybrid: '0'}):
            self.assertEqual(plan({**request, 'execution': 'gpu',
                                   'qwen21_w8a8': False, 'ane_manifest': '',
                                   'inputs': refs[:2]})[0], 0)
            self.assertNotEqual(plan({**request, 'inputs': refs[:2]})[0], 0)
        with patch.dict(os.environ, {local: '1', lora_hybrid: '1'}):
            self.assertNotEqual(plan({**request, 'inputs': refs[:2]})[0], 0)

    def test_qwen21_viggle_resized_512_references_are_diagnostic_only(self):
        adapter = dict(path='Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors',
                       role='transformer', strength=1)
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png') for i in range(3)]
        base = dict(model='qwen-image-2.1', operation='image.edit', prompt='Three objects',
                    width=512, height=512, steps=6, audio=False, frames=1,
                    execution='gpu', allow_approximation=True, qwen21_reference_size=512,
                    inputs=refs, loras=[adapter])
        flag = 'TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC'
        with patch.dict(os.environ, {flag: '0'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag: '1'}):
            for count in (1, 2, 3):
                code, result, error = plan({**base, 'inputs': refs[:count]})
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_viggle_reference_resize_512_diagnostic',
                              result['algorithm_approximations'])
                self.assertIn('qwen21_reference_resize_512', result['algorithm_approximations'])
            for invalid in (dict(steps=20), dict(inputs=[]), dict(inputs=refs + refs[:1]),
                            dict(qwen21_reference_size=256), dict(loras=[]),
                            dict(loras=[{**adapter, 'path': 'unrelated-r256.safetensors'}]),
                            dict(loras=[{**adapter, 'strength': 0.5}]),
                            dict(operation='image.generate', inputs=[]),
                            dict(allow_approximation=False)):
                self.assertNotEqual(plan({**base, **invalid})[0], 0)
            hybrid = {**base, 'execution': 'gpu_ane', 'qwen21_w8a8': True,
                      'ane_manifest': 'base-manifest-not-loaded-during-planning.json'}
            self.assertNotEqual(plan(hybrid)[0], 0)
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC': '1'}):
                code, result, error = plan(hybrid)
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_runtime_lora_base_ane_suffix_only_diagnostic',
                              result['algorithm_approximations'])
                self.assertNotEqual(plan({**hybrid, 'qwen21_gpu_w8a16': True})[0], 0)
                for layers in ('8', '16'):
                    with patch.dict(os.environ, {
                        'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC': layers
                    }):
                        self.assertNotEqual(plan(hybrid)[0], 0)
                with patch.dict(os.environ, {
                    'TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC': '1'
                }):
                    for count in (1, 2, 3):
                        self.assertNotEqual(plan({**hybrid, 'inputs': refs[:count]})[0], 0)
                with patch.dict(os.environ, {
                    'TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC': '1'
                }):
                    self.assertNotEqual(plan(hybrid)[0], 0)
        with patch.dict(os.environ, {flag: 'bad'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_full_reference_w8a8_diagnostic_gate(self):
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                for i in range(3)]
        base = dict(model='qwen-image-2.1', operation='image.edit', prompt='Three objects',
                    width=512, height=512, steps=40, audio=False, frames=1,
                    execution='gpu_ane', allow_approximation=True, qwen21_w8a8=True,
                    ane_manifest='compiled-manifest-not-loaded-during-planning.json',
                    qwen21_reference_size=1024, inputs=refs)
        flag = 'TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC'
        with patch.dict(os.environ, {flag:'0'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag:'1'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertEqual(result['execution'], 'gpu_ane_experimental')
            self.assertIn('qwen21_w8a8_full_reference_diagnostic',
                          result['algorithm_approximations'])
            for invalid in [dict(width=1024, height=1024), dict(qwen21_w8a8=False),
                            dict(qwen21_gpu_full_ffn_blocks=[3,5,7]),
                            dict(inputs=refs + [refs[0]]), dict(allow_approximation=False),
                            dict(ane_manifest='')]:
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)

    def test_qwen21_reference_resize_512_requires_explicit_edit_opt_in(self):
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                for i in range(3)]
        base = dict(model='qwen-image-2.1', operation='image.edit', prompt='Three objects',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu', allow_approximation=True,
                    qwen21_reference_size=512, inputs=refs)
        code, result, error = plan(base)
        self.assertEqual(code, 0, error)
        self.assertIn('qwen21_reference_resize_512', result['algorithm_approximations'])
        for invalid in [dict(allow_approximation=False),
                        dict(operation='image.generate', inputs=[]),
                        dict(inputs=refs + refs[:1]),
                        dict(qwen21_reference_size=768),
                        dict(model='z-image-turbo')]:
            with self.subTest(invalid=invalid):
                self.assertNotEqual(plan({**base, **invalid})[0], 0)

    def test_qwen21_tiled_prefill_is_diagnostic_only(self):
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                for i in range(3)]
        base = dict(model='qwen-image-2.1', operation='image.edit', prompt='Three objects',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu_ane', allow_approximation=True,
                    qwen21_reference_size=512, qwen21_w8a8=True,
                    ane_manifest='manifest.json', inputs=refs)
        flag = 'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC'
        with patch.dict(os.environ, {flag: '0'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertNotIn('qwen21_tiled_prefill_w8a8_diagnostic',
                             result['algorithm_approximations'])
        with patch.dict(os.environ, {flag: '1'}):
            for count in (1, 2, 3):
                code, result, error = plan({**base, 'inputs':refs[:count]})
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_tiled_prefill_w8a8_diagnostic',
                              result['algorithm_approximations'])
            for invalid in [dict(execution='gpu', ane_manifest='', qwen21_w8a8=False),
                            dict(qwen21_reference_size=256), dict(qwen21_reference_size=1024),
                            dict(inputs=refs + refs[:1]), dict(allow_approximation=False)]:
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV': '1'}):
                self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag: '8'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertIn('qwen21_tiled_prefill_last8_w8a8_diagnostic',
                          result['algorithm_approximations'])
        for layers in (16, 20, 24):
            with patch.dict(os.environ, {flag: str(layers)}):
                code, result, error = plan(base)
                self.assertEqual(code, 0, error)
                self.assertIn(f'qwen21_tiled_prefill_last{layers}_w8a8_diagnostic',
                              result['algorithm_approximations'])
        with patch.dict(os.environ, {flag: 'bad'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag: ''}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_last_prefill_block_target_only_is_explicit(self):
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                for i in range(3)]
        base = dict(model='qwen-image-2.1', operation='image.edit', prompt='Three objects',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu', allow_approximation=True,
                    qwen21_reference_size=512, inputs=refs)
        flag = 'TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC'
        with patch.dict(os.environ, {flag: '0'}):
            self.assertNotIn('qwen21_prefill_last_target_only_diagnostic',
                             plan(base)[1]['algorithm_approximations'])
        with patch.dict(os.environ, {flag: '1'}):
            for count in (1, 2, 3):
                code, result, error = plan({**base, 'inputs': refs[:count]})
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_prefill_last_target_only_diagnostic',
                              result['algorithm_approximations'])
            generate = {**base, 'operation': 'image.generate', 'inputs': [],
                        'qwen21_reference_size': 1024}
            self.assertEqual(plan(generate)[0], 0)
            self.assertEqual(plan({**base, 'qwen21_reference_size': 1024})[0], 0)
            hybrid = {**base, 'execution': 'gpu_ane', 'qwen21_w8a8': True,
                      'ane_manifest': 'manifest.json'}
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC': '16'}):
                self.assertEqual(plan(hybrid)[0], 0)
            for invalid in (dict(width=1024, height=1024), dict(inputs=refs + refs[:1]),
                            dict(qwen21_reference_size=256), dict(allow_approximation=False),
                            dict(loras=[dict(path='adapter.safetensors', scale=1)])):
                self.assertNotEqual(plan({**base, **invalid})[0], 0)
            self.assertNotEqual(plan(hybrid)[0], 0)
        with patch.dict(os.environ, {flag: 'invalid'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_hybrid_final_ffn_reuse_is_diagnostic_only(self):
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                for i in range(3)]
        base = dict(model='qwen-image-2.1', operation='image.edit', prompt='Three objects',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu_ane', allow_approximation=True,
                    qwen21_reference_size=512, qwen21_w8a8=True,
                    ane_manifest='manifest.json', inputs=refs)
        flag = 'TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC'
        with patch.dict(os.environ, {flag: '1',
                                    'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC': '16'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertIn('qwen21_hybrid_reuse_final_ffn_diagnostic',
                          result['algorithm_approximations'])
            for invalid in (dict(execution='gpu', qwen21_w8a8=False, ane_manifest=''),
                            dict(steps=2), dict(steps=6), dict(steps=40), dict(qwen21_reference_size=1024),
                            dict(allow_approximation=False), dict(inputs=refs + refs[:1])):
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN': '1'}):
                self.assertNotEqual(plan(base)[0], 0)
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC': '8'}):
                self.assertNotEqual(plan(base)[0], 0)
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC': '0'}):
                self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag: '1',
                                    'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC': '20'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag: 'bad'}):
            self.assertNotEqual(plan(base)[0], 0)
        last16 = 'TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC'
        with patch.dict(os.environ, {last16: '1',
                                    'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC': '16'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertIn('qwen21_hybrid_reuse_final_last16_ffn_diagnostic',
                          result['algorithm_approximations'])
            for incompatible in (flag, 'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN'):
                with patch.dict(os.environ, {incompatible: '1'}):
                    self.assertNotEqual(plan(base)[0], 0)
            for invalid in (dict(execution='gpu', qwen21_w8a8=False, ane_manifest=''),
                            dict(steps=40), dict(qwen21_reference_size=1024),
                            dict(inputs=refs + refs[:1])):
                self.assertNotEqual(plan({**base, **invalid})[0], 0)
        with patch.dict(os.environ, {last16: '1'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_hybrid_penultimate_even_ffn_reuse_requires_final_last16(self):
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                for i in range(3)]
        base = dict(model='qwen-image-2.1', operation='image.edit', prompt='Three objects',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu_ane', allow_approximation=True,
                    qwen21_reference_size=512, qwen21_w8a8=True,
                    ane_manifest='manifest.json', inputs=refs)
        flag = 'TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC'
        with patch.dict(os.environ, {flag: '1',
                                    'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC': '16',
                                    'TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC': '1'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertIn('qwen21_hybrid_reuse_penultimate_even_ffn_diagnostic',
                          result['algorithm_approximations'])
            for invalid in (dict(execution='gpu', qwen21_w8a8=False, ane_manifest=''),
                            dict(steps=40), dict(allow_approximation=False),
                            dict(qwen21_reference_size=1024)):
                self.assertNotEqual(plan({**base, **invalid})[0], 0)
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN': '1',
                                        'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN': '1'}):
                self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag: '1'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag: 'bad'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_tiled_prefill_prefix_kv_is_explicit(self):
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                for i in range(3)]
        base = dict(model='qwen-image-2.1', operation='image.edit', prompt='Three objects',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu_ane', allow_approximation=True, qwen21_w8a8=True,
                    qwen21_reference_size=512, residency='resident',
                    ane_manifest='manifest.json', inputs=refs)
        flag = 'TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC'
        tiled = 'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC'
        prefix = 'TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV'
        with patch.dict(os.environ, {prefix: '1', tiled: '16', flag: '1'}):
            for count in (1, 2, 3):
                code, result, error = plan({**base, 'inputs': refs[:count]})
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_tiled_prefill_prefix_kv_diagnostic',
                              result['algorithm_approximations'])
            for invalid in (dict(execution='gpu', qwen21_w8a8=False, ane_manifest=''),
                            dict(steps=40), dict(qwen21_reference_size=1024),
                            dict(allow_approximation=False), dict(residency='component_staged'),
                            dict(inputs=refs + refs[:1])):
                self.assertNotEqual(plan({**base, **invalid})[0], 0)
            with patch.dict(os.environ, {tiled: '8'}):
                self.assertNotEqual(plan(base)[0], 0)
            with patch.dict(os.environ, {prefix: '0'}):
                self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {prefix: '1', tiled: '16', flag: '0'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag: 'bad'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_tiled_prefix_target_only_needs_resident_tiled_prefix(self):
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                for i in range(3)]
        base = dict(model='qwen-image-2.1', operation='image.edit', prompt='Three objects',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu_ane', allow_approximation=True, qwen21_w8a8=True,
                    qwen21_reference_size=512, residency='resident',
                    ane_manifest='manifest.json', inputs=refs)
        target = 'TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC'
        tiled = 'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC'
        prefix = 'TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV'
        reuse = 'TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC'
        with patch.dict(os.environ, {target: '1', tiled: '16', prefix: '1', reuse: '1'}):
            for count in (1, 2, 3):
                code, result, error = plan({**base, 'inputs': refs[:count]})
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_tiled_prefix_target_only_diagnostic',
                              result['algorithm_approximations'])
            for invalid in (dict(residency='component_staged'), dict(allow_approximation=False),
                            dict(qwen21_reference_size=1024), dict(qwen21_gpu_w8a16=True),
                            dict(inputs=refs + refs[:1])):
                self.assertNotEqual(plan({**base, **invalid})[0], 0)
            with patch.dict(os.environ, {reuse: '0'}):
                self.assertNotEqual(plan(base)[0], 0)
            with patch.dict(os.environ, {prefix: '0'}):
                self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {target: 'invalid'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_fused_qkv_gpu_is_diagnostic_only(self):
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                for i in range(3)]
        base = dict(model='qwen-image-2.1', operation='image.edit', prompt='Three objects',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu', allow_approximation=True,
                    qwen21_reference_size=512, inputs=refs)
        flag = 'TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC'
        with patch.dict(os.environ, {flag: '0'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertNotIn('qwen21_metal_fused_qkv_diagnostic',
                             result['algorithm_approximations'])
        with patch.dict(os.environ, {flag: '1'}):
            for count in (0, 1, 2, 3):
                request = {**base, 'inputs': refs[:count]}
                if not count:
                    request['operation'] = 'image.generate'
                    request['qwen21_reference_size'] = 1024
                code, result, error = plan(request)
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_metal_fused_qkv_diagnostic',
                              result['algorithm_approximations'])
            for invalid in (dict(width=1024, height=1024), dict(allow_approximation=False),
                            dict(qwen21_reference_size=1024),
                            dict(inputs=refs + refs[:1])):
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)
            for incompatible in ('TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE',
                                 'TURBOCIDER_QWEN21_METAL_QK_ROPE',
                                 'TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION'):
                with patch.dict(os.environ, {incompatible: '1'}):
                    self.assertNotEqual(plan(base)[0], 0)
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV': '1'}):
                self.assertEqual(plan(base)[0], 0)
            hybrid = {**base, 'execution': 'gpu_ane', 'qwen21_w8a8': True,
                      'ane_manifest': 'manifest.json'}
            code, result, error = plan(hybrid)
            self.assertEqual(code, 0, error)
            self.assertIn('qwen21_metal_fused_qkv_diagnostic',
                          result['algorithm_approximations'])
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV': '1'}):
                self.assertEqual(plan(hybrid)[0], 0)
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC': '16'}):
                code, combined, error = plan(hybrid)
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_metal_fused_qkv_diagnostic',
                              combined['algorithm_approximations'])
                self.assertIn('qwen21_tiled_prefill_last16_w8a8_diagnostic',
                              combined['algorithm_approximations'])
                with patch.dict(os.environ, {'TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV': '1'}):
                    self.assertNotEqual(plan(hybrid)[0], 0)
            for invalid in (dict(qwen21_gpu_w8a16=True), dict(qwen21_reference_size=1024),
                            dict(qwen21_reference_size=256)):
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**hybrid, **invalid})[0], 0)
        with patch.dict(os.environ, {flag: 'broken'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_prefill_segment_profiler_gate(self):
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                for i in range(3)]
        base = dict(model='qwen-image-2.1', operation='image.edit', prompt='Three objects',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu', qwen21_reference_size=1024, inputs=refs)
        flag = 'TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS'
        with patch.dict(os.environ, {flag:'1', 'TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV':'0',
                                     'TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION':'0'}):
            self.assertEqual(plan(base)[0], 0)
            for invalid in [dict(execution='auto'), dict(operation='image.generate', inputs=[]),
                            dict(width=1024, height=1024), dict(inputs=refs + refs[:1])]:
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)
        with patch.dict(os.environ, {flag:'1', 'TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV':'1'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag:'broken'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_gpu_final_ffn_reuse_gate(self):
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A teapot',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu', allow_approximation=True)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN': '0'}):
            self.assertEqual(plan(base)[1]['algorithm_approximations'], [])
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN': '1'}):
            self.assertEqual(plan(base)[1]['algorithm_approximations'],
                             ['qwen21_gpu_reuse_final_ffn'])
            self.assertEqual(plan({**base, 'steps':2})[1]['algorithm_approximations'], [])
            self.assertNotEqual(plan({**base, 'allow_approximation':False})[0], 0)
            self.assertNotEqual(plan({**base, 'execution':'auto'})[0], 0)
            self.assertNotEqual(plan({**base, 'width':1024, 'height':1024})[0], 0)
            edit = {**base, 'operation':'image.edit', 'qwen21_reference_size':256,
                    'inputs':[dict(kind='image', role='reference', path=f'ref-{i}.png')
                              for i in range(3)]}
            self.assertEqual(plan(edit)[0], 0)
            for count in (1, 2, 3):
                resized = {**edit, 'qwen21_reference_size':512, 'inputs':edit['inputs'][:count]}
                code, result, error = plan(resized)
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_gpu_reuse_final_ffn', result['algorithm_approximations'])
                self.assertIn('qwen21_reference_resize_512', result['algorithm_approximations'])
            self.assertNotEqual(plan({**edit, 'qwen21_reference_size':1024})[0], 0)
            self.assertNotEqual(plan({**edit, 'inputs':edit['inputs'] + [edit['inputs'][0]]})[0], 0)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN': '2'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_metal_qk_norm_rope_gate(self):
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A teapot',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu', allow_approximation=True)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE': '0'}):
            self.assertNotIn('qwen21_metal_qk_norm_rope', plan(base)[1]['algorithm_approximations'])
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE': '1'}):
            self.assertIn('qwen21_metal_qk_norm_rope', plan(base)[1]['algorithm_approximations'])
            for invalid in [dict(execution='auto'), dict(execution='gpu_ane', ane_manifest='probe.json'),
                            dict(allow_approximation=False),
                            dict(width=768, height=768)]:
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)
            hybrid = dict(base, execution='gpu_ane', qwen21_w8a8=True,
                          ane_manifest='diagnostic-only.json')
            self.assertEqual(plan(hybrid)[0], 0)
            self.assertIn('qwen21_metal_qk_norm_rope',
                          plan(hybrid)[1]['algorithm_approximations'])
            self.assertNotEqual(plan({**hybrid, 'qwen21_gpu_w8a16': True})[0], 0)
            runtime = dict(base, execution='gpu_ane', hybrid_mlp_mode='runtime',
                           residency='resident', ane_manifest='runtime-v2.json')
            for count in range(4):
                request = dict(runtime)
                if count:
                    request.update(operation='image.edit', qwen21_reference_size=512,
                                   inputs=[dict(kind='image', role='reference', path=f'ref-{i}.png')
                                           for i in range(count)])
                for lora in (False, True):
                    if lora:
                        request.update(steps=6, lora_strategy='inference_time', loras=[dict(
                            path='runtime-adapter.safetensors', role='transformer', strength=1.0)])
                    with patch.dict(os.environ, {'TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC':
                                                 '1' if count and lora else '0'}):
                        code, result, error = plan(request)
                    self.assertEqual(code, 0, error)
                    self.assertIn('qwen21_metal_qk_norm_rope', result['algorithm_approximations'])
                    self.assertIn('runtime_weight_fp16_token_row_ffn', result['algorithm_approximations'])
            for invalid in (dict(width=768, height=768), dict(allow_approximation=False),
                            dict(qwen21_gpu_w8a16=True), dict(qwen21_w8a8=True),
                            dict(residency='component_staged')):
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**runtime, **invalid})[0], 0)
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_METAL_QK_ROPE': '1'}):
                self.assertNotEqual(plan(runtime)[0], 0)
            for route in (base, runtime, hybrid):
                large = dict(route, width=1024, height=1024, steps=40, residency='resident')
                with patch.dict(os.environ, {'TURBOCIDER_QWEN21_1024_W8A8_DIAGNOSTIC': '1'}):
                    code, result, error = plan(large)
                    self.assertEqual(code, 0, error)
                    self.assertIn('qwen21_metal_qk_norm_rope', result['algorithm_approximations'])
                    for invalid in (dict(operation='image.edit', inputs=[dict(
                                            kind='image', role='reference', path='reference.png')]),
                                    dict(steps=6, lora_strategy='inference_time', loras=[dict(
                                            path='adapter.safetensors', role='transformer', strength=1.0)]),
                                    dict(residency='component_staged'), dict(height=512),
                                    dict(allow_approximation=False), dict(qwen21_gpu_w8a16=True)):
                        with self.subTest(route=route, invalid=invalid):
                            self.assertNotEqual(plan({**large, **invalid})[0], 0)
                    with patch.dict(os.environ, {'TURBOCIDER_QWEN21_METAL_QK_ROPE': '1'}):
                        self.assertNotEqual(plan(large)[0], 0)
            qkv = dict(base, width=1024, height=1024, steps=40,
                       execution='gpu_ane', residency='resident',
                       hybrid_mlp_mode='runtime_qkv', ane_manifest='qkv-runtime.json')
            code, result, error = plan(qkv)
            self.assertEqual(code, 0, error)
            self.assertIn('qwen21_metal_qk_norm_rope', result['algorithm_approximations'])
            self.assertIn('runtime_weight_fp16_token_row_qkv', result['algorithm_approximations'])
            for invalid in (dict(width=512, height=512), dict(residency='component_staged'),
                            dict(allow_approximation=False), dict(qwen21_w8a8=True),
                            dict(loras=[dict(path='adapter.safetensors', role='transformer', strength=1.)],
                                 lora_strategy='inference_time'),
                            dict(operation='image.edit', inputs=[dict(kind='image', role='reference',
                                                                  path='reference.png')])):
                with self.subTest(qkv_invalid=invalid):
                    self.assertNotEqual(plan({**qkv, **invalid})[0], 0)
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_METAL_QK_ROPE': '1'}):
                self.assertNotEqual(plan(qkv)[0], 0)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE': '1',
                                      'TURBOCIDER_QWEN21_METAL_QK_ROPE': '1'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE': '1',
                                      'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN': '1'}):
            labels = plan(base)[1]['algorithm_approximations']
            self.assertIn('qwen21_metal_qk_norm_rope', labels)
            self.assertIn('qwen21_gpu_reuse_final_ffn', labels)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE': '2'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_reference_local_attention_gate(self):
        base = dict(model='qwen-image-2.1', operation='image.edit', prompt='Two teapots',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu', allow_approximation=True, qwen21_reference_size=1024,
                    inputs=[dict(kind='image', role='reference', path=f'ref-{i}.png')
                            for i in range(2)])
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION': '1'}):
            code, request_plan = plan(base)[:2]
            self.assertEqual(code, 0)
            self.assertIn('qwen21_reference_local_attention', request_plan['algorithm_approximations'])
            for invalid in [dict(execution='auto'), dict(allow_approximation=False),
                            dict(width=1024, height=1024), dict(operation='image.generate'),
                            dict(qwen21_reference_size=256),
                            dict(inputs=base['inputs'][:1]),
                            dict(inputs=base['inputs'] * 2)]:
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION': '2'}):
            code, request_plan = plan(base)[:2]
            self.assertEqual(code, 0)
            self.assertIn('qwen21_last_reference_local_attention',
                          request_plan['algorithm_approximations'])
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION': '3'}):
            code, request_plan = plan(base)[:2]
            self.assertEqual(code, 0)
            self.assertIn('qwen21_last16_reference_local_attention_diagnostic',
                          request_plan['algorithm_approximations'])
            hybrid = {**base, 'execution': 'gpu_ane', 'qwen21_w8a8': True,
                      'ane_manifest': 'manifest.json'}
            with patch.dict(os.environ, {'TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC': '1'}):
                code, hybrid_plan, error = plan(hybrid)
                self.assertEqual(code, 0, error)
                self.assertIn('qwen21_last16_reference_local_attention_diagnostic',
                              hybrid_plan['algorithm_approximations'])
                for invalid in [dict(qwen21_w8a8=False), dict(qwen21_gpu_w8a16=True),
                                dict(qwen21_gpu_full_ffn_blocks=[3, 5, 7])]:
                    self.assertNotEqual(plan({**hybrid, **invalid})[0], 0)
                with patch.dict(os.environ, {'TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC': '1'}):
                    code, combined, error = plan(hybrid)
                    self.assertEqual(code, 0, error)
                    self.assertIn('qwen21_prefill_last_target_only_diagnostic',
                                  combined['algorithm_approximations'])
                    with patch.dict(os.environ, {'TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC': '0'}):
                        self.assertNotEqual(plan(hybrid)[0], 0)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION': '2',
                                     'TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC': '1'}):
            self.assertNotEqual(plan({**base, 'execution': 'gpu_ane', 'qwen21_w8a8': True,
                                      'ane_manifest': 'manifest.json'})[0], 0)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION': '4'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_gpu_penultimate_even_ffn_reuse_gate(self):
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A teapot',
                    width=512, height=512, steps=5, audio=False, frames=1,
                    execution='gpu', allow_approximation=True)
        flag = 'TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN'
        with patch.dict(os.environ, {flag: '1', 'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN': '1'}):
            labels = plan(base)[1]['algorithm_approximations']
            self.assertIn('qwen21_gpu_reuse_final_ffn', labels)
            self.assertIn('qwen21_gpu_reuse_penultimate_even_ffn', labels)
            self.assertNotIn('qwen21_gpu_reuse_penultimate_even_ffn',
                             plan({**base, 'steps':2})[1]['algorithm_approximations'])
            for invalid in [dict(execution='auto'), dict(allow_approximation=False),
                            dict(width=1024, height=1024)]:
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)
        with patch.dict(os.environ, {flag: '1', 'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN': '0'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {flag: '2', 'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN': '1'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_viggle_v021_r256_six_step_lora_gate(self):
        adapter = dict(path='download/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors',
                       strength=1.0, role='transformer')
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A fox',
                    width=512, height=512, steps=6, audio=False, frames=1,
                    execution='gpu', allow_approximation=True, loras=[adapter],
                    lora_strategy='inference_time')
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN': '0'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertEqual(result['lora_strategy'], 'inference_time')
            self.assertIn('qwen21_viggle_v021_r256_6step_distillation',
                          result['algorithm_approximations'])
            for invalid in [dict(steps=5), dict(execution='gpu_ane', ane_manifest='probe.json'),
                            dict(allow_approximation=False), dict(width=1024, height=1024),
                            dict(lora_strategy='in_memory_merge'),
                            dict(loras=[{**adapter, 'strength':0.5}]),
                            dict(loras=[{**adapter, 'path':'wrong.safetensors'}])]:
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)
            edit = {**base, 'operation':'image.edit', 'inputs':[
                dict(kind='image', role='reference', path=f'ref-{i}.png') for i in range(3)]}
            self.assertEqual(plan(edit)[0], 0)
            self.assertNotEqual(plan({**edit, 'inputs':edit['inputs'] + [edit['inputs'][0]]})[0], 0)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN': '1'}):
            self.assertNotEqual(plan(base)[0], 0)
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_VIGGLE_LORA_FP16': '1'}):
            code, result, error = plan(base)
            self.assertEqual(code, 0, error)
            self.assertIn('qwen21_viggle_lora_fp16_matmuls', result['algorithm_approximations'])
            code, plain, error = plan({**base, 'loras':[], 'lora_strategy':'auto'})
            self.assertEqual(code, 0, error)
            self.assertNotIn('qwen21_viggle_lora_fp16_matmuls', plain['algorithm_approximations'])
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_VIGGLE_LORA_FP16': 'invalid'}):
            self.assertNotEqual(plan(base)[0], 0)

    def test_qwen21_viggle_v021_r128_six_step_gpu_workflows(self):
        adapter = dict(path='local/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors',
                       strength=1.0, role='transformer')
        base = dict(model='qwen-image-2.1', operation='image.generate', prompt='A fox',
                    width=512, height=512, steps=6, audio=False, frames=1,
                    execution='gpu', residency='resident', allow_approximation=True,
                    loras=[adapter], lora_strategy='inference_time')
        flags = {name: '0' for name in (
            'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN',
            'TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN',
            'TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC',
            'TURBOCIDER_QWEN21_LORA_GATE_UP_DIAGNOSTIC',
            'TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC',
            'TURBOCIDER_QWEN21_VIGGLE_LORA_FP16',
        )}
        refs = [dict(kind='image', role='reference', path=f'ref-{i}.png')
                for i in range(4)]
        with patch.dict(os.environ, flags):
            for count in range(4):
                request = {**base, 'operation': 'image.edit' if count else 'image.generate',
                           'inputs': refs[:count]}
                with self.subTest(references=count):
                    code, result, error = plan(request)
                    self.assertEqual(code, 0, error)
                    self.assertEqual(result['execution'], 'gpu')
                    self.assertEqual(result['lora_strategy'], 'inference_time')
                    self.assertEqual(result['qwen21_reference_size'], 1024)
                    self.assertEqual(result['lora_count'], 1)
                    self.assertIn('qwen21_viggle_v021_r128_6step_distillation',
                                  result['algorithm_approximations'])
                    self.assertNotIn('qwen21_viggle_v021_r256_6step_distillation',
                                     result['algorithm_approximations'])
            code, automatic, error = plan({**base, 'lora_strategy': 'auto'})
            self.assertEqual(code, 0, error)
            self.assertEqual(automatic['lora_strategy'], 'inference_time')
            for invalid in (
                dict(steps=5), dict(steps=40), dict(allow_approximation=False),
                dict(width=768), dict(execution='auto'),
                dict(execution='gpu_ane', ane_manifest='probe.json'),
                dict(lora_strategy='in_memory_merge'), dict(lora_strategy='disk_premerge'),
                dict(loras=[{**adapter, 'role': 'text_encoder'}]),
                dict(loras=[{**adapter, 'strength': 0.0}]),
                dict(loras=[{**adapter, 'strength': 0.5}]),
                dict(loras=[{**adapter, 'strength': 2.0}]),
                dict(loras=[{**adapter, 'path': 'unrelated-r128.safetensors'}]),
                dict(loras=[adapter, adapter]),
                dict(operation='image.generate', inputs=refs[:1]),
                dict(operation='image.edit', inputs=[]),
                dict(operation='image.edit', inputs=refs),
                dict(operation='image.edit', inputs=refs[:1], qwen21_reference_size=512),
            ):
                with self.subTest(invalid=invalid):
                    self.assertNotEqual(plan({**base, **invalid})[0], 0)

    def test_qwen21_viggle_v021_r128_schema2_three_reference_workflow(self):
        references = [dict(kind='image', role='reference', path=f'ordered-{i}.png')
                      for i in range(3)]
        request = dict(
            schema_version=2, model='qwen-image-2.1', operation='image.edit',
            inputs=[dict(kind='text', role='prompt', text='Combine image 1, 2 and 3.'),
                    *references],
            outputs=[dict(kind='image', path='edit.png', width=512, height=512,
                          frames=1, audio=False)],
            sampling=dict(seed=42, steps=6),
            execution=dict(policy='gpu', residency='resident', allow_approximation=True),
            lora_strategy='inference_time',
            loras=[dict(path='Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors',
                        strength=1.0, role='transformer')],
        )
        with patch.dict(os.environ, {'TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN': '0'}):
            code, result, error = plan(request)
            self.assertEqual(code, 0, error)
            self.assertEqual(result['operation'], 'image.edit')
            self.assertEqual(result['lora_strategy'], 'inference_time')
            self.assertIn('qwen21_viggle_v021_r128_6step_distillation',
                          result['algorithm_approximations'])

    def test_qwen21_explicit_w8a8_edit_cli_gate(self):
        refs = [dict(kind='image', role='reference', path=f'qwen-exp-{i}.png')
                for i in range(3)]
        request = dict(model='qwen-image-2.1', operation='image.edit', prompt='Two teapots',
                       width=512, height=512, steps=40, audio=False, frames=1,
                       inputs=refs[:2], execution='gpu_ane', allow_approximation=True,
                       ane_manifest='qwen-w8a8-compiled.json', qwen21_w8a8=True,
                       qwen21_reference_size=256, qwen21_gpu_full_ffn_blocks=[3, 5, 7])
        code, result, error = plan(request)
        self.assertEqual(code, 0, error)
        self.assertEqual(result['precision'], 'bf16_gpu+w8a8_mlp_fp16_io')
        self.assertEqual(result['planned_w8a8_ffn_layer_coverage'], 29 / 32)
        self.assertEqual(result['qwen21_reference_size'], 256)
        self.assertEqual(result['qwen21_gpu_full_ffn_blocks'], [3, 5, 7])
        self.assertIn('qwen21_reference_resize_256', result['algorithm_approximations'])
        self.assertIn('qwen21_decode_mlp_w8a8_per_tensor', result['algorithm_approximations'])
        for size in (1, 2, 3):
            self.assertEqual(plan({**request, 'inputs': refs[:size]})[0], 0)
            code, full, error = plan({**request, 'inputs': refs[:size],
                                      'qwen21_gpu_full_ffn_blocks': []})
            self.assertEqual(code, 0, error)
            self.assertEqual(full['planned_w8a8_ffn_layer_coverage'], 1)
            self.assertEqual(full['qwen21_gpu_full_ffn_blocks'], [])
            code, half_size, error = plan({**request, 'inputs': refs[:size],
                                            'qwen21_reference_size': 512,
                                            'qwen21_gpu_full_ffn_blocks': []})
            self.assertEqual(code, 0, error)
            self.assertIn('qwen21_reference_resize_512', half_size['algorithm_approximations'])
            self.assertEqual(half_size['planned_w8a8_ffn_layer_coverage'], 1)
        schema2 = dict(schema_version=2, model='qwen-image-2.1', operation='image.edit',
                       inputs=[dict(kind='text', role='prompt', text='Two teapots'), *refs[:2]],
                       outputs=[dict(kind='image', path='qwen-edit.png', width=512, height=512,
                                     frames=1, audio=False)], sampling=dict(seed=42, steps=40),
                       execution=dict(policy='gpu_ane', allow_approximation=True,
                                      ane_manifest=request['ane_manifest'], qwen21_w8a8=True,
                                      qwen21_gpu_full_ffn_blocks=[3, 5, 7]),
                       parameters=dict(qwen21_reference_size=256))
        code, _, error = plan(schema2)
        self.assertEqual(code, 0, error)
        schema2['execution']['qwen21_gpu_full_ffn_blocks'] = []
        code, full, error = plan(schema2)
        self.assertEqual(code, 0, error)
        self.assertEqual(full['planned_w8a8_ffn_layer_coverage'], 1)
        for invalid in [dict(qwen21_w8a8=False), dict(qwen21_reference_size=512),
                        dict(qwen21_reference_size=1024),
                        dict(qwen21_gpu_full_ffn_blocks=[3, 5]),
                        dict(qwen21_gpu_full_ffn_blocks=[7, 5, 3]),
                        dict(qwen21_gpu_full_ffn_blocks=[3, 3, 7]),
                        dict(inputs=refs + refs[:1]), dict(width=1024),
                        dict(steps=1), dict(allow_approximation=False),
                        dict(execution='gpu'), dict(qwen21_gpu_w8a16='true')]:
            self.assertNotEqual(plan({**request, **invalid})[0], 0, invalid)
        gpu = {**request, 'execution': 'gpu', 'ane_manifest': '', 'qwen21_w8a8': False,
               'qwen21_gpu_full_ffn_blocks': []}
        self.assertEqual(plan(gpu)[0], 0)
        self.assertNotEqual(plan({**gpu, 'allow_approximation': False})[0], 0)
        other = {**gpu, 'model': 'z-image-turbo'}
        self.assertNotEqual(plan(other)[0], 0)

    def test_device_optimization_profile(self):
        system = json.loads(consume(C.c_void_p(lib.tc_system_json())))
        expected = system['gpu'] == 'Apple M5 Pro' and system['physical_memory_bytes'] == 24 << 30
        copy_only = system['gpu'] == 'Apple M4 Pro' and system['physical_memory_bytes'] == 48 << 30
        profile = system['optimization_profile']
        self.assertEqual(profile['id'], 'm5pro24-v1' if expected else 'm4pro48-coreml-copy-v1' if copy_only else 'legacy')
        for flag in ['z_image_suffix_streaming', 'z_image_hybrid_segments',
                     'z_image_memory_lifecycle', 'z_image_smallest_partition',
                     'external_automatic_partitions', 'coreml_output_copy', 'z_image_int8_streaming']:
            self.assertIs(profile[flag], expected or (copy_only and flag == 'coreml_output_copy'))

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
            self.assertEqual(configured['stages'][1]['iterations'], 8 if model == 'z-image-turbo' else 9)
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
            'feature_binding_seconds',
            'model_prediction_seconds',
            'output_handling_seconds',
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
            '@"feature_binding_seconds_session_total"',
            '@"model_prediction_seconds_session_total"',
            '@"output_handling_seconds_session_total"',
        ]:
            self.assertIn(key,results)
        self.assertIn('input, rows, true, zero_delta ? &*zero_delta : nullptr)',coreml)
        self.assertLess(coreml.index('std::optional<Tensor> zero_delta;'),
                        coreml.index('for (int iteration = 0; iteration < warmups; ++iteration)'))
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
        self.assertIn('r.hybrid_mlp_mode == "lora_fused"',source)
        self.assertIn('? std::vector<LoRAAsset>{} : active_loras_',source)
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

    def test_qwen21_lora_precision_receipt_survives_hybrid_route_selection(self):
        source = (ROOT/'native/models/qwen21/pipeline.cpp').read_text()
        marker = 'result.selection += "; experimental FP16 low-rank LoRA matmuls";'
        self.assertEqual(source.count(marker), 1)
        position = source.index(marker)
        self.assertLess(source.rfind('result.selection ='), position)
        self.assertIn('if (lora_fp16 && !r.loras.empty())', source[position-100:position])

    def test_native_inference_time_lora_keeps_packed_weights(self):
        header=(ROOT/'native/backends/mlx.hpp').read_text()
        source=(ROOT/'native/backends/mlx.cpp').read_text()
        z_image=(ROOT/'native/models/z_image/z_image.cpp').read_text()
        self.assertIn('runtime_loras_',header)
        self.assertIn('bool inference_time = false',header)
        self.assertIn('if (inference_time) {',source)
        # The optional Viggle FP16 low-rank path changes only the rank-sized
        # matmuls; accumulation back into the base output remains FP32.
        self.assertIn('const auto rank_dtype = runtime_lora_fp16_ ? mx::float16 : mx::float32;',source)
        self.assertIn('auto input = mx::astype(x, rank_dtype);',source)
        self.assertIn('auto down = mx::astype(adapter.down, rank_dtype);',source)
        self.assertIn('auto up = mx::astype(adapter.up, rank_dtype);',source)
        self.assertIn('auto low = mx::matmul(input, mx::transpose(down));',source)
        self.assertIn('mx::astype(mx::matmul(low, mx::transpose(up)), mx::float32)',source)
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

    def test_service_routes_ltx_worker_from_effective_plan(self):
        source=(ROOT/'services/turbociderd/service.mm').read_text()
        self.assertIn('route_ltx_plan(plan_value)',source)
        self.assertIn('memory_constrained_native_session',source)
        self.assertIn('@"service_route":@(route.name.c_str())',source)
        self.assertIn('job.route_resolved ? job.external_worker',source)
        route=source[source.index('bool external_ltx_request'):
                     source.index('bool resident_ltx_candidate_request')]
        self.assertIn('value.residency == "component_staged"',route)
        self.assertIn('!value.memory_constrained',route)
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
        self.assertIn('TURBOCIDER_TEST_STREAMING_CATALOG',cli)
        self.assertIn('#ifdef TURBOCIDER_ENABLE_TEST_HOOKS',cli)
        self.assertNotIn('TURBOCIDER_LTX_SMOKE_EXECUTOR',cli)
        self.assertIn('run_ltx_worker',service)
        self.assertIn('active_child_',service)
        self.assertIn('exec_ltx_video_finalizer',session)
        self.assertIn('ltx-video-finalizer',build)
        self.assertTrue(finalizer.is_file())
        self.assertTrue(converter.is_file())
        self.assertIn('ltx_mlx_video_vae_decode_tokens_bf16',
                      finalizer.read_text())
        self.assertIn('TURBOCIDER_LTX_VIDEO_VAE_CHECKPOINT_FD',
                      finalizer.read_text())
        self.assertIn('TURBOCIDER_LTX_PUBLIC_ENVELOPE_FD',
                      finalizer.read_text())
        self.assertIn('streaming_result_envelope',session)
        self.assertIn('verify_and_attach_public_streaming_result',session)
        self.assertIn('component_staged ||',session)
        self.assertIn('exact_streaming && public_stream_lease_',session)
        self.assertIn('ltx_mlx_video_vae_create_fd',finalizer.read_text())
        self.assertIn('F_DUPFD',session)
        self.assertIn('public_stream_lease_->revalidate_after_drain()',session)
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
        self.assertIn('max_refill_slots, 1, 1, &residency_plan', blocks)
        session=(ROOT/'native/platform/apple/ltx_session.mm').read_text()
        self.assertIn('options.max_refill_slots =', session)
        policy=(ROOT/'native/runtime/block_residency.c').read_text()
        self.assertIn('capacity >= active_blocks', policy)
        self.assertIn('return finish_plan(plan, active_blocks, 0u, 1)', policy)
        self.assertIn('request_slot_refills', session)
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
                         'bf16_gpu+coreml_mlp_fp16_io')
        self.assertEqual(configured['precision'], 'bf16')
        self.assertIn('qwen3_encoder_mlp_coreml_approximation',
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
        self.assertEqual(configured['precision'], 'bf16_gpu+coreml_mlp_fp16_io')

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
        self.assertIn('qwen3_vl_encoder_mlp_coreml_approximation',
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
        self.assertIn('gemma4_encoder_mlp_coreml_approximation',
                      configured['algorithm_approximations'])
        self.assertNotIn('qwen3_encoder_mlp_coreml_approximation',
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
