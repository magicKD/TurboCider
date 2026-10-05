#!/usr/bin/env python3
"""Strict Runtime worker wire/env/receipt/IO contracts using only CPU doubles."""
import hashlib
import json
import os
import struct
import subprocess
import tempfile
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(request, options):
    value = {'native_request_v2': request, 'runtime_options': options}
    return hashlib.sha256(b'tc-runtime-worker-request-v1\n' + json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def profile_environment(cache, qwen=False):
    result = {key: value for key, value in os.environ.items() if not key.startswith('TURBOCIDER_')}
    flags = {'ANE_BACKEND': 'auto', 'ALLOW_PRIVATE_ANE': '1', 'PRIVATE_ANE_DATA_PATH': 'w8a8',
             'PRIVATE_ANE_GPU_IO': '1', 'PRIVATE_ANE_CHANNELS': '4096', 'RUNTIME_ANE_CHUNKS': '1',
             'RUNTIME_ANE_PROFILE': '0', 'PRIVATE_ANE_SCALE_CACHE': '1', 'PRIVATE_ANE_PREFETCH': '0',
             'PRIVATE_ANE_LAUNCH_FENCE': '0', 'PRIVATE_ANE_A8_LOOKAHEAD': '0',
             'PRIVATE_ANE_STAGE_SPECIALIZE': '0', 'PRIVATE_ANE_CACHE_DIR': str(cache)}
    if qwen:
        flags.update(QWEN21_RUNTIME_STAGED_DIAGNOSTIC='1', QWEN21_RUNTIME_PREPARE_EARLY='0')
    result.update({'TURBOCIDER_' + key: value for key, value in flags.items()})
    return result


with tempfile.TemporaryDirectory(prefix='tc-runtime-worker-') as temporary:
    work = Path(temporary).resolve()
    binary = work / 'host-test'
    subprocess.run(['clang++', '-std=c++20', '-fobjc-arc', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'bindings/c/include'),
                    str(ROOT / 'tests/native/runtime_worker_contract_test.mm'),
                    '-framework', 'Foundation', '-o', str(binary)], check=True)
    # Actual tiny PNG bytes; no image library, model, Metal or Core ML initialization.
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    png = work / 'fixture.png'
    png.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 512, 512, 8, 2, 0, 0, 0))
                    + chunk(b'IDAT', zlib.compress((b'\0' + b'\x20\x40\x60' * 512) * 512)) + chunk(b'IEND', b''))
    terminal_modes = ['success', 'gpu_fallback', 'partial_fallback', 'all_fallback', 'plan_error',
                      'create_error', 'generate_error', 'cancel', 'missing_artifact', 'symlink_artifact',
                      'wrong_output', 'wrong_dimensions', 'wrong_steps', 'wrong_contract', 'wrong_channels',
                      'public_backend', 'wrong_axis', 'boolean_count', 'negative_count', 'missing_metrics',
                      'unexpected_streaming', 'wrong_execution', 'reason_missing',
                      'nested_reason_missing', 'nested_reason_bool', 'top_only_reason']
    rejection_modes = ['wrong_digest', 'unknown_options', 'unknown_request', 'unknown_execution',
                       'streaming', 'bad_steps', 'bad_descriptor_rows', 'bad_descriptor_biases',
                       'descriptor_symlink', 'input_symlink', 'input_fifo', 'input_empty', 'input_large',
                       'bad_env_value', 'missing_env', 'extra_mpp_env', 'capture_env', 's1_env', 'a8_single_pass_env', 'cache_symlink',
                       'existing_output', 'qwen_three_refs', 'qwen_r256', 'qwen_wrong_lora_strength',
                       'bad_hardware', 'bad_memory', 'boolean_memory']
    qwen_modes = ['qwen_base', 'qwen_one_ref', 'qwen_two_refs', 'qwen_r128']
    for mode in terminal_modes + rejection_modes + qwen_modes:
        case = work / mode
        case.mkdir()
        cache = case / 'cache'
        cache.mkdir()
        output = case / 'output.png'
        descriptor_path = case / 'descriptor.json'
        qwen = mode.startswith('qwen_')
        options = {'profile_id': 'private-w8a8-hadamard-512-v1',
                   'descriptor_path': str(descriptor_path), 'cache_dir': str(cache)}
        descriptor = {'schema_version': 1, 'descriptor_version': 1, 'backend': 'private_runtime_shape',
                      'kind': 'swiglu', 'rows': 1056, 'hidden': 4096 if qwen else 3840,
                      'width': 12288 if qwen else 10240, 'tile_k': 2048, 'tile_n': 1024,
                      'layout': 'out_in', 'biases': False, 'lora_inputs': True}
        request = {'schema_version': 2, 'model': 'qwen-image-2.1' if qwen else 'z-image-turbo',
                   'operation': 'image.generate', 'inputs': [{'kind': 'text', 'role': 'prompt', 'text': '猫 in sunlight'}],
                   'outputs': [{'kind': 'image', 'path': str(output), 'width': 512, 'height': 512,
                                'frames': 1, 'fps': 24, 'audio': False}], 'sampling': {'seed': 42, 'steps': 20 if qwen else 8},
                   'execution': {'policy': 'gpu_ane', 'ane_manifest': str(descriptor_path), 'allow_approximation': True,
                                 'hybrid_mlp_mode': 'runtime', 'residency': 'component_staged' if qwen else 'resident'},
                   'parameters': {'dynamic_text': True, 'compile_gpu': False}}
        if qwen:
            request['parameters']['qwen21_reference_size'] = 1024
        refs = {'qwen_one_ref': 1, 'qwen_two_refs': 2, 'qwen_three_refs': 3}.get(mode, 0)
        if refs:
            request['operation'] = 'image.edit'
            request['inputs'] += [{'kind': 'image', 'role': 'reference', 'path': str(case / f'ref{i}.png')} for i in range(refs)]
        if mode in ['qwen_r128', 'qwen_r256', 'qwen_wrong_lora_strength']:
            rank = 'r256' if mode == 'qwen_r256' else 'r128'
            request['loras'] = [{'path': str(case / f'Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-{rank}.safetensors'),
                                 'strength': 0.5 if mode == 'qwen_wrong_lora_strength' else 1, 'role': 'transformer'}]
            request['lora_strategy'] = 'inference_time'
            request['sampling']['steps'] = 6
        if mode == 'unknown_options': options['channels'] = 2048
        if mode == 'unknown_request': request['cfg'] = 2
        if mode == 'unknown_execution': request['execution']['qwen21_gpu_full_ffn_blocks'] = [1]
        if mode == 'streaming': request['execution']['streaming'] = {'schema_version': 2, 'enabled': True}
        if mode == 'bad_steps': request['sampling']['steps'] = 9
        if mode == 'bad_descriptor_rows': descriptor['rows'] = 512
        if mode == 'bad_descriptor_biases': descriptor['biases'] = 0
        descriptor_path.write_text(json.dumps(descriptor))
        if mode == 'descriptor_symlink':
            source = case / 'source.json'
            descriptor_path.rename(source)
            descriptor_path.symlink_to(source)
        if mode == 'cache_symlink':
            target = case / 'cache-target'
            cache.rename(target)
            cache.symlink_to(target, target_is_directory=True)
        if mode == 'existing_output': output.write_bytes(b'previous output')
        wire = {'protocol_version': 1, 'job_id': '00000000-0000-4000-8000-000000000001',
                'request_id': '00000000-0000-4000-8000-000000000002',
                'model_installation_ref': str(case / 'never-loaded-model'), 'native_request_v2': request,
                'runtime_options': options, 'request_digest': digest(request, options)}
        if mode == 'wrong_digest': wire['request_digest'] = '0' * 64
        input_path = case / 'input.json'
        input_path.write_text(json.dumps(wire, ensure_ascii=False))
        if mode == 'input_symlink':
            source = case / 'input-source.json'
            input_path.rename(source)
            input_path.symlink_to(source)
        if mode == 'input_fifo':
            input_path.unlink()
            os.mkfifo(input_path)
        if mode == 'input_empty': input_path.write_bytes(b'')
        if mode == 'input_large': input_path.write_bytes(b' ' * ((1 << 20) + 1))
        environment = profile_environment(cache, qwen)
        if mode == 'bad_env_value': environment['TURBOCIDER_PRIVATE_ANE_CHANNELS'] = '5120'
        if mode == 'missing_env': environment.pop('TURBOCIDER_PRIVATE_ANE_GPU_IO')
        if mode == 'extra_mpp_env': environment['TURBOCIDER_Z_RUNTIME_LORA_MPP'] = '1'
        if mode == 'capture_env': environment['TURBOCIDER_ANE_CALIBRATION_DIR'] = str(case / 'capture')
        if mode == 's1_env': environment['TURBOCIDER_PRIVATE_ANE_S1_PROFILE'] = str(case / 's1.json')
        if mode == 'a8_single_pass_env': environment['TURBOCIDER_PRIVATE_ANE_A8_SINGLE_PASS'] = '1'
        native_mode = mode if mode in terminal_modes + ['bad_hardware', 'bad_memory', 'boolean_memory'] else 'success' if mode in qwen_modes else 'rejected'
        proc = subprocess.run([str(binary), str(input_path), native_mode, str(png)], env=environment,
                              capture_output=True, text=True, timeout=5)
        succeeded = mode in ['success', 'gpu_fallback', 'partial_fallback', 'all_fallback'] + qwen_modes
        assert proc.returncode == (0 if succeeded else 2 if mode == 'cancel' else 1), (mode, proc.returncode, proc.stdout, proc.stderr)
        if not proc.stdout:
            assert mode in rejection_modes, (mode, proc.stderr)
            assert not output.exists(), mode
            print(mode, 'PASS')
            continue
        reply = json.loads(proc.stdout)
        for key in ['job_id', 'request_id', 'request_digest']: assert reply[key] == wire[key], mode
        assert reply['actual_container'] == 'cli_worker' and reply['runtime_fingerprint'] == 'test-runtime'
        assert reply['runtime_options'] == options
        for key in ['resolution_digest', 'record_digest', 'layout_digest', 'public_streaming_summary']: assert reply[key] is None
        events = [json.loads(line.removeprefix('TC_EVENT\t')) for line in proc.stderr.splitlines() if line.startswith('TC_EVENT\t')]
        if mode in terminal_modes and mode not in ['plan_error', 'create_error'] or mode in qwen_modes:
            assert [event['kind'] for event in events] == ['progress', 'progress'], (mode, events)
            assert [event['sequence'] for event in events] == [1, 2]
            for event in events:
                for key in ['job_id', 'request_id', 'request_digest']: assert event[key] == wire[key]
        else: assert not events
        if succeeded:
            assert reply['status'] == 'succeeded' and reply['error'] is None
            # Native exporter shape, including success and all fallback forms:
            # no hybrid.failure_reason, and a required nested string instead.
            native_hybrid = reply['result']['hybrid']
            assert 'failure_reason' not in native_hybrid, mode
            assert isinstance(native_hybrid['runtime_weight']['failure_reason'], str), mode
            assert reply['artifact'] == {'path': str(output), 'size': png.stat().st_size,
                                         'sha256': hashlib.sha256(png.read_bytes()).hexdigest()}
            actual = reply['runtime_receipt']
            gpu = mode in ['gpu_fallback', 'all_fallback']
            assert actual['backend'] == ('gpu' if gpu else 'private_ane')
            assert actual['actual_execution'] == ('gpu' if gpu else 'gpu_ane')
            assert actual['selected_backend'] == (None if mode == 'gpu_fallback' else 'private_ane')
            assert actual['ane_channels'] == (0 if gpu else 4096)
            assert actual['partial_fallback'] == (mode == 'partial_fallback')
            assert bool(actual['fallback_reason']) == (mode in ['gpu_fallback', 'all_fallback', 'partial_fallback'])
            if checker := os.environ.get('TURBOCIDER_TEST_RUNTIME_WORKER_TERMINAL'):
                terminal_path = case / 'terminal.json'
                terminal_path.write_text(proc.stdout)
                subprocess.run([checker, str(input_path), str(terminal_path)], check=True)
        else:
            assert reply['status'] == ('cancelled' if mode == 'cancel' else 'error')
            assert reply['artifact'] is None and reply['runtime_receipt'] is None and 'result' not in reply
        if mode == 'existing_output': assert output.read_bytes() == b'previous output'
        print(mode, 'PASS')
    missing = work / 'missing.json'
    for token in [b'', b'\0', b'\1']:
        proc = subprocess.run([str(binary), str(missing), 'rejected', str(png), '--supervised'],
                              input=token, capture_output=True, timeout=5)
        expected = b'worker_input_open_failed' if token == b'\1' else b'worker_start_not_admitted'
        assert proc.returncode == 1 and not proc.stdout and expected in proc.stderr
    proc = subprocess.Popen([str(binary), str(missing), 'rejected', str(png), '--supervised'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        time.sleep(0.1)
        assert proc.poll() is None, 'runtime worker started before admission'
        proc.stdin.close()
        proc.stdin = None
        stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode == 1 and not stdout and b'worker_start_not_admitted' in stderr
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    print('supervised admission/EOF PASS')
