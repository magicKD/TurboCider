#!/usr/bin/env python3
"""Real wrapper, isolated native doubles; no GPU/PNG/public qualification claim."""
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix='tc-generate-contract-') as tmp:
    work = Path(tmp).resolve()
    binary = work / 'test'
    subprocess.run(['clang++', '-std=c++20', '-fobjc-arc', '-Wall', '-Wextra', '-Werror',
                    '-I', str(root / 'bindings/c/include'),
                    str(root / 'tests/native/worker_query_contract_test.mm'),
                    '-framework', 'Foundation', '-o', str(binary)], check=True)
    for mode in ['generate_success', 'bad_exact', 'existing', 'generate_error', 'generate_cancel',
                 'empty_result', 'unverified', 'wrong_layout', 'wrong_record', 'wrong_source',
                 'wrong_output', 'wrong_dimensions', 'missing_receipt', 'missing_artifact', 'symlink_artifact']:
        case = work / mode
        case.mkdir()
        output = case / 'output.png'
        if mode == 'existing':
            output.write_bytes(b'previous artifact')
        request = json.loads((root / 'tests/fixtures/streaming/worker-request-v1.json').read_text())
        request['outputs'][0].update(path=str(output), frames=1, fps=24, audio=False)
        request['parameters'] = {'dynamic_text': True}
        digest = hashlib.sha256(b'tc-worker-request-v1\n' + json.dumps(
            request, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        wire = {'protocol_version': 1, 'job_id': '00000000-0000-4000-8000-000000000001',
                'request_id': '00000000-0000-4000-8000-000000000002', 'request_digest': digest,
                'model_installation_ref': '/tmp/model', 'native_request_v2': request}
        input_path = case / 'input.json'
        input_path.write_text(json.dumps(wire))
        proc = subprocess.run([str(binary), str(input_path), mode, 'generate'],
                              capture_output=True, text=True, timeout=5)
        expected = 0 if mode == 'generate_success' else 2 if mode == 'generate_cancel' else 1
        assert proc.returncode == expected, (mode, proc.returncode, proc.stdout, proc.stderr)
        result = json.loads(proc.stdout)
        for key in ['job_id', 'request_id', 'request_digest']:
            assert result[key] == wire[key], mode
        assert result['actual_container'] == 'cli_worker' and result['runtime_fingerprint'] == 'test-runtime'
        if mode == 'generate_success':
            assert result['status'] == 'succeeded' and result['error'] is None
            assert result['resolution_digest'] == 'resolution' and result['record_digest'] == 'record'
            assert result['layout_digest'] == 'layout' and result['public_streaming_summary']['actual_plan_verified'] is True
            assert result['artifact'] == {'path': str(output), 'size': len(b'stub artifact\n'),
                                          'sha256': hashlib.sha256(b'stub artifact\n').hexdigest()}
            assert result['result']['output'] == str(output)
            if checker := os.environ.get('TURBOCIDER_TEST_WORKER_TERMINAL'):
                terminal_path = case / 'terminal.json'
                terminal_path.write_text(proc.stdout)
                subprocess.run([checker, str(input_path), str(terminal_path)], check=True)
        else:
            assert result['status'] == ('cancelled' if mode == 'generate_cancel' else 'error')
            assert result['artifact'] is None and result['public_streaming_summary'] is None
            assert result['resolution_digest'] is None and 'result' not in result
        if mode == 'existing':
            assert output.read_bytes() == b'previous artifact'
        if mode == 'bad_exact':
            assert not output.exists()
        print(mode, 'PASS')
