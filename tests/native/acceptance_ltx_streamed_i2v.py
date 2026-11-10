#!/usr/bin/env python3
"""Opt-in real-model regression: first-frame AV must survive weight-slot refills.

Runs the production Swift LTX worker at two different residency budgets and
compares all stage latents. Requires local model files, a compiled worker, and
the packaged CLI. It never downloads assets or touches the desktop draft.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditioning-cache', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    request = json.loads(args.request.read_text())
    assert request['operation'] == 'video.image' and request['audio'] is True
    assert len(request['inputs']) == 1 and request['inputs'][0]['role'] == 'first_frame'
    env = os.environ.copy()
    env['TURBOCIDER_LTX_CONDITIONING_CACHE_DIR'] = str(args.conditioning_cache.resolve())
    runs = []
    for budget in (12, 10):
        folder = args.output / f'{budget}gib'
        folder.mkdir()
        payload = {**request, 'residency': 'streamed', 'execution': 'gpu',
                   'ltx_backend': 'c_metal', 'memory_budget_bytes': budget << 30,
                   'dynamic_text': request.get('dynamic_text', True),
                   'output': str((folder / 'video.mp4').resolve()),
                   'dump_tensors': str((folder / 'tensors').resolve())}
        payload.pop('dump', None)
        request_path = folder / 'request.json'
        request_path.write_text(json.dumps(payload, indent=2) + '\n')
        started = time.monotonic()
        with (folder / 'result.json').open('w') as output, (folder / 'events.jsonl').open('w') as events:
            subprocess.run([str(args.worker.resolve()), str(args.model.resolve()),
                            str(args.cli.resolve()), str(request_path.resolve())],
                           env=env, stdout=output, stderr=events, check=True)
        result = json.loads((folder / 'result.json').read_text())
        assert result['operation'] == 'video.image' and result['audio'] is True
        assert result['worker_artifact']['media_verified']
        assert result['worker_artifact']['audio_pcm_verified']
        assert result['plan']['residency'] == 'streamed'
        assert result['plan']['memory_budget_bytes'] == budget << 30
        assert len(result['denoise_step_seconds']['stage1']) == 8
        assert len(result['denoise_step_seconds']['stage2']) == 3
        hashes = {}
        for name in ('stage1_video', 'stage1_audio', 'stage2_video', 'stage2_audio'):
            data = (folder / 'tensors' / f'{name}.bf16').read_bytes()
            assert data and len(data) % 2 == 0
            values = (np.frombuffer(data, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
            assert np.isfinite(values).all(), name
            hashes[name] = hashlib.sha256(data).hexdigest()
        runs.append({'budget_gib': budget, 'worker_wall_seconds': time.monotonic() - started,
                     'native_wall_seconds': result['timings_seconds']['request_wall'],
                     'streamed_blocks': result['ltx_streaming_observation']['streamed_blocks'],
                     'latent_sha256': hashes, 'artifact': result['worker_artifact']})
        print(json.dumps(runs[-1]), flush=True)
    assert runs[0]['streamed_blocks'] != runs[1]['streamed_blocks'], 'Test must move the resident/refill boundary'
    assert runs[0]['latent_sha256'] == runs[1]['latent_sha256'], 'First-frame conditioning changed across slot residency'
    report = {'complete': True, 'all_latents_finite': True,
              'different_residency_boundaries_byte_identical': True, 'runs': runs}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: streamed first-frame AV; all stage latents identical across different refill boundaries', flush=True)


if __name__ == '__main__':
    main()
