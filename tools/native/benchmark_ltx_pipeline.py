#!/usr/bin/env python3
"""Measure packaged LTX generation with cold/warm prompt caches and step timings."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('fresh output directory required')
    output = args.output.resolve()
    output.mkdir(parents=True)
    request = json.loads(args.request.read_text())
    if request.get('ltx_backend') != 'c_metal' or not isinstance(request.get('audio'), bool):
        parser.error('C/Metal request with an explicit audio boolean required')
    env = os.environ.copy()
    for key in ('TURBOCIDER_LTX_RAW_CONVROT', 'TURBOCIDER_LTX_CONVROT_W8A8',
                'TURBOCIDER_LTX_GEMMA_FUSED_MLP', 'TURBOCIDER_LTX_GEMMA_RESIDENT_WEIGHTS',
                'TURBOCIDER_LTX_C_PROFILE'):
        env.pop(key, None)
    env['TURBOCIDER_LTX_CONDITIONING_CACHE_DIR'] = str(output / 'conditioning-cache')
    cli = args.cli.resolve()
    report = {'schema': 'tc-ltx-packaged-pipeline-timings-v1',
              'method': 'Fresh CLI process per run; production automatic precision; cold/warm refer to prompt conditioning cache, not OS file or shader caches; profiling is a separate diagnostic run',
              'cli': str(cli), 'cli_sha256': file_sha256(cli),
              'runtime_artifacts': {name: file_sha256(cli.with_name(name)) for name in
                                    ('libturbocider.dylib', 'ltx-video-finalizer', 'ltx_shaders.metal',
                                     'libmlx.dylib', 'mlx.metallib')},
              'model': str(args.model.resolve()), 'request': request, 'runs': [], 'complete': False}

    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    modes = ['prompt-cache-miss', 'prompt-cache-hit'] + (['profile'] if args.profile else [])
    for mode in modes:
        folder = output / mode
        folder.mkdir()
        value = dict(request, output=str(folder / 'video.mp4'), dump_tensors=str(folder / 'tensors'))
        request_path = folder / 'request.json'
        request_path.write_text(json.dumps(value, indent=2) + '\n')
        run_env = dict(env)
        if mode == 'profile':
            run_env['TURBOCIDER_LTX_C_PROFILE'] = '1'
        started = time.perf_counter()
        with (folder / 'stdout.log').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
            process = subprocess.run([str(cli), 'generate', str(args.model.resolve()), str(request_path)],
                                     env=run_env, stdout=stdout, stderr=stderr, timeout=1200)
        wall = time.perf_counter() - started
        if process.returncode:
            raise RuntimeError((folder / 'stderr.log').read_text()[-6000:])
        result = None
        for line in (folder / 'stdout.log').read_text().splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if 'timings_seconds' in item:
                result = item
        if result is None:
            raise ValueError('native result missing')
        if (result.get('steps') != request['steps'] or result.get('seed') != request['seed']
                or result.get('warmup') is not False):
            raise ValueError('App result correlation fields missing or inconsistent')
        timings = result['timings_seconds']
        if result.get('audio') != request['audio']:
            raise ValueError('audio output mode mismatch')
        if request['audio']:
            if result.get('audio_sample_rate') != 48000 or result.get('audio_channels') != 2:
                raise ValueError('48 kHz stereo audio result missing')
            for field in ('audio_vae_decode','audio_vocoder','audio_bwe','audio_decode','audio_mux'):
                if not math.isfinite(timings[field]) or timings[field] <= 0:
                    raise ValueError('invalid audio stage timing: ' + field)
        for field in ('text_encoder_load', 'text_encode', 'connector', 'denoise',
                      'stage1', 'stage2', 'video_vae_decode', 'request_wall'):
            if not math.isfinite(timings[field]) or timings[field] < 0:
                raise ValueError('invalid timing: ' + field)
        for stage, count in (('stage1', 8), ('stage2', 3)):
            steps = result['denoise_step_seconds'][stage]
            if (len(steps) != count or any(not math.isfinite(x) or x <= 0 for x in steps)
                    or sum(steps) > timings[stage] + 0.01):
                raise ValueError('invalid stage/step boundaries: ' + stage)
        expected_hit = mode != 'prompt-cache-miss'
        if result['conditioning_cache_hit'] != expected_hit:
            raise ValueError('conditioning cache experiment mismatch')
        encoder = timings['text_encoder_load'] + timings['text_encode']
        if (expected_hit and encoder != 0) or (not expected_hit and encoder <= 0):
            raise ValueError('encoder cache timing mismatch')
        profiles = []
        for line in (folder / 'stderr.log').read_text().splitlines():
            if line.startswith('ltx_c_profile '):
                profiles.append({key: float(number) for key, number in
                                 re.findall(r'(\w+)=([\d.eE+-]+)', line)})
        row = {'mode': mode, 'client_wall_seconds': wall, 'result': result,
               'encoder_total_seconds': encoder, 'dit_total_seconds': timings['denoise'],
               'stage1_mean_step_seconds': sum(result['denoise_step_seconds']['stage1']) / 8,
               'stage2_mean_step_seconds': sum(result['denoise_step_seconds']['stage2']) / 3,
               'native_block_profile': profiles}
        report['runs'].append(row)
        save()
        print(json.dumps({key: val for key, val in row.items() if key not in ('result', 'native_block_profile')}), flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
