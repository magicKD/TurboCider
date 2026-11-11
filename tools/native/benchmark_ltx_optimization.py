#!/usr/bin/env python3
"""Matched complete-process LTX optimization ablations with fixed PrunaVAED."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def assignments(items):
    result = {}
    for item in items:
        key, separator, value = item.partition('=')
        if not separator or not key.startswith('TURBOCIDER_'):
            raise ValueError('expected TURBOCIDER_NAME=VALUE')
        result[key] = value
    return result


def tuning_fields(items):
    result = {}
    for item in items:
        key, separator, value = item.partition('=')
        if (not separator or key in ['ltx_video_vae', 'ltx_video_vae_path'] or
                not (key.startswith('ltx_') or key in ['allow_approximation', 'residency', 'memory_budget_bytes'])):
            raise ValueError('only implementation/approximation controls may change between arms')
        result[key] = json.loads(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--candidate-env', action='append', required=True)
    parser.add_argument('--baseline-env', action='append', default=[])
    parser.add_argument('--candidate-request', action='append', default=[], help='implementation FIELD=JSON_VALUE')
    parser.add_argument('--baseline-request', action='append', default=[])
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--reverse-order', action='store_true', help='start measured samples with BAAB')
    parser.add_argument('--conditioning-cache', type=Path, help='optional shared cache for a fixed verified encoder')
    parser.add_argument('--skip-latent-dump', action='store_true',
                        help='measure normal output without diagnostic latent exports')
    parser.add_argument('--candidate-first-warmup', action='store_true',
                        help='record the candidate with an initially empty prompt cache first')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('positive rounds required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    payload = json.loads(args.request.read_text())
    if payload.get('ltx_video_vae') != 'prunavaed':
        parser.error('explicit PrunaVAED baseline required')
    options = {'baseline': assignments(args.baseline_env), 'candidate': assignments(args.candidate_env)}
    tuning = {'baseline': tuning_fields(args.baseline_request), 'candidate': tuning_fields(args.candidate_request)}
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('TURBOCIDER_LTX_', 'TURBOCIDER_PRUNAVAED_'))}
    cache = args.conditioning_cache.resolve() if args.conditioning_cache else args.output / 'conditioning-cache'
    env['TURBOCIDER_LTX_CONDITIONING_CACHE_DIR'] = str(cache)
    cli = args.cli.resolve()
    artifacts = {name: sha(cli.with_name(name)) for name in
                 ['turbocider', 'libturbocider.dylib', 'ltx-video-finalizer', 'ltx_shaders.metal']}
    manifest = cli.parent / 'runtime-build/runtime-build-manifest.json'
    report = dict(complete=False, method='One complete warm-up per arm, then ABBA/BAAB; fresh CLI process per sample; common conditioning cache; compare outputs after timing',
                  artifacts_sha256=artifacts, request=payload, environments=options, request_overrides=tuning,
                  source_build_manifest=json.loads(manifest.read_text()) if manifest.exists() else None,
                  conditioning_cache=str(cache),
                  latent_dump_enabled=not args.skip_latent_dump,
                  samples=[], summaries={})

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    order = [('baseline', True), ('candidate', True)]
    if args.candidate_first_warmup:
        order.reverse()
    for index in range(args.rounds):
        order.extend((arm, False) for arm in
                     (['baseline', 'candidate', 'candidate', 'baseline'] if (index + args.reverse_order) % 2 == 0
                      else ['candidate', 'baseline', 'baseline', 'candidate']))
    report['planned_sequence'] = [{'arm': arm, 'warmup': warmup} for arm, warmup in order]
    save()
    for index, (arm, warmup) in enumerate(order):
        folder = args.output / f'{index:02}-{arm}'
        folder.mkdir()
        request = {**payload, **tuning[arm], 'output': str(folder / 'video.mp4')}
        if args.skip_latent_dump:
            request.pop('dump_tensors', None)
        else:
            request['dump_tensors'] = str(folder / 'tensors')
        request_path = folder / 'request.json'
        request_path.write_text(json.dumps(request, indent=2) + '\n')
        started = time.monotonic()
        with (folder / 'stdout.log').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
            result = subprocess.run([str(cli), 'generate', str(args.model.resolve()), str(request_path)],
                                    env={**env, **options[arm]}, stdout=stdout, stderr=stderr)
        if result.returncode:
            raise RuntimeError((folder / 'stderr.log').read_text()[-4000:])
        records = []
        for line in (folder / 'stdout.log').read_text().splitlines():
            try:
                data = json.loads(line)
                if 'timings_seconds' in data:
                    records.append(data)
            except ValueError:
                pass
        if len(records) != 1:
            raise ValueError('one native result required')
        data = records[0]
        for key in ['model', 'operation', 'width', 'height', 'frames', 'steps', 'seed', 'audio']:
            if data.get(key) != request[key]:
                raise ValueError('result identity mismatch: ' + key)
        if data.get('ltx_video_vae') != 'prunavaed' or data.get('warmup') is not False:
            raise ValueError('PrunaVAED actual receipt missing')
        row = dict(arm=arm, warmup=warmup, folder=str(folder),
                   process_wall_seconds=time.monotonic() - started, result=data)
        report['samples'].append(row)
        save()
        print(json.dumps(dict(arm=arm, warmup=warmup, wall=row['process_wall_seconds'],
                              timings=data['timings_seconds'])), flush=True)
    timing_keys = set.intersection(*(set(row['result']['timings_seconds'])
                                     for row in report['samples']))
    for key in sorted(timing_keys):
        values = {arm: [row['result']['timings_seconds'][key] for row in report['samples']
                        if row['arm'] == arm and not row['warmup']] for arm in options}
        medians = {arm: statistics.median(samples) for arm, samples in values.items()}
        report['summaries'][key] = dict(medians=medians, samples=values,
                                         speedup=(medians['baseline'] / medians['candidate']
                                                  if medians['candidate'] else None))
    tensors = {}
    for name in ['stage1_video', 'stage1_audio', 'stage2_input_video', 'stage2_video', 'stage2_audio']:
        if args.skip_latent_dump:
            break
        hashes = []
        for row in report['samples']:
            path = Path(row['folder']) / 'tensors' / (name + '.bf16')
            data = np.fromfile(path, dtype='<u2')
            if not data.size or not np.isfinite((data.astype(np.uint32) << 16).view(np.float32)).all():
                raise ValueError('invalid latent: ' + str(path))
            hashes.append(sha(path))
        tensors[name] = dict(byte_exact=len(set(hashes)) == 1, hashes=hashes)
    report.update(latent_comparison=tensors,
                  all_latents_byte_exact=(all(x['byte_exact'] for x in tensors.values())
                                         if tensors else None),
                  complete=True)
    save()
    print(json.dumps(dict(summaries=report['summaries'], all_latents_byte_exact=report['all_latents_byte_exact']), indent=2), flush=True)


if __name__ == '__main__':
    main()
