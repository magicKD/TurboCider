#!/usr/bin/env python3
"""Matched ABBA/BAAB comparison of PrunaVAED implementation candidates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            result.update(data)
    return result.hexdigest()


def compare(left, right):
    a = np.memmap(left, mode='r', dtype='<u2')
    b = np.memmap(right, mode='r', dtype='<u2')
    if a.shape != b.shape or not a.size:
        raise ValueError('output shape mismatch')
    square_error = square_reference = 0.0
    maximum = 0.0
    differences = 0
    for start in range(0, a.size, 1 << 20):
        aa = a[start:start + (1 << 20)]
        bb = b[start:start + (1 << 20)]
        differences += int(np.count_nonzero(aa != bb))
        x = (aa.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        y = (bb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('non-finite VAE pixels')
        delta = x - y
        square_error += float(np.dot(delta, delta))
        square_reference += float(np.dot(x, x))
        maximum = max(maximum, float(np.abs(delta).max()))
    return dict(all_finite=True, elements=int(a.size), differing_bf16_values=differences,
                byte_exact=differences == 0, rmse=(square_error / a.size) ** 0.5,
                relative_l2=(square_error / max(square_reference, 1e-30)) ** 0.5,
                max_abs=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--latent', type=Path, required=True)
    parser.add_argument('--shape', type=int, nargs=3, required=True)
    parser.add_argument('--candidate-env', action='append', required=True)
    parser.add_argument('--baseline-env', action='append', default=[],
                        help='defaults to the original MLX Conv3D backend')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or min(args.shape) < 1:
        parser.error('positive rounds and shape required')
    environments = {'baseline': {'TURBOCIDER_PRUNAVAED_CONV_TENSOROPS': '0'}, 'candidate': {}}
    for arm, items in [('baseline', args.baseline_env), ('candidate', args.candidate_env)]:
        for item in items:
            key, separator, value = item.partition('=')
            if not separator or not key.startswith('TURBOCIDER_PRUNAVAED_'):
                parser.error('switches must be TURBOCIDER_PRUNAVAED_NAME=VALUE')
            environments[arm][key] = value
    args.output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('TURBOCIDER_PRUNAVAED_', 'TURBOCIDER_LTX_VAE_'))
           and k != 'TURBOCIDER_LTX_VIDEO_VAE_CHECKPOINT_FD'}
    report = dict(complete=False, method='One complete warm-up per arm, then matched ABBA/BAAB; fresh process per decode; output comparison after all timings',
                  binary_sha256=digest(args.binary), latent_sha256=digest(args.latent),
                  checkpoint_sha256=digest(args.checkpoint), shape=args.shape,
                  baseline_environment=environments['baseline'],
                  candidate_environment=environments['candidate'], samples=[])

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    order = [('baseline', True), ('candidate', True)]
    for index in range(args.rounds):
        order.extend((arm, False) for arm in
                     (['baseline', 'candidate', 'candidate', 'baseline'] if index % 2 == 0
                      else ['candidate', 'baseline', 'baseline', 'candidate']))
    for index, (arm, warmup) in enumerate(order):
        timing = args.output / f'{index:02}-{arm}.json'
        output = args.output / f'{arm}.bf16'
        started = time.monotonic()
        process = subprocess.run([str(args.binary.resolve()), str(args.checkpoint.resolve()),
                                  str(args.latent.resolve()), *map(str, args.shape),
                                  str(output.resolve()), str(timing.resolve())],
                                 env={**env, **environments[arm]},
                                 capture_output=True, text=True)
        if process.returncode:
            (args.output / f'{index:02}-error.log').write_text(process.stderr)
            raise RuntimeError(process.stderr[-4000:])
        row = dict(arm=arm, warmup=warmup, process_wall_seconds=time.monotonic() - started,
                   **json.loads(timing.read_text()))
        if row['ltx_video_vae'] != 'prunavaed':
            raise ValueError('PrunaVAED did not execute')
        report['samples'].append(row)
        save()
        print(json.dumps(row), flush=True)
    medians = {arm: statistics.median(row['decode_compute_seconds'] for row in report['samples']
                                      if row['arm'] == arm and not row['warmup'])
               for arm in ['baseline', 'candidate']}
    report.update(median_decode_seconds=medians, speedup=medians['baseline'] / medians['candidate'],
                  pixel_comparison=compare(args.output / 'baseline.bf16', args.output / 'candidate.bf16'),
                  complete=True)
    save()
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
