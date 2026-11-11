#!/usr/bin/env python3
"""ABBA benchmark of native LTX/Gemma FFNs using real weight components.

Inputs are deterministic synthetic BF16. This measures resident component
execution, not complete video speed or visual quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def quality(reference, candidate):
    def read(path):
        return (np.fromfile(path, dtype='<u2').astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    ref, actual = read(reference), read(candidate)
    if ref.size != actual.size or not ref.size or not np.isfinite(ref).all() or not np.isfinite(actual).all():
        raise ValueError('FFN output geometry/nonfinite failure')
    rnorm, anorm = np.linalg.norm(ref), np.linalg.norm(actual)
    return dict(relative_l2=float(np.linalg.norm(actual - ref) / max(rnorm, 1e-30)),
                cosine=float(np.dot(actual, ref) / max(rnorm * anorm, 1e-30)),
                max_abs=float(np.max(np.abs(actual - ref))), byte_exact=bool(np.array_equal(actual, ref)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--component', type=Path, required=True, help='directory containing weights.safetensors and provenance.json')
    parser.add_argument('--rows', type=int, nargs='+', default=[32, 128, 1024])
    parser.add_argument('--runs', type=int, default=11)
    parser.add_argument('--output', type=Path, required=True, help='fresh benchmark directory')
    parser.add_argument('--max-relative-l2', type=float, default=0.05)
    parser.add_argument('--min-cosine', type=float, default=0.998)
    args = parser.parse_args()
    if not 1 <= args.runs <= 100 or any(not 1 <= n <= 4096 for n in args.rows):
        parser.error('rows must be 1..4096; runs 1..100')
    if args.output.exists():
        parser.error('fresh benchmark output required')
    provenance = json.loads((args.component / 'provenance.json').read_text())
    weights = args.component / 'weights.safetensors'
    if sha(weights) != provenance['fixture_sha256']:
        raise ValueError('component fixture changed after extraction')
    kind = 'gated' if provenance['kind'] == 'gemma-mlp' else 'mlp'
    sources = [args.probe.resolve(), ROOT / 'tools/native/ltx_convrot_mlp_probe.c',
               ROOT / 'native/models/ltx_runtime/ltx_gpu.m', ROOT / 'native/platform/apple/ltx_w8a8.mm',
               ROOT / 'native/models/ltx_runtime/ltx_shaders.metal',
               ROOT / 'native/third_party/cider/kernels/w8a8_matmul.metal',
               ROOT / 'native/third_party/cider/kernels/convrot_quantize.metal']
    report = dict(schema='tc-ltx25-real-weight-ffn-abba-v1',
                  scope='real weights, synthetic BF16 inputs, resident isolated FFN; not full video speed/quality',
                  platform=platform.platform(), source_sha256={str(p): sha(p) for p in sources},
                  component_provenance=provenance, samples=[], summaries=[], complete=False)
    args.output.mkdir(parents=True)
    report_path = args.output / 'report.json'

    def save():
        temp = args.output / 'report.json.tmp'
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(report_path)

    save()
    for rows in args.rows:
        samples = []
        reference = None
        for index, mode in enumerate(('bf16', 'w8a8', 'w8a8', 'bf16')):
            destination = args.output / f'rows-{rows}-{index}-{mode}.bf16'
            env = os.environ.copy()
            env.pop('TURBOCIDER_LTX_RAW_CONVROT', None)
            env['TURBOCIDER_LTX_CONVROT_W8A8'] = '1' if mode == 'w8a8' else '0'
            command = [str(args.probe.resolve()), str(weights.resolve()), str(sources[4]),
                       str(rows), str(args.runs), kind, str(destination.resolve())]
            process = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
            row = dict(mode=mode, rows=rows, command=command, returncode=process.returncode,
                       stderr=process.stderr, output=str(destination),
                       result=json.loads(process.stdout) if process.returncode == 0 else None)
            report['samples'].append(row)
            save()
            if process.returncode:
                raise RuntimeError(process.stderr)
            result = row['result']
            if result['w8a8'] != (mode == 'w8a8'):
                raise ValueError('requested precision path did not execute')
            if mode == 'w8a8' and (result['projections'] != (args.runs + 2) * (3 if kind == 'gated' else 2) or
                                   result['activation_quantizations'] != (args.runs + 2) * 2):
                raise ValueError('projection/activation reuse dispatch budget mismatch')
            reference = reference or destination
            row['quality'] = quality(reference, destination)
            row['sha256'] = sha(destination)
            samples.append(row)
            save()
            print(json.dumps(dict(mode=mode, rows=rows, result=result, quality=row['quality'])), flush=True)
        timing = {mode: statistics.median(s['result']['median_seconds'] for s in samples if s['mode'] == mode)
                  for mode in ('bf16', 'w8a8')}
        candidate = [s['quality'] for s in samples if s['mode'] == 'w8a8']
        stable = all(s['quality']['byte_exact'] for s in samples if s['mode'] == 'bf16')
        stable = stable and len({s['sha256'] for s in samples if s['mode'] == 'w8a8'}) == 1
        acceptable = stable and all(q['relative_l2'] <= args.max_relative_l2 and q['cosine'] >= args.min_cosine for q in candidate)
        report['summaries'].append(dict(rows=rows, **timing, speedup=timing['bf16'] / timing['w8a8'],
                                        quality_acceptable=acceptable, deterministic=stable, quality=candidate[0]))
        save()
        if not acceptable:
            raise ValueError('real-weight FFN quality/determinism gate failed')
    report['complete'] = True
    save()
    print(json.dumps(report['summaries']), flush=True)


if __name__ == '__main__':
    main()
