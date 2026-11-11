#!/usr/bin/env python3
"""Real-weight FFN ablation of W8A8 output/tile/sync options; not a video speed claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def options(items):
    result = {}
    for item in items:
        key, separator, value = item.partition('=')
        if not separator or not (key.startswith('TURBOCIDER_LTX_W8A8_') or key == 'TURBOCIDER_LTX_CONVROT_SIMD'):
            raise ValueError('expected a W8A8 or ConvRot SIMD option')
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--kind', choices=['mlp', 'gated'], default='mlp')
    parser.add_argument('--shader', type=Path, help='defaults to the immutable shader beside the probe')
    parser.add_argument('--rows', type=int, nargs='+', default=[1001, 4004])
    parser.add_argument('--runs', type=int, default=15)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--baseline-env', action='append', default=[])
    parser.add_argument('--candidate-env', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.runs <= 100 or args.rounds < 1 or any(not 1 <= n <= 4096 for n in args.rows):
        parser.error('positive rounds, runs 1..100 and rows 1..4096 required')
    args.output.mkdir(parents=True, exist_ok=False)
    environments = {
        arm: {'TURBOCIDER_LTX_W8A8_FUSED_OUTPUT': str(index),
              'TURBOCIDER_LTX_W8A8_TILE_M': '128',
              'TURBOCIDER_LTX_W8A8_NO_TG_BARRIER': '0', **options(items)}
        for index, (arm, items) in enumerate([('baseline', args.baseline_env), ('candidate', args.candidate_env)])}
    probe = args.probe.resolve()
    shader = args.shader.resolve() if args.shader else probe.with_name('ltx_shaders.metal')
    report = dict(complete=False, scope=__doc__, probe_sha256=sha(probe),
                  library_sha256=sha(probe.with_name('libturbocider.dylib')),
                  shader_sha256=sha(shader),
                  kind=args.kind,
                  environments=environments,
                  weights_sha256=sha(args.weights), samples=[], summaries=[])

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    for rows in args.rows:
        samples = []
        for index in range(args.rounds):
            order = ['baseline', 'candidate', 'candidate', 'baseline'] if index % 2 == 0 else ['candidate', 'baseline', 'baseline', 'candidate']
            for arm in order:
                env = {k: v for k, v in os.environ.items() if not k.startswith('TURBOCIDER_LTX_')}
                env.update(TURBOCIDER_LTX_CONVROT_W8A8='1', **environments[arm])
                output = args.output / f'{rows}-{len(samples)}-{arm}.bf16'
                process = subprocess.run([str(args.probe.resolve()), str(args.weights.resolve()),
                                          str(shader),
                                          str(rows), str(args.runs), args.kind, str(output.resolve())],
                                         env=env, capture_output=True, text=True, check=True)
                result = json.loads(process.stdout)
                if result['w8a8'] is not True:
                    raise ValueError('requested W8A8 route did not execute')
                if result.get('gated') is not (args.kind == 'gated'):
                    raise ValueError('requested FFN kind did not execute')
                if environments[arm].get('TURBOCIDER_LTX_CONVROT_SIMD') == '1' and result.get('convrot_simd') is not True:
                    raise ValueError('requested SIMD rotation did not execute')
                if (environments[arm].get('TURBOCIDER_LTX_W8A8_FUSED_GELU_CONVROT') == '1'
                        and result.get('gelu_convrot_dispatches') != args.runs + 2):
                    raise ValueError('requested GELU/ConvRot fusion did not execute for every sample')
                if (environments[arm].get('TURBOCIDER_LTX_W8A8_FUSED_ROT_QUANT') == '1'
                        and result.get('rot_quant_dispatches') != 2 * (args.runs + 2)):
                    raise ValueError('requested rotation/quantization fusion did not execute for every projection')
                if (environments[arm].get('TURBOCIDER_LTX_W8A8_FUSED_GELU_OUTPUT') == '1'
                        and args.kind == 'mlp'
                        and environments[arm].get('TURBOCIDER_LTX_W8A8_FUSED_OUTPUT') == '1'
                        and result.get('gelu_output_dispatches') != args.runs + 2):
                    raise ValueError('requested GELU output fusion did not execute for every FFN')
                row = dict(rows=rows, arm=arm, result=result, output=str(output), stderr=process.stderr)
                samples.append(row)
                report['samples'].append(row)
                save()
                print(json.dumps({k: v for k, v in row.items() if k not in ['stderr', 'output']}), flush=True)
        # Avoid CPU output scans between timed samples.
        hashes = [sha(Path(row['output'])) for row in samples]
        if len(set(hashes)) != 1:
            raise ValueError('FFN output changed or became nondeterministic')
        medians = {arm: statistics.median(row['result']['median_seconds'] for row in samples if row['arm'] == arm)
                   for arm in ['baseline', 'candidate']}
        report['summaries'].append(dict(rows=rows, medians=medians, byte_exact=True,
                                        sha256=hashes[0], speedup=medians['baseline'] / medians['candidate']))
        save()
    report['complete'] = True
    save()
    print(json.dumps(report['summaries'], indent=2), flush=True)


if __name__ == '__main__':
    main()
