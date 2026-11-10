#!/usr/bin/env python3
"""Complete native Gemma encoder ABBA comparison: BF16 vs original-code W8A8.

Both modes use the existing fused gated MLP. Measures first/warm encodes,
checks all raw video/audio/mask outputs and verifies actual GPU dispatch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess

from ltx_gemma_quality import tensor_quality


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--shader', type=Path, required=True)
    parser.add_argument('--prompt', action='append', required=True)
    parser.add_argument('--warm-runs', type=int, default=1)
    parser.add_argument('--gpu-taps', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-relative-l2', type=float, default=0.05)
    parser.add_argument('--min-cosine', type=float, default=0.998)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.warm_runs <= 50:
        parser.error('fresh output directory and warm-runs 1..50 required')
    with args.checkpoint.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < length < 64 << 20:
            raise ValueError('invalid checkpoint header length')
        header = json.loads(stream.read(length))
    layer_pattern = re.compile(r'model\.layers\.(\d+)\..*\.weight$')
    i8 = {name: item for name, item in header.items() if layer_pattern.fullmatch(name) and item['dtype'] == 'I8'}
    layers = {int(layer_pattern.fullmatch(name)[1]) for name in i8}
    if len(layers) != 48 or not i8:
        raise ValueError('unexpected Gemma layer layout')
    expected_projections = len(i8) * (args.warm_runs + 1)
    expected_quantizations = 4 * len(layers) * (args.warm_runs + 1)
    report = dict(schema='tc-ltx25-full-gemma-w8a8-abba-v1',
                  method='ABBA processes; same fused MLP; first encode plus warm encodes; all raw outputs checked',
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                  tokenizer_sha256=sha(args.tokenizer), probe_sha256=sha(args.probe), shader_sha256=sha(args.shader),
                  gpu_taps=args.gpu_taps, samples=[], summaries=[], complete=False)
    args.output.mkdir(parents=True)

    def save():
        temp = args.output / 'report.json.tmp'
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(args.output / 'report.json')

    save()
    for case, prompt in enumerate(args.prompt):
        samples, reference = [], None
        for index, mode in enumerate(('bf16', 'w8a8', 'w8a8', 'bf16')):
            folder = args.output / f'prompt-{case}-{index}-{mode}'
            env = os.environ.copy()
            env.pop('TURBOCIDER_LTX_RAW_CONVROT', None)
            env['TURBOCIDER_LTX_CONVROT_W8A8'] = '1' if mode == 'w8a8' else '0'
            env['TURBOCIDER_LTX_GEMMA_FUSED_MLP'] = '1'
            env['TURBOCIDER_LTX_GEMMA_RESIDENT_WEIGHTS'] = '0'
            env.pop('TURBOCIDER_LTX_GEMMA_GPU_TAPS', None)
            if args.gpu_taps:
                env['TURBOCIDER_LTX_GEMMA_GPU_TAPS'] = '1'
            command = [str(args.probe.resolve()), str(args.checkpoint.resolve()), str(args.tokenizer.resolve()),
                       str(args.shader.resolve()), prompt, str(folder.resolve()), str(args.warm_runs)]
            process = subprocess.run(['/usr/bin/time', '-l', *command], env=env, text=True,
                                     capture_output=True, timeout=1200)
            folder.mkdir(exist_ok=True)
            (folder / 'stdout.log').write_text(process.stdout)
            (folder / 'stderr.log').write_text(process.stderr)
            row = dict(case=case, prompt=prompt, mode=mode, command=command, returncode=process.returncode)
            report['samples'].append(row)
            save()
            if process.returncode:
                raise RuntimeError(process.stderr[-6000:])
            payload = json.loads(process.stdout)
            row['result'] = payload
            peak = re.search(r'(\d+)\s+maximum resident set size', process.stderr)
            row['peak_rss_bytes'] = int(peak[1]) if peak else None
            enabled = mode == 'w8a8'
            if (payload['convrot_w8a8'] != enabled or
                    payload['convrot_w8a8_projections'] != (expected_projections if enabled else 0) or
                    payload['convrot_w8a8_quantizations'] != (expected_quantizations if enabled else 0)):
                raise ValueError('full Gemma precision/reuse dispatch budget mismatch')
            if payload['ane_used'] or not payload['fused_mlp'] or payload['gpu_taps'] != args.gpu_taps:
                raise ValueError('encoder configuration differs from benchmark')
            reference = reference or folder
            qualities = {name: tensor_quality(folder / filename, reference / filename)
                         for name, filename in [('video', 'raw_video_context.bf16'), ('audio', 'raw_audio_context.bf16')]}
            if any(not q['all_finite'] or not q['shape_match'] for q in qualities.values()):
                raise ValueError('Gemma raw conditioning shape/nonfinite failure')
            row['quality'] = qualities
            row['mask_equal'] = (folder / 'raw_text_mask.bf16').read_bytes() == (reference / 'raw_text_mask.bf16').read_bytes()
            row['output_sha256'] = {name: sha(folder / name) for name in
                                    ['raw_video_context.bf16', 'raw_audio_context.bf16', 'raw_text_mask.bf16']}
            row['first_seconds'] = payload['create_seconds'] + payload['prepare_seconds'] + payload['first_seconds']
            row['warm_seconds'] = statistics.median(payload['warm_seconds'])
            samples.append(row)
            save()
            print(json.dumps(dict(case=case, mode=mode, first=row['first_seconds'], warm=row['warm_seconds'],
                                  rows=payload['rows'], quality=qualities)), flush=True)
        timing = {mode: statistics.median(s['warm_seconds'] for s in samples if s['mode'] == mode)
                  for mode in ('bf16', 'w8a8')}
        deterministic = all(len({tuple(sorted(s['output_sha256'].items())) for s in samples if s['mode'] == mode}) == 1
                            for mode in ('bf16', 'w8a8'))
        acceptable = deterministic and all(s['mask_equal'] and
            all(q['relative_l2'] <= args.max_relative_l2 and q['cosine'] >= args.min_cosine for q in s['quality'].values())
            for s in samples)
        report['summaries'].append(dict(case=case, **timing, speedup=timing['bf16'] / timing['w8a8'],
                                        deterministic=deterministic, quality_acceptable=acceptable))
        save()
    report['complete'] = True
    report['quality_acceptable'] = all(s['quality_acceptable'] for s in report['summaries'])
    save()
    print(json.dumps(report['summaries']), flush=True)


if __name__ == '__main__':
    main()
