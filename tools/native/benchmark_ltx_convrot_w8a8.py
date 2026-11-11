#!/usr/bin/env python3
"""Measure native LTX QKV W8A8 against BF16 activations with ABBA ordering.

Synthetic projection timings are not end-to-end video quality or speed claims.
Build first: bash tools/native/build_ltx_attention_probe.sh
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, default=ROOT / 'build/native/ltx-qkv-projection-probe')
    parser.add_argument('--rows', type=int, nargs='+', default=[128, 1024])
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.runs <= 100 or any(not 1 <= n <= 16384 for n in args.rows):
        parser.error('rows must be 1..16384 and runs 1..100')
    sources = [args.probe, ROOT / 'native/models/ltx_runtime/ltx_gpu.m',
               ROOT / 'native/platform/apple/ltx_w8a8.mm',
               ROOT / 'native/models/ltx_runtime/ltx_shaders.metal',
               ROOT / 'native/third_party/cider/kernels/w8a8_matmul.metal',
               ROOT / 'native/third_party/cider/kernels/convrot_quantize.metal']
    report = dict(schema='tc-ltx-convrot-w8a8-synthetic-qkv-v1',
                  scope='synthetic QKV projections only; not end-to-end video speed or quality',
                  platform=platform.platform(),
                  chip=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  order=['bf16', 'w8a8', 'w8a8', 'bf16'], samples=[], summaries=[], complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(report, out, indent=2)

    def save():
        temp = args.output.with_name(args.output.name + '.tmp')
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(args.output)

    for rows in args.rows:
        samples = []
        for mode in report['order']:
            env = os.environ.copy()
            env.pop('TURBOCIDER_LTX_RAW_CONVROT', None)
            env['TURBOCIDER_LTX_CONVROT_W8A8'] = '1' if mode == 'w8a8' else '0'
            command = [str(args.probe.resolve()), str(sources[3]), str(rows), str(args.runs)]
            run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
            record = dict(mode=mode, rows=rows, command=command, returncode=run.returncode,
                          stderr=run.stderr, result=json.loads(run.stdout) if run.returncode == 0 else None)
            report['samples'].append(record)
            save()
            if run.returncode:
                raise RuntimeError(run.stderr)
            samples.append(record)
            print(json.dumps(record), flush=True)
        timings = {mode: statistics.median(s['result']['separate_seconds'] for s in samples if s['mode'] == mode)
                   for mode in ('bf16', 'w8a8')}
        report['summaries'].append(dict(rows=rows, **timings, speedup=timings['bf16'] / timings['w8a8']))
        save()
    report['complete'] = True
    save()
    print(json.dumps(report['summaries']), flush=True)


if __name__ == '__main__':
    main()
