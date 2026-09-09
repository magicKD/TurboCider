#!/usr/bin/env python3
"""Native Wan eager/compiled ABBA timing; development only.

First request is not OS-cold: no cache purges or thermal-state controls are
performed. Warm requests reuse one pipeline and must report prompt-cache hits.
Outputs and logs are retained; this tool does not assert video-quality parity.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


ROW = re.compile(r'; seconds=([\d.eE+-]+); run=(\d+); prompt_cache_hit=([01]); '
                 r'compiled=([01]); mlx_peak_bytes=(\d+)')


def parse_rows(stdout, compiled, repeats):
    rows = []
    for match in ROW.finditer(stdout):
        seconds, run, hit, mode, peak = match.groups()
        row = dict(seconds=float(seconds), run=int(run), prompt_cache_hit=hit == '1',
                   compiled=mode == '1', mlx_peak_bytes=int(peak))
        if not math.isfinite(row['seconds']) or row['seconds'] <= 0:
            raise ValueError('invalid timing')
        if (row['run'] != len(rows) or row['compiled'] != compiled or
                row['prompt_cache_hit'] != (len(rows) > 0)):
            raise ValueError('unexpected mode/run/cache sequence')
        rows.append(row)
    if len(rows) != repeats:
        raise ValueError('missing request measurements')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--decoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, choices=[5, 81], default=5)
    parser.add_argument('--prompt', default='A red fox in snow')
    args = parser.parse_args()
    probe, model, decoder = (p.resolve(strict=True) for p in
                             [args.probe, args.model, args.decoder])
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'native eager/compiled ABBA, not baseline-migration or quality gate',
              'os_caches_purged': False, 'geometry': [832, 480, args.frames],
              'prompt': args.prompt, 'probe_sha256': hashlib.sha256(probe.read_bytes()).hexdigest(),
              'runs': []}
    for index, compiled in enumerate([False, True, True, False]):
        mode = 'compiled' if compiled else 'eager'
        print(f'{index + 1}/4: {mode}, three same-session requests', flush=True)
        command = [str(probe), str(model), str(decoder), args.prompt,
                   str(output / f'{index}-{mode}.mp4'), '832', '480', str(args.frames),
                   '--repeat', '3']
        if compiled:
            command.append('--compiled')
        with (output / f'{index}.stdout.log').open('x') as stdout, \
                (output / f'{index}.stderr.log').open('x') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, cwd=output)
        if result.returncode:
            raise SystemExit(f'probe failed ({result.returncode}); inspect {output}')
        rows = parse_rows((output / f'{index}.stdout.log').read_text(), compiled, 3)
        report['runs'].append({'mode': mode, 'command': command, 'requests': rows})
    report['summary'] = {}
    for mode in ['eager', 'compiled']:
        for category, select in [('first', lambda r: r['run'] == 0),
                                 ('warm', lambda r: r['run'] > 0)]:
            samples = [r['seconds'] for run in report['runs'] if run['mode'] == mode
                       for r in run['requests'] if select(r)]
            report['summary'][f'{mode}_{category}'] = {
                'samples_seconds': samples, 'median_seconds': statistics.median(samples),
                'min_seconds': min(samples), 'max_seconds': max(samples)}
    with (output / 'report.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
