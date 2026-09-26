#!/usr/bin/env python3
"""Compare bounded Z-Image lookahead at equal budgets; keep cold/warm runs separate.

Each case gets a fresh native process. Later rounds reverse depth order. Loading
and conversion time is summed across workers and must not be added to wall time.
The requested lookahead can be reduced by the budget; refill_slots reports the
actual allocation, including the current layer. A fully resident plan has zero.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from quality_gate import image_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--request', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--budgets', type=int, nargs='+', default=[6, 8])
    p.add_argument('--depths', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--runs', type=int, default=4, help='one cold run followed by warm runs')
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--reference', help='resident image for the identical prompt/seed/execution')
    a = p.parse_args()
    if a.runs < 2 or a.rounds < 1 or any(v < 6 for v in a.budgets) or any(v < 1 or v > 8 for v in a.depths):
        p.error('use >=2 runs, >=1 rounds, budgets >=6 GiB, and depths 1..8')
    request = json.loads(Path(a.request).read_text())
    if request.get('schema_version') == 2 or request.get('execution') not in ('gpu', 'gpu_ane'):
        p.error('supply a flat Z-Image request with explicit gpu or gpu_ane execution')
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {'library_sha256': hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
               'request': request, 'cases': [], 'scope': 'one cold run excluded from warm statistics; same budget includes all prefetch slots'}
    for round_index in range(a.rounds):
        depths = a.depths if round_index % 2 == 0 else list(reversed(a.depths))
        for budget in a.budgets:
            for depth in depths:
                name = f'r{round_index + 1}-{budget}g-p{depth}'
                folder = out / name
                request_path = out / (name + '.json')
                case_request = dict(request, residency='streamed', memory_budget_bytes=budget << 30)
                request_path.write_text(json.dumps(case_request, ensure_ascii=False, indent=2))
                env = dict(os.environ, TURBOCIDER_Z_STREAM_PREFETCH=str(depth))
                command = [sys.executable, str(Path(__file__).with_name('benchmark_native.py')),
                           '--library', str(Path(a.library).resolve()), '--model', str(Path(a.model).resolve()),
                           '--model-id', 'z-image-turbo', '--request', str(request_path),
                           '--output', str(folder), '--runs', str(a.runs), '--record-vm']
                print(f'Start {name}', flush=True)
                with (out / (name + '.log')).open('w') as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads((folder / 'report.json').read_text())
                warm = report['runs'][1:]
                case = {'name': name, 'budget_gib': budget, 'prefetch_requested': depth,
                        'cold_wall_seconds': report['runs'][0]['request_wall_including_export'],
                        'warm_wall_seconds': [v['request_wall_including_export'] for v in warm],
                        'median_wall_seconds': statistics.median(v['request_wall_including_export'] for v in warm),
                        'median_wait_seconds': statistics.median(v['metrics']['block_residency']['request_wait_seconds'] for v in warm),
                        'median_denoise_seconds': statistics.median(v['metrics']['timings_seconds']['denoise'] for v in warm),
                        'peak_mlx_bytes': max(v['metrics']['memory']['mlx_peak_bytes'] for v in warm),
                        'plan': warm[0]['metrics']['block_residency']}
                if a.reference:
                    case['quality'] = [image_metrics(Path(a.reference), folder / f'{i}.png') for i in range(a.runs)]
                    if any(v['mae_255'] != 0 for v in case['quality']):
                        raise RuntimeError(f'{name}: streaming output differs from resident reference')
                summary['cases'].append(case)
                (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                print(f"Done {name}: median {case['median_wall_seconds']:.3f}s, wait {case['median_wait_seconds']:.3f}s, "
                      f"pinned {case['plan']['pinned_blocks']}, slots {case['plan']['refill_slots']}", flush=True)


if __name__ == '__main__':
    main()
