#!/usr/bin/env python3
"""Summarize diagnostic Stage-2 Sol routes, including the dense blocks/steps.

Exact score-pair sparsity is not a FLOP reduction or a speedup: routed-out
tiles can still contribute pooled summaries, and routing itself costs work.
"""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(request, rows):
    if not request.get('ltx_sol_stage2') or request.get('ltx_sol_stage1'):
        raise ValueError('Stage-2-only Sol request required')
    edge_blocks = request.get('ltx_sol_dense_edge_blocks', 1)
    edge_steps = request.get('ltx_sol_dense_edge_steps', 0)
    expected = {(step, block) for step in range(edge_steps, 3 - edge_steps)
                for block in range(edge_blocks, 48 - edge_blocks)}
    actual = {(row['step'], row['block']) for row in rows}
    if len(actual) != len(rows) or actual != expected or not rows:
        raise ValueError('missing, duplicate or unexpected Sol calls')
    tokens = ((request['frames'] - 1) // 8 + 1) * (request['width'] // 32) * (request['height'] // 32)
    heads = rows[0]['heads']
    pairs_per_call = heads * tokens * tokens
    block_pairs = heads * ((tokens + 63) // 64) ** 2
    for row in rows:
        if (row['stage'] != 2 or row['rows'] != tokens or row['heads'] != heads or
                row['total_token_pairs'] != pairs_per_call or row['total_blocks'] != block_pairs or
                row['mode'] != request.get('ltx_sparse_mode', 0) or
                not 0 <= row['exact_token_pairs'] <= pairs_per_call or
                not 0 <= row['exact_blocks'] <= block_pairs):
            raise ValueError('inconsistent route geometry/counters')
    selected_exact = sum(row['exact_token_pairs'] for row in rows)
    selected_total = len(rows) * pairs_per_call
    dense_calls = 3 * 48 - len(rows)
    all_exact = selected_exact + dense_calls * pairs_per_call
    return {'stage2_video_rows': tokens, 'heads': heads, 'selected_calls': len(rows),
            'dense_calls': dense_calls, 'total_calls': 3 * 48,
            'selected_exact_pair_density': selected_exact / selected_total,
            'selected_exact_pair_sparsity': 1 - selected_exact / selected_total,
            'stage2_self_exact_pair_density': all_exact / (3 * 48 * pairs_per_call),
            'stage2_self_exact_pair_sparsity': 1 - all_exact / (3 * 48 * pairs_per_call),
            'minimum_call_density': min(row['exact_token_pairs'] / pairs_per_call for row in rows),
            'maximum_call_density': max(row['exact_token_pairs'] / pairs_per_call for row in rows),
            'calls': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('fresh output required')
    request = json.loads(args.request.read_text())
    result = json.loads(args.result.read_text())
    if (result.get('warmup') is not False or result.get('model') != request['model'] or
            result.get('seed') != request['seed'] or
            len(result.get('denoise_step_seconds', {}).get('stage2', [])) != 3):
        raise ValueError('completed matching native result required')
    prefix = 'ltx_sol_route_stats='
    rows = [json.loads(line[len(prefix):]) for line in args.log.read_text().splitlines() if line.startswith(prefix)]
    report = {'complete': True, 'scope': __doc__, **summarize(request, rows),
              'request': request, 'diagnostic_only': True,
              'inputs_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [args.request, args.log, args.result]}}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['calls', 'request', 'scope']}))


if __name__ == '__main__':
    main()
