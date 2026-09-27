#!/usr/bin/env python3
"""Exploratory paired public-semantics D/Q comparison, never release evidence.

Each sample gets a fresh process, source verification and full lifetime memory
sampling. Both catalogs must differ only in D/Q (apart from catalog/review fields).
OS caches are not flushed; there is no artificial pressure or extra warmup.
"""
import argparse
import copy
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'model', 'request', 'baseline-catalog', 'candidate-catalog', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-image-sha256', required=True)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='Continue complete raw samples in the frozen schedule; never rerun or overwrite a sample')
    parser.add_argument('--retain-inconclusive-memory', action='store_true', help='Continue timing observations while preserving INCONCLUSIVE memory verdicts; never qualifies memory')
    args = parser.parse_args()
    if args.pairs < 2 or args.pairs % 2:
        parser.error('--pairs must be even and at least 2')
    if len(args.expected_image_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.expected_image_sha256):
        parser.error('expected image digest must be lowercase SHA-256')
    records, catalogs = {}, {'baseline': args.baseline_catalog.resolve(), 'candidate': args.candidate_catalog.resolve()}
    configurations = {}
    for variant, path in catalogs.items():
        records[variant] = json.loads(path.read_text())['records']
        if len(records[variant]) != 1:
            raise ValueError('requires one exact record per catalog')
        records[variant] = records[variant][0]
        configurations[variant] = copy.deepcopy(records[variant]['plan'])
    for key in ('source', 'workload', 'runtime', 'device'):
        if records['baseline'][key] != records['candidate'][key]:
            raise ValueError(f'comparison changes {key}')
    stripped = {}
    for variant, plan in configurations.items():
        normalized = copy.deepcopy(plan)
        normalized.pop('layout_digest')
        stage = normalized['canonical_config']['stages']['denoiser']
        stage.pop('prefetch_distance'); stage.pop('io_workers')
        stripped[variant] = normalized
    if stripped['baseline'] != stripped['candidate']:
        raise ValueError('comparison changes more than D/Q')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    tools = Path(__file__).resolve().parent
    schedule = [dict(pair=i+1, variant=v) for i in range(args.pairs)
                for v in (['baseline', 'candidate'] if i % 2 == 0 else ['candidate', 'baseline'])]
    plan = dict(schema_version=1, scope=__doc__, schedule=schedule,
                library=str(args.library.resolve()), library_sha256=digest(args.library),
                request=json.loads(args.request.read_text()), model=str(args.model.resolve()),
                catalogs={k: dict(path=str(p), sha256=digest(p)) for k, p in catalogs.items()},
                configurations=configurations, expected_image_sha256=args.expected_image_sha256,
                source_sha256={p.name: digest(p) for p in [Path(__file__), tools/'run_public_streaming_smoke.py',
                               tools/'collect_streaming_memory.py', tools/'process_tree_sampler.py', tools/'verify_process_tree_samples.py']})
    if args.resume:
        frozen = json.loads((output/'plan.json').read_text())
        previous_sources = frozen.pop('source_sha256')
        frozen.pop('retain_inconclusive_memory', None)
        current_sources = plan.pop('source_sha256')
        if frozen != plan or any(previous_sources[k] != v for k, v in current_sources.items() if k != Path(__file__).name):
            raise ValueError('resume changed frozen inputs or measurement tools')
        # Preserve original plan and explicitly record driver-only amendment.
        with (output/'driver-history.jsonl').open('a') as history:
            history.write(json.dumps(dict(driver_sha256=current_sources[Path(__file__).name],
                retain_inconclusive_memory=args.retain_inconclusive_memory, original_driver_sha256=previous_sources[Path(__file__).name]))+'\n')
    else:
        plan['retain_inconclusive_memory'] = args.retain_inconclusive_memory
        write(output/'plan.json', plan)
    rows = []
    for index, item in enumerate(schedule):
        name = f'{index+1:02d}-{item["variant"]}'
        folder = output/name
        existing = folder.exists()
        if existing and not args.resume:
            raise ValueError(f'sample already exists: {folder}')
        if not existing:
            folder.mkdir()
        command = [sys.executable, str(tools/'collect_streaming_memory.py'), '--output', str(folder/'memory.jsonl'),
                   '--summary', str(folder/'memory-summary.json'), '--correlation-id', name, '--',
                   sys.executable, str(tools/'run_public_streaming_smoke.py'), '--library', str(args.library.resolve()),
                   '--model', str(args.model.resolve()), '--request', str(args.request.resolve()),
                   '--catalog', str(catalogs[item['variant']]), '--output', str(folder/'run')]
        if existing:
            if json.loads((folder/'command.json').read_text()) != command:
                raise ValueError('resume command differs')
            # Incomplete/interrupted folders require investigation; never retry silently.
            memory = json.loads((folder/'memory-summary.json').read_text())
            json.loads((folder/'run/observation.json').read_text())
            code = 0 if memory['complete'] else 3
        else:
            write(folder/'command.json', command)
            with (folder/'run.log').open('w') as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        row = dict(index=index+1, **item, exit_code=code)
        rows.append(row)
        write(output/'progress.json', rows)
        if code and not args.retain_inconclusive_memory:
            raise RuntimeError(f'{name} failed; evidence retained at {folder}')
        memory = json.loads((folder/'memory-summary.json').read_text())
        observation = json.loads((folder/'run/observation.json').read_text())
        result = json.loads((folder/'run/result.json').read_text())['result']
        actual = result['block_streaming']['actual_layout']
        stage = configurations[item['variant']]['canonical_config']['stages']['denoiser']
        if (memory.get('command_exit_code') != 0 or memory.get('status') not in ('complete', 'inconclusive')
                or (not memory['complete'] and not args.retain_inconclusive_memory)
                or observation['status'] != 'succeeded' or not observation['cleanup_returned']):
            raise RuntimeError('sample incomplete')
        if observation['image_sha256'] != args.expected_image_sha256:
            raise RuntimeError('image differs from independent reference')
        if (not actual['drained'] or not actual['source_lease_verified'] or
                any(actual[k] != stage[k] for k in ('resident_prefix_blocks', 'block_group_size', 'slot_count', 'prefetch_distance', 'io_workers')) or
                actual['receipt']['fills'] != actual['group_count'] * actual['pass_count']):
            raise RuntimeError('actual execution differs from intended tuple')
        row.update(memory_status=memory['status'], memory_reasons=memory['reasons'], max_gap_ns=memory['max_gap_ns'], wall_seconds=observation['wall_seconds'], request_seconds=result['timings_seconds']['request_wall'],
                   denoise_seconds=result['timings_seconds']['denoise'], wait_seconds=result['block_streaming']['request_wait_seconds'],
                   tree_peak_bytes=memory['tree_peak_phys_footprint_bytes'], mlx_peak_bytes=result['memory']['mlx_peak_bytes'],
                   swap_out_bytes=memory['swap_out_bytes'], image_sha256=observation['image_sha256'],
                   receipt_digest=result['public_streaming']['receipt_digest'], memory_sha256=digest(folder/'memory.jsonl'))
        write(output/'progress.json', rows)
        print(name, json.dumps(row), flush=True)
    summary = dict(schema_version=1, scope='Exploratory paired samples; not formal P0/P1/P2/P3 or release qualification',
                   memory_qualification='INCONCLUSIVE' if any(r['memory_status'] != 'complete' for r in rows) else 'NOT_QUALIFIED_SINGLE_WORKLOAD_EXPLORATION',
                   plan_sha256=digest(output/'plan.json'), samples=rows, variants={})
    for variant in catalogs:
        selected = [r for r in rows if r['variant'] == variant]
        summary['variants'][variant] = {k: statistics.median(r[k] for r in selected)
            for k in ('wall_seconds', 'request_seconds', 'denoise_seconds', 'wait_seconds', 'tree_peak_bytes', 'mlx_peak_bytes')}
    summary['candidate_reduction_percent'] = {k: 100*(1-summary['variants']['candidate'][k]/summary['variants']['baseline'][k])
        for k in ('wall_seconds', 'request_seconds', 'denoise_seconds')}
    write(output/'summary.json', summary)
    print(json.dumps(summary['candidate_reduction_percent']), flush=True)


if __name__ == '__main__':
    main()
