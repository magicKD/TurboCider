"""Sequential resolution/route crossover, first-call and warm timings kept separate.

Uses only local artifacts through the public C ABI; never exports or downloads.
Each group has its own engine process. AB/BA reverses order; GPU and INT8 output
quality are compared separately and are not asserted bit-identical.
"""
import argparse, hashlib, json, statistics, subprocess, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--manifest-512', required=True)
p.add_argument('--manifest-1024', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
library = root / 'build/native/libturbocider.dylib'
identity = hashlib.sha256(library.read_bytes()).hexdigest()
results = {}
for size in (512, 1024):
    records = {}
    for round_index, order in enumerate((('gpu', 'compiled', 'hybrid'), ('hybrid', 'compiled', 'gpu'))):
        for mode in order:
            if hashlib.sha256(library.read_bytes()).hexdigest() != identity:
                raise RuntimeError('Engine changed during benchmark')
            directory = out / f'{size}-{mode}-{round_index}'
            request = dict(model='flux2-klein-4b', operation='image.generate',
                prompt='A red fox in a snowy forest.', width=size, height=size,
                steps=4, seed=42, dynamic_text=True,
                execution='gpu_ane' if mode == 'hybrid' else 'gpu',
                compile_gpu=mode == 'compiled')
            if mode == 'hybrid':
                request.update(ane_manifest=getattr(a, f'manifest_{size}'), allow_approximation=True)
            file = out / 'request.json'
            file.write_text(json.dumps(request))
            print(f'RUN {size} {mode} round {round_index}', flush=True)
            subprocess.run([sys.executable, str(root / 'tools/native/benchmark_native.py'),
                '--library', str(library), '--model', a.model, '--request', str(file),
                '--output', str(directory), '--runs', '4'], check=True)
            report = json.loads((directory / 'report.json').read_text())
            records.setdefault(mode, []).append(report)
            print(f'DONE {size} {mode}: ' + str([round(r['request_wall_including_export'], 3) for r in report['runs']]), flush=True)
    summaries = {}
    for mode, groups in records.items():
        warm = [r['request_wall_including_export'] for group in groups for r in group['runs'][1:]]
        denoise = [r['metrics']['timings_seconds']['denoise'] for group in groups for r in group['runs'][1:]]
        hashes = {hashlib.sha256(Path(r['metrics']['output']).read_bytes()).hexdigest() for group in groups for r in group['runs']}
        if len(hashes) != 1:
            raise RuntimeError(f'Non-deterministic output in {size} {mode}')
        summaries[mode] = dict(warm_seconds=warm, warm_median=statistics.median(warm),
            denoise_median=statistics.median(denoise), png_sha256=next(iter(hashes)),
            first_request_seconds=[g['runs'][0]['request_wall_including_export'] for g in groups])
    if summaries['gpu']['png_sha256'] != summaries['compiled']['png_sha256']:
        raise RuntimeError(f'Compiled GPU changed output at {size}')
    for mode in summaries:
        summaries[mode]['speedup_vs_gpu'] = summaries['gpu']['warm_median'] / summaries[mode]['warm_median']
    results[str(size)] = summaries
    (out / 'summary.json').write_text(json.dumps(dict(library_sha256=identity,
        method='Two fresh processes per route; forward/reverse order; 4 requests per process, first excluded; 6 warm samples per route. Interactive desktop background load is not controlled.',
        results=results), indent=2))
