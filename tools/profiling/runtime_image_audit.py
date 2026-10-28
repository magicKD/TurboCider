#!/usr/bin/env python3
"""Independent post-inference source/PNG/selector audit; contaminated != speed.

Run only after every model window is terminal. CPU pixel scans and full model
hashes are deliberately excluded from the inference collectors.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

from runtime_image_collect import ROOT, snapshot, validate_run, sha256_file, write_json
from expanded_audit import archive_binding, PINS
from gpu_dual_request_audit import memory_contract
from analyze import observer_contract
from audit import quality
import numpy as np
from PIL import Image

LOAD_FAILURES = frozenset(('competing inference CPU load observed; comparison is not qualified',
                          'load observation gap exceeded'))


def normalized_request(q):
    return {k: v for k, v in q.items() if k not in ('execution', 'hybrid_mlp_mode', 'ane_manifest', 'output')}


def pair_speed(a, b):
    # Never report even a tempting ratio from a timeline, shape sweep, synthetic
    # fixture, shortened sampling, failed or contaminated measurement window.
    if any(x['status'] != 'complete' or x['mode'] != 'stock' or
           x['spec'].get('expected_lora_projections', 0) or len(x['spec']['requests']) < 3 for x in (a, b)):
        return None
    qa, qb = ([normalized_request(q) for q in x['spec']['requests']] for x in (a, b))
    if qa != qb or len({(q['width'], q['height'], q['operation']) for q in qa}) != 1:
        return None
    aa, bb = (statistics.median(r['request_wall_including_export'] for r in x['report']['runs'][1:]) for x in (a, b))
    return dict(control_seconds=aa, candidate_seconds=bb, ratio=aa / bb)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--native-source-archive', type=Path, required=True)
    p.add_argument('--window', type=Path, action='append', required=True)
    p.add_argument('--quality-pair', nargs=3, action='append', default=[], metavar=('LABEL', 'GPU', 'CANDIDATE'))
    p.add_argument('--pair', nargs=3, action='append', default=[], metavar=('LABEL', 'GPU', 'CANDIDATE'))
    p.add_argument('--allow-contaminated-functionality', action='store_true', help='explicit functional/quality audit only; never a timing qualification')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('fresh output required')
    library = a.library.resolve(strict=True)
    manifest = json.loads((library.parent / 'runtime-build/runtime-build-manifest.json').read_text())
    archive_binding(a.native_source_archive, manifest)
    full = {str(library): sha256_file(library)}
    records, summaries, images = [], {}, {}
    for root in a.window:
        root = root.resolve(strict=True)
        s = json.loads((root / 'summary.json').read_text())
        r = json.loads((root / 'trial/report.json').read_text())
        contaminated = s.get('status') == 'failed' and s.get('error') in LOAD_FAILURES
        if (r['status'] != 'complete' or s['runtime_build_id'] != manifest['runtime_build_id'] or
                r['loaded_runtime_id'] != manifest['runtime_build_id'] or
                (s['status'] != 'complete' and not (contaminated and a.allow_contaminated_functionality))):
            raise ValueError('complete source-bound inference required; contamination needs explicit functional-only admission')
        before = s['source_snapshot']
        if json.loads(json.dumps(snapshot(list(before)))) != before:
            raise ValueError('actual source generations/prefixes changed')
        if str(library) not in before:
            raise ValueError('actual selected library is not bound')
        for name, body in json.loads((root / 'collector-source.json').read_text()).items():
            if (name not in before or hashlib.sha256(body.encode()[:65536]).hexdigest() != before[name]['prefix_sha256'] or
                    body != Path(name).read_text()):
                raise ValueError('actual captured/current collector source differs')
        q = [json.loads((root / f'trial/request-{i}.json').read_text()) for i in range(len(s['spec']['requests']))]
        if [normalized_request(x) for x in q] != [normalized_request(x) for x in s['spec']['requests']]:
            raise ValueError('actual requests differ from the admitted spec')
        validate_run(r, s['spec'])
        memory_contract(root, s)
        load_hash = sha256_file(root / 'load.jsonl')
        if not contaminated:
            load = s.get('load') or {}
            if load.get('complete') is not True or load.get('evidence_sha256') != load_hash:
                raise ValueError('original successful load evidence changed or is incomplete')
        for name in before:
            if name not in full:
                full[name] = sha256_file(Path(name))
                if Path(name).name in PINS and full[name] != PINS[Path(name).name]:
                    raise ValueError('complete known original asset SHA differs')
        after = json.loads(json.dumps(snapshot(list(before))))
        if after != before:
            raise ValueError('source generation changed during independent full hashing')
        pngs = []
        for i, request in enumerate(q):
            path = root / f'trial/{i}.png'
            if sha256_file(path) != s['png_sha256'][path.name]:
                raise ValueError('original PNG changed')
            with Image.open(path) as im:
                im.load()
                if im.format != 'PNG' or im.size != (request['width'], request['height']) or im.mode not in ('RGB', 'RGBA'):
                    raise ValueError('complete original-sized native RGB(A) output required')
                if s['spec']['model'] == 'qwen-image-2.1' and im.mode != 'RGBA':
                    raise ValueError('native Qwen RGBA output required')
                pngs.append(np.asarray(im.convert('RGBA')).copy())
        details = dict(root=str(root), images=len(pngs), actual_steps=sum(x['metrics']['actual_denoise_steps'] for x in r['runs']),
                       inference_complete=True, performance_load_gate_passed=not contaminated,
                       load_failure_reason=s.get('error'), load_evidence_sha256=load_hash,
                       performance_scope='synthetic capability only' if s['spec'].get('expected_lora_projections') else 'finite local sample')
        if s['mode'] == 'timeline':
            raw = json.loads((root / 'trial/observer.json').read_text())
            if raw.get('source_sha256') != sha256_file(ROOT / 'tools/profiling/qwen_base/observer.mm'):
                raise ValueError('actual compiled selector observer source differs')
            details['observer'] = observer_contract(raw, r['runs'], s['runtime'])
            previous = 0
            for i, row in enumerate(r['runs']):
                calls = (row['metrics'].get('hybrid') or {}).get('runtime_calls_session_total', 0)
                counts = Counter(x['name'] for x in raw['events'] if x['request'] == i)
                if counts['ane.evaluate_api'] != calls - previous + (3 if i == 0 and s['runtime'] else 0):
                    raise ValueError('actual model predictions differ from successful native workload/capability calls')
                previous = calls
        if len(q) > 1 and normalized_request(q[0]) == normalized_request(q[-1]):
            details['restored_rgba_exact'] = bool(np.array_equal(pngs[0], pngs[-1]))
            if not details['restored_rgba_exact']:
                raise ValueError('same-seed restored request changed pixels')
        records.append(details)
        summaries[str(root)] = dict(s, report=r)
        images[str(root)] = pngs
        print(json.dumps(dict(window=str(root), audited=True, load_gate_passed=not contaminated)), flush=True)
    pairs = []
    for is_quality, (label, left, right) in [(True, x) for x in a.quality_pair] + [(False, x) for x in a.pair]:
        left, right = str(Path(left).resolve()), str(Path(right).resolve())
        x, y = summaries[left], summaries[right]
        if [normalized_request(q) for q in x['spec']['requests']] != [normalized_request(q) for q in y['spec']['requests']]:
            raise ValueError('quality pair must retain identical model/task/prompt/steps/seeds')
        if x['spec']['source_files'] != y['spec']['source_files']:
            raise ValueError('quality pair must retain identical original checkpoint/conditioning sources')
        speed = None if is_quality else pair_speed(x, y)
        if not is_quality and speed is None:
            raise ValueError('speed pairs require clean stock repeat windows; timelines, shape sweeps and synthetic LoRAs are never denominators')
        pairs.append(dict(label=label, speedup=speed, scope='quality-only, never timing qualification' if is_quality else 'matched clean finite stock sample',
                          quality=[quality(u, v) for u, v in zip(images[left], images[right])]))
    write_json(a.output, dict(status='complete', qualification_passed=False,
        scope='independent original source/PNG/request/memory/selectors audit; no global semantic or physical-engine qualification',
        library_sha256=full[str(library)], native_source_archive_sha256=sha256_file(a.native_source_archive),
        runtime_build_id=manifest['runtime_build_id'], audit_script_sha256=sha256_file(Path(__file__)),
        full_source_sha256=full, windows=records, pairs=pairs))


if __name__ == '__main__':
    main()
