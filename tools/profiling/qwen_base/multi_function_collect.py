#!/usr/bin/env python3
"""One real ANE model/two procedures/shared-pitch capability, never image speed.

Full raw-array oracles run only in --audit after the native child is terminal.
"""
import argparse
from collections import Counter
import ctypes as C
import json
import math
from pathlib import Path
import shutil
import sys

from collect import ROOT, Observer, sha256_file, write_json
from analyze import observer_evidence
from expanded_audit import archive_binding
from gpu_dual_request_audit import memory_contract
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

SOURCES = ('tools/profiling/qwen_base/multi_function_collect.py',
           'tools/profiling/qwen_base/multi_function.hpp',
           'tools/profiling/qwen_base/multi_function_probe.mm',
           'tools/native/build_ane_multi_function_probe.sh',
           'tools/profiling/qwen_base/observer.mm')

def source_files(kind):
    if kind == 'fp16':
        return SOURCES
    return (SOURCES[0], 'tools/profiling/qwen_base/w8_functions_probe.mm',
            'tools/native/build_ane_w8_functions_probe.sh', SOURCES[-1])


def check_w8_selectors(raw, report):
    rows = observer_evidence(raw)
    if raw['source_sha256'] != sha256_file(ROOT / 'tools/profiling/qwen_base/observer.mm'):
        raise ValueError('actual compiled observer differs')
    if sum(x['name'] == 'ane.evaluate_api' and x['request'] == 0 for x in rows) != 6:
        raise ValueError('both real W8 self-tests require three capability predictions')
    models = [set(),set()]
    for point in report['runs']:
        for arm, value in enumerate(point['arms']):
            scope = 1 + point['index'] * 2 + arm
            owned = [x for x in rows if x['request'] == scope]
            counts = Counter(x['name'] for x in owned)
            if counts['ane.evaluate_api'] != value['calls'] or counts['ane.compile_api'] or counts['ane.load_api']:
                raise ValueError('W8 family/control actual predictions or hot compilation differ')
            models[arm].update(x['model'] for x in owned if x['name'] == 'ane.evaluate_api')
    if any(len(x) != 1 for x in models) or models[0] == models[1]:
        raise ValueError('one retained model for each distinct legacy/family arm required')
    return dict(legacy_model_count=1, family_model_count=1, no_hot_compile_load=True,
                scope='actual Private selectors, no physical INT8/overlap qualification')


def child(a):
    observer = Observer(a.observer)
    scope = C.CFUNCTYPE(None, C.c_int)
    hook = scope(('tc_qwen_profile_scope', observer.lib))
    hook(-1)
    lib = C.CDLL(str(a.probe))
    fn = lib.tc_ane_multi_function_probe if a.kind == 'fp16' else lib.tc_w8_functions_probe
    fn.argtypes = [C.c_char_p, scope]
    try:
        if fn(str(a.output).encode(), hook):
            raise RuntimeError('multi-function native capability failed; original report retained')
    finally:
        hook(-3)
        if observer.lib.tc_qwen_profile_wait(2000):
            raise RuntimeError('completion observer did not drain')
        write_json(a.output / 'observer.json', observer.take('tc_qwen_profile_finish'))


def check_selectors(raw, report):
    rows = observer_evidence(raw)
    if raw['source_sha256'] != sha256_file(ROOT / 'tools/profiling/qwen_base/observer.mm'):
        raise ValueError('actual compiled observer source differs')
    models = {x['model'] for x in rows if x['name'] == 'ane.evaluate_api' and x['request'] > 0}
    if len(models) != 1:
        raise ValueError('one actual ANE model required')
    scopes = []
    for i in range(1, len(report['runs']) + 1):
        counts = Counter(x['name'] for x in rows if x['request'] == i)
        if counts['ane.evaluate_api'] != 1 or counts['ane.compile_api'] or counts['ane.load_api']:
            raise ValueError('each function requires one evaluate, no compile/load')
        scopes.append(dict(scope=i, counts=dict(counts)))
    return dict(actual_model_count=1, function_scopes=scopes, no_hot_compile_load=True,
                scope='actual host selectors, not physical ANE counters or hardware precision')


def full_array_oracle(x, w, y, count):
    import numpy as np
    if (count not in (32,64) or x.shape != (128,64) or w.shape != (128,128) or y.shape != (128,64) or
            any(not np.all(np.isfinite(v)) for v in (x,w,y))):
        raise ValueError('finite complete capability arrays and exact geometry required')
    # Explicit full K contraction, without the platform BLAS matmul's ambient
    # floating-point status. Never accept NaN error through a false comparison.
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        expected = np.einsum('ik,kj->ij', w.astype(np.float64), x[:, :count].astype(np.float64),
                             optimize=False).astype(np.float16)
    if not np.all(np.isfinite(expected)):
        raise ValueError('CPU half oracle is nonfinite')
    error = float(np.max(np.abs(y[:, :count].astype(np.float64) - expected.astype(np.float64))))
    tail = bool(np.all(y[:, count:] == np.float16(1234)))
    if not math.isfinite(error) or error > .015625 or not tail:
        raise ValueError('full-K CPU oracle or shared-pitch tail failed')
    return error, tail


def audit_w8_arrays(root, report):
    import numpy as np
    if (report.get('same_slot_bytes') is not True or report.get('column_view_negative_controls') != 3 or
            [x['rows'] for x in report['runs']] != [32,64,96,32]):
        raise ValueError('shared W8 slots/leading-column negatives and shape restoration required')
    results, outputs = [], []
    for point in report['runs']:
        i, count, variant = point['index'], point['rows'], point['weight_variant']
        x = np.fromfile(root / f'trial/run-{i}-input.bin', dtype='<f4').reshape(128,128)
        wg, wu, wd = [np.fromfile(root / f'trial/weight-{variant}-{j}.bin', dtype='<f4').reshape(shape)
                      for j, shape in enumerate(((512,128),(512,128),(128,512)))]
        if any(not np.all(np.isfinite(v)) for v in (x,wg,wu,wd)):
            raise ValueError('complete W8 source arrays must be finite')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            gate = np.einsum('rh,fh->rf',x[:count].astype(np.float64),wg.astype(np.float64),optimize=False)
            up = np.einsum('rh,fh->rf',x[:count].astype(np.float64),wu.astype(np.float64),optimize=False)
            hidden = gate/(1.+np.exp(-gate))*up
            expected = np.einsum('rf,hf->rh',hidden,wd.astype(np.float64),optimize=False)
        if not np.all(np.isfinite(expected)):
            raise ValueError('full W8 CPU reference is nonfinite')
        actuals = []
        for arm, row in enumerate(point['arms']):
            name = 'family' if arm else 'legacy'
            y = np.fromfile(root / f'trial/run-{i}-{name}.bin', dtype='<f4').reshape(128,128)
            error = float(np.linalg.norm(y[:count].astype(np.float64)-expected)/max(np.linalg.norm(expected),1e-30))
            p = row['policy']
            if (not np.all(np.isfinite(y)) or not math.isfinite(error) or error >= .06 or
                    not np.all(y[row['launch_rows']:] == 1234) or row['calls'] != (count+63)//64 or
                    p != dict(fixed=True,instances=1,blocked_rebuilds=0,disabled=False)):
                raise ValueError('complete W8 full-K oracle, padding, fixed program or consumer failed')
            actuals.append(y)
            results.append(dict(index=i,arm=name,rows=count,relative_l2_cpu_reference=error,
                                output_sha256=sha256_file(root / f'trial/run-{i}-{name}.bin')))
        delta = float(np.linalg.norm(actuals[1][:count].astype(np.float64)-actuals[0][:count].astype(np.float64))/max(np.linalg.norm(actuals[0][:count].astype(np.float64)),1e-30))
        if not math.isfinite(delta) or delta >= .005:
            raise ValueError('W8 family arithmetic delta exceeds the declared component budget')
        outputs.append(actuals)
    if any(not np.array_equal(outputs[0][i].view(np.uint32),outputs[-1][i].view(np.uint32)) for i in range(2)):
        raise ValueError('W8 A/B/A shape/source restoration changed output bits')
    return results


def audit(a):
    import numpy as np
    root = a.audit.resolve(strict=True)
    s = json.loads((root / 'summary.json').read_text())
    r = json.loads((root / 'trial/report.json').read_text())
    if s.get('kind','fp16') != a.kind:
        raise ValueError('explicit capability kind differs')
    if s['status'] != 'complete' or r['status'] != 'complete' or r['loaded_runtime_id'] != s['runtime_build_id']:
        raise ValueError('complete bound native capability required')
    manifest = json.loads((a.library.parent / 'runtime-build/runtime-build-manifest.json').read_text())
    if manifest['runtime_build_id'] != s['runtime_build_id']:
        raise ValueError('actual native build differs')
    archive_binding(a.native_source_archive, manifest)
    if any(sha256_file(ROOT / name) != value for name, value in s['source_sha256'].items()):
        raise ValueError('captured tool source differs')
    for key in ('library', 'probe', 'observer'):
        if sha256_file(getattr(a, key)) != s['assets'][key]:
            raise ValueError('actual native/helper/observer binary differs')
    memory_contract(root, s)
    if a.kind == 'w8':
        if r.get('compiled_native_id') != s['runtime_build_id']:
            raise ValueError('actual compiled W8 helper/native ABI differs')
        if r.get('activation') != s.get('activation'):
            raise ValueError('actual W8 activation differs from fixed process recipe')
        values = audit_w8_arrays(root,r)
        selectors = check_w8_selectors(json.loads((root/'trial/observer.json').read_text()),r)
        if r['cpp_sha256'] != s['source_sha256']['tools/profiling/qwen_base/w8_functions_probe.mm']:
            raise ValueError('compiled W8 helper source differs')
        result = dict(status='complete',qualification_passed=False,scope='synthetic full W8 SwiGLU consumer; no Qwen image/physical INT8 speed qualification',
                      runtime_build_id=s['runtime_build_id'],native_source_archive_sha256=sha256_file(a.native_source_archive),
                      full_array_oracles=values,selectors=selectors,restored_bits_exact=True,audit_script_sha256=sha256_file(Path(__file__)))
        if a.output.exists():raise ValueError('fresh independent audit output required')
        write_json(a.output,result);print(json.dumps(result,indent=2));return
    if (r['cpp_sha256'] != s['source_sha256']['tools/profiling/qwen_base/multi_function_probe.mm'] or
            r.get('mil_header_sha256') != s['source_sha256']['tools/profiling/qwen_base/multi_function.hpp'] or
            [x['rows'] for x in r['runs']] != [32, 64, 32] or
            len({x['procedure_id'] for x in r['runs']}) != 2 or
            not r['unknown_function_rejected'] or not r['implicit_binding_rejected']):
        raise ValueError('both real procedures and rejected binding controls required')
    outputs, values = [], []
    for i, row in enumerate(r['runs']):
        prefix = root / f'trial/run-{i}'
        x = np.fromfile(str(prefix) + '-input.bin', dtype='<f2').reshape(128, 64)
        w = np.fromfile(str(prefix) + '-weight.bin', dtype='<f2').reshape(128, 128)
        y = np.fromfile(str(prefix) + '-output.bin', dtype='<f2').reshape(128, 64)
        count = row['rows']
        error, tail = full_array_oracle(x,w,y,count)
        if (row.get('physical_complete') is not True or row.get('same_surface_set') is not True):
            raise ValueError('full-K CPU oracle/physical completion/shared-pitch tail failed')
        outputs.append(y)
        values.append(dict(index=i, elements=128*count, full_k=128, maximum_abs_half_error=error,
                           untouched_tail=tail, output_sha256=sha256_file(Path(str(prefix) + '-output.bin'))))
    if (not np.array_equal(outputs[0].view(np.uint16), outputs[2].view(np.uint16)) or
            np.array_equal(outputs[0].view(np.uint16), outputs[1].view(np.uint16))):
        raise ValueError('A/B/A source/shape restoration or stale-output negative control failed')
    selectors = check_selectors(json.loads((root / 'trial/observer.json').read_text()), r)
    result = dict(status='complete', qualification_passed=False, scope='synthetic FP16 capability only; no FFN/image speed or W8A8 proof',
                  native_source_archive_sha256=sha256_file(a.native_source_archive), runtime_build_id=s['runtime_build_id'],
                  full_array_oracles=values, selectors=selectors, restored_bits_exact=True,
                  audit_script_sha256=sha256_file(Path(__file__)))
    if a.output.exists():
        raise ValueError('fresh audit output required')
    write_json(a.output, result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('library', 'probe', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--observer', type=Path, default=ROOT / 'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib')
    p.add_argument('--audit', type=Path)
    p.add_argument('--kind', choices=('fp16','w8'),default='fp16')
    p.add_argument('--activation',choices=('exp','sigmoid-complement'),default='exp')
    p.add_argument('--native-source-archive', type=Path)
    p.add_argument('--child', action='store_true')
    a = p.parse_args()
    for key in ('library', 'probe', 'observer'):
        setattr(a, key, getattr(a, key).resolve(strict=True))
    a.output = a.output.absolute()
    if a.child:
        child(a)
        return
    if a.audit:
        if not a.native_source_archive:
            p.error('exact native source archive required for independent audit')
        audit(a)
        return
    manifest = json.loads((a.library.parent / 'runtime-build/runtime-build-manifest.json').read_text())
    if any(sha256_file(ROOT / x['path']) != x['sha256'] for x in manifest['inputs']['sources']):
        p.error('actual native sources differ from selected library')
    if shutil.disk_usage(ROOT).free < (2 << 30) + (128 << 20):
        p.error('2GiB reserve required')
    a.output.mkdir(parents=True, exist_ok=False)
    trial = a.output / 'trial'
    trial.mkdir()
    summary = dict(status='running', qualification_passed=False, performance_qualified=False,
                   runtime_build_id=manifest['runtime_build_id'],kind=a.kind,activation=a.activation,
                   assets={k: sha256_file(getattr(a, k)) for k in ('library', 'probe', 'observer')},
                   source_sha256={x: sha256_file(ROOT / x) for x in source_files(a.kind)})
    write_json(a.output / 'summary.json', summary)
    write_json(a.output / 'source.json', {x: (ROOT / x).read_text() for x in source_files(a.kind)})
    load = LoadObservation(a.output / 'load.jsonl')
    env = {k: v for k, v in benchmark_environment().items() if not k.startswith('TURBOCIDER_')}
    env.update(TURBOCIDER_ALLOW_PRIVATE_ANE='1', TURBOCIDER_ANE_BACKEND='private')
    if a.kind=='w8':env['TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION']=a.activation
    command = [sys.executable, '-B', __file__, '--child', '--library', str(a.library), '--probe', str(a.probe),
               '--observer', str(a.observer), '--output', str(trial),'--kind',a.kind,'--activation',a.activation]
    try:
        with (a.output / 'stdout.log').open('x') as stdout, (a.output / 'stderr.log').open('x') as stderr:
            summary['memory'] = run_sampled(command, repo=ROOT, output=a.output, stem='trial', env=env, stdout=stdout,
                                            stderr=stderr, timeout=180, interval_ms=100, max_gap_ms=500, observer=load)
        report = json.loads((trial / 'report.json').read_text())
        cpp_name='tools/profiling/qwen_base/'+('multi_function_probe.mm' if a.kind=='fp16' else 'w8_functions_probe.mm')
        if (report['status'] != 'complete' or report['loaded_runtime_id'] != summary['runtime_build_id'] or
                report['cpp_sha256'] != summary['source_sha256'][cpp_name] or
                (a.kind=='w8' and report.get('activation')!=a.activation) or
                (a.kind=='w8' and report.get('compiled_native_id')!=summary['runtime_build_id']) or
                (a.kind=='fp16' and report.get('mil_header_sha256') != summary['source_sha256']['tools/profiling/qwen_base/multi_function.hpp'])):
            raise ValueError('actual complete bound capability required')
        summary['selectors'] = (check_selectors if a.kind=='fp16' else check_w8_selectors)(json.loads((trial / 'observer.json').read_text()), report)
        summary['status'] = 'complete'
    except Exception as e:
        summary.update(status='failed', error=str(e))
    try:
        summary['load'] = load.verify()
    except ValueError as e:
        summary['load_error'] = str(e)  # Still never a performance qualification.
    write_json(a.output / 'summary.json', summary)
    if summary['status'] != 'complete':
        raise RuntimeError(summary['error'])
    print(json.dumps(dict(status='complete', output=str(a.output), performance_qualified=False)))


if __name__ == '__main__':
    main()
