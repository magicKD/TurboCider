"""Small, checkpoint-free runtime INT8 numerical and placement probe.

CPU_AND_NE is a policy, and a compute plan is not proof of physical INT8/ANE
execution. Latencies are diagnostic; this is not a production speed benchmark.
Explicit temporary model packages/compiled artifacts are removed on exit.
"""
import argparse
import collections
import gc
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
def errors(actual, reference):
    assert actual.shape == reference.shape and np.isfinite(actual).all()
    d = actual.astype(np.float64) - reference.astype(np.float64)
    return {'relative_l2': float(np.linalg.norm(d) / max(np.linalg.norm(reference), 1e-12)),
            'max_absolute': float(np.max(np.abs(d)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--medium", action="store_true", help="128 x 512 x 768 instead of 32 x 64 x 96")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, hidden, width = (128, 512, 768) if args.medium else (32, 64, 96)
    sys.path[:0] = [str(ROOT / 'tools/coreml'), str(ROOT / 'tools/validation')]
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.proto import MIL_pb2
    from runtime_ane_int8_candidate import export, specification, make_program, prepare_inputs
    from qwen21_ane_placement import inspect_plan
    from google.protobuf.text_format import MessageToString
    OUT = args.output.resolve()
    report = {'scope': 'synthetic projection; CPU_AND_NE policy, plan is not a hardware trace',
              'platform': platform.platform(), 'coremltools': ct.__version__, 'shape': [rows,hidden,width],
              'timing_scope': 'group=1 preparation + prediction including in-graph restoration; warmed ABBA, 4 samples per variant',
              'models': {}, 'hardware_int8_verified': False, 'production_speedup_verified': False}
    rng = np.random.default_rng(71)
    x = rng.normal(size=(rows,hidden)).astype(np.float32)
    w = rng.normal(size=(width,hidden)).astype(np.float32) * .2
    x[0] = 0; w[0] = 0
    cases = [('A', x, w), ('B', x * .03, w * -17), ('A_again', x, w),
             ('zero', np.zeros_like(x), np.zeros_like(w)),
             ('finite_restore', np.full_like(x,6000), np.full_like(w,.0001))]

    with tempfile.TemporaryDirectory(prefix='tc-int8-tiny-') as scratch:
        models = {}; specs = {}; all_outputs = {}
        for precision in ('fp16','qdq_int8'):
            path = Path(scratch) / precision
            spec = specification(rows,hidden,width,precision)
            t = time.perf_counter()
            export(path, spec, program_factory=make_program, compute_precision=ct.precision.FLOAT32)
            row = report['models'][precision] = {'export_compile_seconds': time.perf_counter()-t}
            proto = ct.models.MLModel(str(path/'graph.mlpackage'), skip_model_load=True).get_spec()
            (OUT/f'{precision}-converted-mil.txt').write_text(MessageToString(proto.mlProgram))
            ops = list(proto.mlProgram.functions['main'].block_specializations.values())[0].operations
            counts = collections.Counter(op.type for op in ops)
            row['converted_op_counts'] = dict(counts)
            row['converted_outputs'] = [{'operator': op.type, 'outputs': [
                {'name': output.name, 'dtype': MIL_pb2.DataType.Name(output.type.tensorType.dataType)}
                for output in op.outputs]} for op in ops if op.type != 'const']
            assert counts['matmul'] == 1
            assert counts['quantize'] == (2 if precision == 'qdq_int8' else 0)
            assert counts['dequantize'] == (2 if precision == 'qdq_int8' else 0)
            assert all(MIL_pb2.DataType.Name(op.outputs[0].type.tensorType.dataType)=='FLOAT32'
                       for op in ops if op.type=='mul')
            assert set(proto.mlProgram.functions['main'].inputs[i].name for i in range(4)) == {'x','w','x_scale','w_scale'}
            t = time.perf_counter()
            model = ct.models.CompiledMLModel(str(path/'graph.mlmodelc'),compute_units=ct.ComputeUnit.CPU_AND_NE)
            row['load_seconds'] = time.perf_counter()-t
            inputs = prepare_inputs(x,w,spec)
            t = time.perf_counter(); model.predict(inputs); row['first_prediction_ms'] = (time.perf_counter()-t)*1000
            row['cases'] = []; all_outputs[precision] = {}
            for group in (1,64):
                grouped_spec = specification(rows,hidden,width,precision,group)
                for name,a,b in cases:
                    prepared = prepare_inputs(a,b,grouped_spec)
                    actual = model.predict(prepared)['y']
                    reference = np.einsum("ik,jk->ij", a.astype(np.float64), b.astype(np.float64))
                    error = errors(actual,reference)
                    # Record accuracy; a synthetic relative-error value alone does not
                    # qualify image quality or a production backend.
                    assert np.isfinite(reference).all()
                    all_outputs[precision][(group,name)] = actual
                    row['cases'].append({'group': group,'case': name, **error})
                assert np.array_equal(all_outputs[precision][(group,'A')],all_outputs[precision][(group,'A_again')])
            row['planned_devices'] = inspect_plan(MLComputePlan.load_from_path(str(path/'graph.mlmodelc'),compute_units=ct.ComputeUnit.CPU_AND_NE))
            models[precision] = model; specs[precision] = spec
        samples = {p: [] for p in models}
        for _ in range(2):
            for p in ('fp16','qdq_int8','qdq_int8','fp16'):
                t=time.perf_counter(); inputs=prepare_inputs(x,w,specs[p]); t2=time.perf_counter()
                models[p].predict(inputs); t3=time.perf_counter()
                samples[p].append({'prepare_ms':(t2-t)*1000,'prediction_ms':(t3-t2)*1000,'total_ms':(t3-t)*1000})
        for p in models:
            report['models'][p]['samples']=samples[p]
            report['models'][p]['median_ms']={k:statistics.median(s[k] for s in samples[p]) for k in samples[p][0]}
        report['qdq_vs_control']=[{'group': group,'case':name,**errors(all_outputs['qdq_int8'][(group,name)],all_outputs['fp16'][(group,name)])} for group in (1,64) for name,_,_ in cases]
        report['total_latency_ratio_fp16_over_qdq']=report['models']['fp16']['median_ms']['total_ms']/report['models']['qdq_int8']['median_ms']['total_ms']
        del model, models
        gc.collect()
    report['explicit_artifacts_removed'] = not Path(scratch).exists()
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
