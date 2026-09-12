"""Read-only runtime diagnosis: stage times and memory/VM deltas, no tuning.

Runs existing native C ABI with identical requests; samples only at stage
boundaries, not with a high-frequency polling thread. Host VM counters are
system-wide: positive deltas are evidence of concurrent pressure, not proof
that this process alone caused it.
"""
import argparse
import ctypes as c
import json
from pathlib import Path
import time

import mlx.core as mx

ROOT=Path(__file__).resolve().parents[2]
HOST_KEYS=('free_pages','compressor_pages','pageins','pageouts','compressions',
           'decompressions','swapins','swapouts')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execution',choices=('gpu','gpu_ane'),required=True)
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--cache-mib',type=int,help='Diagnostic-only MLX free-cache limit; does not change App settings')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    if a.cache_mib is not None:
        if a.cache_mib<0:raise ValueError('cache limit must be nonnegative')
        mx.set_cache_limit(a.cache_mib<<20)
    helper=c.CDLL(str(ROOT/'build/native/task-footprint.dylib'))
    helper.tc_task_footprint.argtypes=[c.POINTER(c.c_uint64)]
    helper.tc_host_vm.argtypes=[c.POINTER(c.c_uint64)]
    lib=c.CDLL(str(ROOT/'build/native/libturbocider.dylib'))
    lib.tc_string_free.argtypes=[c.c_void_p]
    lib.tc_engine_create_model.argtypes=[c.c_char_p,c.c_char_p,c.POINTER(c.c_void_p),c.POINTER(c.c_void_p)]
    lib.tc_engine_generate.argtypes=[c.c_void_p,c.c_char_p,c.c_void_p,c.c_void_p,c.POINTER(c.c_void_p),c.POINTER(c.c_void_p)]
    lib.tc_engine_free.argtypes=[c.c_void_p]

    def consume(ptr):
        if not ptr.value:return None
        value=c.string_at(ptr).decode();lib.tc_string_free(ptr);return value

    def snapshot(label):
        physical=(c.c_uint64*3)();host=(c.c_uint64*8)()
        if helper.tc_task_footprint(physical) or helper.tc_host_vm(host):raise RuntimeError('memory query failed')
        return dict(label=label,monotonic=time.perf_counter(),
                    phys_footprint_bytes=physical[0],ledger_peak_bytes=physical[2],
                    mlx_active_bytes=mx.get_active_memory(),mlx_cache_bytes=mx.get_cache_memory(),
                    host_vm=dict(zip(HOST_KEYS,list(host))))

    request=json.loads((ROOT/'examples/requests/z-image-turbo.json').read_text())
    request['execution']['policy']=a.execution
    request['outputs'][0].update(width=1024,height=1024)
    request['sampling'].update(seed=42,steps=9)
    if a.execution=='gpu_ane':
        manifest=ROOT/'models/coreml/z_image_convrot_native_a4096_s8_o32_b4128_all/compiled-cache/manifest-b4729a0ede31c479c0baf128bb8a211b48e51547d845223a6a9930575a3b2978.json'
        request['execution'].update(ane_manifest=str(manifest),allow_approximation=True)
    e,error=c.c_void_p(),c.c_void_p()
    root=ROOT/'outputs/quantization-20260910/int8_convrot-text-bf16/layout'
    status=lib.tc_engine_create_model(b'z-image-turbo',str(root).encode(),c.byref(e),c.byref(error))
    if status:raise RuntimeError(consume(error))
    report=dict(request=request,runs=[],cache_limit_mib=a.cache_mib,
                scope='Host VM counters are system-wide; self footprint is not ANE-exclusive memory')
    stages=[]
    @c.CFUNCTYPE(None,c.c_char_p,c.c_void_p)
    def event(text,ctx):
        event=json.loads(text)
        phase,done,total=event['phase'],event['completed'],event['total']
        label=None
        if phase=='denoise' and done in (0,total):label=f'denoise_{done}'
        if phase=='export' and done==0:label='after_vae_before_export'
        if label:
            try:stages.append(snapshot(label))
            except Exception as err:stages.append({'label':label,'error':str(err)})
    try:
        for i in range(a.runs):
            stages=[]
            before=snapshot('request_start')
            request['outputs'][0]['path']=str((a.output/f'{i}.png').resolve())
            result,error=c.c_void_p(),c.c_void_p()
            start=time.perf_counter()
            status=lib.tc_engine_generate(e,json.dumps(request).encode(),event,None,c.byref(result),c.byref(error))
            elapsed=time.perf_counter()-start
            text,message=consume(result),consume(error)
            if status:raise RuntimeError(message)
            row=dict(run=i,wall_seconds=elapsed,before=before,stages=stages,
                     after=snapshot('request_end'),metrics=json.loads(text))
            report['runs'].append(row)
            (a.output/'report.json').write_text(json.dumps(report,indent=2))
            print(i,row['metrics']['timings_seconds'],flush=True)
    finally:lib.tc_engine_free(e)


if __name__=='__main__':main()
