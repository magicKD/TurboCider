#!/usr/bin/env python3
"""Private ConvRot GPU/runtime-FP16 model consumer probe; not qualification."""
import argparse
import ctypes as C
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from benchmark_z_image_streaming import process_memory,vm_counters
from benchmark_z_image_metal import loaded_runtime_libraries


def digest(path):
    with path.open("rb") as source: return hashlib.file_digest(source,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--route",choices=["gpu","runtime"],required=True)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--chunks",type=int,default=1)
    p.add_argument("--size",type=int,default=512)
    p.add_argument("--steps",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--runs",type=int,default=1)
    p.add_argument("--dump",action="store_true")
    p.add_argument("--prompt",default="A studio photograph of an adult ceramic artist, both hands visible while holding a small blue cup, neutral background, natural skin texture.")
    a=p.parse_args()
    if a.output.exists() or a.output.is_symlink(): p.error("output already exists")
    if (a.route=="runtime")!=(a.manifest is not None): p.error("runtime requires manifest; GPU must omit it")
    if not 0<=a.chunks<=128 or not 1<=a.runs<=8: p.error("invalid chunks/runs")
    controls={"TURBOCIDER_Z_IMAGE_TRANSFORMER":str(a.checkpoint.resolve())}
    if a.route=="runtime":
        controls.update(TURBOCIDER_Z_RUNTIME_CONVROT="1",TURBOCIDER_RUNTIME_ANE_CHUNKS=str(a.chunks))
    elif os.environ.get("TURBOCIDER_Z_RUNTIME_CONVROT","0")!="0": p.error("GPU baseline must not enable runtime ConvRot")
    for key,value in controls.items():
        if key in os.environ and os.environ[key]!=value: p.error("conflicting environment: "+key)
    os.environ.update(controls)
    a.output.mkdir(parents=True)
    with open("/tmp/turbocider-z-image-gpu-benchmark.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lib=C.CDLL(str(a.library.resolve()))
        lib.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.tc_engine_free.argtypes=[C.c_void_p];lib.tc_string_free.argtypes=[C.c_void_p]
        def take(pointer):
            if not pointer.value:return None
            result=C.string_at(pointer).decode();lib.tc_string_free(pointer);return result
        report={"schema":"tc-convrot-runtime-consumer-screen-v1","production_qualified":False,
            "scope":"one explicit legacy packed ConvRot inverse-H256 FP16 consumer; no whole-request/INT8/performance qualification",
            "route":a.route,"controls":controls,"library_sha256":digest(a.library),
            "loaded_libraries":loaded_runtime_libraries(),"source_sha256":digest(a.checkpoint),
            "manifest_sha256":digest(a.manifest) if a.manifest else None,"runs":[],"status":"running"}
        engine,error=C.c_void_p(),C.c_void_p()
        status=lib.tc_engine_create_model(b"z-image-turbo",str(a.model.resolve()).encode(),C.byref(engine),C.byref(error))
        if status:raise RuntimeError(take(error))
        try:
            for index in range(a.runs):
                image=a.output/f"run-{index}.png"
                request={"schema_version":2,"model":"z-image-turbo","operation":"image.generate",
                    "inputs":[{"kind":"text","role":"prompt","text":a.prompt}],
                    "outputs":[{"kind":"image","path":str(image.resolve()),"width":a.size,"height":a.size,"audio":False}],
                    "sampling":{"seed":a.seed,"steps":a.steps},"execution":{"policy":"gpu","warmup_iterations":0},
                    "parameters":{"dynamic_text":True}}
                if a.route=="runtime":request["execution"].update(policy="gpu_ane",ane_manifest=str(a.manifest.resolve()),
                    hybrid_mlp_mode="runtime",allow_approximation=True)
                if a.dump:request["dump_tensors"]=str((a.output/f"run-{index}-tensors").resolve())
                samples=[];errors=[];stopped=threading.Event()
                def sample():
                    while not stopped.is_set():
                        try:samples.append({"time":time.monotonic(),**process_memory()})
                        except Exception as error:errors.append(str(error))
                        stopped.wait(.01)
                sampler=threading.Thread(target=sample,daemon=True)
                before=vm_counters();sampler.start()
                with (a.output/f"run-{index}-events.jsonl").open("x") as events:
                    @C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
                    def event(raw,_):
                        progress=json.loads(raw);events.write(json.dumps({**progress,"time_monotonic":time.monotonic()})+"\n");events.flush()
                    value,error=C.c_void_p(),C.c_void_p();start=time.perf_counter();print("START",a.route,index,flush=True)
                    try:status=lib.tc_engine_generate(engine,json.dumps(request).encode(),event,None,C.byref(value),C.byref(error))
                    finally:stopped.set();sampler.join()
                elapsed=time.perf_counter()-start;result,message=take(value),take(error);after=vm_counters()
                report["runs"].append({"request":request,"status":status,"error":message,"wall_seconds":elapsed,
                    "metrics":json.loads(result) if result else None,"memory_samples":samples,"sampling_errors":errors,
                    "vm_deltas":{key:after[key]-before[key] for key in before},
                    "png_sha256":digest(image) if image.exists() else None})
                (a.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
                print("DONE",a.route,index,status,round(elapsed,3),message,flush=True)
                if status:raise RuntimeError(message)
            report["status"]="completed";(a.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        finally:lib.tc_engine_free(engine)


if __name__=="__main__":main()
