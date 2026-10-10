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
from z_image_probe_identity import bind_components,revalidate_components


def digest(path):
    with path.open("rb") as source: return hashlib.file_digest(source,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--route",choices=["gpu","runtime"],required=True)
    p.add_argument("--gpu-recipe",choices=["legacy","compiled_dense","compiled_butterfly"],default="legacy")
    p.add_argument("--validate-source-blocks",action="store_true")
    p.add_argument("--measurement",choices=["diagnostic","timing","memory"],default="diagnostic")
    p.add_argument("--warmup",type=int,default=0)
    p.add_argument("--bind-components",action="store_true")
    p.add_argument("--gpu-cache-bytes",type=int,help="compiled GPU allocator hint 0..4GiB; not RAM cap")
    p.add_argument("--retain-gpu-cache",action="store_true",help="explicitly retain bounded global allocator bins; caller hint restored, not a RAM cap")
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
    if not 0<=a.warmup<=8:p.error("warmup 0..8")
    if a.gpu_recipe!="legacy" and a.route!="gpu":p.error("compiled ConvRot recipe requires GPU route")
    if a.measurement!="diagnostic" and (a.dump or a.validate_source_blocks):p.error("dump/source validation requires diagnostic")
    if a.validate_source_blocks and a.gpu_recipe=="legacy":p.error("source validation requires compiled ConvRot")
    if a.gpu_cache_bytes is not None and (a.gpu_recipe=="legacy" or a.route!="gpu" or not 0<=a.gpu_cache_bytes<=4<<30):p.error("cache hint requires compiled GPU recipe and 0..4GiB")
    if a.retain_gpu_cache and a.gpu_cache_bytes is None:p.error("retained GPU cache requires explicit cache hint")
    if a.measurement!="diagnostic" and os.environ.get("TURBOCIDER_Z_CONVROT_VALIDATE_BLOCKS","0")!="0":p.error("source validation cannot contaminate timing/memory")
    controls={"TURBOCIDER_Z_IMAGE_TRANSFORMER":str(a.checkpoint.resolve())}
    if a.route=="runtime":
        controls.update(TURBOCIDER_Z_RUNTIME_CONVROT="1",TURBOCIDER_RUNTIME_ANE_CHUNKS=str(a.chunks))
    elif os.environ.get("TURBOCIDER_Z_RUNTIME_CONVROT","0")!="0": p.error("GPU baseline must not enable runtime ConvRot")
    if a.route=="gpu":controls["TURBOCIDER_Z_CONVROT_GPU_RECIPE"]=a.gpu_recipe
    if a.validate_source_blocks:controls["TURBOCIDER_Z_CONVROT_VALIDATE_BLOCKS"]="1"
    if a.gpu_cache_bytes is not None:controls["TURBOCIDER_Z_CONVROT_CACHE_BYTES"]=str(a.gpu_cache_bytes)
    if a.retain_gpu_cache:controls["TURBOCIDER_Z_CONVROT_CACHE_RETAIN"]="1"
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
        lib.tc_system_json.argtypes=[];lib.tc_system_json.restype=C.c_void_p
        def take(pointer):
            if not pointer.value:return None
            result=C.string_at(pointer).decode();lib.tc_string_free(pointer);return result
        components,component_states,binding_seconds=bind_components(a.model,True,{}) if a.bind_components else (None,{},0)
        report={"schema":"tc-convrot-runtime-consumer-screen-v2","production_qualified":False,
            "scope":"explicit ConvRot GPU/compiled/runtime consumer; no whole-request/INT8/performance qualification",
            "route":a.route,"controls":controls,"library_sha256":digest(a.library),
            "hardware":json.loads(take(C.c_void_p(lib.tc_system_json()))),"baseline_bf16":False,
            "prompt_cache_policy":"hit","dit_source_sha256":digest(a.checkpoint),
            "measurement":a.measurement,"gpu_recipe":a.gpu_recipe,"component_binding":components,
            "component_binding_seconds":binding_seconds,"runtime_environment":{k:v for k,v in os.environ.items() if k.startswith("TURBOCIDER_")},
            "loaded_libraries":loaded_runtime_libraries(),"source_sha256":digest(a.checkpoint),
            "manifest_sha256":digest(a.manifest) if a.manifest else None,"runs":[],"status":"running"}
        engine,error=C.c_void_p(),C.c_void_p()
        status=lib.tc_engine_create_model(b"z-image-turbo",str(a.model.resolve()).encode(),C.byref(engine),C.byref(error))
        if status:raise RuntimeError(take(error))
        try:
            for index in range(a.runs+a.warmup):
                image=a.output/f"run-{index}.png"
                request={"schema_version":2,"model":"z-image-turbo","operation":"image.generate",
                    "inputs":[{"kind":"text","role":"prompt","text":a.prompt}],
                    "outputs":[{"kind":"image","path":str(image.resolve()),"width":a.size,"height":a.size,"audio":False}],
                    "sampling":{"seed":a.seed,"steps":a.steps},"execution":{"policy":"gpu","warmup_iterations":0},
                    "parameters":{"dynamic_text":True}}
                if a.route=="runtime":request["execution"].update(policy="gpu_ane",ane_manifest=str(a.manifest.resolve()),
                    hybrid_mlp_mode="runtime",allow_approximation=True)
                if a.gpu_recipe!="legacy":request["execution"]["allow_approximation"]=True;request["parameters"]["compile_gpu"]=True
                if a.dump:request["dump_tensors"]=str((a.output/f"run-{index}-tensors").resolve())
                samples=[];errors=[];stopped=threading.Event()
                def sample():
                    while not stopped.is_set():
                        try:samples.append({"time":time.monotonic(),**process_memory()})
                        except Exception as error:errors.append(str(error))
                        stopped.wait(.01)
                sampler=threading.Thread(target=sample,daemon=True)
                before=vm_counters()
                if a.measurement!="timing":sampler.start()
                with (a.output/f"run-{index}-events.jsonl").open("x") as events:
                    @C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
                    def event(raw,_):
                        progress=json.loads(raw);events.write(json.dumps({**progress,"time_monotonic":time.monotonic()})+"\n");events.flush()
                    value,error=C.c_void_p(),C.c_void_p();start=time.perf_counter();print("START",a.route,index,flush=True)
                    try:status=lib.tc_engine_generate(engine,json.dumps(request).encode(),None if a.measurement=="timing" else event,None,C.byref(value),C.byref(error))
                    finally:
                        stopped.set()
                        if a.measurement!="timing":sampler.join()
                elapsed=time.perf_counter()-start;result,message=take(value),take(error);after=vm_counters()
                revalidate_components(component_states)
                report["runs"].append({"request":request,"status":status,"error":message,"wall_seconds":elapsed,
                    "metrics":json.loads(result) if result else None,"memory_samples":samples,"sampling_errors":errors,
                    "measurement":a.measurement,"warmup":index<a.warmup,
                    "memory_sample_max_gap_seconds":max((b["time"]-c["time"] for c,b in zip(samples,samples[1:])),default=None),
                    "vm_deltas":{key:after[key]-before[key] for key in before},
                    "png_sha256":digest(image) if image.exists() else None})
                (a.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
                print("DONE",a.route,index,status,round(elapsed,3),message,flush=True)
                if status:raise RuntimeError(message)
            report["status"]="completed";(a.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        finally:lib.tc_engine_free(engine)


if __name__=="__main__":main()
