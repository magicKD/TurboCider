#!/usr/bin/env python3
"""Real Z GGUF bounded-execution probe. Results are experimental, not certification."""
import argparse
import ctypes as C
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

from benchmark_z_image_streaming import process_memory, vm_counters
from benchmark_z_image_metal import loaded_runtime_libraries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prefetch",type=int,nargs="+",default=[0,1])
    parser.add_argument("--runs",type=int,default=1)
    parser.add_argument("--size",type=int,default=512)
    parser.add_argument("--steps",type=int,default=4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--precision",choices=["z-source-mixed-v1","z-source-mixed-f16-v1","z-source-exact-f32-v1","z-source-native-affine-v1","z-mlx-compat-affine-v1","z-mlx-compat-f16-v1","z-mlx-compat-f32-v1"],default="z-source-mixed-v1")
    parser.add_argument("--prompt",default="A studio photograph of an adult ceramic artist, both hands visible while holding a small blue cup, neutral background, natural skin texture.")
    parser.add_argument("--dump",action="store_true",help="save diagnostic latents/pixels; not a performance run")
    parser.add_argument("--cancel-once-at-block",type=int,help="cancel the first request at a main block, then test retry")
    parser.add_argument("--cancel-once-at-encoder-layer",type=int,help="cancel first uncached Qwen3 encode, then test retry")
    args=parser.parse_args()
    if any(p not in (-1,0,1) for p in args.prefetch) or not 1<=args.runs<=24: parser.error("prefetch -1=native packed, 0/1=bounded; runs 1..24")
    if args.output.exists() or args.output.is_symlink(): parser.error("output already exists")
    if args.cancel_once_at_block is not None and (not 0<=args.cancel_once_at_block<30 or args.prefetch[0]<0):
        parser.error("cancellation requires a bounded first request and block 0..29")
    if args.cancel_once_at_encoder_layer is not None and (not 0<=args.cancel_once_at_encoder_layer<35 or args.cancel_once_at_block is not None):
        parser.error("encoder cancellation requires layer 0..34 and no block cancellation")
    args.output.mkdir(parents=True)
    with open("/tmp/turbocider-z-image-gpu-benchmark.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        library=C.CDLL(str(args.library.resolve()))
        library.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        library.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        library.tc_engine_free.argtypes=[C.c_void_p]
        library.tc_engine_cancel.argtypes=[C.c_void_p]
        library.tc_string_free.argtypes=[C.c_void_p]
        def take(pointer):
            if not pointer.value: return None
            value=C.string_at(pointer).decode(); library.tc_string_free(pointer); return value
        engine,error=C.c_void_p(),C.c_void_p()
        status=library.tc_engine_create_model(b"z-image-turbo-gguf",str(args.model.resolve()).encode(),C.byref(engine),C.byref(error))
        message=take(error)
        if status: raise RuntimeError(message)
        report={"schema_version":1,"scope":"real private GGUF execution; not whole-request memory/performance qualification",
            "library_sha256":hashlib.sha256(args.library.read_bytes()).hexdigest(),"loaded_libraries":loaded_runtime_libraries(),
            "runs":[],"status":"running"}
        indexes={}; cancel_used=False
        try:
            for p in args.prefetch:
                for run in range(args.runs):
                    index=indexes.get(p,0);indexes[p]=index+1
                    name=f"p{p}-{index}" if p>=0 else f"packed-{index}"; image=args.output/(name+".png")
                    request={"schema_version":2,"model":"z-image-turbo-gguf","operation":"image.generate",
                        "inputs":[{"kind":"text","role":"prompt","text":args.prompt}],
                        "outputs":[{"kind":"image","path":str(image.resolve()),"width":args.size,"height":args.size,"audio":False}],
                        "sampling":{"seed":args.seed,"steps":args.steps},
                        "execution":{"policy":"gpu","quantized_execution":{"schema_version":1,"enabled":True,"prefetch_layers":p,"precision_profile":args.precision}},
                        "parameters":{"dynamic_text":True}}
                    if args.dump: request["dump_tensors"]=str((args.output/(name+"-tensors")).resolve())
                    if p<0: request["execution"].pop("quantized_execution")
                    samples=[]; errors=[]; stopped=threading.Event()
                    def sample():
                        while not stopped.is_set():
                            try: samples.append({"time":time.monotonic(),**process_memory()})
                            except Exception as exception: errors.append(str(exception))
                            stopped.wait(.02)
                    sampler=threading.Thread(target=sample,daemon=True)
                    before=vm_counters(); sampler.start()
                    with (args.output/(name+"-events.jsonl")).open("x") as events:
                        should_cancel=(args.cancel_once_at_block is not None or args.cancel_once_at_encoder_layer is not None) and not cancel_used
                        triggered=[False]
                        @C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
                        def event(raw,_):
                            events.write(raw.decode()+"\n"); events.flush()
                            progress=json.loads(raw)
                            cancel_phase = "z_image_text_encode" if args.cancel_once_at_encoder_layer is not None else "z_image_denoise_block"
                            cancel_at = args.cancel_once_at_encoder_layer if args.cancel_once_at_encoder_layer is not None else args.cancel_once_at_block
                            if should_cancel and not triggered[0] and progress.get("phase")==cancel_phase and progress.get("completed")==cancel_at:
                                library.tc_engine_cancel(engine);triggered[0]=True
                        value,error=C.c_void_p(),C.c_void_p(); start=time.perf_counter()
                        try:
                            print("START",name,flush=True)
                            status=library.tc_engine_generate(engine,json.dumps(request).encode(),event,None,C.byref(value),C.byref(error))
                        finally: stopped.set(); sampler.join()
                    elapsed=time.perf_counter()-start; raw=take(value); message=take(error); after=vm_counters()
                    row={"prefetch":p,"index":index,"request":request,"status":status,"error":message,
                        "wall_seconds":elapsed,"metrics":json.loads(raw) if raw else None,"memory_samples":samples,
                        "sampling_errors":errors,"vm_deltas":{key:after[key]-before[key] for key in before},
                        "png_sha256":hashlib.sha256(image.read_bytes()).hexdigest() if image.exists() else None}
                    row["cancellation_triggered"]=triggered[0]
                    report["runs"].append(row)
                    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
                    print("DONE",name,status,round(elapsed,3),message,flush=True)
                    if triggered[0]:
                        cancel_used=True
                        if not status or not message or "cancel" not in message.lower() or image.exists():
                            raise RuntimeError("cancellation did not fail cleanly")
                    elif status: raise RuntimeError(message)
            report["status"]="completed"
            successful=[row for row in report["runs"] if row["status"]==0]
            report["png_bytes_exact"]=len({row["png_sha256"] for row in successful})==1 if len(successful)>1 else None
            bounded=[row for row in successful if row["prefetch"]>=0]
            report["bounded_png_bytes_exact"]=len({row["png_sha256"] for row in bounded})==1 if len(bounded)>1 else None
            (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
            print("PNG exact:",report["png_bytes_exact"],flush=True)
        finally: library.tc_engine_free(engine)

if __name__=="__main__": main()
