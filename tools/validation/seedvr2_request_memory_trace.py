#!/usr/bin/env python3
"""Read public MLX allocator counters at existing full-request callbacks.

No extra GPU eval/synchronization, tensors, peak resets or cache controls. A
cumulative peak first observed at an event locates an interval, not an exclusive
operator allocation or independent per-phase high water. Not a speed benchmark.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from seedvr2_framework_screen import sha256,write_json,verify_output,validate_native_build_binding,check_native_execution


def summarize(rows,peak):
    if not rows or type(peak) is not int or peak<=0:raise ValueError("positive observed request peak required")
    previous=0;first=None
    for index,row in enumerate(rows):
        active,high,cache=(row.get(name) for name in ("active_bytes","peak_bytes","cache_bytes"))
        if any(type(value) is not int or value<0 for value in (active,high,cache)) or high<active or high<previous:
            raise ValueError("invalid or nonmonotonic cumulative allocator trace")
        if high>peak:raise ValueError("observer and native peak counters disagree")
        if high==peak and first is None:first=index
        previous=high
    if first is None:raise ValueError("native request maximum was never observed")
    return dict(first_observed_peak_event=rows[first],preceding_event=rows[first-1] if first else None,
        scope="peak was reached no later than first observed event, after the preceding observation; no exclusive operator or independently reset phase maxima")


def frame_hashes(path):
    text=subprocess.check_output(["ffmpeg","-v","error","-i",str(path),"-map","0:v:0","-fps_mode","passthrough",
        "-pix_fmt","rgb24","-f","framemd5","pipe:1"],text=True)
    return [line.strip() for line in text.splitlines() if line and not line.startswith("#")]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("library","model","source","reference-export","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--mode",choices=("gpu","dual"),required=True)
    parser.add_argument("--hot-manifest",type=Path)
    parser.add_argument("--upscale-manifest",type=Path)
    parser.add_argument("--width",type=int,default=1024);parser.add_argument("--height",type=int,default=1024)
    parser.add_argument("--frames",type=int,default=25);parser.add_argument("--fps",type=int,default=24)
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh memory trace directory required")
    for name in ("library","model","source","reference_export"):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    for name,environment in (("hot_manifest","TURBOCIDER_SEEDVR2_VAE_CAUSAL"),("upscale_manifest","TURBOCIDER_SEEDVR2_VAE_UPSCALE_CAUSAL")):
        manifest=getattr(args,name)
        if args.mode=="dual":
            if manifest is None:parser.error("dual trace requires both explicit original banks")
            manifest=manifest.resolve(strict=True);setattr(args,name,manifest)
            os.environ[environment+"_MANIFEST"]=str(manifest);os.environ[environment+"_COMPUTE"]="ne"
        elif manifest is not None or os.environ.get(environment+"_MANIFEST"):
            parser.error("GPU trace must not inherit or select a causal bank")
    if os.environ.get("TURBOCIDER_SEEDVR2_VIDEO_RAW_CAPTURE"):parser.error("raw captures must run separately")
    # Import the managed Python binding of the same libmlx used by the native
    # engine. Positive matching native/Python peaks below verify shared counters.
    import mlx.core as mx
    args.output=args.output.absolute();args.output.mkdir(parents=True)
    manifest=args.library.parent/"runtime-build/runtime-build-manifest.json"
    runtime=validate_native_build_binding(args.library,json.loads(manifest.read_text()))
    options=dict(encoder="streaming",decoder="streaming",history="frames",share_upscale_history=True,allocator_cache_bytes=1<<30)
    request=dict(schema_version=1,model="seedvr2-3b",operation="video.upscale",prompt="",steps=1,seed=42,
        inputs=[dict(kind="video",role="source",path=str(args.source))],output=str(args.output/"output.mp4"),
        width=args.width,height=args.height,frames=args.frames,fps=args.fps,audio=False,execution="gpu",residency="component_staged",seedvr2_vae=options)
    write_json(args.output/"request.json",request)
    report=dict(schema="seedvr2-request-cumulative-memory-trace-v1",status="running",mode=args.mode,events=[],
        library_sha256=sha256(args.library),runtime_build_id=runtime,source_sha256=sha256(args.source),
        reference_export_sha256=sha256(args.reference_export),mlx_python_extension=str(mx.__file__),
        startup_active_bytes=mx.get_active_memory(),scope=__doc__)
    library=C.CDLL(str(args.library));engine=C.c_void_p();callback_type=C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
    library.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    library.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,callback_type,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    library.tc_engine_unload.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    library.tc_engine_cancel.argtypes=[C.c_void_p];library.tc_engine_free.argtypes=[C.c_void_p];library.tc_string_free.argtypes=[C.c_void_p]
    def consume(pointer):
        if not pointer.value:return None
        value=C.string_at(pointer).decode();library.tc_string_free(pointer);return value
    errors=[];started=time.perf_counter()
    @callback_type
    def callback(data,_):
        try:
            event=json.loads(data);active=mx.get_active_memory();peak=mx.get_peak_memory();cache=mx.get_cache_memory()
            report["events"].append(dict(event,observer_index=len(report["events"]),observer_seconds=time.perf_counter()-started,
                active_bytes=active,peak_bytes=peak,cache_bytes=cache))
        except BaseException as error:
            errors.append(str(error));library.tc_engine_cancel(engine)
    error=C.c_void_p()
    try:
        if library.tc_engine_create_model(b"seedvr2-3b",str(args.model).encode(),C.byref(engine),C.byref(error)):raise ValueError(consume(error))
        result,error=C.c_void_p(),C.c_void_p();started=time.perf_counter()
        code=library.tc_engine_generate(engine,json.dumps(request).encode(),callback,None,C.byref(result),C.byref(error))
        text,message=consume(result),consume(error)
        report.update(code=code,error=message,callback_errors=errors,metrics=json.loads(text) if text else None)
        if code or errors or not text:raise ValueError("memory-traced request failed")
        metrics=report["metrics"]
        check_native_execution(SimpleNamespace(route="native-gpu",kind="video",frames=args.frames,width=args.width,height=args.height,
            residency="component_staged",vae_options=options,experimental_causal_manifest=args.hot_manifest,
            experimental_upscale_causal_manifest=args.upscale_manifest),metrics)
        if mx.get_peak_memory()!=metrics["mlx_peak_bytes"] or max(row["active_bytes"] for row in report["events"])<=1<<30:
            raise ValueError("Python/native MLX allocator counters are not shared")
        report["summary"]=summarize(report["events"],metrics["mlx_peak_bytes"])
        output=Path(request["output"]);report["media"]=verify_output(output,SimpleNamespace(kind="video",width=args.width,height=args.height,frames=args.frames,fps=args.fps))
        original,observed=frame_hashes(args.reference_export),frame_hashes(output)
        if len(original)!=args.frames or original!=observed:raise ValueError("read-only observer changed exported video pixels")
        report["all_exported_frames_bitwise_equal"]=True;report["output_sha256"]=sha256(output)
        if sha256(args.source)!=report["source_sha256"] or sha256(args.reference_export)!=report["reference_export_sha256"]:
            raise ValueError("memory trace source/reference changed")
        result,error=C.c_void_p(),C.c_void_p()
        if library.tc_engine_unload(engine,C.byref(result),C.byref(error)):raise ValueError(consume(error))
        report["unload"]=json.loads(consume(result));report["python_active_after_unload"]=mx.get_active_memory()
        if report["unload"].get("mlx_active_bytes")!=0 or report["python_active_after_unload"]!=0:raise ValueError("memory trace retained tensor allocations")
        report["status"]="complete";print(json.dumps(dict(status="complete",mode=args.mode,summary=report["summary"])),flush=True)
    except BaseException as error:report.update(status="failed",failure=str(error));raise
    finally:
        if engine.value:library.tc_engine_free(engine)
        write_json(args.output/"report.json",report)


if __name__=="__main__":main()
