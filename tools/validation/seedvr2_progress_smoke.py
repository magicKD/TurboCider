#!/usr/bin/env python3
"""Functional complete-video progress/cancel/recovery screen, not a benchmark.

Uses a small output canvas to exercise every source frame affordably. Native
math is unchanged; compare every decoded RGB pixel to an explicit prior build.
"""
import argparse
import ctypes as C
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from image_framework_native import generate_requests
from seedvr2_framework_screen import sha256, write_json, validate_native_build_binding, verify_output
from seedvr2_app_media_screen import PixelStats, read_frame, video_frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--before-library",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--frames",type=int,default=97)
    p.add_argument("--fps",type=int,default=24)
    p.add_argument("--size",type=int,default=256)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--before-worker",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    if not args.before_worker:
        if args.output.exists(): p.error("fresh output required")
        args.output.mkdir(parents=True)
    source_sha=sha256(args.source)
    report=dict(schema="seedvr2-progress-smoke-v1",status="running",source_sha256=source_sha,
                frames=args.frames,fps=args.fps,size=args.size,runs=[],scope="functional smoke, not performance qualification")
    def request(name,stream=False):
        return dict(schema_version=2,model="seedvr2-3b",operation="video.upscale",
            inputs=[dict(kind="video",role="source",path=str(args.source))],
            outputs=[dict(kind="video",path=str(args.output/(name+".mp4")),width=args.size,height=args.size,
                          frames=args.frames,fps=args.fps,audio=False)],sampling=dict(seed=42,steps=1),
            execution=dict(policy="gpu",residency="resident",seedvr2_vae=dict(
                encoder="streaming" if stream else "buffered",decoder="streaming" if stream else "buffered")))
    if args.before_worker:
        manifest=json.loads((args.before_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
        identity=validate_native_build_binding(args.before_library,manifest)
        old=generate_requests(args.before_library,"seedvr2-3b",args.model,
                              [request("before"),request("before-streaming",True)],args.output,"before")
        write_json(args.output/"before.json",dict(runtime_build_id=identity,**old))
        return
    # Objective-C class names are process-global; two runtime revisions must
    # never be dlopened together, even with RTLD_LOCAL.
    subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--before-worker"],check=True)
    report["before_runtime_build_id"]=json.loads((args.output/"before.json").read_text())["runtime_build_id"]
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    report["after_runtime_build_id"]=validate_native_build_binding(args.library,manifest)
    lib=C.CDLL(str(args.library))
    callback_type=C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
    lib.tc_string_free.argtypes=[C.c_void_p]
    lib.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    lib.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,callback_type,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    lib.tc_engine_cancel.argtypes=[C.c_void_p]
    lib.tc_engine_free.argtypes=[C.c_void_p]
    def consume(pointer):
        if not pointer.value:return ""
        try:return C.string_at(pointer).decode()
        finally:lib.tc_string_free(pointer)
    engine,error=C.c_void_p(),C.c_void_p()
    if lib.tc_engine_create_model(b"seedvr2-3b",str(args.model).encode(),C.byref(engine),C.byref(error)):
        raise RuntimeError(consume(error))
    try:
        for name,stream,cancel in (("buffered",False,False),("cancelled",True,True),("streaming-recovery",True,False)):
            events=[]
            @callback_type
            def callback(payload,_):
                event=json.loads(payload.decode());events.append(event)
                if cancel and event["phase"]=="seedvr2_decode_progress" and event["completed"]>=3:
                    lib.tc_engine_cancel(engine)
            r=request(name,stream)
            result,error=C.c_void_p(),C.c_void_p()
            status=lib.tc_engine_generate(engine,json.dumps(r).encode(),callback,None,C.byref(result),C.byref(error))
            text,failure=consume(result),consume(error)
            write_json(args.output/(name+"-events.json"),events)
            output=Path(r["outputs"][0]["path"])
            if cancel:
                if status!=2 or output.exists() or list(args.output.glob(output.stem+".seedvr2-*")):
                    raise ValueError("typed cancellation/staging cleanup failed")
                report["runs"].append(dict(name=name,status=status,error=failure,output_absent=True))
            else:
                if status:raise ValueError(failure)
                metrics=json.loads(text)
                verify_output(output,SimpleNamespace(kind="video",width=args.size,height=args.size,frames=args.frames,fps=args.fps))
                for phase,total in (("seedvr2_encode_progress",args.frames),("seedvr2_dit",32),("seedvr2_decode_progress",args.frames)):
                    rows=[e for e in events if e["phase"]==phase]
                    counts=[e["completed"] for e in rows]
                    if not rows or counts[0]!=0 or counts[-1]!=total or counts!=sorted(counts) or any(e["total"]!=total for e in rows):
                        raise ValueError("incomplete/resetting whole-video progress: "+phase)
                if metrics["streamed_input_frames"]!=(args.frames if stream else 0) or metrics["streamed_output_frames"]!=(args.frames if stream else 0):
                    raise ValueError("streaming selection did not execute")
                stats=PixelStats()
                reference=args.output/("before-streaming.mp4" if stream else "before.mp4")
                with video_frames(reference,args.size) as left,video_frames(output,args.size) as right:
                    for _ in range(args.frames):
                        a,b=read_frame(left,args.size*args.size*3),read_frame(right,args.size*args.size*3)
                        if a is None or b is None:raise ValueError("short decoded output")
                        stats.add(a,b)
                    if read_frame(left,args.size*args.size*3) is not None or read_frame(right,args.size*args.size*3) is not None:
                        raise ValueError("extra decoded output")
                comparison=stats.result()
                if not comparison["decoded_pixels_bitwise_equal"]:raise ValueError("progress-only change altered exported pixels")
                report["runs"].append(dict(name=name,status=status,metrics=metrics,comparison=comparison,progress_monotonic=True))
            write_json(args.output/"report.json",report)
            print("PASS",name,flush=True)
        if sha256(args.source)!=source_sha:raise ValueError("source changed")
        report["status"]="complete"
    except BaseException as error:
        report.update(status="failed",error=str(error));raise
    finally:
        lib.tc_engine_free(engine)
        write_json(args.output/"report.json",report)


if __name__=="__main__":main()
