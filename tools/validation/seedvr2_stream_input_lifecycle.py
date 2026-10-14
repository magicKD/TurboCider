#!/usr/bin/env python3
"""Actual public-engine streaming-input cancellation, EOF guard and recovery.

Run separately from timing cohorts. Preserves an existing destination on errors,
checks that extra source frames fail before sampling, and runs a complete recovery.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import time

from seedvr2_framework_screen import sha256, write_json, verify_output
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("library","model","source","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh lifecycle directory required")
    for name in ("library","model","source"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output.mkdir(parents=True)
    report=dict(schema="seedvr2-stream-input-lifecycle-v1",status="running",runs=[],
        library_sha256=sha256(args.library),source_sha256=sha256(args.source),
        scope="real <=121-frame public engine; cancellation/EOF-before-sampling/recovery, no speed or low-RAM claim")
    library=C.CDLL(str(args.library));engine=C.c_void_p()
    callback_type=C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
    library.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    library.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,callback_type,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    library.tc_engine_cancel.argtypes=[C.c_void_p]
    library.tc_engine_unload.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    library.tc_engine_free.argtypes=[C.c_void_p];library.tc_string_free.argtypes=[C.c_void_p]
    def consume(pointer):
        if not pointer.value:return None
        value=C.string_at(pointer).decode();library.tc_string_free(pointer);return value
    error=C.c_void_p()
    try:
        if library.tc_engine_create_model(b"seedvr2-3b",str(args.model).encode(),C.byref(engine),C.byref(error)):
            raise ValueError(consume(error))
        sentinel=args.output/"preserve-existing.mp4"
        sentinel.write_bytes(b"existing destination must survive source failure")
        sentinel_sha=sha256(sentinel)
        def execute(name,frames,cancel_at):
            events=[];callback_errors=[];cancelled=False
            output=sentinel if name!="recovery" else args.output/"recovery.mp4"
            request=dict(schema_version=1,model="seedvr2-3b",operation="video.upscale",prompt="",steps=1,
                inputs=[dict(kind="video",role="source",path=str(args.source))],output=str(output),
                frames=frames,fps=24,width=512,height=512,seed=42,audio=False,execution="gpu",residency="component_staged",
                seedvr2_vae=dict(encoder="streaming",decoder="streaming",history="frames",allocator_cache_bytes=1<<30))
            @callback_type
            def callback(data,_):
                nonlocal cancelled
                try:
                    value=json.loads(data);events.append(value)
                    if (cancel_at is not None and not cancelled and value.get("phase")=="seedvr2_video_import" and
                        value.get("completed")==cancel_at):
                        cancelled=True;library.tc_engine_cancel(engine)
                except BaseException as failure:callback_errors.append(str(failure))
            result,error=C.c_void_p(),C.c_void_p();started=time.perf_counter()
            code=library.tc_engine_generate(engine,json.dumps(request).encode(),callback,None,C.byref(result),C.byref(error))
            text,message=consume(result),consume(error)
            row=dict(name=name,code=code,seconds=time.perf_counter()-started,error=message,cancel_called=cancelled,
                events=events,callback_errors=callback_errors,metrics=json.loads(text) if text else None)
            report["runs"].append(row);write_json(args.output/"report.json",report)
            if callback_errors:raise ValueError("lifecycle callback failed")
            if name=="cancel":
                if code!=2 or not cancelled or row["metrics"] is not None:raise ValueError("streaming input did not return typed cancellation")
            elif name=="extra-frame":
                if code!=1 or row["metrics"] is not None:raise ValueError("undeclared source tail was accepted")
                if any(event.get("phase") in ("seedvr2_dit_stage_load","seedvr2_dit","seedvr2_vae_decode") for event in events):
                    raise ValueError("source EOF failure reached sampling/decode")
            else:
                metrics=row["metrics"] or {}
                if (code!=0 or metrics.get("vae_encoder_recipe")!="streaming" or metrics.get("streamed_input_frames")!=25 or
                    metrics.get("streamed_output_frames")!=25):raise ValueError("same-engine complete recovery did not succeed")
                row["media"]=verify_output(output,SimpleNamespace(kind="video",width=512,height=512,frames=25,fps=24))
            if sha256(sentinel)!=sentinel_sha or sha256(args.source)!=report["source_sha256"]:
                raise ValueError("failed/cancelled request changed destination or source")
            leftovers=[path.name for path in args.output.iterdir() if ".seedvr2-" in path.name or ".mp4." in path.name]
            if leftovers:raise ValueError("streaming source failure leaked private staging: "+str(leftovers))
            print(f"PASS {name} code={code}",flush=True)
        execute("cancel",25,1)
        execute("extra-frame",24,None)
        execute("recovery",25,None)
        result,error=C.c_void_p(),C.c_void_p()
        if library.tc_engine_unload(engine,C.byref(result),C.byref(error)):raise ValueError(consume(error))
        report["unload"]=json.loads(consume(result));report["status"]="complete"
    except BaseException as failure:report.update(status="failed",failure=str(failure));raise
    finally:
        if engine.value:library.tc_engine_free(engine)
        write_json(args.output/"report.json",report)


if __name__=="__main__":main()
