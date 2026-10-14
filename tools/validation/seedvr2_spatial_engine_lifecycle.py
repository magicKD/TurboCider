#!/usr/bin/env python3
"""Public-engine file-row decode cancellation/quota failure/operation recovery.

Generated sentinel and scratch are private fixtures. User source/media are never
modified. Run outside performance cohorts because all requests execute model work.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import time
from PIL import Image
from seedvr2_framework_screen import sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("library","model","video","image","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--encoder-spatial",choices=("full","file_rows"),default="full")
    parser.add_argument("--cancel-stage",choices=("decoder","encoder"),default="decoder")
    args=parser.parse_args()
    if args.output.exists() or (args.cancel_stage=="encoder" and args.encoder_spatial!="file_rows"):
        parser.error("fresh lifecycle directory and explicit encoder rows for encoder cancellation required")
    for name in ("library","model","video","image"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.absolute();args.output.mkdir(parents=True)
    scratch=args.output/"scratch";scratch.mkdir();os.environ["TMPDIR"]=str(scratch)
    report=dict(schema="seedvr2-spatial-engine-lifecycle-v1",status="running",runs=[],
        library_sha256=sha256(args.library),video_sha256=sha256(args.video),image_sha256=sha256(args.image),
        encoder_spatial=args.encoder_spatial,cancel_stage=args.cancel_stage,
        scope="actual public engine, selected encoder/decoder cancel, file quota failure/same-engine image recovery; no performance/low-RAM claim")
    library=C.CDLL(str(args.library));engine=C.c_void_p();callback_type=C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
    library.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    library.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,callback_type,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    library.tc_engine_cancel.argtypes=[C.c_void_p];library.tc_engine_free.argtypes=[C.c_void_p];library.tc_string_free.argtypes=[C.c_void_p]
    library.tc_engine_unload.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    def consume(pointer):
        if not pointer.value:return None
        value=C.string_at(pointer).decode();library.tc_string_free(pointer);return value
    error=C.c_void_p()
    try:
        if library.tc_engine_create_model(b"seedvr2-3b",str(args.model).encode(),C.byref(engine),C.byref(error)):raise ValueError(consume(error))
        sentinel=args.output/"preserve-existing.mp4";sentinel.write_bytes(b"owned sentinel must survive spatial failure")
        sentinel_sha=sha256(sentinel)
        cancel_name="cancel-encode" if args.cancel_stage=="encoder" else "cancel-decode"
        for name in (cancel_name,"quota-failure","image-recovery"):
            is_video=name!="image-recovery";events=[];errors=[];cancelled=False
            output=sentinel if is_video else args.output/"recovery.png"
            budget=64<<20 if name=="quota-failure" else 8<<30
            request=dict(schema_version=1,model="seedvr2-3b",operation="video.upscale" if is_video else "image.upscale",prompt="",
                inputs=[dict(kind="video" if is_video else "image",role="source",path=str(args.video if is_video else args.image))],
                output=str(output),width=1024,height=1024,frames=25 if is_video else 1,fps=24,steps=1,seed=42,audio=False,
                execution="gpu",residency="component_staged",seedvr2_vae=dict(spatial="file_rows",encoder_spatial=args.encoder_spatial,spatial_rows=64,
                    spatial_disk_budget_bytes=budget,history="frames",allocator_cache_bytes=1<<30,
                    encoder="streaming" if is_video else "buffered",decoder="streaming" if is_video else "buffered"))
            @callback_type
            def callback(data,_):
                nonlocal cancelled
                try:
                    event=json.loads(data);events.append(event)
                    cancel_phase="seedvr2_vae_encode" if args.cancel_stage=="encoder" else "seedvr2_vae_export_frame"
                    if name==cancel_name and event.get("phase")==cancel_phase and event.get("completed")==1:
                        cancelled=True;library.tc_engine_cancel(engine)
                except BaseException as failure:errors.append(str(failure))
            result,error=C.c_void_p(),C.c_void_p();start=time.perf_counter()
            code=library.tc_engine_generate(engine,json.dumps(request).encode(),callback,None,C.byref(result),C.byref(error))
            text,message=consume(result),consume(error);metrics=json.loads(text) if text else None
            row=dict(name=name,code=code,error=message,seconds=time.perf_counter()-start,cancel_called=cancelled,
                events=events,callback_errors=errors,metrics=metrics);report["runs"].append(row);write_json(args.output/"report.json",report)
            if errors:raise ValueError("lifecycle callback failed")
            if name==cancel_name:
                if code!=2 or not cancelled or metrics is not None:raise ValueError("spatial cancellation status was not typed")
                if args.cancel_stage=="decoder" and any(v.get("phase")=="seedvr2_vae_export_frame" and v.get("completed",0)>1 for v in events):raise ValueError("cancelled spatial decoder read ahead")
                if args.cancel_stage=="encoder" and any(v.get("phase") in ("seedvr2_dit_stage_load","seedvr2_dit","seedvr2_vae_export_frame") for v in events):
                    raise ValueError("cancelled encoder advanced into sampling/export")
            elif name=="quota-failure":
                if code!=1 or metrics is not None or "budget" not in (message or ""):raise ValueError("file quota did not fail explicitly")
            else:
                if (code or not metrics or metrics.get("vae_spatial_recipe")!="file_rows" or
                    metrics.get("vae_spatial_tile_convolutions",0)<=0 or metrics.get("vae_spatial_cleanup_failures")!=0):
                    raise ValueError("same-engine spatial operation recovery failed")
                if args.encoder_spatial=="file_rows" and (metrics.get("vae_encoder_spatial_recipe")!="file_rows" or
                    metrics.get("vae_encoder_spatial_tile_convolutions",0)<=0 or metrics.get("vae_encoder_spatial_cleanup_failures")!=0):
                    raise ValueError("same-engine encoder spatial recovery failed")
                with Image.open(output) as pixels:
                    if pixels.size!=(1024,1024):raise ValueError("recovery image canvas changed")
            if sha256(sentinel)!=sentinel_sha or sha256(args.video)!=report["video_sha256"] or sha256(args.image)!=report["image_sha256"]:
                raise ValueError("failed spatial request changed destination or sources")
            leftovers=[path.name for path in scratch.iterdir() if path.name.startswith("turbocider-vae-spatial-")]
            media_leftovers=[path.name for path in args.output.iterdir() if ".seedvr2-" in path.name or ".mp4." in path.name]
            if leftovers or media_leftovers:raise ValueError("spatial failure left private scratch/media staging")
            print(f"PASS {name} code={code}",flush=True)
        result,error=C.c_void_p(),C.c_void_p()
        if library.tc_engine_unload(engine,C.byref(result),C.byref(error)):raise ValueError(consume(error))
        report.update(unload=json.loads(consume(result)),status="complete")
    except BaseException as failure:report.update(status="failed",failure=str(failure));raise
    finally:
        if engine.value:library.tc_engine_free(engine)
        write_json(args.output/"report.json",report)


if __name__=="__main__":main()
