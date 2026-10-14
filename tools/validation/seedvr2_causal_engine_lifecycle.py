#!/usr/bin/env python3
"""Actual experimental-engine cancellation, bank rejection and route recovery.

Cancel after frame two, hence after genuine subsequent-frame predictions. This
is not a cancellation-inside-prediction test (covered by the component probe).
Never modify the original bank, source or an existing user destination.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from seedvr2_framework_screen import sha256,write_json,verify_output,validate_native_build_binding


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("library","model","source","manifest","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--ordinary-prepare-only",action="store_true",help="reject both experiment selectors in an ordinary library, then recover; no inference")
    parser.add_argument("--upscale-manifest",type=Path,help="also test the independent upscale causal layer and all on/off combinations")
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh lifecycle output required")
    if os.environ.get("TURBOCIDER_SEEDVR2_VIDEO_RAW_CAPTURE"):parser.error("raw captures must be run separately")
    if os.environ.get("TURBOCIDER_SEEDVR2_VAE_UPSCALE_CAUSAL_MANIFEST") and not args.upscale_manifest:
        parser.error("inherited upscale selector must be explicitly recorded")
    for name in ("library","model","source","manifest"):setattr(args,name,getattr(args,name).resolve(strict=True))
    if args.upscale_manifest:args.upscale_manifest=args.upscale_manifest.resolve(strict=True)
    args.output=args.output.absolute();args.output.mkdir(parents=True)
    identity=args.library.parent/"runtime-build/runtime-build-manifest.json"
    runtime=validate_native_build_binding(args.library,json.loads(identity.read_text()))
    report=dict(schema="seedvr2-causal-engine-lifecycle-v1",status="running",runs=[],runtime_build_id=runtime,
        library_sha256=sha256(args.library),source_sha256=sha256(args.source),manifest_sha256=sha256(args.manifest),
        upscale_manifest_sha256=sha256(args.upscale_manifest) if args.upscale_manifest else None,
        scope="actual full-source requests; cancellation after real subsequent-frame work, tampered owned bank, same-engine recovery/on-off/residency; not performance/public qualification")
    if args.ordinary_prepare_only:
        report["scope"]="ordinary library: reject experimental bank/raw-capture selectors, then prepare/unload on same engine; no model inference"
    library=C.CDLL(str(args.library));engine=C.c_void_p();callback_type=C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
    library.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    library.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,callback_type,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    library.tc_engine_prepare.argtypes=[C.c_void_p,C.c_char_p,C.c_int,callback_type,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    library.tc_engine_cancel.argtypes=[C.c_void_p];library.tc_engine_free.argtypes=[C.c_void_p];library.tc_string_free.argtypes=[C.c_void_p]
    library.tc_engine_unload.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    def consume(pointer):
        if not pointer.value:return None
        value=C.string_at(pointer).decode();library.tc_string_free(pointer);return value
    error=C.c_void_p()
    try:
        if library.tc_engine_create_model(b"seedvr2-3b",str(args.model).encode(),C.byref(engine),C.byref(error)):raise ValueError(consume(error))
        sentinel=args.output/"preserve-existing.mp4";sentinel.write_bytes(b"owned sentinel must survive experimental request failure")
        sentinel_sha=sha256(sentinel)
        if args.ordinary_prepare_only:
            cases=[("reject-selector",args.manifest,None,"component_staged",1),("reject-capture",None,None,"component_staged",1)]
            if args.upscale_manifest:cases.append(("reject-upscale-selector",None,args.upscale_manifest,"component_staged",1))
            cases.append(("ordinary-prepare-recovery",None,None,"component_staged",0))
        else:
            target=args.upscale_manifest or args.manifest
            owned=args.output/"tampered-bank";shutil.copytree(target.parent,owned)
            spec=json.loads(target.read_text());file=owned/"conv.mlmodelc"/spec["compiled"]["identity"]["files"][0]["path"]
            with file.open("ab") as stream:stream.write(b"owned tamper fixture")
            cases=[("tampered-upscale-bank",args.manifest,owned/target.name,"component_staged",1) if args.upscale_manifest else
                ("tampered-bank",owned/target.name,None,"component_staged",1),
                ("cancel-after-predictions",args.manifest,args.upscale_manifest,"component_staged",2),
                ("experimental-recovery",args.manifest,args.upscale_manifest,"component_staged",0),
                ("gpu-off-resident",None,None,"resident",0)]
            if args.upscale_manifest:
                cases.extend([("upscale-only-resident",None,args.upscale_manifest,"resident",0),("hot-only-staged",args.manifest,None,"component_staged",0)])
            cases.append(("experimental-on-staged",args.manifest,args.upscale_manifest,"component_staged",0))
        os.environ["TURBOCIDER_SEEDVR2_VAE_CAUSAL_COMPUTE"]="ne"
        os.environ["TURBOCIDER_SEEDVR2_VAE_UPSCALE_CAUSAL_COMPUTE"]="ne"
        for name,manifest,upscale_manifest,residency,expected_code in cases:
            if manifest is None:os.environ.pop("TURBOCIDER_SEEDVR2_VAE_CAUSAL_MANIFEST",None)
            else:os.environ["TURBOCIDER_SEEDVR2_VAE_CAUSAL_MANIFEST"]=str(manifest)
            if upscale_manifest is None:os.environ.pop("TURBOCIDER_SEEDVR2_VAE_UPSCALE_CAUSAL_MANIFEST",None)
            else:os.environ["TURBOCIDER_SEEDVR2_VAE_UPSCALE_CAUSAL_MANIFEST"]=str(upscale_manifest)
            capture=args.output/"capture-must-not-be-created"
            if name=="reject-capture":os.environ["TURBOCIDER_SEEDVR2_VIDEO_RAW_CAPTURE"]=str(capture)
            else:os.environ.pop("TURBOCIDER_SEEDVR2_VIDEO_RAW_CAPTURE",None)
            destination=sentinel if expected_code else args.output/(name+".mp4")
            request=dict(schema_version=1,model="seedvr2-3b",operation="video.upscale",prompt="",steps=1,seed=42,
                inputs=[dict(kind="video",role="source",path=str(args.source))],output=str(destination),
                width=512,height=512,frames=25,fps=24,audio=False,execution="gpu",residency=residency,
                seedvr2_vae=dict(encoder="streaming",decoder="streaming",history="frames",share_upscale_history=True,allocator_cache_bytes=1<<30))
            events=[];callback_errors=[];cancelled=False
            @callback_type
            def callback(data,_):
                nonlocal cancelled
                try:
                    value=json.loads(data);events.append(value)
                    if (name=="cancel-after-predictions" and not cancelled and
                        value.get("phase")=="seedvr2_vae_export_frame" and value.get("completed")==2):
                        cancelled=True;library.tc_engine_cancel(engine)
                except BaseException as failure:callback_errors.append(str(failure))
            result,error=C.c_void_p(),C.c_void_p();start=time.perf_counter()
            if args.ordinary_prepare_only:
                code=library.tc_engine_prepare(engine,json.dumps(request).encode(),0,callback,None,C.byref(result),C.byref(error))
            else:code=library.tc_engine_generate(engine,json.dumps(request).encode(),callback,None,C.byref(result),C.byref(error))
            text,message=consume(result),consume(error);metrics=json.loads(text) if text else None
            row=dict(name=name,code=code,error=message,seconds=time.perf_counter()-start,events=events,metrics=metrics,
                callback_errors=callback_errors,cancel_called=cancelled)
            report["runs"].append(row);write_json(args.output/"report.json",report)
            if callback_errors or code!=expected_code:raise ValueError("experimental lifecycle status/callback differs")
            if code:
                if metrics is not None:raise ValueError("failed request returned success metrics")
                reason=("requires an isolated experimental build" if args.ordinary_prepare_only else "compiled file identity changed")
                if code==1 and reason not in (message or ""):raise ValueError("experimental rejection reason differs")
                if code==2 and (not cancelled or max((v.get("completed",0) for v in events if v.get("phase")=="seedvr2_vae_export_frame"),default=0)!=2):
                    raise ValueError("typed cancellation did not stop after real subsequent-frame work")
            elif args.ordinary_prepare_only:
                if not metrics or metrics.get("prepared") is not True or metrics.get("warmup") is not False:
                    raise ValueError("ordinary preparation did not recover after selector rejection")
            else:
                causal=manifest is not None;upscale=upscale_manifest is not None
                if (not metrics or metrics.get("execution")!=("gpu_vae_cpu_ne_experimental" if causal or upscale else "gpu") or
                    metrics.get("weight_residency_recipe")!=("stage_dit" if residency=="component_staged" else "resident")):
                    raise ValueError("same-engine route/residency recovery differs")
                for prefix,enabled in (("vae_causal_conv",causal),("vae_upscale_causal_conv",upscale)):
                    if (metrics.get(prefix+"_experimental_enabled",False) is not enabled or metrics.get(prefix+"_calls",0)!=(1536 if enabled else 0) or
                        metrics.get(prefix+"_backing_hits",0)!=(1536 if enabled else 0) or metrics.get(prefix+"_frames",0)!=(24 if enabled else 0)):
                        raise ValueError("same-engine per-layer execution recovery differs: "+prefix)
                row["media"]=verify_output(destination,SimpleNamespace(kind="video",width=512,height=512,frames=25,fps=24))
            if sha256(sentinel)!=sentinel_sha or sha256(args.source)!=report["source_sha256"] or sha256(args.manifest)!=report["manifest_sha256"]:
                raise ValueError("experimental failure changed protected input/destination/bank")
            if args.upscale_manifest and sha256(args.upscale_manifest)!=report["upscale_manifest_sha256"]:
                raise ValueError("experimental failure changed original upscale bank")
            if any(".seedvr2-" in path.name or ".mp4." in path.name for path in args.output.iterdir()):
                raise ValueError("experimental request leaked private media staging")
            if args.ordinary_prepare_only and (capture.exists() or any(path.name.endswith(".png") or path.name=="ordinary-prepare-recovery.mp4" for path in args.output.iterdir())):
                raise ValueError("ordinary preparation unexpectedly generated output/captures")
            print(f"PASS {name} code={code}",flush=True)
        result,error=C.c_void_p(),C.c_void_p()
        if library.tc_engine_unload(engine,C.byref(result),C.byref(error)):raise ValueError(consume(error))
        unloaded=json.loads(consume(result))
        if unloaded.get("released") is not True or unloaded.get("mlx_active_bytes")!=0:
            raise ValueError("experimental engine retained MLX allocations after unload")
        report.update(unload=unloaded,status="complete")
    except BaseException as failure:report.update(status="failed",failure=str(failure));raise
    finally:
        if engine.value:library.tc_engine_free(engine)
        write_json(args.output/"report.json",report)


if __name__=="__main__":main()
