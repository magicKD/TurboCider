#!/usr/bin/env python3
"""Identity-bound stage/file tiled decoder screen, not full-request qualification.

Trims only an explicitly selected causal prefix for an offline experiment; the
App's complete-source contract and model frame cap are not changed. All actual
emitted pixels are checked against both saved native and optional FP32 witnesses.
"""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/coreml"))
from export_seedvr2_vae_conv import CHECKPOINT_SHA
from seedvr2_framework_screen import sha256, write_json
from seedvr2_vae_coreml_conv_screen import quality


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("probe","model","native-fixture","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fp32-fixture",type=Path)
    parser.add_argument("--frames",type=int,default=5)
    parser.add_argument("--rows",type=int,default=64)
    parser.add_argument("--threshold-bytes",type=int,default=256<<20)
    args=parser.parse_args()
    if args.output.exists() or args.frames<1 or (args.frames-1)%4 or not 8<=args.rows<=128 or args.rows%8:
        parser.error("fresh output, complete padded causal prefix and 8-aligned rows required")
    for name in ("probe","model","native_fixture"):setattr(args,name,getattr(args,name).resolve(strict=True))
    if args.fp32_fixture:args.fp32_fixture=args.fp32_fixture.resolve(strict=True)
    checkpoint=args.model/"vae/seedvr2_ema_vae_fp16.safetensors"
    if sha256(checkpoint)!=CHECKPOINT_SHA:parser.error("original VAE checkpoint SHA required")
    import numpy as np
    from safetensors.numpy import load_file,save_file
    source=load_file(str(args.native_fixture));pixels=source.get("decoded",source.get("pixels"));latent=source["latent"]
    if (pixels is None or pixels.dtype!=np.float16 or pixels.ndim!=5 or pixels.shape[:2]!=(1,3) or
        pixels.shape[2]<args.frames or latent.ndim!=5 or latent.shape[:2]!=(1,16) or latent.shape[2]<(args.frames-1)//4+1):
        parser.error("complete native decoder latent/FP16 pixel witness required")
    args.output.mkdir(parents=True)
    fixture=args.output/"prefix.safetensors"
    if pixels.shape[2]==args.frames and latent.shape[2]==(args.frames-1)//4+1:
        # Keep the exact serialized latent identity when qualifying an already
        # prepared prefix against its separately executed FP32 oracle.
        shutil.copy2(args.native_fixture,fixture)
    else:
        save_file(dict(latent=np.ascontiguousarray(latent[:,:,:((args.frames-1)//4+1)]),
            pixels=np.ascontiguousarray(pixels[:,:,:args.frames])),str(fixture))
    shape=(args.frames,pixels.shape[3],pixels.shape[4],3)
    report=dict(schema="seedvr2-spatial-decoder-comparison-v1",status="running",qualified=False,
        probe_sha256=sha256(args.probe),checkpoint_sha256=CHECKPOINT_SHA,native_fixture_sha256=sha256(args.native_fixture),
        prefix_fixture_sha256=sha256(fixture),fp32_fixture_sha256=sha256(args.fp32_fixture) if args.fp32_fixture else None,
        frames=args.frames,rows=args.rows,threshold_bytes=args.threshold_bytes,routes=[],
        scope="explicit offline causal prefix decoder, same original coefficients; sampled process peak/host quality/export included; no full request/App/paired warm-speed/low-RAM qualification")
    try:
        fp32=None
        if args.fp32_fixture:
            value=load_file(str(args.fp32_fixture));fp32=value.get("decoded",value.get("pixels"))
            if fp32 is None or fp32.dtype!=np.float32 or fp32.shape[:2]!=(1,3) or fp32.shape[2]<args.frames or fp32.shape[3:]!=pixels.shape[3:]:
                raise ValueError("matching complete original-FP32 pixel witness required")
            fp32=np.ascontiguousarray(fp32[0,:,:args.frames].transpose(1,2,3,0))
        for route in ("baseline","spatial"):
            destination=args.output/route
            command=[str(args.probe),str(args.model),str(fixture),str(destination),route,str(args.threshold_bytes),str(args.rows)]
            print(f"spatial screen {route} {shape}",flush=True)
            result=subprocess.run(command,capture_output=True,text=True,timeout=600)
            receipt=dict(route=route,command=command,exit_code=result.returncode,stderr=result.stderr,stdout=result.stdout)
            report["routes"].append(receipt)
            if (destination/"report.json").is_file():receipt["observation"]=json.loads((destination/"report.json").read_text())
            write_json(args.output/"report.json",report)
            if result.returncode:raise ValueError("spatial decoder failed; preserved receipt: "+result.stderr)
            path=destination/"decoded.fp16.bin"
            if path.stat().st_size!=np.prod(shape)*2:raise ValueError("bounded decoder export extent differs")
            decoded=np.memmap(path,mode="r",dtype=np.float16,shape=shape)
            receipt["export_sha256"]=sha256(path)
            if fp32 is not None:
                receipt["original_fp32_quality"]=quality(fp32,decoded)
                if not receipt["original_fp32_quality"]["passed"]:raise ValueError("unchanged original-FP32 pixel quality gate failed")
            if not receipt["observation"]["quality_passed"]:raise ValueError("saved native pixel quality gate failed")
        if sha256(args.native_fixture)!=report["native_fixture_sha256"] or sha256(checkpoint)!=CHECKPOINT_SHA:
            raise ValueError("spatial screen source changed")
        report["status"]="complete"
        print(json.dumps(dict(status="complete",frames=args.frames,routes=[row["observation"] for row in report["routes"]])),flush=True)
    except BaseException as failure:report.update(status="failed",failure=str(failure));raise
    finally:write_json(args.output/"report.json",report)


if __name__=="__main__":main()
