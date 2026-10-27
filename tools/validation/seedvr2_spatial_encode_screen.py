#!/usr/bin/env python3
"""Complete encoder-head spatial schedule versus original native VAE.

Never trims a source. Numerical parity is against the original native encoder,
not an independent FP32/whole-request/App/low-memory-machine qualification.
"""
import argparse
import json
from pathlib import Path
import subprocess
from seedvr2_framework_screen import sha256,write_json
from seedvr2_vae_coreml_conv_screen import quality

CHECKPOINT_SHA="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"


def validate_observation(value,route,frames,height,width,threshold):
    if (value.get("schema")!="seedvr2-spatial-encode-observation-v1" or value.get("route")!=route or
        (value.get("source_frames"),value.get("height"),value.get("width"))!=(frames,height,width) or
        value.get("finite") is not True or value.get("qualified") is not False or
        value.get("file_live_bytes_after_encode")!=0 or value.get("host_live_bytes_after_encode")!=0 or
        value.get("cleanup_failures")!=0):
        raise ValueError("incomplete encoder/context/ownership observation")
    eligible=height*width*128*2>=threshold
    if route=="spatial" and eligible:
        if value["tile_convolutions"]<=0 or value["global_normalizations"]!=frames*4:
            raise ValueError("eligible encoder head did not schedule all four norms per frame")
        if (frames==1 and value["file_written_bytes"]!=0) or (frames>1 and value["file_written_bytes"]<=0):
            raise ValueError("encoder file history does not match image/video lifetime")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("probe","model","input","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--rows",type=int,default=64)
    parser.add_argument("--threshold-bytes",type=int,default=256<<20)
    args=parser.parse_args()
    if args.output.exists() or args.rows<8 or args.rows>128 or args.rows%8 or args.threshold_bytes<=0:
        parser.error("fresh output and bounded explicit row settings required")
    for name in ("probe","model","input"):setattr(args,name,getattr(args,name).resolve(strict=True))
    checkpoint=args.model/"vae/seedvr2_ema_vae_fp16.safetensors"
    if sha256(checkpoint)!=CHECKPOINT_SHA:parser.error("original checkpoint identity required")
    import numpy as np
    from safetensors import safe_open
    with safe_open(args.input,framework="numpy") as file:shape=file.get_slice("pixels").get_shape()
    if len(shape)!=5 or shape[:2]!=[1,3] or shape[2]<1 or shape[2]>121 or shape[3]%8 or shape[4]%8 or (shape[2]-1)%4:
        parser.error("complete BCTHW RGB input, T=4n+1 required")
    args.output.mkdir(parents=True)
    report=dict(schema="seedvr2-spatial-encode-comparison-v1",status="running",qualified=False,
        input_sha256=sha256(args.input),checkpoint_sha256=CHECKPOINT_SHA,probe_sha256=sha256(args.probe),
        native_source_sha256=sha256(Path(__file__).resolve().parents[2]/"native/models/seedvr2/vae.cpp"),
        source_shape=shape,rows=args.rows,threshold_bytes=args.threshold_bytes,routes=[],
        scope="complete original-parameter encoder, singleton high-resolution head scheduling only; native latent parity; no independent FP32/request/App/device-budget qualification")
    try:
        from safetensors.numpy import load_file
        for route in ("baseline","spatial"):
            target=args.output/route
            command=[str(args.probe),str(args.model),str(args.input),str(target),route,str(args.threshold_bytes),str(args.rows)]
            print(f"encode route={route} shape={shape}",flush=True)
            process=subprocess.run(command,capture_output=True,text=True,timeout=600)
            row=dict(route=route,command=command,exit_code=process.returncode,stdout=process.stdout,stderr=process.stderr)
            report["routes"].append(row);write_json(args.output/"report.json",report)
            if process.returncode:raise ValueError("encoder failed; original receipt preserved: "+process.stderr)
            row["observation"]=json.loads((target/"report.json").read_text())
            validate_observation(row["observation"],route,*shape[2:],args.threshold_bytes)
            row["latent_sha256"]=sha256(target/"latent.safetensors");write_json(args.output/"report.json",report)
        paths=[args.output/route/"latent.safetensors" for route in ("baseline","spatial")]
        values=[load_file(str(path))["latent"] for path in paths]
        if any(value.dtype!=np.float16 or value.shape!=(1,16,(shape[2]-1)//4+1,shape[3]//8,shape[4]//8) for value in values):
            raise ValueError("complete encoder latent datatype/shape changed")
        report["native_quality"]=quality(*values)
        report["native_quality"]["bitwise_equal"]=bool(np.array_equal(values[0].view(np.uint16),values[1].view(np.uint16)))
        if not report["native_quality"]["passed"]:raise ValueError("unchanged native latent numerical gate failed")
        if sha256(args.input)!=report["input_sha256"] or sha256(checkpoint)!=CHECKPOINT_SHA:raise ValueError("encoder input/checkpoint changed")
        report["status"]="complete";print(json.dumps(report),flush=True)
    except BaseException as failure:report.update(status="failed",failure=str(failure));raise
    finally:write_json(args.output/"report.json",report)


if __name__=="__main__":main()
