#!/usr/bin/env python3
"""Complete captured-real spatial projection/bias/shuffle boundary comparison.

Each timed request processes the complete selected layer, not just one tile.
This is not a full VAE/App route, raw whole-VAE quality, or physical ANE proof.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/coreml"))
from export_ltx_gemma_mlp import mapped_tensor,read_header,tensor_info
from export_seedvr2_ffn import sha256,tree_identity
from export_seedvr2_vae_upscale import SCHEMA,LAYER,CHECKPOINT_SHA
from seedvr2_framework_screen import write_json


def validate_manifest(value):
    if (value.get("schema")!=SCHEMA or value.get("qualified") is not False or
        value.get("checkpoint_sha256")!=CHECKPOINT_SHA or value.get("layer")!=LAYER or
        value.get("original_shape")!=[1024,256,1,1,1] or value.get("precision")!="fp16" or
        value.get("bias_applied") is not False or value.get("channel_order")!="spatial_row_spatial_col_channel" or
        value.get("compute_units")!="cpuAndNeuralEngine" or value.get("variant") not in ("projection","shuffle")):
        raise ValueError("original complete spatial projection bank required")
    rows,cols=value["rows"],value["cols"]
    if type(rows) is not int or type(cols) is not int or rows not in (32,64,128) or cols!=rows:
        raise ValueError("bounded square fixed tile required")
    expected=[1,256,rows*2,cols*2] if value["variant"]=="shuffle" else [1,1024,rows,cols]
    if value["input_shape"]!=[1,256,rows,cols] or value["output_shape"]!=expected:
        raise ValueError("bank input/output layout differs")


def validate_observation(value,route,manifest,size,iterations):
    tiles=(size//manifest["rows"])**2
    coreml=route not in ("gpu_matmul","gpu_fused","gpu_conv2d","gpu_tiled")
    if (value.get("schema")!="seedvr2-upscale-boundary-observation-v1" or value.get("route")!=route or
        value.get("rows")!=size or value.get("cols")!=size or value.get("tiles")!=tiles or
        value.get("variant")!=manifest["variant"] or value.get("physical_ane_occupancy_measured") is not False or
        value.get("signed_input_aba_checked") is not True or value.get("quality",{}).get("passed") is not True):
        raise ValueError("incomplete/mismatched projection observation")
    expected_calls=tiles*(iterations+3) if coreml else 0
    if value.get("coreml_calls")!=expected_calls or value.get("output_backing_identity_hits")!=expected_calls:
        raise ValueError("actual Core ML coverage/backing hits differ")
    if coreml and (value.get("bitwise_pack_checked_values")!=size*size*256 or
        value.get("bitwise_consumer_checked_values")!=size*size*1024 or value.get("slot_bytes",0)<=0):
        raise ValueError("complete independent shared-IO witnesses required")
    if route=="gpu_fused" and value.get("bitwise_default_checked_values")!=3*size*size*1024:
        raise ValueError("fused GPU must match native FP16 words under signed ABA")
    observations=value.get("observations",[])
    if len(observations)!=iterations+3 or sum(v["warm"] for v in observations)!=iterations:
        raise ValueError("warm request coverage differs")
    if [v["sign"] for v in observations[:3]]!=[1,-1,1]:
        raise ValueError("signed-input ABA was not executed")
    for row in observations:
        quality=row["quality"]
        if (not quality["passed"] or not quality["finite"] or
            not math.isfinite(quality["relative_l2"]) or not math.isfinite(quality["cosine"]) or
            quality["relative_l2"]>.05 or quality["cosine"]<.999 or
            not math.isfinite(row["total_seconds"]) or row["total_seconds"]<=0):
            raise ValueError("unchanged per-request original-FP32 numerical gate failed")


def prepare_reference(pixels,weight,path):
    import numpy as np
    # Blocked BLAS avoids a full FP32 input copy; the output file is one FP32
    # unbiased projection. Bias/sign and original phase indexing are separately
    # checked in native code against every final output value.
    h,w,c=pixels.shape;matrix=weight[:,:,0,0,0].astype(np.float32).T
    result=np.memmap(path,mode="w+",dtype="<f4",shape=(h*w,1024))
    flat=pixels.reshape(h*w,c)
    for first in range(0,h*w,4096):
        stop=min(first+4096,h*w)
        block=flat[first:stop].astype(np.float32)@matrix
        if not np.all(np.isfinite(block)):raise ValueError("FP32 projection reference is nonfinite")
        result[first:stop]=block
    result.flush()
    maximum=0.;points=0
    for pixel in (0,(h*w)//2,h*w-1):
        for channel in (0,255,511,767,1023):
            expected=float(np.dot(flat[pixel].astype(np.float64),weight[channel,:,0,0,0].astype(np.float64)))
            maximum=max(maximum,abs(expected-float(result[pixel,channel])));points+=1
            if abs(expected-float(result[pixel,channel]))>1e-4*max(1.,abs(expected)):
                raise ValueError("FP32 reference disagrees with independent FP64 products")
    del result
    return dict(finite=True,sampled_fp64_points=points,sampled_fp64_max_abs_error=maximum)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","capture","source-prefix","projection-manifest","shuffle-manifest","probe","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--frame",type=int,choices=(0,24),default=0)
    parser.add_argument("--size",type=int,choices=(128,256,512),default=512)
    parser.add_argument("--iterations",type=int,default=6)
    parser.add_argument("--rounds",type=int,default=2)
    args=parser.parse_args()
    if args.output.exists() or not 3<=args.iterations<=15 or not 1<=args.rounds<=2:
        parser.error("fresh output and bounded observations required")
    if sha256(args.checkpoint)!=CHECKPOINT_SHA:parser.error("original checkpoint identity required")
    capture=json.loads((args.capture/"capture.json").read_text())
    if (capture.get("status")!="complete" or capture.get("decoded_frames")!=25 or
        capture.get("observed_projection_frames")!=25 or capture.get("captured_frames")!=[0,24]):
        parser.error("complete causal t25 capture required")
    banks={name:(path,json.loads(path.read_text())) for name,path in
        (("projection",args.projection_manifest),("shuffle",args.shuffle_manifest))}
    for name,(path,value) in banks.items():
        validate_manifest(value)
        if value["variant"]!=name or tree_identity(path.parent/value["compiled"]["path"])!=value["compiled"]["identity"]:
            parser.error("compiled boundary identity/variant differs")
    if any(v[1]["rows"]!=banks["projection"][1]["rows"] for v in banks.values()):parser.error("matched tile shapes required")
    if args.size%banks["projection"][1]["rows"]:parser.error("complete layer must be divisible by tile")
    args.output.mkdir(parents=True)
    source=args.capture/f"input-frame-{args.frame}.safetensors";probe=args.probe.resolve(strict=True)
    report=dict(schema="seedvr2-upscale-real-boundary-screen-v1",status="running",qualified=False,runs=[],
        checkpoint_sha256=CHECKPOINT_SHA,captured_input_sha256=sha256(source),capture_report_sha256=sha256(args.capture/"capture.json"),
        declared_source_prefix_sha256=sha256(args.source_prefix),probe_sha256=sha256(probe),
        capture_source_sha256=sha256(ROOT/"tools/native/seedvr2_vae_upscale_capture.cpp"),
        native_source_sha256=sha256(ROOT/"tools/native/seedvr2_vae_upscale_probe.mm"),
        io_source_sha256=sha256(ROOT/"tools/native/seedvr2_vae_upscale_io.hpp"),source_sha256=sha256(Path(__file__)),
        frame=args.frame,source_shape=[1,1,512,512,256],rows=args.size,cols=args.size,crop_origin=[0,0],
        complete_layer=args.size==512,physical_ane_occupancy_measured=False,
        manifests={name:dict(path=str(path),sha256=sha256(path)) for name,(path,_) in banks.items()},
        scope="captured real complete-context operands, selected spatial projection+FP16 bias+shuffle; pack/sync/assembly in timing; not complete VAE/App/physical occupancy/low-RAM qualification")
    try:
        import numpy as np
        from safetensors import safe_open
        with safe_open(source,framework="numpy") as file:
            metadata=file.metadata();raw=file.get_tensor("input")
        if (raw.dtype!=np.float16 or raw.shape!=(1,1,512,512,256) or metadata.get("layer")!=LAYER or
            metadata.get("output_frame")!=str(args.frame) or metadata.get("initial")!=("true" if args.frame==0 else "false")):
            raise ValueError("real input datatype/shape/phase metadata differs")
        pixels=np.ascontiguousarray(raw[0,0,:args.size,:args.size]);del raw
        if not np.all(np.isfinite(pixels)):raise ValueError("captured operand is nonfinite")
        offset,header=read_header(args.checkpoint)
        shape=(1024,256,1,1,1);info=tensor_info(header,LAYER+".weight","F16",shape)
        weight=np.ascontiguousarray(mapped_tensor(args.checkpoint,offset,info,np.float16,shape))
        shift=np.ascontiguousarray(mapped_tensor(args.checkpoint,offset,tensor_info(header,LAYER+".bias","F16",(1024,)),np.float16,(1024,)))
        report["reference_verification"]=prepare_reference(pixels,weight,args.output/"reference.fp32.bin")
        pixels.tofile(args.output/"input.fp16.bin");weight[:,:,0,0,0].tofile(args.output/"weight.fp16.bin");shift.tofile(args.output/"bias.fp16.bin")
        report["input_range"]=[float(pixels.min()),float(pixels.max())]
        report["operand_hashes"]={name:sha256(args.output/name) for name in
            ("input.fp16.bin","weight.fp16.bin","bias.fp16.bin","reference.fp32.bin")}
        fixtures={}
        for name,(path,value) in banks.items():
            fixture=dict(schema="seedvr2-upscale-boundary-fixture-v1",rows=args.size,cols=args.size,
                tile_rows=value["rows"],tile_cols=value["cols"],iterations=args.iterations,variant=name,
                compiled=str((path.parent/value["compiled"]["path"]).resolve()),
                **{key:str((args.output/(key+suffix)).resolve()) for key,suffix in
                    (("input",".fp16.bin"),("weight",".fp16.bin"),("bias",".fp16.bin"),("reference",".fp32.bin"))})
            fixtures[name]=args.output/(name+".json");write_json(fixtures[name],fixture)
        del pixels,weight,shift
        routes=[("projection",v) for v in ("gpu_matmul","ne_serial","ne_pipeline","gpu_conv2d","gpu_fused","gpu_tiled","cpu_serial")]
        routes+=[("shuffle",v) for v in ("ne_serial","ne_pipeline","cpu_serial")]
        for cohort in range(args.rounds):
            for name,route in (routes if cohort%2==0 else list(reversed(routes))):
                print(f"cohort={cohort} boundary={name} route={route} size={args.size} frame={args.frame}",flush=True)
                command=[str(probe),str(fixtures[name].resolve()),route]
                process=subprocess.run(command,capture_output=True,text=True,timeout=240)
                row=dict(cohort=cohort,variant=name,route=route,command=command,exit_code=process.returncode,stderr=process.stderr)
                if process.returncode==0:
                    row["observation"]=json.loads(process.stdout)
                    validate_observation(row["observation"],route,banks[name][1],args.size,args.iterations)
                else:row["stdout"]=process.stdout
                report["runs"].append(row);write_json(args.output/"report.json",report)
        summary={}
        for name,route in routes:
            rows=[row for row in report["runs"] if (row["variant"],row["route"])==(name,route)]
            times=[v["total_seconds"] for row in rows if "observation" in row for v in row["observation"]["observations"] if v["warm"]]
            summary[name+"/"+route]=dict(component_passed=all("observation" in row for row in rows),
                warm_median_seconds=statistics.median(times) if times else None,
                process_warm_medians=[row["observation"]["warm_median_seconds"] for row in rows if "observation" in row])
        for name,(path,value) in banks.items():
            if sha256(path)!=report["manifests"][name]["sha256"] or tree_identity(path.parent/value["compiled"]["path"])!=value["compiled"]["identity"]:
                raise ValueError("bank identity changed during run")
        if sha256(source)!=report["captured_input_sha256"]:raise ValueError("captured input changed during run")
        report.update(status="complete",routes=summary,component_quality_passed=all(v["component_passed"] for v in summary.values()))
        if not report["component_quality_passed"]:raise ValueError("boundary route failed; original failure records preserved")
        print(json.dumps(summary),flush=True)
    except BaseException as failure:report.update(status="failed",failure=str(failure));raise
    finally:write_json(args.output/"report.json",report)


if __name__=="__main__":main()
