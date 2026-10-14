#!/usr/bin/env python3
"""Every exported frame of matched complete GPU/experimental VAE requests.

This complements the independently audited same-latent FP32 component gate;
8-bit exports cannot qualify intermediates, physical ANE occupancy or speed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from seedvr2_framework_screen import sha256,write_json
from seedvr2_app_media_screen import PixelStats,video_frames,read_frame,pixel_quality_passed


def validate_pair(reference,candidate):
    if (reference.get("status")!="complete" or candidate.get("status")!="complete" or
        reference.get("source_sha256")!=candidate.get("source_sha256") or reference.get("workload")!=candidate.get("workload") or
        reference.get("native_artifact_identity",{}).get("library_sha256")!=candidate.get("native_artifact_identity",{}).get("library_sha256") or
        not reference.get("runs") or len(reference["runs"])!=len(candidate.get("runs",[]))):
        raise ValueError("matched complete source/workload/native-library requests required")
    for left,right in zip(reference["runs"],candidate["runs"]):
        a,b=left["metrics"],right["metrics"]
        if (a.get("seed")!=b.get("seed") or a.get("execution")!="gpu" or b.get("execution")!="gpu_vae_cpu_ne_experimental" or
            a.get("vae_causal_conv_experimental_enabled") is True or b.get("vae_causal_conv_experimental_enabled") is not True):
            raise ValueError("request seed/experimental execution differs")


def load_capture(run,workload,source_sha,experimental):
    binding=run["metrics"].get("video_raw_capture") or {}
    path=Path(binding.get("manifest",""))
    if (not path.is_file() or path.is_symlink() or binding.get("diagnostic_only") is not True or
        binding.get("frames")!=workload["frames"] or binding.get("sha256")!=sha256(path)):
        raise ValueError("request-bound complete raw capture required")
    capture=json.loads(path.read_text())
    expected=dict(schema="seedvr2-causal-request-pixels-v1",status="complete",fp16_layout="THWC",
        width=workload["width"],height=workload["height"],frame_count=workload["frames"],fps=workload["fps"],
        seed=run["metrics"]["seed"],source_sha256=source_sha,experimental_causal_enabled=experimental,
        causal_manifest_sha256=run["metrics"].get("vae_causal_conv_manifest_sha256") if experimental else None,
        checkpoint_sha256="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1")
    if any(capture.get(key)!=value for key,value in expected.items()) or len(capture.get("frames",[]))!=workload["frames"]:
        raise ValueError("raw capture source/workload/route differs")
    predecode=capture.get("predecode",{})
    shape=[1,16,1+(workload["frames"]-1+3)//4,workload["height"]//8,workload["width"]//8]
    for name in ("condition","velocity","latent"):
        tensor=predecode.get(name,{})
        size=2 if name=="condition" else 4
        digest=tensor.get("sha256","")
        if (tensor.get("shape")!=shape or tensor.get("dtype")!=("float16" if size==2 else "float32") or
            tensor.get("bytes")!=math.prod(shape)*size or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("raw capture predecode tensor metadata differs")
    capture["root"]=path.parent
    capture["expected_rgb24_sha256"]=run["metrics"].get("video_preencode_rgb24_sha256")
    if run["metrics"].get("video_preencode_pixel_hash_diagnostic") is not True or not capture["expected_rgb24_sha256"]:
        raise ValueError("actual preencoder hash required for raw capture")
    return capture


def capture_frame(capture,index,width,height):
    row=capture["frames"][index]
    if row.get("index")!=index:raise ValueError("raw capture frame coverage differs")
    values=[]
    for name,suffix,dtype,size in (("fp16","fp16","<f2",2),("rgb24","rgb24","u1",1)):
        item=row.get(name,{})
        if item.get("path")!=f"frame-{index}.{suffix}":raise ValueError("raw frame path escapes capture")
        path=capture["root"]/item["path"];expected=width*height*3*size
        if (path.is_symlink() or not path.is_file() or path.stat().st_size!=expected or item.get("bytes")!=expected or
            item.get("sha256")!=sha256(path)):
            raise ValueError("captured frame identity/extent changed")
        values.append(np.fromfile(path,dtype=dtype).reshape(height,width,3))
    if not np.all(np.isfinite(values[0])):raise ValueError("nonfinite raw FP16 pixels")
    return values


def compare_raw(reference,candidate,workload):
    if reference["predecode"]!=candidate["predecode"]:
        raise ValueError("condition/velocity/latent are not bitwise bound to the same operands")
    frames=[];total=PixelStats();raw_error=raw_reference=raw_actual=raw_dot=0.;maximum=0.
    hashes=[hashlib.sha256(),hashlib.sha256()]
    for index in range(workload["frames"]):
        a,x=capture_frame(reference,index,workload["width"],workload["height"])
        b,y=capture_frame(candidate,index,workload["width"],workload["height"])
        raw_bitwise_equal=np.array_equal(a.view(np.uint16),b.view(np.uint16))
        for digest,values in zip(hashes,(x,y)):digest.update(values.tobytes())
        a,b=a.astype(np.float64),b.astype(np.float64);delta=a-b
        error=float(np.sum(delta*delta));norm=float(np.sum(a*a));other=float(np.sum(b*b));dot=float(np.sum(a*b))
        relative=math.sqrt(error/norm) if norm else (0. if error==0 else math.inf)
        cosine=dot/math.sqrt(norm*other) if norm and other else (1. if error==0 else 0.)
        if not (math.isfinite(relative) and relative<=.05 and cosine>=.999):raise ValueError("unchanged raw-frame numerical gate failed")
        values=PixelStats();values.add(x,y);rgb=values.result();total.add(x,y)
        if not pixel_quality_passed(rgb):raise ValueError("unchanged preencoder RGB numerical gate failed")
        if index==0 and (not raw_bitwise_equal or not rgb["decoded_pixels_bitwise_equal"]):
            raise ValueError("native first frame changed")
        extent=float(np.max(np.abs(delta)));maximum=max(maximum,extent)
        frames.append(dict(frame=index,raw_fp16_relative_l2=relative,raw_fp16_cosine=cosine,raw_fp16_maximum_error=extent,preencode_rgb24=rgb))
        raw_error+=error;raw_reference+=norm;raw_actual+=other;raw_dot+=dot
    if any(digest.hexdigest()!=capture["expected_rgb24_sha256"] for digest,capture in zip(hashes,(reference,candidate))):
        raise ValueError("captured RGB bytes differ from actual encoder input hash")
    return dict(predecode_bitwise_equal=True,frames=frames,raw_fp16_maximum_error=maximum,
        raw_fp16_relative_l2=math.sqrt(raw_error/raw_reference) if raw_reference else 0.,
        raw_fp16_cosine=raw_dot/math.sqrt(raw_reference*raw_actual) if raw_reference and raw_actual else 1.,
        preencode_rgb24=total.result())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("reference","candidate","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--require-raw-capture",action="store_true",help="bind predecode tensors and inspect every preencoder FP16/RGB24 frame")
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh complete-request media receipt required")
    reference=json.loads(args.reference.read_text());candidate=json.loads(args.candidate.read_text());validate_pair(reference,candidate)
    workload=reference["workload"]
    if workload["kind"]!="video":parser.error("this bounded verifier expects video requests")
    count=workload["frames"];frame_values=workload["width"]*workload["height"]*3
    report=dict(schema="seedvr2-causal-complete-request-media-v1",status="running",runs=[],
        reference_sha256=sha256(args.reference),candidate_sha256=sha256(args.candidate),
        scope="all exported RGB frames under unchanged per-frame .05/.999 gate; not tensor/FP32/speed/occupancy qualification")
    try:
        for index,(left,right) in enumerate(zip(reference["runs"],candidate["runs"])):
            raw=None
            if args.require_raw_capture:
                captures=[load_capture(row,workload,reference["source_sha256"],experimental) for row,experimental in ((left,False),(right,True))]
                raw=compare_raw(*captures,workload)
            paths=[Path(row["metrics"]["output"]) for row in (left,right)]
            for path,row in zip(paths,(left,right)):
                if sha256(path)!=row["output_sha256"]:raise ValueError("completed request media changed")
            total=PixelStats();frames=[]
            with video_frames(paths[0],workload["width"]) as a,video_frames(paths[1],workload["width"]) as b:
                for frame in range(count):
                    x,y=read_frame(a,frame_values),read_frame(b,frame_values)
                    if x is None or y is None:raise ValueError("incomplete exported video")
                    total.add(x,y);values=PixelStats();values.add(x,y);row=dict(frame=frame,**values.result());frames.append(row)
                    if not pixel_quality_passed(row):raise ValueError("unchanged per-frame exported-media gate failed")
                if read_frame(a,frame_values) is not None or read_frame(b,frame_values) is not None:raise ValueError("undeclared video tail")
            report["runs"].append(dict(index=index,frames=frames,raw_capture=raw,passed=True,**total.result()))
        if args.require_raw_capture:report["scope"]+="; identical predecode hashes and every raw FP16/preencoder RGB frame checked; still not broad perceptual/temporal/public qualification"
        report["status"]="complete";print(json.dumps(dict(status="complete",runs=[{key:row[key] for key in ("index","decoded_frames","relative_l2","cosine","maximum_error_8bit")} for row in report["runs"]])),flush=True)
    except BaseException as error:report.update(status="failed",failure=str(error));raise
    finally:write_json(args.output,report)


if __name__=="__main__":main()
