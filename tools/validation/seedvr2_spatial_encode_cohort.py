#!/usr/bin/env python3
"""Fresh encode on every repetition, explicit ABBA, no full-request claims."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
from seedvr2_framework_screen import sha256,write_json
from seedvr2_spatial_encode_screen import CHECKPOINT_SHA,validate_observation
from seedvr2_vae_coreml_conv_screen import quality

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("probe","model","input","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=7);parser.add_argument("--rows",type=int,default=64)
    parser.add_argument("--threshold-bytes",type=int,default=256<<20)
    args=parser.parse_args()
    if args.output.exists() or not 4<=args.repeats<=15 or not 8<=args.rows<=128 or args.rows%8:parser.error("fresh output and bounded repeats/rows required")
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import load_file
    for name in ("probe","model","input"):setattr(args,name,getattr(args,name).resolve(strict=True))
    checkpoint=args.model/"vae/seedvr2_ema_vae_fp16.safetensors"
    if sha256(checkpoint)!=CHECKPOINT_SHA:parser.error("original checkpoint identity required")
    with safe_open(args.input,framework="numpy") as source:shape=source.get_slice("pixels").get_shape()
    args.output.mkdir(parents=True)
    report=dict(schema="seedvr2-spatial-encode-abba-v1",status="running",qualified=False,source_shape=shape,
        checkpoint_sha256=CHECKPOINT_SHA,input_sha256=sha256(args.input),probe_sha256=sha256(args.probe),
        order=["baseline","spatial","spatial","baseline"],rows=args.rows,threshold_bytes=args.threshold_bytes,runs=[],
        scope="complete encoder only, fresh encode per repetition, one cold then warmed, original weights/global context; no DiT/media/App/physical-RAM qualification")
    try:
        for index,route in enumerate(report["order"]):
            target=args.output/f"{index}-{route}";command=[str(args.probe),str(args.model),str(args.input),str(target),route,str(args.threshold_bytes),str(args.rows),str(args.repeats)]
            print(f"encoder ABBA index={index} route={route}",flush=True)
            process=subprocess.run(command,capture_output=True,text=True,timeout=1200)
            row=dict(index=index,route=route,command=command,exit_code=process.returncode,stderr=process.stderr,stdout=process.stdout)
            report["runs"].append(row);write_json(args.output/"report.json",report)
            if process.returncode:raise ValueError("encoder cohort failed; receipt preserved")
            raw=json.loads((target/"report.json").read_text());row["observation"]=raw
            validate_observation(raw,route,*shape[2:],args.threshold_bytes)
            values=raw["iterations"]
            if len(values)!=args.repeats or any(v["iteration"]!=i or not math.isfinite(v["encode_seconds"]) or v["encode_seconds"]<=0 for i,v in enumerate(values)):
                raise ValueError("complete fresh encoder repetitions required")
            row["latent_sha256"]=sha256(target/"latent.safetensors")
            row["warm_seconds"]=[v["encode_seconds"] for v in values[1:]];row["warm_peaks"]=[v["mlx_active_peak_bytes"] for v in values[1:]]
            write_json(args.output/"report.json",report)
        expected=load_file(str(args.output/"0-baseline/latent.safetensors"))["latent"];report["quality"]={}
        for row in report["runs"][1:]:
            value=load_file(str(Path(row["command"][3])/"latent.safetensors"))["latent"]
            result=quality(expected,value);result["bitwise_equal"]=bool(np.array_equal(expected.view(np.uint16),value.view(np.uint16)))
            report["quality"][str(row["index"])]=result
            if not result["passed"]:raise ValueError("unchanged encoder numerical gate failed")
        for route in ("baseline","spatial"):
            rows=[row for row in report["runs"] if row["route"]==route]
            report[route+"_warm_median_seconds"]=statistics.median(v for row in rows for v in row["warm_seconds"])
            report[route+"_active_peak_bytes"]=max(v for row in rows for v in row["warm_peaks"])
        if sha256(args.input)!=report["input_sha256"] or sha256(args.probe)!=report["probe_sha256"] or sha256(checkpoint)!=CHECKPOINT_SHA:
            raise ValueError("encoder cohort identities changed")
        report.update(status="complete",speed_gate_passed=report["spatial_warm_median_seconds"]<=report["baseline_warm_median_seconds"])
        print(json.dumps(report),flush=True)
    except BaseException as failure:report.update(status="failed",failure=str(failure));raise
    finally:write_json(args.output/"report.json",report)

if __name__=="__main__":main()
