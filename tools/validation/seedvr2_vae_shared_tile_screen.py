#!/usr/bin/env python3
"""Matched full-spatial convolution boundary and public shared-IO screening.

This is not full-VAE/real-video qualification. Timed native schedules include
GPU tile/temporal pack, all producer/consumer waits, CPU(+NE), output assembly
and optional disjoint GPU output channels. CPU quality scans are outside timing.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/coreml"))
from export_ltx_gemma_mlp import mapped_tensor, read_header, tensor_info
from export_seedvr2_ffn import sha256, tree_identity
from export_seedvr2_vae_conv import CHECKPOINT_SHA
from seedvr2_framework_screen import write_json
from seedvr2_vae_native_io_screen import validate_manifest
from seedvr2_vae_coreml_conv_screen import reference_3d, check_reference


def validate_pair(full,prefix,size):
    validate_manifest(full)
    if full["output_channels"]!=full["total_output_channels"]:
        raise ValueError("complete-output bank required")
    if type(size) is not int or size not in (64,128,256,512,1024) or size%full["rows"] or size%full["cols"]:
        raise ValueError("full spatial geometry must be divisible by the fixed tile")
    out,cin,taps,kh,kw=full["original_shape"]
    if cin*taps*(size+kh-1)*(size+kw-1)*2+out*size*size*6>(2<<30):
        raise ValueError("complete fixture exceeds 2-GiB bound")
    if prefix is not None:
        validate_manifest(prefix)
        for key in ("layer","original_shape","checkpoint_sha256","rows","cols","input_rows","input_cols"):
            if prefix[key]!=full[key]:raise ValueError("prefix bank differs from full bank")
        if not 0<prefix["output_channels"]<out:raise ValueError("disjoint nonempty GPU suffix required")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--hybrid-manifest",type=Path)
    parser.add_argument("--probe",type=Path,required=True)
    parser.add_argument("--size",type=int,default=128)
    parser.add_argument("--iterations",type=int,default=9)
    parser.add_argument("--rounds",type=int,default=2)
    parser.add_argument("--kernel",choices=("scalar","transpose"),default="scalar")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not 3<=args.iterations<=30 or not 1<=args.rounds<=4:parser.error("fresh output and bounded repeats required")
    if sha256(args.checkpoint)!=CHECKPOINT_SHA:parser.error("original checkpoint identity required")
    full=json.loads(args.manifest.read_text())
    prefix=json.loads(args.hybrid_manifest.read_text()) if args.hybrid_manifest else None
    validate_pair(full,prefix,args.size)
    banks=[(args.manifest,full)]+([(args.hybrid_manifest,prefix)] if prefix else [])
    for path,manifest in banks:
        if tree_identity(path.parent/manifest["compiled"]["path"])!=manifest["compiled"]["identity"]:
            parser.error("compiled bank identity changed")
    probe=args.probe.resolve(strict=True)
    args.output.mkdir(parents=True)
    report=dict(schema="seedvr2-vae-shared-tile-screen-v1",status="running",qualified=False,
        checkpoint_sha256=CHECKPOINT_SHA,manifest_sha256=sha256(args.manifest),
        hybrid_manifest_sha256=sha256(args.hybrid_manifest) if prefix else None,probe_sha256=sha256(probe),
        source_sha256=sha256(Path(__file__)),native_source_sha256=sha256(ROOT/"tools/native/seedvr2_vae_shared_tile_probe.mm"),
        hardware=dict(chip=subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip(),
            memory_bytes=int(subprocess.check_output(["sysctl","-n","hw.memsize"],text=True)),
            macos=subprocess.check_output(["sw_vers","-productVersion"],text=True).strip()),
        size=args.size,tile_rows=full["rows"],tile_cols=full["cols"],layer=full["layer"],kernel=args.kernel,runs=[],
        physical_ane_occupancy_measured=False,
        scope="complete synthetic normalized spatial convolution, original FP16, no bias/norm; native pack/sync/consumer/suffix included, quality/poison scans excluded; not VAE/request/physical occupancy qualification")
    try:
        import numpy as np
        offset,header=read_header(args.checkpoint);shape=tuple(full["original_shape"])
        info=tensor_info(header,full["layer"]+".weight","F16",shape)
        weight=np.ascontiguousarray(mapped_tensor(args.checkpoint,offset,info,np.float16,shape))
        pixels=np.random.default_rng(31).normal(0,.5,(1,shape[1],shape[2],args.size+shape[3]-1,args.size+shape[4]-1)).astype(np.float16)
        reference=reference_3d(pixels,weight)
        report["reference_verification"]=check_reference(pixels,weight,reference)
        np.ascontiguousarray(pixels.transpose(0,2,3,4,1)).tofile(args.output/"input.fp16.bin")
        np.ascontiguousarray(weight.transpose(0,2,3,4,1)).tofile(args.output/"weight.fp16.bin")
        np.ascontiguousarray(reference,dtype="<f4").tofile(args.output/"reference.fp32.bin")
        base=dict(schema="seedvr2-vae-shared-tile-fixture-v1",cin=shape[1],taps=shape[2],ks=shape[3],cout=shape[0],
            rows=args.size,cols=args.size,tile_rows=full["rows"],tile_cols=full["cols"],iterations=args.iterations,
            input=str((args.output/"input.fp16.bin").resolve()),weight=str((args.output/"weight.fp16.bin").resolve()),
            reference=str((args.output/"reference.fp32.bin").resolve()))
        fixtures={}
        for name,(path,manifest) in zip(("full","prefix"),banks):
            fixture=dict(base,core_cout=manifest["output_channels"],compiled=str((path.parent/manifest["compiled"]["path"]).resolve()))
            fixtures[name]=args.output/(name+".json");write_json(fixtures[name],fixture)
        report["fixture_hashes"]={path.name:sha256(path) for path in fixtures.values()}
        report["operand_hashes"]={name:sha256(args.output/name) for name in ("input.fp16.bin","weight.fp16.bin","reference.fp32.bin")}
        routes=["gpu3d","ne_serial","ne_pipeline","gpu2d","cpu_serial"]
        if prefix:routes+= ["hybrid3d","hybrid2d"]
        for cohort in range(args.rounds):
            order=routes if cohort%2==0 else list(reversed(routes))
            for route in order:
                print(f"cohort={cohort} route={route} size={args.size}",flush=True)
                fixture=fixtures["prefix" if route.startswith("hybrid") else "full"]
                command=[str(probe),str(fixture.resolve()),route]+([args.kernel] if args.kernel!="scalar" else [])
                process=subprocess.run(command,capture_output=True,text=True,timeout=180)
                row=dict(cohort=cohort,route=route,command=command,exit_code=process.returncode,stderr=process.stderr)
                if process.returncode==0:row["observation"]=json.loads(process.stdout)
                else:row.update(status="failed",stdout=process.stdout)
                report["runs"].append(row);write_json(args.output/"report.json",report)
        summaries={}
        for route in routes:
            rows=[row for row in report["runs"] if row["route"]==route]
            success=all("observation" in row and row["observation"]["quality"]["passed"] for row in rows)
            times=[value["total_seconds"] for row in rows if "observation" in row
                for value in row["observation"]["observations"] if value["warm"]]
            summaries[route]=dict(component_passed=success,warm_median_seconds=statistics.median(times) if times else None,
                process_warm_medians=[row["observation"]["warm_median_seconds"] for row in rows if "observation" in row])
        for path,manifest in banks:
            if tree_identity(path.parent/manifest["compiled"]["path"])!=manifest["compiled"]["identity"]:
                raise ValueError("compiled bank changed during boundary screen")
        report.update(status="complete",routes=summaries,component_quality_passed=all(row["component_passed"] for row in summaries.values()))
        print(json.dumps(dict(status="complete",routes=summaries,component_quality_passed=report["component_quality_passed"])),flush=True)
        if not report["component_quality_passed"]:raise RuntimeError("one or more shared-tile component routes failed; see preserved receipts")
    except BaseException as error:report.update(status="failed",failure=str(error));raise
    finally:write_json(args.output/"report.json",report)


if __name__=="__main__":main()
