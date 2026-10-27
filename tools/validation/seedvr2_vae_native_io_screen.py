#!/usr/bin/env python3
"""Original-weight native Core ML IO/tile screen, not a production VAE route.

Prepares FP32 witnesses and bounded binary inputs, measures resident MLX GPU
convolutions, then runs public Core ML reusable-IO controls in separate processes.
Full-frame scheduling, GPU/Core ML cooperation and full-VAE quality are excluded.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/coreml"))
from export_ltx_gemma_mlp import mapped_tensor, read_header, tensor_info
from export_seedvr2_ffn import sha256, tree_identity
from export_seedvr2_vae_conv import CHECKPOINT_SHA, SCHEMA, FIXED_TILE_EXTENTS, temporal_conv2d_weights
from seedvr2_framework_screen import write_json
from seedvr2_vae_coreml_conv_screen import reference_3d, check_reference, quality


def validate_manifest(manifest):
    """Reject recipe/shape drift before compiling or allocating any fixture."""
    if (manifest.get("schema")!=SCHEMA or manifest.get("checkpoint_sha256")!=CHECKPOINT_SHA or
        manifest.get("bias_applied") is not False or manifest.get("precision")!="fp16" or
        manifest.get("source_operand_recipe")!="original-fp16-temporal-channel-stack"):
        raise ValueError("original-FP16 temporal-stack, no-bias probe required")
    values = [manifest.get(key) for key in ("output_channels","total_output_channels","rows","cols","input_rows","input_cols")]
    original = manifest.get("original_shape")
    if not isinstance(original,list) or len(original)!=5 or any(type(value) is not int for value in values+original):
        raise ValueError("integer fixed geometry required")
    out,cin,taps,kh,kw = original
    channels,total,rows,cols,hi,wi = values
    if (not 0<cin<=512 or not 0<channels<=out<=4096 or total!=out or taps not in (1,3) or
        kh!=kw or kh not in (1,3) or rows not in FIXED_TILE_EXTENTS or cols not in FIXED_TILE_EXTENTS or
        hi!=rows+kh-1 or wi!=cols+kw-1 or manifest.get("input_layout")!="N(T*C)HW" or
        manifest.get("output_layout")!="NOHW" or manifest.get("temporal_taps")!=taps or
        manifest.get("input_channels")!=cin):
        raise ValueError("fixed halo/temporal/prefix geometry differs")
    if (cin*taps*hi*wi*2+channels*rows*cols*6)>(512<<20):
        raise ValueError("fixture exceeds bounded 512-MiB allowance")


def prepare_fixture(manifest, checkpoint, directory, iterations):
    import numpy as np
    offset,header = read_header(checkpoint)
    shape = tuple(manifest["original_shape"])
    info = tensor_info(header,manifest["layer"]+".weight","F16",shape)
    weight = np.ascontiguousarray(mapped_tensor(checkpoint,offset,info,np.float16,shape)[:manifest["output_channels"]])
    pixels = np.random.default_rng(17).normal(0,.5,(1,shape[1],shape[2],manifest["input_rows"],manifest["input_cols"])).astype(np.float16)
    prepared = np.ascontiguousarray(pixels.transpose(0,2,1,3,4).reshape(1,shape[1]*shape[2],manifest["input_rows"],manifest["input_cols"]))
    reference = reference_3d(pixels,weight)
    reference_check = check_reference(pixels,weight,reference)
    prepared.tofile(directory/"input.fp16.bin")
    np.ascontiguousarray(pixels.transpose(0,2,3,4,1)).tofile(directory/"gpu_input.fp16.bin")
    np.ascontiguousarray(weight.transpose(0,2,3,4,1)).tofile(directory/"gpu_weight.fp16.bin")
    np.ascontiguousarray(reference,dtype="<f4").tofile(directory/"reference.fp32.bin")
    fixture = dict(schema="seedvr2-vae-native-io-fixture-v1",cin=shape[1]*shape[2],cout=manifest["output_channels"],
        input_rows=manifest["input_rows"],input_cols=manifest["input_cols"],rows=manifest["rows"],cols=manifest["cols"],
        input=str((directory/"input.fp16.bin").resolve()),reference=str((directory/"reference.fp32.bin").resolve()),
        gpu_input=str((directory/"gpu_input.fp16.bin").resolve()),gpu_weight=str((directory/"gpu_weight.fp16.bin").resolve()),
        cin_per_tap=shape[1],taps=shape[2],ks=shape[3],
        iterations=iterations)
    return pixels,weight,reference,reference_check,fixture


def gpu_controls(pixels,weight,reference,iterations):
    import numpy as np
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    x3d = mx.array(np.ascontiguousarray(pixels.transpose(0,2,3,4,1)))
    w3d = mx.array(np.ascontiguousarray(weight.transpose(0,2,3,4,1)))
    w2d = mx.array(np.ascontiguousarray(temporal_conv2d_weights(weight,weight.shape[0]).transpose(0,2,3,1)))
    def pack():
        return mx.reshape(mx.contiguous(mx.transpose(x3d,(0,2,3,1,4))),
            (1,pixels.shape[3],pixels.shape[4],pixels.shape[1]*pixels.shape[2]))
    x2d = pack()
    mx.eval(x3d,w3d,w2d,x2d)
    routes = []
    for name in ("gpu_conv3d_resident","gpu_conv2d_prepacked","gpu_conv2d_pack_each_call"):
        def predict():
            return mx.conv3d(x3d,w3d) if name=="gpu_conv3d_resident" else mx.conv2d(x2d if name=="gpu_conv2d_prepacked" else pack(),w2d)
        samples = []
        for repetition in range(iterations+2):
            start = time.perf_counter(); result = predict(); mx.eval(result)
            seconds = time.perf_counter()-start
            if repetition>=2:samples.append(seconds)
        start = time.perf_counter()
        actual = np.asarray(result)
        if name=="gpu_conv3d_resident":actual = actual[:,0]
        actual = actual.transpose(0,3,1,2).copy()
        readback = time.perf_counter()-start
        routes.append(dict(route=name,warm_seconds=samples,warm_median_seconds=statistics.median(samples),
            output_readback_seconds=readback,quality=quality(reference,actual),
            scope="resident FP16 GPU operands; Python dispatch and completion included; output readback reported separately"))
    return routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,action="append",required=True)
    parser.add_argument("--probe",type=Path,required=True)
    parser.add_argument("--gpu-probe",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--iterations",type=int,default=5)
    parser.add_argument("--reverse-routes",action="store_true",help="reverse CPU/NE and IO order for an order-sensitivity screen")
    args = parser.parse_args()
    if args.output.exists() or not 3<=args.iterations<=30:parser.error("fresh output directory and 3...30 iterations required")
    if sha256(args.checkpoint)!=CHECKPOINT_SHA:parser.error("original VAE checkpoint identity required")
    probe = args.probe.resolve(strict=True)
    gpu_probe = args.gpu_probe.resolve(strict=True) if args.gpu_probe else None
    manifests = []
    for path in args.manifest:
        manifest = json.loads(path.read_text())
        compiled = path.parent/manifest["compiled"]["path"]
        validate_manifest(manifest)
        if tree_identity(compiled)!=manifest["compiled"]["identity"]:parser.error("compiled bank changed")
        manifests.append((path,manifest,compiled))
    args.output.mkdir(parents=True)
    report = dict(schema="seedvr2-vae-native-io-screen-v1",status="running",qualified=False,
        probe_sha256=sha256(probe),source_sha256=sha256(Path(__file__)),checkpoint_sha256=CHECKPOINT_SHA,
        gpu_probe_sha256=sha256(gpu_probe) if gpu_probe else None,reverse_routes=args.reverse_routes,
        hardware=dict(chip=subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip(),
            memory_bytes=int(subprocess.check_output(["sysctl","-n","hw.memsize"],text=True)),
            macos=subprocess.check_output(["sw_vers","-productVersion"],text=True).strip()),tiles=[],
        physical_ane_occupancy_measured=False,
        scope="synthetic normalized prepared tiles; no bias, full frame, real activation, full VAE or overlap qualification; Core ML internal copies not excluded by backing identity")
    try:
        for index,(path,manifest,compiled) in enumerate(manifests):
            directory = args.output/f"tile-{index}"
            directory.mkdir()
            print(f"preparing {manifest['rows']}x{manifest['cols']} cout={manifest['output_channels']}",flush=True)
            pixels,weight,reference,reference_check,fixture = prepare_fixture(manifest,args.checkpoint,directory,args.iterations)
            fixture["compiled"] = str(compiled.resolve())
            write_json(directory/"fixture.json",fixture)
            item = dict(layer=manifest["layer"],rows=manifest["rows"],cols=manifest["cols"],cout=manifest["output_channels"],
                cin=fixture["cin"],manifest=str(path),manifest_sha256=sha256(path),reference_verification=reference_check,
                fixture_sha256=sha256(directory/"fixture.json"),input_sha256=sha256(directory/"input.fp16.bin"),
                reference_sha256=sha256(directory/"reference.fp32.bin"),
                macs=manifest["rows"]*manifest["cols"]*weight.size,
                gpu_routes=gpu_controls(pixels,weight,reference,args.iterations),native_routes=[])
            if gpu_probe:
                item["native_gpu_routes"] = []
                gpu_routes = ["conv3d","conv2d","pack_conv2d"]
                if fixture["taps"]==1 and fixture["ks"]==1:gpu_routes.append("matmul")
                for route in (list(reversed(gpu_routes)) if args.reverse_routes else gpu_routes):
                    command = [str(gpu_probe),str(directory/"fixture.json"),route]
                    process = subprocess.run(command,capture_output=True,text=True,timeout=180)
                    receipt = dict(command=command,exit_code=process.returncode,stderr=process.stderr)
                    if process.returncode==0:receipt["observation"] = json.loads(process.stdout)
                    else:receipt.update(status="failed",stdout=process.stdout)
                    item["native_gpu_routes"].append(receipt)
            report["tiles"].append(item)
            write_json(args.output/"report.json",report)
            policies = ["cpu","ne"]
            io_routes = [("array","none"),("array","backing"),("wrapped","backing"),("surface","backing")]
            for policy in (list(reversed(policies)) if args.reverse_routes else policies):
                for mode,backing in (list(reversed(io_routes)) if args.reverse_routes else io_routes):
                    print(f"native {policy}/{mode}/{backing}",flush=True)
                    command = [str(probe),str(directory/"fixture.json"),policy,mode,backing]
                    process = subprocess.run(command,capture_output=True,text=True,timeout=180)
                    receipt = dict(command=command,exit_code=process.returncode,stderr=process.stderr)
                    if process.returncode==0:
                        receipt["observation"] = json.loads(process.stdout)
                    else:
                        receipt.update(status="failed",stdout=process.stdout)
                    item["native_routes"].append(receipt)
                    write_json(args.output/"report.json",report)
            item["quality_passed"] = all(route["quality"]["passed"] for route in item["gpu_routes"]) and all(
                route.get("observation",{}).get("quality",{}).get("passed",False)
                for route in item["native_routes"]+item.get("native_gpu_routes",[]))
            if tree_identity(compiled)!=manifest["compiled"]["identity"]:raise ValueError("compiled bank changed during screen")
        report.update(status="complete",component_quality_passed=all(item["quality_passed"] for item in report["tiles"]))
        print(json.dumps(dict(status="complete",tiles=len(report["tiles"]),component_quality_passed=report["component_quality_passed"])),flush=True)
    except BaseException as error:
        report.update(status="failed",failure=str(error));raise
    finally:
        write_json(args.output/"report.json",report)


if __name__=="__main__":main()
