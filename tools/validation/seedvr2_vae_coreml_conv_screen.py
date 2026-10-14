#!/usr/bin/env python3
"""Fixed prepared-tile convolution feasibility; not whole-VAE/ANE qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/coreml"))
from export_ltx_gemma_mlp import mapped_tensor, read_header, tensor_info
from export_seedvr2_ffn import sha256, tree_identity
from export_seedvr2_vae_conv import CHECKPOINT_SHA, SCHEMA
from seedvr2_framework_screen import write_json


def reference_3d(pixels,weight):
    import numpy as np
    # Original NCTHW/OCTHW ordering, FP32 arithmetic, bounded host columns.
    n,c,t,h,w = pixels.shape
    out,_,kt,kh,kw = weight.shape
    view = np.lib.stride_tricks.sliding_window_view(pixels.astype(np.float32),(kt,kh,kw),axis=(2,3,4))
    view = view.transpose(0,2,3,4,1,5,6,7)
    result = np.empty((n,t-kt+1,h-kh+1,w-kw+1,out),dtype=np.float32)
    matrix = weight.astype(np.float32).reshape(out,-1).T
    for row in range(0,result.shape[2],16):
        stop = min(row+16,result.shape[2])
        columns = np.ascontiguousarray(view[:,:,row:stop]).reshape(-1,matrix.shape[0])
        result[:,:,row:stop] = (columns@matrix).reshape(n,result.shape[1],stop-row,result.shape[3],out)
    return result[:,0].transpose(0,3,1,2)


def quality(reference,actual):
    import numpy as np
    if reference.shape != actual.shape: raise ValueError("convolution output shape changed")
    x,y = reference.astype(np.float64).ravel(),actual.astype(np.float64).ravel()
    finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
    if not finite: return dict(finite=False,passed=False,relative_l2=None,cosine=None)
    nx,ny = np.linalg.norm(x),np.linalg.norm(y)
    error = float(np.linalg.norm(x-y)/max(nx,1e-20))
    cosine = float(np.dot(x,y)/max(nx*ny,1e-20))
    return dict(finite=True,relative_l2=error,cosine=cosine,passed=error <= .05 and cosine >= .999)


def check_reference(pixels,weight,reference):
    import numpy as np
    if not np.isfinite(reference).all(): raise ValueError("nonfinite independent convolution reference")
    errors = []
    for channel in (0,weight.shape[0]//2,weight.shape[0]-1):
        for row in (0,reference.shape[2]//2,reference.shape[2]-1):
            for col in (0,reference.shape[3]//2,reference.shape[3]-1):
                patch = pixels[0,:,:,row:row+weight.shape[3],col:col+weight.shape[4]].astype(np.float64)
                expected = float(np.sum(patch*weight[channel].astype(np.float64)))
                error = abs(float(reference[0,channel,row,col])-expected)
                if error > 5e-5*max(1.,abs(expected)): raise ValueError("FP32 reference failed independent sampled FP64 verification")
                errors.append(error)
    return dict(finite=True,sampled_fp64_points=len(errors),sampled_fp64_max_abs_error=max(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--iterations",type=int,default=9)
    args = parser.parse_args()
    if args.output.exists() or not 3 <= args.iterations <= 30: parser.error("fresh output and 3...30 warm iterations required")
    manifest = json.loads(args.manifest.read_text())
    compiled = args.manifest.parent/manifest["compiled"]["path"]
    if manifest["schema"] != SCHEMA or sha256(args.checkpoint) != CHECKPOINT_SHA:
        parser.error("identity-bound original VAE convolution probe required")
    if tree_identity(compiled) != manifest["compiled"]["identity"]: parser.error("compiled bank identity changed")
    import numpy as np
    import coremltools as ct
    import mlx.core as mx
    offset,header = read_header(args.checkpoint)
    shape = tuple(manifest["original_shape"])
    info = tensor_info(header,manifest["layer"]+".weight","F16",shape)
    weight = np.ascontiguousarray(mapped_tensor(args.checkpoint,offset,info,np.float16,shape)[:manifest["output_channels"]])
    rng = np.random.default_rng(17)
    pixels = rng.normal(0,.5,(1,shape[1],shape[2],manifest["input_rows"],manifest["input_cols"])).astype(np.float16)
    prepared = np.ascontiguousarray(pixels.transpose(0,2,1,3,4).reshape(1,shape[1]*shape[2],manifest["input_rows"],manifest["input_cols"]))
    reference = reference_3d(pixels,weight)
    reference_check = check_reference(pixels,weight,reference)
    report = dict(schema="seedvr2-vae-conv-feasibility-screen-v1",status="running",qualified=False,
        manifest_sha256=sha256(args.manifest),input_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),routes=[],
        scope="synthetic normalized FP16 prepared tile, no bias; GPU readback/CoreML Python copies included; packing, full VAE and cooperative routing excluded",
        physical_ane_occupancy_measured=False,reference_verification=reference_check)
    try:
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(str(compiled),compute_units=ct.ComputeUnit.CPU_AND_NE)
        placement = []
        for function in plan.model_structure.program.functions.values():
            for op in function.block.operations:
                if op.operator_name.rsplit(".",1)[-1] != "conv": continue
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                placement.append(dict(operator=op.operator_name,preferred=type(usage.preferred_compute_device).__name__,
                                      supported=[type(device).__name__ for device in usage.supported_compute_devices]))
        report["anticipated_placement"] = placement
    except Exception as error: report["anticipated_placement_error"] = str(error)
    try:
        for route in ("gpu","cpu","cpu_ne"):
            started = time.perf_counter()
            if route == "gpu":
                x = mx.array(np.ascontiguousarray(pixels.transpose(0,2,3,4,1)))
                w = mx.array(np.ascontiguousarray(weight.transpose(0,2,3,4,1)))
                mx.eval(x,w)
                def predict():
                    result = mx.conv3d(x,w)
                    mx.eval(result)
                    return np.asarray(result)[:,0].transpose(0,3,1,2).copy()
            else:
                model = ct.models.CompiledMLModel(str(compiled),compute_units=ct.ComputeUnit.CPU_ONLY if route == "cpu" else ct.ComputeUnit.CPU_AND_NE)
                def predict(): return model.predict({"x":prepared})["y"]
            load_seconds = time.perf_counter()-started
            started = time.perf_counter(); actual = predict(); cold = time.perf_counter()-started
            times = []
            for _ in range(args.iterations):
                started = time.perf_counter(); actual = predict(); times.append(time.perf_counter()-started)
            report["routes"].append(dict(route=route,load_seconds=load_seconds,cold_seconds=cold,warm_seconds=times,
                                          warm_median_seconds=statistics.median(times),quality=quality(reference,actual)))
        report.update(status="complete",component_quality_passed=all(row["quality"]["passed"] for row in report["routes"]))
        print(json.dumps(report),flush=True)
    except BaseException as error:
        report.update(status="failed",failure=str(error)); raise
    finally: write_json(args.output,report)


if __name__ == "__main__": main()
