#!/usr/bin/env python3
"""Explicit FP32 VAE-only oracle on pinned local Comfy, no user service/patches.

Older restore-oracle dtype=fp32 describes the DiT, not its FP16 VAE. This tool
verifies every floating dispatch tensor and parameter is FP32 during decode.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from seedvr2_framework_screen import sha256, write_json
from seedvr2_oracle import configure_comfy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("model","input","comfy","comfy-extra","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","mps"),default="mps")
    parser.add_argument("--output-storage",choices=("device","cpu"),default="device",
        help="use the existing full-context output-buffer API to avoid accumulating GPU slice outputs")
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh oracle output required")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):parser.error("automatic MPS fallback is not an FP32 oracle")
    for name in ("model","input","comfy","comfy_extra"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output.mkdir(parents=True)
    checkpoint=args.model/"vae/seedvr2_ema_vae_fp16.safetensors"
    source=args.comfy/"comfy/ldm/seedvr/vae.py"
    report=dict(schema="seedvr2-vae-fp32-decoder-oracle-v1",status="running",device=args.device,
        checkpoint_sha256=sha256(checkpoint),input_sha256=sha256(args.input),comfy_vae_sha256=sha256(source),
        comfy_commit=subprocess.check_output(["git","-C",str(args.comfy),"rev-parse","HEAD"],text=True).strip(),
        comfy_patch_sha256=hashlib.sha256(subprocess.check_output(["git","-C",str(args.comfy),"diff","--"])).hexdigest(),
        oracle_source_sha256=sha256(Path(__file__)),output_storage=args.output_storage,
        scope="VAE-only original-FP16 coefficients promoted to FP32, same latent; all floating dispatch tensors checked; no server/full-request/performance claim")
    write_json(args.output/"report.json",report)
    try:
        if report["checkpoint_sha256"]!="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1":
            raise ValueError("original VAE identity required")
        import torch
        from safetensors.torch import load_file,save_file
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves
        report["torch_version"]=torch.__version__
        if args.device=="mps" and not torch.backends.mps.is_available():raise ValueError("MPS unavailable; no CPU fallback")
        configure_comfy(args.comfy,args.comfy_extra,args.device,"sdpa")
        from comfy.ldm.seedvr.vae import VideoAutoencoderKLWrapper
        class FP32Dispatch(TorchDispatchMode):
            def __init__(self):super().__init__();self.calls=0;self.floating_checked=0
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs={} if kwargs is None else kwargs
                result=func(*args,**kwargs)
                self.calls+=1
                for value in tree_leaves((args,kwargs,result)):
                    if isinstance(value,torch.Tensor) and value.is_floating_point():
                        if value.dtype!=torch.float32:raise ValueError(f"non-FP32 VAE dispatch {func}: {value.dtype}")
                        self.floating_checked+=1
                return result
        model=VideoAutoencoderKLWrapper()
        state=load_file(str(checkpoint),device="cpu");model.load_state_dict(state,strict=True,assign=True);del state
        model.eval().to(device=args.device,dtype=torch.float32)
        parameter_types=sorted({str(value.dtype) for value in model.parameters()})
        if parameter_types!=["torch.float32"]:raise ValueError("VAE parameters did not promote exactly to FP32")
        latent=load_file(str(args.input),device="cpu")["latent"].to(device=args.device,dtype=torch.float32)
        output_buffer=None
        if args.output_storage=="cpu":
            shape=model.decoded_shape(latent.shape) if hasattr(model,"decoded_shape") else (
                latent.shape[0],3,1+4*(latent.shape[2]-1),latent.shape[3]*8,latent.shape[4]*8)
            output_buffer=torch.empty(shape,device="cpu",dtype=torch.float32)
        report.update(parameter_dtypes=parameter_types,latent_shape=list(latent.shape),stage="decode_started")
        write_json(args.output/"report.json",report)
        audit=FP32Dispatch();started=time.perf_counter()
        with torch.inference_mode(),audit:pixels=model.decode(latent,output_buffer=output_buffer) if output_buffer is not None else model.decode(latent)
        if args.device=="mps":torch.mps.synchronize()
        if pixels.dtype!=torch.float32 or not torch.isfinite(pixels).all().item():raise ValueError("nonfinite/non-FP32 VAE output")
        save_file(dict(pixels=pixels.cpu().contiguous(),latent=latent.cpu().contiguous()),str(args.output/"decode.safetensors"))
        report.update(status="complete",decode_seconds_with_audit=time.perf_counter()-started,
            parameter_dtypes=parameter_types,output_dtype=str(pixels.dtype),output_shape=list(pixels.shape),
            floating_dispatch_values_checked=audit.floating_checked,dispatch_calls=audit.calls,output_sha256=sha256(args.output/"decode.safetensors"))
        print(json.dumps(report),flush=True)
    except BaseException as failure:report.update(status="failed",failure=str(failure));raise
    finally:write_json(args.output/"report.json",report)


if __name__=="__main__":main()
