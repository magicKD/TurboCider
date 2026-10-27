#!/usr/bin/env python3
"""Pinned local original-coefficient FP32 encoder reference; no user service.

Every floating dispatch input/output and parameter is checked as FP32. Same
normalized input file, including native's documented FP16-input/scaling error.
No independent crop/temporal tiling or automatic MPS fallback is allowed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from seedvr2_framework_screen import sha256,write_json
from seedvr2_oracle import configure_comfy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("model","input","comfy","comfy-extra","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","mps"),default="mps")
    args=parser.parse_args()
    if args.output.exists() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        parser.error("fresh output and no automatic MPS fallback required")
    for name in ("model","input","comfy","comfy_extra"):setattr(args,name,getattr(args,name).resolve(strict=True))
    checkpoint=args.model/"vae/seedvr2_ema_vae_fp16.safetensors";source=args.comfy/"comfy/ldm/seedvr/vae.py"
    args.output.mkdir(parents=True)
    report=dict(schema="seedvr2-vae-fp32-encoder-oracle-v1",status="running",device=args.device,
        checkpoint_sha256=sha256(checkpoint),input_sha256=sha256(args.input),comfy_vae_sha256=sha256(source),
        comfy_commit=subprocess.check_output(["git","-C",str(args.comfy),"rev-parse","HEAD"],text=True).strip(),
        comfy_patch_sha256=hashlib.sha256(subprocess.check_output(["git","-C",str(args.comfy),"diff","--"])).hexdigest(),
        oracle_source_sha256=sha256(Path(__file__)),scope="complete same-input VAE encoder, original FP16 coefficients promoted to FP32, all floating dispatch tensors audited; no performance/App/DiT claim")
    write_json(args.output/"report.json",report)
    try:
        if report["checkpoint_sha256"]!="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1":
            raise ValueError("original VAE checkpoint identity required")
        import torch
        from safetensors.torch import load_file,save_file
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves
        if args.device=="mps" and not torch.backends.mps.is_available():raise ValueError("MPS unavailable; no CPU fallback")
        configure_comfy(args.comfy,args.comfy_extra,args.device,"sdpa")
        from comfy.ldm.seedvr.vae import VideoAutoencoderKLWrapper
        class Audit(TorchDispatchMode):
            def __init__(self):super().__init__();self.calls=0;self.checked=0
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs={} if kwargs is None else kwargs;result=func(*args,**kwargs);self.calls+=1
                for value in tree_leaves((args,kwargs,result)):
                    if isinstance(value,torch.Tensor) and value.is_floating_point():
                        if value.dtype!=torch.float32:raise ValueError(f"non-FP32 encoder dispatch {func}: {value.dtype}")
                        self.checked+=1
                return result
        model=VideoAutoencoderKLWrapper();state=load_file(str(checkpoint),device="cpu")
        model.load_state_dict(state,strict=True,assign=True);del state
        model.eval().to(device=args.device,dtype=torch.float32)
        kinds=sorted({str(value.dtype) for value in model.parameters()})
        if kinds!=["torch.float32"]:raise ValueError("encoder parameters did not promote to FP32")
        pixels=load_file(str(args.input),device="cpu")["pixels"].to(dtype=torch.float32)
        if pixels.ndim!=5 or pixels.shape[:2]!=(1,3) or (pixels.shape[2]-1)%4 or not torch.isfinite(pixels).all().item():
            raise ValueError("complete finite normalized RGB BCTHW input required")
        # Keep source storage on CPU; the existing wrapper moves each original
        # initial5/subsequent4 slice to MPS while preserving its causal cache.
        report.update(torch_version=torch.__version__,parameter_dtypes=kinds,input_shape=list(pixels.shape),
            input_dtype=str(pixels.dtype),source_storage="cpu",stage="encode_started")
        write_json(args.output/"report.json",report);audit=Audit();begin=time.perf_counter()
        with torch.inference_mode(),audit:
            latent=model.comfy_format_encoded(model.encode(pixels,device=torch.device(args.device)))
        if args.device=="mps":torch.mps.synchronize()
        if latent.dtype!=torch.float32 or not torch.isfinite(latent).all().item():raise ValueError("nonfinite/non-FP32 encoder output")
        expected=[pixels.shape[0],16,(pixels.shape[2]-1)//4+1,pixels.shape[3]//8,pixels.shape[4]//8]
        if list(latent.shape)!=expected:raise ValueError("encoder reference did not retain complete frame/spatial shape")
        save_file(dict(latent=latent.cpu().contiguous()),str(args.output/"encode.safetensors"))
        report.update(status="complete",stage="encode_complete",encode_seconds_with_audit=time.perf_counter()-begin,
            output_dtype=str(latent.dtype),output_shape=list(latent.shape),floating_dispatch_values_checked=audit.checked,
            dispatch_calls=audit.calls,output_sha256=sha256(args.output/"encode.safetensors"))
        print(json.dumps(report),flush=True)
    except BaseException as failure:report.update(status="failed",failure=str(failure));raise
    finally:write_json(args.output/"report.json",report)


if __name__=="__main__":main()
