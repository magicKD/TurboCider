#!/usr/bin/env python3
"""Qualify existing complete decoder exports against a verified FP32 VAE oracle.

Creates a separate receipt; never relabels or changes an earlier run/failed gate.
"""
import argparse
import json
from pathlib import Path
from seedvr2_framework_screen import sha256,write_json
from seedvr2_vae_coreml_conv_screen import quality


def validate_oracle(screen,oracle):
    if (screen.get("status")!="complete" or oracle.get("status")!="complete" or
        oracle.get("schema")!="seedvr2-vae-fp32-decoder-oracle-v1" or
        oracle.get("parameter_dtypes")!=["torch.float32"] or oracle.get("output_dtype")!="torch.float32" or
        oracle.get("floating_dispatch_values_checked",0)<=0 or
        oracle.get("checkpoint_sha256")!=screen.get("checkpoint_sha256") or
        oracle.get("input_sha256")!=screen.get("prefix_fixture_sha256")):
        raise ValueError("complete same-input all-FP32 VAE oracle required")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("screen","oracle","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--per-frame",action="store_true",help="add unchanged per-frame finite/cosine/relative-L2 gates")
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh separate FP32 comparison receipt required")
    screen=json.loads(args.screen.read_text());oracle=json.loads(args.oracle.read_text());validate_oracle(screen,oracle)
    fixture=args.screen.parent/"prefix.safetensors";reference=args.oracle.parent/"decode.safetensors"
    if sha256(fixture)!=screen["prefix_fixture_sha256"] or sha256(reference)!=oracle["output_sha256"]:
        parser.error("same latent prefix/oracle output identity changed")
    import numpy as np
    from safetensors.numpy import load_file
    expected=load_file(str(reference))["pixels"]
    if expected.dtype!=np.float32 or list(expected.shape)!=oracle["output_shape"]:parser.error("oracle datatype/shape differs")
    report=dict(schema="seedvr2-spatial-same-latent-fp32-quality-v1",status="running",routes=[],
        screen_sha256=sha256(args.screen),oracle_sha256=sha256(args.oracle),original_checkpoint_sha256=oracle["checkpoint_sha256"],
        scope="all emitted pixels against original-coefficient all-FP32 VAE on exactly the same latent; no DiT/speed/App/low-RAM/default qualification")
    try:
        expected=np.ascontiguousarray(expected[0].transpose(1,2,3,0))
        if expected.shape[0]!=screen["frames"]:raise ValueError("oracle does not cover every emitted frame")
        for row in screen["routes"]:
            if row["exit_code"] or not row["observation"]["quality_passed"]:raise ValueError("decoder route is not a completed pixel gate")
            path=Path(row["command"][3])/"decoded.fp16.bin"
            if sha256(path)!=row["export_sha256"] or path.stat().st_size!=expected.size*2:raise ValueError("decoder export identity/extent changed")
            actual=np.memmap(path,mode="r",dtype=np.float16,shape=expected.shape)
            values=quality(expected,actual)
            if args.per_frame:
                values["per_frame"]=[dict(frame=index,**quality(expected[index],actual[index])) for index in range(expected.shape[0])]
                if not all(frame["passed"] for frame in values["per_frame"]):
                    raise ValueError("unchanged original-FP32 per-frame VAE quality gate failed")
            report["routes"].append(dict(route=row["route"],**values))
            if not values["passed"]:raise ValueError("unchanged .999-cosine/.05-relative-L2 original-FP32 VAE quality gate failed")
        report["status"]="complete";print(json.dumps(report),flush=True)
    except BaseException as failure:report.update(status="failed",failure=str(failure));raise
    finally:write_json(args.output,report)


if __name__=="__main__":main()
