#!/usr/bin/env python3
"""Separate whole-latent gate against an audited same-input FP32 encoder."""
import argparse
import json
from pathlib import Path
from seedvr2_framework_screen import sha256,write_json
from seedvr2_vae_coreml_conv_screen import quality

def validate_reference(screen,reference):
    if (screen.get("status")!="complete" or reference.get("status")!="complete" or
        reference.get("schema")!="seedvr2-vae-fp32-encoder-oracle-v1" or
        reference.get("parameter_dtypes")!=["torch.float32"] or reference.get("input_dtype")!="torch.float32" or
        reference.get("output_dtype")!="torch.float32" or reference.get("floating_dispatch_values_checked",0)<=0 or
        reference.get("checkpoint_sha256")!=screen.get("checkpoint_sha256") or
        reference.get("input_sha256")!=screen.get("input_sha256") or reference.get("input_shape")!=screen.get("source_shape")):
        raise ValueError("complete same-input audited FP32 encoder reference required")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("screen","reference","output"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh independent comparison receipt required")
    screen=json.loads(args.screen.read_text());reference=json.loads(args.reference.read_text());validate_reference(screen,reference)
    path=args.reference.parent/"encode.safetensors"
    if sha256(path)!=reference["output_sha256"]:parser.error("reference output identity changed")
    import numpy as np
    from safetensors.numpy import load_file
    expected=load_file(str(path))["latent"]
    if expected.dtype!=np.float32 or list(expected.shape)!=reference["output_shape"]:parser.error("reference dtype/shape differs")
    report=dict(schema="seedvr2-spatial-encode-fp32-quality-v1",status="running",routes=[],
        screen_sha256=sha256(args.screen),reference_sha256=sha256(args.reference),checkpoint_sha256=screen["checkpoint_sha256"],
        scope="all encoded latent values, complete source, independent original-coefficient FP32 VAE; not DiT/request/App/performance qualification")
    try:
        for row in screen["routes"]:
            path=Path(row["command"][3])/"latent.safetensors"
            if row["exit_code"] or sha256(path)!=row["latent_sha256"]:raise ValueError("native encoder observation/export identity differs")
            values=quality(expected,load_file(str(path))["latent"]);report["routes"].append(dict(route=row["route"],**values))
            if not values["passed"]:raise ValueError("unchanged .05/.999 original-FP32 encoder gate failed")
        report["status"]="complete";print(json.dumps(report),flush=True)
    except BaseException as failure:report.update(status="failed",failure=str(failure));raise
    finally:write_json(args.output,report)

if __name__=="__main__":main()
