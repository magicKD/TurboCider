#!/usr/bin/env python3
"""Original-FP16 spatial projection/shuffle probe, never a production VAE bank.

The original channel order is [spatial_row, spatial_col, channel], not the
usual channel-first depth-to-space order. Bias remains outside the graph so
the materialized FP16 projection boundary precedes a separate FP16 add.
"""
import argparse
import json
from pathlib import Path
import shutil
from export_ltx_gemma_mlp import mapped_tensor, read_header, tensor_info
from export_seedvr2_ffn import sha256, tree_identity
from export_seedvr2_vae_conv import CHECKPOINT_SHA

SCHEMA = "turbocider-seedvr2-vae-upscale-probe-v1"
LAYER = "decoder.up_blocks.2.upsamplers.0.upscale_conv"


def projection_weights(value):
    import numpy as np
    if value.dtype != np.float16 or value.shape != (1024,256,1,1,1):
        raise ValueError("original 256-to-1024 FP16 spatial projection required")
    return np.ascontiguousarray(value[:,:,0,:,:])


def spatial_shuffle(value):
    """Independent NumPy index witness; input/output NCHW, no bias here."""
    import numpy as np
    if value.ndim != 4 or value.shape[1] % 4:
        raise ValueError("complete four-phase NCHW projection required")
    b, expanded, h, w = value.shape
    return np.ascontiguousarray(value.reshape(b,2,2,expanded//4,h,w)
        .transpose(0,3,4,1,5,2).reshape(b,expanded//4,h*2,w*2))


def build_model(weight, rows, cols, variant):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    if variant not in ("projection","shuffle"):
        raise ValueError("known fixed projection boundary required")
    @mb.program(input_specs=[mb.TensorSpec(shape=(1,256,rows,cols),dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def program(x):
        y = mb.conv(x=x,weight=weight,pad_type="valid",strides=[1,1],name="projection")
        if variant == "shuffle":
            y = mb.reshape(x=y,shape=[1,2,2,256,rows,cols])
            y = mb.transpose(x=y,perm=[0,3,4,1,5,2])
            y = mb.reshape(x=y,shape=[1,256,rows*2,cols*2])
        return mb.identity(x=y,name="y")
    return ct.convert(program,convert_to="mlprogram",minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,skip_model_load=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--rows",type=int,choices=(32,64,128),default=64)
    parser.add_argument("--cols",type=int,choices=(32,64,128),default=64)
    parser.add_argument("--variant",choices=("projection","shuffle"),default="projection")
    args=parser.parse_args()
    if args.output.exists() or sha256(args.checkpoint)!=CHECKPOINT_SHA:
        parser.error("fresh output and original checkpoint identity required")
    import coremltools as ct
    import numpy as np
    offset,header=read_header(args.checkpoint)
    info=tensor_info(header,LAYER+".weight","F16",(1024,256,1,1,1))
    original=mapped_tensor(args.checkpoint,offset,info,np.float16,(1024,256,1,1,1))
    model=build_model(projection_weights(original),args.rows,args.cols,args.variant)
    args.output.mkdir(parents=True)
    package=args.output/"upscale.mlpackage";compiled=args.output/"upscale.mlmodelc"
    model.save(str(package))
    loaded=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_AND_NE)
    shutil.copytree(Path(loaded.get_compiled_model_path()),compiled)
    expanded=args.variant=="shuffle"
    manifest=dict(schema=SCHEMA,qualified=False,checkpoint_sha256=CHECKPOINT_SHA,layer=LAYER,
        original_shape=[1024,256,1,1,1],precision="fp16",variant=args.variant,
        rows=args.rows,cols=args.cols,input_shape=[1,256,args.rows,args.cols],
        output_shape=[1,256,args.rows*2,args.cols*2] if expanded else [1,1024,args.rows,args.cols],
        channel_order="spatial_row_spatial_col_channel",bias_applied=False,
        compute_units="cpuAndNeuralEngine",compiled=dict(path=compiled.name,identity=tree_identity(compiled)),
        scope="offline fixed spatial projection/shuffle, separate FP16 bias on GPU; not VAE/full-request/ANE-occupancy qualification")
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest),flush=True)


if __name__=="__main__":main()
