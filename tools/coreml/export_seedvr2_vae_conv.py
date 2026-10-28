#!/usr/bin/env python3
"""Offline, identity-bound VAE convolution probe; not a production VAE ANE route.

Stack the original temporal taps into Conv2D input channels. Spatial halos are
supplied explicitly, and bias/residual/normalization remain outside the bank.
Core ML reduction order/FP16 arithmetic requires independent qualification.
"""
import argparse
import json
from pathlib import Path
import shutil

from export_ltx_gemma_mlp import mapped_tensor, read_header, tensor_info
from export_seedvr2_ffn import sha256, tree_identity

CHECKPOINT_SHA = "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"
SCHEMA = "turbocider-seedvr2-vae-conv-probe-v1"
# A 3x3 valid convolution adds two halo columns. Include extents whose
# halo-expanded FP16 rows can avoid the observed 66 -> 96-element pitch.
# These are experimental banks, not a portable performance recommendation.
FIXED_TILE_EXTENTS = (32, 62, 64, 94, 128, 256)
CAUSAL_TILE_EXTENTS = FIXED_TILE_EXTENTS[:-1]


def temporal_conv2d_weights(value, channels):
    import numpy as np
    if value.ndim != 5 or not 0 < channels <= value.shape[0]:
        raise ValueError("original O,C,T,H,W filter and valid output prefix required")
    if value.dtype != np.float16:
        raise ValueError("original FP16 coefficients required")
    out, cin, taps, kh, kw = value.shape
    if taps not in (1,3) or kh != kw or kh not in (1,3):
        raise ValueError("probe supports original singleton/three-tap 1x1/3x3 filters")
    return np.ascontiguousarray(value[:channels].transpose(0,2,1,3,4).reshape(channels,taps*cin,kh,kw))


def build_model(weight, rows, cols):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    halo = weight.shape[2]-1

    @mb.program(input_specs=[mb.TensorSpec(shape=(1,weight.shape[1],rows+halo,cols+halo),dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def program(x):
        return mb.conv(x=x,weight=weight,pad_type="valid",strides=[1,1],name="y")

    # No bias here: a fused Core ML bias can change the native recipe's FP16
    # conv-output rounding before its bias add. GPU assembly handles that add.
    model = ct.convert(program,convert_to="mlprogram",minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16,skip_model_load=True)
    model.user_defined_metadata.update({"turbocider.schema":SCHEMA,"turbocider.bias_applied":"false",
                                       "turbocider.source_operand_recipe":"original-fp16-temporal-channel-stack"})
    return model


def export(args):
    import coremltools as ct
    import numpy as np
    checkpoint = args.checkpoint.resolve(strict=True)
    if sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("original SeedVR2 VAE checkpoint SHA-256 required")
    if args.output.exists():
        raise ValueError("fresh VAE convolution probe directory required")
    offset,header = read_header(checkpoint)
    name = args.layer+".weight"
    if name not in header or header[name].get("dtype") != "F16":
        raise ValueError("named original FP16 VAE convolution not found")
    shape = tuple(header[name]["shape"])
    info = tensor_info(header,name,"F16",shape)
    original = mapped_tensor(checkpoint,offset,info,np.float16,shape)
    channels = args.channels or shape[0]
    weight = temporal_conv2d_weights(original,channels)
    model = build_model(weight,args.rows,args.cols)
    args.output.mkdir(parents=True)
    package = args.output/"conv.mlpackage"
    compiled = args.output/"conv.mlmodelc"
    model.save(str(package))
    loaded = ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_AND_NE)
    shutil.copytree(Path(loaded.get_compiled_model_path()),compiled)
    manifest = dict(schema=SCHEMA,checkpoint_sha256=CHECKPOINT_SHA,checkpoint_bytes=checkpoint.stat().st_size,
        layer=args.layer,original_shape=list(shape),source_operand_recipe="original-fp16-temporal-channel-stack",
        input_layout="N(T*C)HW",output_layout="NOHW",temporal_taps=shape[2],input_channels=shape[1],
        output_channels=channels,total_output_channels=shape[0],rows=args.rows,cols=args.cols,
        input_rows=args.rows+shape[3]-1,input_cols=args.cols+shape[4]-1,
        bias_applied=False,precision="fp16",compute_units="cpuAndNeuralEngine",qualified=False,
        qualification_scope="offline convolution probe only; no VAE routing, quality/speed gate or physical ANE occupancy proof",
        compiled=dict(path=compiled.name,identity=tree_identity(compiled)))
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({key:manifest[key] for key in ("schema","layer","original_shape","output_channels","qualified")}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--layer",default="decoder.up_blocks.3.resnets.0.conv1")
    parser.add_argument("--channels",type=int,default=64)
    parser.add_argument("--rows",type=int,default=128)
    parser.add_argument("--cols",type=int,default=128)
    args = parser.parse_args()
    if args.channels < 0 or args.rows not in FIXED_TILE_EXTENTS or args.cols not in FIXED_TILE_EXTENTS:
        parser.error("nonnegative prefix (0=all) and fixed 32/62/64/94/128/256 spatial extents required")
    export(args)


if __name__ == "__main__": main()
