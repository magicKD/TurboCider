#!/usr/bin/env python3
"""Time-embedding operator diagnostics on identical pinned-Comfy inputs.

This isolates rounding inside each operator; it is not full-model quality or
performance qualification. Reference tensors come from non-mutating hooks.
"""
import argparse
import json
import math
from pathlib import Path


def mlx_leaf(args):
    import mlx.core as mx
    import numpy as np
    if args.output.exists():
        raise ValueError("fresh leaf output required")
    args.output.mkdir(parents=True)
    mx.set_default_device(mx.gpu)
    trace = mx.load(str(args.reference/"time-detail.safetensors"))
    weights = mx.load(str(args.checkpoint))
    output = {}
    for name,input_name in (("proj_in","raw"),("proj_hid","act_in"),("proj_out","act_hid")):
        x = trace[input_name]
        w = weights[f"emb_in.{name}.weight"].astype(mx.bfloat16)
        b = weights[f"emb_in.{name}.bias"].astype(mx.bfloat16)
        output[name+".addmm"] = mx.addmm(b,x,w.T)
        output[name+".separate_bias"] = x@w.T+b
        output[name+".fp32_addmm"] = mx.addmm(b.astype(mx.float32),x.astype(mx.float32),
            w.astype(mx.float32).T).astype(x.dtype)
        output[name+".fp32_separate_then_cast"] = (x.astype(mx.float32)@w.astype(mx.float32).T+
            b.astype(mx.float32)).astype(x.dtype)
        output[name+".fp32_matmul_cast_before_bias"] = (x.astype(mx.float32)@w.astype(mx.float32).T).astype(x.dtype)+b
    for name,input_name in (("act_in","proj_in"),("act_hid","proj_hid")):
        x = trace[input_name]
        f = x.astype(mx.float32)
        output[name+".fp32_multiply_sigmoid"] = (f*mx.sigmoid(f)).astype(x.dtype)
        output[name+".fp32_divide_exp"] = (f/(1+mx.exp(-f))).astype(x.dtype)
        output[name+".bf16_multiply_sigmoid"] = x*mx.sigmoid(x)
    exponent = -np.float32(np.log(10000.))*np.arange(128,dtype=np.float32)/np.float32(128)
    angles = np.exp(exponent)*np.float32(1000)
    output["raw.numpy_fp32"] = mx.array(np.concatenate((np.sin(angles),np.cos(angles)))[None,:],dtype=mx.bfloat16)
    exponent64 = -np.log(10000.)*np.arange(128,dtype=np.float64)/128.
    angles32 = np.exp(exponent64).astype(np.float32)*np.float32(1000)
    output["raw.host_fp64_frequency_fp32_angles"] = mx.array(
        np.concatenate((np.sin(angles32),np.cos(angles32)))[None,:],dtype=mx.bfloat16)
    mx.eval(*output.values())
    mx.save_safetensors(str(args.output/"leaf.safetensors"),output)
    print("isolated operator outputs saved",flush=True)


def compare(args):
    import numpy as np
    from safetensors.torch import load_file
    reference = load_file(str(args.reference/args.detail_file))
    actual = load_file(str(args.output/args.leaf_file))
    rows = []
    for name,value in sorted(actual.items()):
        expected = reference[name.split(".",1)[0]]
        if value.shape != expected.shape:
            raise ValueError("leaf shape mismatch")
        x,y = expected.float().numpy().astype(np.float64).ravel(),value.float().numpy().astype(np.float64).ravel()
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        if not finite:
            raise ValueError("nonfinite leaf")
        nx,ny = np.linalg.norm(x),np.linalg.norm(y)
        row = dict(name=name,unequal=int(np.count_nonzero(x!=y)),elements=int(x.size),
            relative_l2=float(np.linalg.norm(x-y)/max(nx,1e-20)),
            cosine=float(np.dot(x,y)/max(nx*ny,1e-20)),max_abs=float(np.max(np.abs(x-y))))
        rows.append(row)
        print(json.dumps(row),flush=True)
    report = dict(schema="seedvr2-time-leaf-diagnostic-v1",rows=rows,
        scope="isolated operators with identical inputs; not full-model quality or speed qualification")
    (args.output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")


def attention_leaf(args):
    import mlx.core as mx
    import numpy as np
    if args.output.exists():
        raise ValueError("fresh attention leaf output required")
    args.output.mkdir(parents=True)
    mx.set_default_device(mx.gpu)
    ref = mx.load(str(args.reference/"block0-detail.safetensors"))
    weights = mx.load(str(args.checkpoint))
    nv,nt = ref["attn_vid_qkv"].shape[0],ref["attn_txt_qkv"].shape[0]
    side = math.isqrt(nv)
    if side*side != nv:
        raise ValueError("attention leaf requires an image/square/single-window fixture")
    frequency = weights["blocks.0.attn.rope.rope.freqs"].astype(mx.float32)
    vv = ref["attn_vid_qkv"].reshape(nv,3,20,128)[:,2]
    tv = ref["attn_txt_qkv"].reshape(nt,3,20,128)[:,2]
    pack = lambda a,b:mx.concatenate((a,b),axis=0).transpose(1,0,2)[None,...]
    q,k,v = pack(ref["attn_vid_ropeq"],ref["attn_txt_ropeq"]),pack(
        ref["attn_vid_ropek"],ref["attn_txt_ropek"]),pack(vv,tv)
    scale = 1/math.sqrt(128)
    result = {}
    outputs = dict(fast_bf16=mx.fast.scaled_dot_product_attention(q,k,v,scale=scale),
        fast_fp32=mx.fast.scaled_dot_product_attention(q.astype(mx.float32),k.astype(mx.float32),v.astype(mx.float32),scale=scale).astype(q.dtype))
    for target in (128,256):
        if q.shape[-2] < target:
            padded = mx.concatenate((q,mx.zeros((1,20,target-q.shape[-2],128),dtype=q.dtype)),axis=-2)
            outputs[f"fast_padq{target}"] = mx.fast.scaled_dot_product_attention(padded,k,v,scale=scale)[...,:q.shape[-2],:]
    scores = (q@k.swapaxes(-1,-2)).astype(mx.float32)*scale
    outputs["bf16_score_fp32_softmax"] = (mx.softmax(scores,axis=-1)@v.astype(mx.float32)).astype(q.dtype)
    for tile in (16,32,64):
        maximum = mx.full(q.shape[:-1]+(1,),-float("inf"),dtype=mx.float32)
        total = mx.zeros(q.shape[:-1]+(1,),dtype=mx.float32)
        accumulator = mx.zeros(q.shape,dtype=mx.float32)
        scale2 = float(np.float32(scale)*np.float32(1.44269504089))
        for first in range(0,k.shape[-2],tile):
            scores = (q.astype(mx.float32)@k[...,first:first+tile,:].astype(mx.float32).swapaxes(-1,-2))*scale2
            updated = mx.maximum(maximum,mx.max(scores,axis=-1,keepdims=True))
            factor = mx.power(mx.array(2.,dtype=mx.float32),maximum-updated)
            probability = mx.power(mx.array(2.,dtype=mx.float32),scores-updated)
            accumulator = accumulator*factor + probability.astype(q.dtype).astype(mx.float32)@v[...,first:first+tile,:].astype(mx.float32)
            total = total*factor+mx.sum(probability,axis=-1,keepdims=True)
            maximum = updated
            mx.eval(accumulator,total,maximum)
        outputs[f"online_bf16_tile{tile}"] = (accumulator/total).astype(q.dtype)
    scores = (q.astype(mx.float32)@k.astype(mx.float32).swapaxes(-1,-2))*scale
    outputs["composite_fp32"] = (mx.softmax(scores,axis=-1)@v.astype(mx.float32)).astype(q.dtype)
    outputs["fp32_score_bf16_probability"] = mx.softmax(scores,axis=-1).astype(q.dtype)@v
    factor = math.sqrt(scale)
    scores = (q.astype(mx.float32)*factor)@(k.astype(mx.float32)*factor).swapaxes(-1,-2)
    outputs["prescale_fp32"] = (mx.softmax(scores,axis=-1)@v.astype(mx.float32)).astype(q.dtype)
    scores = (q@k.swapaxes(-1,-2))*scale
    outputs["composite_bf16"] = mx.softmax(scores,axis=-1)@v
    for name,out in outputs.items():
        flat = out[0].transpose(1,0,2).reshape(nv+nt,2560)
        result["attn_vid_raw."+name] = flat[:nv]
        result["attn_txt_raw."+name] = flat[nv:]
    kernel = mx.fast.metal_kernel("tc_seedvr2_rope_leaf_fma",["x","cs","sn"],["y"],r'''
        uint i=thread_position_in_grid.x;
        if (i >= uint(x_shape[0])*20*64) return;
        uint pair=i%64, token=i/(20*64), offset=i*2;
        if (pair==63) { y[offset]=x[offset]; y[offset+1]=x[offset+1]; return; }
        float c=cs[token*63+pair], s=sn[token*63+pair];
        float e=float(x[offset]), o=float(x[offset+1]);
        y[offset]=T(metal::fma(o,-s,e*c));
        y[offset+1]=T(metal::fma(o,c,e*s));
    ''')
    freq_np = np.array(frequency.tolist(),dtype=np.float32)
    for branch,count in (("vid",nv),("txt",nt)):
        positions = np.array([(nt,h,w) for h in range(side) for w in range(side)],dtype=np.float32) if branch=="vid" else None
        angles = (mx.array(positions)[...,None]*frequency[None,None,:]).reshape(count,63) if branch=="vid" else mx.tile(
            mx.arange(count,dtype=mx.float32)[:,None]*frequency[None,:],(1,3))
        host_angles = (positions[...,None]*freq_np[None,None,:]).reshape(count,63) if branch=="vid" else np.tile(
            np.arange(count,dtype=np.float32)[:,None]*freq_np[None,:],(1,3))
        for label,cs,sn in (("mlx_trig",mx.cos(angles),mx.sin(angles)),
            ("fp16_angles",mx.cos(angles.astype(mx.float16).astype(mx.float32)),mx.sin(angles.astype(mx.float16).astype(mx.float32))),
            ("host_trig",mx.array(np.cos(host_angles)),mx.array(np.sin(host_angles)))):
            for suffix in ("q","k"):
                x = ref[f"attn_{branch}_norm{suffix}"]
                pairs = x[...,:126].astype(mx.float32).reshape(count,20,63,2)
                e,o = pairs[...,0],pairs[...,1]
                y = mx.stack((e*cs[:,None,:]-o*sn[:,None,:],e*sn[:,None,:]+o*cs[:,None,:]),axis=-1).reshape(count,20,126).astype(x.dtype)
                result[f"attn_{branch}_rope{suffix}.{label}_separate"] = mx.concatenate((y,x[...,126:]),axis=-1)
                result[f"attn_{branch}_rope{suffix}.{label}_fma"] = kernel(inputs=[x,cs,sn],output_shapes=[x.shape],
                    output_dtypes=[x.dtype],grid=(count*20*64,1,1),threadgroup=(256,1,1),template=[("T",x.dtype)])[0]
    mx.eval(*result.values())
    mx.save_safetensors(str(args.output/"attention.safetensors"),result)
    print("isolated attention/RoPE outputs saved",flush=True)


def torch_attention(args):
    import torch
    from safetensors.torch import load_file,save_file
    if args.output.exists():
        raise ValueError("fresh Torch attention leaf output required")
    args.output.mkdir(parents=True)
    ref = load_file(str(args.reference/"block0-detail.safetensors"),device="mps")
    nv,nt = ref["attn_vid_qkv"].shape[0],ref["attn_txt_qkv"].shape[0]
    pack = lambda a,b:torch.cat((a,b),dim=0).transpose(0,1)[None,...]
    q,k,v = pack(ref["attn_vid_ropeq"],ref["attn_txt_ropeq"]),pack(
        ref["attn_vid_ropek"],ref["attn_txt_ropek"]),pack(ref["attn_vid_qkv"].reshape(nv,3,20,128)[:,2],
        ref["attn_txt_qkv"].reshape(nt,3,20,128)[:,2])
    result = {}
    with torch.inference_mode():
        for name,out in (("direct_bf16",torch.nn.functional.scaled_dot_product_attention(q,k,v)),
            ("direct_fp32",torch.nn.functional.scaled_dot_product_attention(q.float(),k.float(),v.float()).to(q.dtype))):
            flat = out[0].transpose(0,1).reshape(nv+nt,2560)
            result["attn_vid_raw."+name] = flat[:nv].cpu().contiguous()
            result["attn_txt_raw."+name] = flat[nv:].cpu().contiguous()
    save_file(result,str(args.output/"attention.safetensors"))
    print("direct pinned Torch attention outputs saved",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("mlx","attention_mlx","torch_attention","compare"))
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--detail-file",default="time-detail.safetensors")
    parser.add_argument("--leaf-file",default="leaf.safetensors")
    args = parser.parse_args()
    if args.command in ("mlx","attention_mlx"):
        if args.checkpoint is None:
            parser.error("original FP16 checkpoint required")
        (mlx_leaf if args.command=="mlx" else attention_leaf)(args)
    elif args.command == "torch_attention":
        torch_attention(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
