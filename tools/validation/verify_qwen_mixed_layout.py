"""Independent NumPy decoding versus MLX adapter on real Qwen mixed layers."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from qwen_mixed_probe import header, prepare, similarity


def fp8(raw):
    raw=np.asarray(raw,dtype=np.uint8)
    e=(raw>>3)&15
    m=raw&7
    value=np.where(e==0,m.astype(np.float32)*2**-9,
                   (1+m.astype(np.float32)/8)*np.exp2(e.astype(np.float32)-7))
    value=np.where((e==15)&(m==7),np.nan,value)
    return np.where(raw&128,-value,value).astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new output file')
    _,formats=header(a.weights)
    source=mx.load(str(a.weights))
    report={}
    for prefix in ['model.layers.0.self_attn.q_proj','model.layers.0.mlp.down_proj',
                   'model.layers.10.mlp.gate_proj']:
        names=[prefix+s for s in ['.weight','.weight_scale','.weight_scale_2']]
        original={k:source[k] for k in names}
        raw=np.array(original[names[0]])
        scale_raw=np.array(original[names[1]])
        global_scale=original[names[2]].item()
        rows,cols=raw.shape[0],raw.shape[1]*2
        rr,cc=np.arange(rows)[:,None],np.arange(cols//16)[None,:]
        ct=(cols//16+3)//4
        offsets=((((rr//128*ct+cc//4)*32+rr%32)*4+rr%128//32)*4+cc%4)
        scales=fp8(scale_raw).reshape(-1)[offsets]
        lut=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float32)
        digits=np.stack([raw>>4,raw&15],axis=-1).reshape(rows,cols)
        reference=lut[digits]*np.repeat(scales*global_scale,16,axis=1)
        adapted=dict(original)
        prepare(adapted,{prefix:'nvfp4'},False)
        decoded=mx.dequantize(adapted[names[0]],adapted[names[1]],group_size=16,
                              bits=4,mode='nvfp4',dtype=mx.float32)*global_scale
        mx.eval(decoded)
        diff=similarity(reference,np.array(decoded))
        if diff['max_abs']!=0:raise ValueError(f'Layout mismatch: {prefix}')
        x=mx.sin(mx.arange(32*cols,dtype=mx.float32)).reshape(32,cols)
        actual=mx.quantized_matmul(x,adapted[names[0]],adapted[names[1]],transpose=True,
                                   group_size=16,bits=4,mode='nvfp4')*global_scale
        expected=x@mx.array(reference).T
        mx.eval(actual,expected)
        report[prefix]=dict(decode=diff,matmul=similarity(np.array(expected),np.array(actual)),
                            input_dtype=str(x.dtype),output_dtype=str(actual.dtype))
        if not report[prefix]['matmul']['finite']:raise ValueError('Nonfinite projection')
        print(prefix,report[prefix],flush=True)
    prefix=next(k for k,v in formats.items() if v=='float8_e4m3fn')
    raw=np.array(source[prefix+'.weight'])
    decoded=mx.from_fp8(source[prefix+'.weight'],dtype=mx.float32)
    mx.eval(decoded)
    diff=similarity(fp8(raw),np.array(decoded))
    if diff['max_abs']!=0:raise ValueError('FP8 decode mismatch')
    report[prefix]={'fp8_decode':diff}
    packed={prefix+s:source[prefix+s] for s in ['.weight','.weight_scale']}
    prepare(packed,{prefix:'float8_e4m3fn'},False,True)
    decoded_packed=mx.dequantize(packed[prefix+'.weight'],packed[prefix+'.mlx_block_scales'],
                                group_size=32,bits=8,mode='mxfp8',dtype=mx.float32)
    mx.eval(decoded_packed)
    diff=similarity(fp8(raw),np.array(decoded_packed))
    if diff['max_abs']!=0:raise ValueError('MXFP8 repack changed FP8 values')
    report[prefix]['mxfp8_decode']=diff
    cols=raw.shape[-1]
    x=mx.sin(mx.arange(32*cols,dtype=mx.float32)).reshape(32,cols)
    actual=mx.quantized_matmul(x,packed[prefix+'.weight'],packed[prefix+'.mlx_block_scales'],
                               transpose=True,group_size=32,bits=8,mode='mxfp8')*packed[prefix+'.weight_scale']
    reference=x@(mx.array(fp8(raw))*packed[prefix+'.weight_scale']).T
    mx.eval(actual,reference)
    report[prefix]['mxfp8_matmul']=similarity(np.array(reference),np.array(actual))
    if not report[prefix]['mxfp8_matmul']['finite']:raise ValueError('Nonfinite FP8 projection')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
