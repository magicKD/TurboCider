"""Isolated MLX Qwen3 conditioning probe; does not change the App loader.

Comfy NVFP4 is layout-repacked without requantizing; FP8 projections are
decoded on demand, once with --fp8-resident-bf16, or losslessly packed with
--fp8-mxfp8 (E8M0 block scales of 1; retain external Comfy tensor scale).
This is weight-only
execution, not NVIDIA W4A4. Timings include Python dispatch, exclude loading.
The conditioning contract mirrors native/components/text/qwen3.cpp.
"""
import argparse
import collections
import json
from pathlib import Path
import resource
import statistics
import struct
import time

import mlx.core as mx
import numpy as np


def header(path):
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        data = json.loads(stream.read(size))
        formats = {}
        for key, entry in data.items():
            if key.endswith('.comfy_quant'):
                start, end = entry['data_offsets']
                stream.seek(8 + size + start)
                formats[key[:-12]] = json.loads(stream.read(end - start))['format']
    return data, formats


def similarity(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return dict(cosine=float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                max_abs=float(np.max(np.abs(a-b))), finite=bool(np.isfinite(b).all()))


def prepare(weights, formats, resident_fp8, packed_fp8=False):
    for prefix, fmt in formats.items():
        w = weights[prefix + '.weight']
        if fmt == 'nvfp4':
            if prefix + '.pre_quant_scale' in weights:
                raise ValueError('pre_quant_scale is not implemented')
            scales = weights[prefix + '.weight_scale']
            rows, columns = w.shape[0], w.shape[1] // 8
            rt, ct = (rows + 127)//128, (columns + 3)//4
            assert w.dtype == mx.uint8 and scales.dtype == mx.uint8
            assert scales.size == rt*ct*512
            w = ((w << 4) | (w >> 4)).view(mx.uint32)
            scales = scales.reshape(rt, ct, 32, 4, 4).transpose(0, 3, 2, 1, 4)
            scales = scales.reshape(rt*128, ct*4)[:rows, :columns]
            mx.eval(w, scales)
            g = weights[prefix + '.weight_scale_2'].item()
            if not np.isfinite(g) or g <= 0:
                raise ValueError('invalid global scale')
            weights[prefix + '.weight'] = w
            weights[prefix + '.weight_scale'] = scales
        elif fmt == 'float8_e4m3fn':
            assert w.dtype == mx.uint8
            if packed_fp8:
                if w.shape[-1] % 32: raise ValueError('FP8 columns must align to 32')
                scales=mx.full((w.shape[0],w.shape[1]//32),127,dtype=mx.uint8)
                packed=w.view(mx.uint32)
                mx.eval(scales,packed)
                weights[prefix+'.weight']=packed
                weights[prefix+'.mlx_block_scales']=scales
            elif resident_fp8:
                dense = (mx.from_fp8(w, dtype=mx.float32) * weights[prefix+'.weight_scale']).astype(mx.bfloat16)
                mx.eval(dense)
                weights[prefix+'.weight'] = dense
        else:
            raise ValueError('unsupported mixed format: ' + fmt)


def encode(weights, formats, count, resident_fp8, packed_fp8=False, token_ids=None):
    def linear(x, prefix):
        w = weights[prefix+'.weight']
        fmt = formats.get(prefix)
        if fmt == 'nvfp4':
            y = mx.quantized_matmul(x, w, weights[prefix+'.weight_scale'],
                                    transpose=True, group_size=16, bits=4, mode='nvfp4')
            return (y.astype(mx.float32)*weights[prefix+'.weight_scale_2']).astype(x.dtype)
        if fmt == 'float8_e4m3fn' and packed_fp8:
            y=mx.quantized_matmul(x,w,weights[prefix+'.mlx_block_scales'],
                                  transpose=True,group_size=32,bits=8,mode='mxfp8')
            return (y.astype(mx.float32)*weights[prefix+'.weight_scale']).astype(x.dtype)
        if fmt == 'float8_e4m3fn' and not resident_fp8:
            w = (mx.from_fp8(w, dtype=mx.float32)*weights[prefix+'.weight_scale']).astype(mx.bfloat16)
        if fmt is None and w.dtype == mx.uint32:
            scales = weights[prefix+'.scales']
            bits = w.shape[-1]*32 // x.shape[-1]
            return mx.quantized_matmul(x, w, scales, weights.get(prefix+'.biases'),
                                        transpose=True, group_size=32, bits=bits)
        return x @ w.T

    def rms(x, name):
        return mx.fast.rms_norm(x.astype(mx.float32), weights[name].astype(mx.float32), 1e-6).astype(x.dtype)

    def heads(x, n):
        return x.reshape(1, count, n, 128).transpose(0, 2, 1, 3)

    if token_ids is not None and len(token_ids) != count:
        raise ValueError('token count mismatch')
    ids = mx.array([token_ids if token_ids is not None else
                    [100+(i*7919)%100000 for i in range(count)]], mx.int32)
    x = weights['model.embed_tokens.weight'][ids].astype(mx.float32)
    frequencies = 1 / mx.power(mx.array(1000000., mx.float32), mx.arange(0,128,2,dtype=mx.float32)/128)
    angles = mx.arange(count,dtype=mx.float32).reshape(1,count,1)*frequencies.reshape(1,1,64)
    angles = mx.concatenate([angles,angles],axis=-1)
    cosine, sine = mx.cos(angles)[:,None], mx.sin(angles)[:,None]
    p = mx.arange(count)
    mask = mx.where(p[None,:]>p[:,None], mx.array(-float('inf'),mx.float32), mx.array(0.,mx.float32))[None,None]

    def rotate(v):
        a,b=mx.split(v,2,axis=-1)
        return v*cosine+mx.concatenate([-b,a],axis=-1)*sine

    for layer in range(35):
        prefix=f'model.layers.{layer}'
        z=rms(x,prefix+'.input_layernorm.weight')
        q=heads(linear(z,prefix+'.self_attn.q_proj'),32)
        k=heads(linear(z,prefix+'.self_attn.k_proj'),8)
        v=heads(linear(z,prefix+'.self_attn.v_proj'),8)
        q=rotate(rms(q,prefix+'.self_attn.q_norm.weight'))
        k=rotate(rms(k,prefix+'.self_attn.k_norm.weight'))
        k,v=mx.repeat(k,4,axis=1),mx.repeat(v,4,axis=1)
        z=mx.fast.scaled_dot_product_attention(q,k,v,scale=float(np.float32(1/np.sqrt(128))),mask=mask)
        z=z.transpose(0,2,1,3).reshape(1,count,4096)
        x=x+linear(z,prefix+'.self_attn.o_proj')
        z=rms(x,prefix+'.post_attention_layernorm.weight')
        gate=linear(z,prefix+'.mlp.gate_proj')
        x=x+linear(silu(gate)*linear(z,prefix+'.mlp.up_proj'),prefix+'.mlp.down_proj')
        mx.eval(x)
    x=x.astype(mx.bfloat16)
    mx.eval(x)
    return x


@mx.compile
def silu(x):
    return x*mx.sigmoid(x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,choices=(32,128),default=128)
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--reference',type=Path)
    fp8=p.add_mutually_exclusive_group()
    fp8.add_argument('--fp8-resident-bf16',action='store_true')
    fp8.add_argument('--fp8-mxfp8',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    metadata,formats=header(a.weights)
    mx.reset_peak_memory()
    start=time.perf_counter()
    weights=mx.load(str(a.weights))
    mx.eval(list(weights.values()))
    raw_seconds=time.perf_counter()-start
    raw_active=mx.get_active_memory()
    start=time.perf_counter()
    prepare(weights,formats,a.fp8_resident_bf16,a.fp8_mxfp8)
    mx.eval(list(weights.values()))
    prep_seconds=time.perf_counter()-start
    report=dict(mlx=mx.__version__,weights=str(a.weights),file_bytes=a.weights.stat().st_size,
                formats=dict(collections.Counter(formats.values())),tokens=a.tokens,
                fp8_policy='packed_mxfp8' if a.fp8_mxfp8 else ('resident_bf16' if a.fp8_resident_bf16 else 'decode_per_projection'),
                raw_load_seconds=raw_seconds,raw_active_bytes=raw_active,adapt_seconds=prep_seconds,
                load_and_adapt_peak_bytes=mx.get_peak_memory(),resident_weight_bytes=mx.get_active_memory(),
                samples=[],scope='MLX allocator; excludes OS/file cache; Python dispatch included')
    for i in range(a.runs+1):
        mx.clear_cache()
        mx.reset_peak_memory()
        start=time.perf_counter()
        result=encode(weights,formats,a.tokens,a.fp8_resident_bf16,a.fp8_mxfp8)
        elapsed=time.perf_counter()-start
        row=dict(warmup=i==0,seconds=elapsed,mlx_peak_bytes=mx.get_peak_memory(),
                 mlx_active_bytes=mx.get_active_memory(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report['samples'].append(row)
        print(row,flush=True)
        if i==a.runs:
            mx.save_safetensors(str(a.output/'conditioning.safetensors'),{'conditioning':result})
            if a.reference:
                ref=mx.load(str(a.reference))['conditioning']
                report['vs_reference']=similarity(np.array(ref.astype(mx.float32)),np.array(result.astype(mx.float32)))
        del result
    report['warm_median_seconds']=statistics.median(s['seconds'] for s in report['samples'][1:])
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
