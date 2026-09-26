"""Validate image-conditioned DiT against mflux blocks with a dense oracle mask.

The sequence geometry follows ComfyUI QwenImage21.build_sequence. Using one
explicit full mask provides an independent check of native segmented attention,
including bottom-right causal alignment after earlier image blocks.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mflux", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.mflux / "src"))
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    for dtype in (mx.float32, mx.bfloat16):
        ModelConfig.precision = dtype
        mx.random.seed(472)
        model = Qwen21Transformer(in_channels=4, out_channels=4, num_layers=2,
                                 attention_head_dim=8, num_attention_heads=2, context_in_dim=16,
                                 axes_dims_rope=(2,2,4))
        params = [(k, (mx.random.normal(v.shape) * 0.15).astype(dtype))
                  for k,v in tree_flatten(model.parameters()) if k.endswith(".weight")]
        model.load_weights(params, strict=False)
        for count in (1, 2, 10):
            height, width, text_length = 2, 3, 7
            refs = [(1 + i % 3, 2 + i % 2, min(i, text_length)) for i in range(count)]
            text = mx.random.normal((1,text_length,16)).astype(dtype)
            changed_text = mx.random.normal(text.shape).astype(dtype)
            latents = mx.random.normal((1,height*width,4)).astype(dtype)
            images = [mx.random.normal((1,h*w,4)).astype(dtype) for h,w,_ in refs]
            changed_image = mx.random.normal(images[0].shape).astype(dtype)
            positions, segments = [], []
            cursor, pos = 0, 0
            for h,w,slot in refs + [(height,width,text_length)]:
                start = len(positions)
                for _ in range(cursor,slot):
                    positions.append([pos,pos,pos]); pos += 1
                if len(positions) > start:
                    segments.append((start,len(positions),True))
                start = len(positions)
                positions.extend([[pos, r-(h-h//2)+.5*(h%2-height%2), c-(w-w//2)+.5*(w%2-width%2)]
                                  for r in range(h) for c in range(w)])
                segments.append((start,len(positions),False))
                pos += max(h,w)
                cursor = slot
            n = len(positions)
            allowed = np.zeros((n,n),dtype=bool)
            for start,end,causal in segments:
                for query in range(start,end):
                    allowed[query,:query+1 if causal else end] = True
            mask = mx.array(np.where(allowed,0.,-np.inf).astype(np.float32))[None,None].astype(dtype)
            position_array = np.array(positions,dtype=np.float32)
            angles = np.concatenate([position_array[:,axis,None] * (1.0 / (10000**(np.arange(0,dim,2,dtype=np.float32)/dim)))[None,:]
                                     for axis,dim in enumerate((2,2,4))],axis=-1)
            cos,sin = mx.array(np.cos(angles)),mx.array(np.sin(angles))
            prefix = n-height*width

            def oracle(condition, reference_images, t):
                projected = model.txt_in(condition)
                parts, previous = [], 0
                for (_,_,slot), img in zip(refs+[(height,width,text_length)], reference_images+[latents]):
                    if slot > previous: parts.append(projected[:,previous:slot])
                    parts.append(model.img_in(img)); previous = slot
                hidden = mx.concatenate(parts,axis=1)
                temb = model.time_text_embed(mx.array([t,0.]))
                m1,m2 = mx.split(model.modulation(temb),2,axis=-1)
                m1=model._select_modulation_rows(m1,prefix,height*width)
                m2=model._select_modulation_rows(m2,prefix,height*width)
                for block in model.transformer_blocks:
                    hidden=block(hidden,m1,m2,cos,sin,mask)
                scale = model.norm_out.linear(nn.silu(temb[:1]))[:,None]
                return model.proj_out(model.norm_out(hidden[:,prefix:],scale))

            expected = {"first":oracle(text,images,.8),"cached":oracle(text,images,.3),
                        "changed":oracle(changed_text,images,.3),
                        "changed_image":oracle(text,[changed_image]+images[1:],.3)}
            mx.eval(list(expected.values()))
            with tempfile.TemporaryDirectory(prefix="tc-qwen21-refs-") as directory:
                root=Path(directory)
                mx.save_safetensors(str(root/"weights.safetensors"),{k.replace("modulation.layers.1","modulation.1"):v for k,v in params})
                inputs={"text":text,"text_changed":changed_text,"latents":latents,"reference_changed":changed_image}
                metadata={"layers":"2","heads":"2","head_dim":"8","axis0":"2","axis1":"2","axis2":"4",
                          "height":str(height),"width":str(width),"reference_count":str(count)}
                for i,(h,w,slot) in enumerate(refs):
                    key=f"reference{i}"; inputs[key]=images[i]
                    metadata.update({key+"_height":str(h),key+"_width":str(w),key+"_slot":str(slot)})
                mx.save_safetensors(str(root/"inputs.safetensors"),inputs,metadata)
                subprocess.run([str(args.probe.resolve()),str(root/"weights.safetensors"),str(root/"inputs.safetensors"),str(root/"out.safetensors")],check=True)
                actual=mx.load(str(root/"out.safetensors"))
                errors={k:mx.max(mx.abs(v.astype(mx.float32)-actual[k].astype(mx.float32))).item() for k,v in expected.items()}
                for a,b in (("cached","uncached"),("cached","split_cached"),("changed","changed_reference"),("changed_image","changed_image_uncached")):
                    errors[a+"_vs_"+b]=mx.max(mx.abs(actual[a].astype(mx.float32)-actual[b].astype(mx.float32))).item()
                print(json.dumps({"dtype":str(dtype),"references":count,"max_abs":errors}),flush=True)
                assert max(errors.values()) < (3e-5 if dtype==mx.float32 else .015),errors


if __name__ == "__main__":
    main()
