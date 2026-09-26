"""Qwen3-VL visual tower and all deepstack mergers vs unmodified mflux."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import mlx.core as mx
from mlx.utils import tree_flatten


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mflux",type=Path,required=True)
    p.add_argument("--probe",type=Path,required=True)
    p.add_argument("--weights",type=Path)
    args=p.parse_args()
    sys.path.insert(0,str(args.mflux/"src"))
    from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
    for dtype in ([mx.bfloat16] if args.weights else [mx.float32,mx.bfloat16]):
        mx.random.seed(278)
        hidden,heads,depth,patch,side,out,intermediate,deep=(1152,16,27,16,48,4096,4304,[8,16,24]) if args.weights else (16,2,3,2,4,24,32,[0,1,2])
        model=Qwen3VLVisionModel(patch_size=patch,hidden_size=hidden,num_heads=heads,intermediate_size=intermediate,
                                depth=depth,num_position_embeddings=side*side,out_hidden_size=out,deepstack_visual_indexes=deep)
        if args.weights:
            source=mx.load(str(args.weights))
            params=[]
            for key,value in tree_flatten(model.parameters()):
                if key.endswith((".weight",".bias")):
                    value=source["model.visual."+key]
                    if key=="patch_embed.proj.weight": value=value.transpose(0,2,3,4,1)
                    params.append((key,value))
            del source
        else:
            params=[(k,(mx.random.normal(v.shape)*.1).astype(dtype)) for k,v in tree_flatten(model.parameters()) if k.endswith((".weight",".bias"))]
        model.load_weights(params,strict=False)
        source_weights={"model.visual."+k:(v.transpose(0,4,1,2,3) if k=="patch_embed.proj.weight" else v) for k,v in params}
        for h,w in ((4,6),(6,4)):
            patches=mx.random.normal((h*w,3*2*patch*patch)).astype(dtype)
            merged,features=model(patches,mx.array([[1,h,w]]),return_deepstack=True)
            expected={"merged":merged,**{f"deep{i}":v for i,v in enumerate(features)}}
            mx.eval(list(expected.values()))
            with tempfile.TemporaryDirectory(prefix="tc-qwen21-vision-") as directory:
                root=Path(directory)
                if args.weights: weight_path=args.weights.resolve()
                else:
                    weight_path=root/"weights.safetensors"
                    mx.save_safetensors(str(weight_path),source_weights)
                mx.save_safetensors(str(root/"inputs.safetensors"),{"patches":patches},
                    {k:str(v) for k,v in dict(patch=patch,hidden=hidden,heads=heads,layers=depth,position_side=side,height=h,width=w,
                                             deep0=deep[0],deep1=deep[1],deep2=deep[2]).items()})
                subprocess.run([str(args.probe.resolve()),str(weight_path),str(root/"inputs.safetensors"),str(root/"output.safetensors")],check=True)
                actual=mx.load(str(root/"output.safetensors"))
                errors={}
                for key,value in expected.items():
                    delta=actual[key].astype(mx.float32)-value.astype(mx.float32)
                    errors[key]={"max_abs":mx.max(mx.abs(delta)).item(),"relative_rmse":mx.sqrt(mx.mean(delta*delta)/mx.mean(value.astype(mx.float32)**2)).item()}
                print(json.dumps({"dtype":str(dtype),"grid":[h,w],"real_weights":bool(args.weights),"errors":errors}),flush=True)
                assert all(e["relative_rmse"]<(.01 if args.weights else .003) for e in errors.values()),errors


if __name__=="__main__": main()
