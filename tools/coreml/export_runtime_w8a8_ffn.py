"""Public macOS26 normalized W8A8 SwiGLU, no checkpoint/adapter constants."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile
import numpy as np


def geometry(rows, hidden, width, tile_k=1024, lora_inputs=False, basis="sylvester_dh"):
    if (any(type(v) is not int or v<=0 for v in (rows,hidden,width,tile_k)) or rows>4224 or
            hidden>4096 or hidden%128 or width>16384 or width%512 or tile_k<128 or tile_k>2048 or
            type(lora_inputs) is not bool or basis not in ("sylvester_dh","comfy_h256") or
            (basis=="comfy_h256" and hidden%256)):
        raise ValueError("unsupported Public W8A8 FFN geometry/recipe")
    inputs={"x":[hidden,rows],"tx":[1,rows],"wg":[width,hidden],"sg":[width,1],
            "wu":[width,hidden],"su":[width,1],"wd":[hidden,width],"headroom":[1,1]}
    if lora_inputs:inputs.update(dg=[width,rows],du=[width,rows])
    return {"schema_version":1,"graph_version":2 if lora_inputs else 1,"lora_inputs":lora_inputs,
        "backend":"runtime_weight_w8a8","kind":"swiglu","rows":rows,"hidden":hidden,"width":width,
        "tile_k":tile_k,"tile_n":512,"layout":"out_in","biases":False,"basis":basis,
        "headroom_input":True,"normalization_scale":1/128,"a8_group_size":0,
        "inputs":inputs,"outputs":{"y":[hidden+1+(width if lora_inputs else 0),rows]},
        "arithmetic_evidence":"unknown","observed_ane_residency":"unknown","production_qualified":False}


def rotation(width,basis):
    block=256 if basis=="comfy_h256" else 512
    out=np.arange(block,dtype=np.uint64)[:,None];col=np.arange(block,dtype=np.uint64)[None,:]
    if basis=="comfy_h256":
        h4=np.array([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],np.int8)
        signs=np.ones((block,block),np.int8)
        for shift in (0,2,4,6):signs*=h4[((out>>shift)&3).astype(int),((col>>shift)&3).astype(int)]
    else:
        bits=out&col;parity=np.zeros((block,block),np.uint8)
        for shift in range(9):parity^=((bits>>shift)&1).astype(np.uint8)
        signs=1-2*parity.astype(np.int8)
        # Identical unsigned64 recipe to ane_w8a8_math.hpp, not Python hash.
        with np.errstate(over="ignore"):
            key=np.uint64(20260930)+np.arange(block,dtype=np.uint64)*np.uint64(0x9e3779b97f4a7c15)
            key=(key^(key>>30))*np.uint64(0xbf58476d1ce4e5b9)
            key=(key^(key>>27))*np.uint64(0x94d049bb133111eb)
            key^=key>>31
        signs*=np.where(key&1,-1,1).astype(np.int8)[None,:]
    return np.tile((signs/np.sqrt(block)).astype(np.float16),(width//block,1)).reshape(width,block,1,1)


def make_program(spec):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    if not hasattr(ct.target,"macOS26"):raise RuntimeError("Public INT8 FFN needs isolated coremltools>=9")
    m,h,f=spec["rows"],spec["hidden"],spec["width"]
    def dot(w,x):
        partial=None
        for start in range(0,w.shape[1],spec["tile_k"]):
            end=min(start+spec["tile_k"],w.shape[1])
            wt=mb.dequantize(input=mb.slice_by_index(x=w,begin=[0,start],end=[w.shape[0],end]),scale=np.float16(1/128))
            xt=mb.dequantize(input=mb.slice_by_index(x=x,begin=[start,0],end=[end,m]),scale=np.float16(1/128))
            value=mb.matmul(x=wt,y=xt)
            partial=value if partial is None else mb.add(x=partial,y=value)
        return partial
    def body(x,tx,wg,sg,wu,su,wd,headroom,dg=None,du=None):
        gate=mb.mul(x=mb.mul(x=dot(wg,x),y=sg),y=tx)
        up=mb.mul(x=mb.mul(x=dot(wu,x),y=mb.real_div(x=su,y=headroom)),y=tx)
        if dg is not None:
            gate=mb.add(x=gate,y=dg);up=mb.add(x=up,y=mb.real_div(x=du,y=headroom))
        activated=mb.real_div(x=gate,y=mb.add(x=np.float16(1),y=mb.exp(x=mb.mul(x=gate,y=np.float16(-1)))))
        hidden=mb.mul(x=activated,y=up)
        block=256 if spec["basis"]=="comfy_h256" else 512
        hr4=mb.conv(x=mb.reshape(x=hidden,shape=[1,f,1,m]),weight=rotation(f,spec["basis"]),groups=f//block)
        rotated=mb.reshape(x=hr4,shape=[f,m])
        peak=mb.maximum(x=mb.reduce_max(x=mb.abs(x=rotated),axes=[0],keep_dims=True),y=np.float16(2**-12))
        q=mb.quantize(input=mb.mul(x=mb.real_div(x=rotated,y=peak),y=np.float16(127)),scale=np.float16(1),output_dtype="int8")
        output=dot(wd,q);scale=mb.mul(x=peak,y=np.float16(128/127))
        parts=[output,scale]+([hidden] if dg is not None else [])
        return mb.identity(x=mb.concat(values=parts,axis=0),name="y")
    signatures=[mb.TensorSpec(shape=tuple(shape),dtype=types.int8 if name in ("x","wg","wu","wd") else types.fp16)
                for name,shape in spec["inputs"].items()]
    if spec["lora_inputs"]:
        @mb.program(input_specs=signatures,opset_version=ct.target.macOS26)
        def program(x,tx,wg,sg,wu,su,wd,headroom,dg,du):return body(x,tx,wg,sg,wu,su,wd,headroom,dg,du)
    else:
        @mb.program(input_specs=signatures,opset_version=ct.target.macOS26)
        def program(x,tx,wg,sg,wu,su,wd,headroom):return body(x,tx,wg,sg,wu,su,wd,headroom)
    return program


def export(destination,spec):
    import coremltools as ct
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():raise ValueError("output already exists")
    program=make_program(spec);destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".public-w8-ffn-",dir=destination.parent) as temporary:
        root=Path(temporary)
        model=ct.convert(program,convert_to="mlprogram",minimum_deployment_target=ct.target.macOS26,
            inputs=[ct.TensorType(name=name,shape=shape,dtype=np.int8 if name in ("x","wg","wu","wd") else np.float16)
                    for name,shape in spec["inputs"].items()],outputs=[ct.TensorType(name="y",dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,skip_model_load=True)
        model.save(str(root/"graph.mlpackage"));compiled=Path(ct.models.utils.compile_model(str(root/"graph.mlpackage")))
        try:shutil.copytree(compiled,root/"graph.mlmodelc")
        finally:shutil.rmtree(compiled)
        metadata={**spec,"compiled_model":"graph.mlmodelc","coremltools":ct.__version__,"macos":platform.mac_ver()[0],
            "files":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}}
        (root/"manifest.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n");root.rename(destination)
    return metadata


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("rows","hidden","width"):p.add_argument("--"+name,type=int,required=True)
    p.add_argument("--tile-k",type=int,default=1024);p.add_argument("--lora-inputs",action="store_true")
    p.add_argument("--basis",choices=("sylvester_dh","comfy_h256"),default="sylvester_dh");p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.output,geometry(a.rows,a.hidden,a.width,a.tile_k,a.lora_inputs,a.basis))))
