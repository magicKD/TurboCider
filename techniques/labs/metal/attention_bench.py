#!/usr/bin/env python3
"""Exact attention layout/segmentation experiments with an independent CPU row oracle.

Original replay captures are explicit optional inputs. Synthetic contract cases
cover paths/masks/tails, never model quality. One candidate per fresh process;
all outputs, preparation/resident bytes and failed gates remain in the report.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import shutil
import time

from attention_math import allowed_keys, paired, row_oracle, validate_segments
from attention_source import digest, load

CONTROLS={
    "tail":(4,4,33,41,128,"none"),
    "causal-offset":(4,4,17,29,64,"causal"),
    "empty-fused":(4,4,17,29,64,"empty-first-row"),
    "empty-fallback":(4,4,17,29,96,"empty-first-row"),
    "head80":(4,4,33,41,80,"none"),
    "gqa-vector":(4,2,3,41,128,"causal"),
    "gqa-full":(4,2,17,41,128,"none"),
}


def metrics(reference,actual,np):
    if reference.shape!=actual.shape or not np.isfinite(actual).all():
        return {"finite":False,"relative_l2":None,"max_abs":None}
    a,b=reference.astype(np.float64).ravel(),actual.astype(np.float64).ravel()
    delta=a-b;energy=float(np.sum(a*a));error=float(np.sum(delta*delta))
    return {"finite":True,"relative_l2":math.sqrt(error/energy) if energy else (0. if not error else None),
            "max_abs":float(np.max(np.abs(delta)))}


def metric_pass(value):
    number=value.get("relative_l2")
    return value.get("finite") is True and type(number) in (int,float) and math.isfinite(number) and 0<=number<=.01


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root",type=Path)
    parser.add_argument("--phase",choices=("prefill","steady"))
    parser.add_argument("--control",choices=tuple(CONTROLS))
    parser.add_argument("--candidate",choices=("single-mask","contiguous-v","head-chunks","noop"),required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if (args.capture_root is None)==(args.control is None) or (args.capture_root is not None)!=(args.phase is not None):
        parser.error("explicit capture+phase or one synthetic control required")
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent;(output/"source").mkdir()
    snapshots=[]
    for name in ("attention_bench.py","attention_math.py","attention_source.py"):
        shutil.copy2(root/name,output/"source"/name)
        snapshots.append({"path":"source/"+name,"sha256":digest(root/name)})
    report={"schema":"tc-attention-experiment-v1","status":"starting","production_qualified":False,
            "synthetic":args.control is not None,"candidate":args.candidate,"samples":[],"source_snapshots":snapshots,
            "gates":{"full_original_relative_l2_max":.01,"independent_row_relative_l2_max":.01,"empty_row_max_abs":1e-6},
            "warmup_trials":4,"measured_trials":8,"order":"ABBA","errors":[],"physical_device_timestamps":None}

    def save():
        (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    save();mx=None
    try:
        if platform.system()!="Darwin" or shutil.disk_usage(output).free<512*1024*1024:
            raise ValueError("requires macOS GPU and bounded disk headroom")
        import numpy as np
        import mlx.core as mx

        mx.set_default_device(mx.gpu);info=mx.device_info()
        report["environment"]={"macos":platform.mac_ver()[0],"mlx":version("mlx"),"numpy":np.__version__,
            "device":{key:info.get(key) for key in ("device_name","architecture","memory_size")},
            "library_assets":[{"name":path.name,"sha256":digest(path)} for path in
                [Path(mx.__file__),*sorted((Path(mx.__file__).parent/"lib").glob("*"))]
                if path.is_file() and path.suffix in (".so",".dylib",".metallib")]}
        reference=None;segments=[];record=None
        if args.capture_root:
            arrays,record,identity=load(args.capture_root.resolve(),args.phase,np)
            q,k,v=[arrays[name] for name in ("q","k","v")];reference=arrays["reference"]
            mode="segments" if args.phase=="prefill" else "none";segments=record["segments"]
            report.update(source_identity=identity,source_record=record,point=args.phase)
        else:
            hq,hk,ql,kl,dim,mode=CONTROLS[args.control]
            rng=np.random.default_rng(29)
            def random(shape):
                raw=rng.normal(0,.2,shape).astype(np.float32)
                bits=raw.view(np.uint32);rounded=((bits+0x7fff+((bits>>16)&1))&0xffff0000).view(np.float32)
                return rounded
            q,k,v=random((1,hq,ql,dim)),random((1,hk,kl,dim)),random((1,hk,kl,dim))
            report["point"]=args.control
        b,hq,ql,dim=q.shape;hk,kl=k.shape[1:3]
        if b!=1 or not 1<=hq<=32 or not 1<=hk<=hq or hq%hk or not 1<=ql<=1536 or not 1<=kl<=1536 or dim not in (64,80,96,128):
            raise ValueError("bounded attention geometry/GQA required")
        if mode=="segments":validate_segments(segments,ql)
        report.update(shape_q=list(q.shape),shape_k=list(k.shape),shape_v=list(v.shape),mask_mode=mode,segments=segments,
                      input_tensor_sha256={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in zip(("q","k","v"),(q,k,v))})
        scale=1/math.sqrt(dim)
        allowed=[allowed_keys(row,ql,kl,mode,segments) for row in range(ql)]
        heads=sorted({0,hq//2,hq-1});rows=sorted({0,ql//2,ql-1,*([segments[0]["end"]-1,segments[0]["end"]] if segments else [])})
        witness=[]
        for head in heads:
            source_head=head//(hq//hk)
            keys,values=k[0,source_head].tolist(),v[0,source_head].tolist()
            witness.append([row_oracle(q[0,head,row].tolist(),keys,values,allowed[row],scale) for row in rows])
        cpu=np.asarray(witness,dtype=np.float64)
        report["oracle"]={"source_kind":"derived","arithmetic":"independent CPU Python float/math.fsum, all K and output columns per selected row",
                          "heads":heads,"queries":rows,"empty_row_convention":"zero; explicit local gate, not assumed universal backend promise"}
        gpu=[];layout=[]
        for name,value in zip(("q","k","v"),(q,k,v)):
            tensor=mx.array(value,dtype=mx.bfloat16)
            if record:
                metadata=next(item for item in record["tensors"] if item["name"]==name)
                shape=value.shape;row_major=[shape[1]*shape[2]*shape[3],shape[2]*shape[3],shape[3],1]
                interleaved=[shape[1]*shape[2]*shape[3],shape[3],shape[1]*shape[3],1]
                strides=metadata["replay_element_strides"]
                if strides==interleaved and strides!=row_major:
                    tensor=mx.transpose(mx.contiguous(mx.transpose(tensor,(0,2,1,3))),(0,2,1,3))
                elif strides!=row_major:raise ValueError("unsupported captured replay layout; no silent repack")
                layout.append({"name":name,"requested_element_strides":strides})
            gpu.append(tensor)
        mx.eval(gpu)
        stride_probe=mx.fast.metal_kernel(name="tc_attention_actual_strides",input_names=["X"],output_names=["Y"],
            source="Y[0]=X_strides[0];Y[1]=X_strides[1];Y[2]=X_strides[2];Y[3]=X_strides[3];",ensure_row_contiguous=False)
        for slot,tensor in enumerate(gpu):
            actual=stride_probe(inputs=[tensor],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(4,)],output_dtypes=[mx.int64])[0]
            mx.eval(actual);strides=actual.tolist()
            if record and strides!=layout[slot]["requested_element_strides"]:raise ValueError("GPU physical stride witness differs")
            if not record:layout.append({"name":("q","k","v")[slot]})
            layout[slot]["actual_element_strides"]=strides
        report["gpu_layouts"]=layout
        vshape=list(v.shape)
        report["candidate_mechanism"]={"kind":args.candidate,
            "contiguous_v_would_copy":layout[2]["actual_element_strides"]!=[vshape[1]*vshape[2]*vshape[3],vshape[2]*vshape[3],vshape[3],1],
            "head_chunks_would_split":hq>8,"single_mask_replaces_segments":mode=="segments"}
        started=time.perf_counter_ns();full_mask=mx.array(np.asarray(allowed,dtype=np.bool_));mx.eval(full_mask)
        report["mask_preparation_ns"]=time.perf_counter_ns()-started
        report["prepared_mask_logical_bytes"]=full_mask.nbytes
        report["input_logical_bytes"]=sum(value.nbytes for value in gpu)

        def restore(value):return mx.transpose(value,(0,2,1,3)).reshape(1,value.shape[2],hq*value.shape[3])

        def call(qv,kv,vv,mask,method):
            if method=="head-chunks" and hq>8:
                if hk!=hq:raise ValueError("head-chunks only supports matching real Q/KV heads")
                return restore(mx.concatenate([mx.fast.scaled_dot_product_attention(qv[:,first:first+8],kv[:,first:first+8],vv[:,first:first+8],scale=scale,mask=mask)
                    for first in range(0,hq,8)],axis=1))
            return restore(mx.fast.scaled_dot_product_attention(qv,kv,vv,scale=scale,mask=mask))

        def implementation(method):
            def apply(qv,kv,vv):
                if method=="contiguous-v":vv=mx.contiguous(vv)
                if method=="single-mask":return call(qv,kv,vv,full_mask,method)
                if mode=="segments":
                    return mx.concatenate([call(qv[:,:,item["start"]:item["end"]],kv[:,:,:item["end"]],vv[:,:,:item["end"]],
                        "causal" if item["causal"] else None,method) for item in segments],axis=1)
                return call(qv,kv,vv,"causal" if mode=="causal" else full_mask if mode=="empty-first-row" else None,method)
            return mx.compile(apply)

        baseline,candidate=implementation("baseline"),implementation(args.candidate)
        for trial in range(-4,8):
            is_candidate=trial%4 in (1,2);function=candidate if is_candidate else baseline
            started=time.perf_counter_ns();result=function(*gpu);mx.eval(result);wall=time.perf_counter_ns()-started
            actual=np.array(result.astype(mx.float32))
            head_actual=actual.reshape(1,ql,hq,dim).transpose(0,2,1,3)
            sampled=head_actual[0,heads][:,rows]
            oracle=metrics(cpu,sampled,np)
            original=metrics(reference,head_actual,np) if reference is not None else None
            empty_rows=[row for row in range(ql) if not any(allowed[row])]
            empty_max=(float(np.max(np.abs(head_actual[:,:,empty_rows,:]))) if np.isfinite(head_actual[:,:,empty_rows,:]).all() else None) if empty_rows else 0.
            passed=(list(result.shape)==[1,ql,hq*dim] and result.dtype==mx.bfloat16 and metric_pass(oracle)
                    and (original is None or metric_pass(original)) and empty_max is not None and empty_max<=1e-6)
            report["samples"].append({"trial":trial,"warmup":trial<0,"pair":trial//2,"point":report["point"],
                "variant":"candidate" if is_candidate else "baseline","wall_ns":wall,"source_kind":"measured",
                "boundary":"prepared operator host output-ready; layout/consumer restore inside; not device kernel time",
                "component_numeric_pass":passed,"independent_rows":oracle,"vs_original_replay":original,"empty_row_max_abs":empty_max})
            save()
        report["gpu_inputs_unchanged"]=all(np.array_equal(np.array(tensor.astype(mx.float32)),value) for tensor,value in zip(gpu,(q,k,v)))
        report["source_inputs_unchanged"]=all(digest(root/Path(item["path"]).name)==item["sha256"] for item in snapshots)
        if not report["gpu_inputs_unchanged"] or not report["source_inputs_unchanged"]:raise ValueError("attention source/operand mutated")
        report["statistics"]=paired(report["samples"])
        report["identity_control"]=args.candidate=="noop"
        report["status"]="component-numeric-pass" if report["statistics"] is not None else "component-numeric-rejected"
    except Exception as error:
        report["status"]="failed";report["errors"].append({"type":type(error).__name__,"message":str(error)})
    finally:
        if mx is not None:
            try:mx.synchronize()
            except Exception as error:report["status"]="failed";report["errors"].append({"type":type(error).__name__,"message":str(error)})
        save()
    print(json.dumps({"status":report["status"],"point":report.get("point"),"statistics":report.get("statistics"),"production_qualified":False}))
    return int(report["status"]!="component-numeric-pass")


if __name__=="__main__":
    raise SystemExit(main())
