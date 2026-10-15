#!/usr/bin/env python3
"""Correctness-only CPU_ONLY/CPU_AND_NE probe on the EXACT same compiled bundle.

ABBA load-policy calls, deterministic real input, independent sampled FP64.
No compilation, downloads, global logs, cache deletion or physical-op claims.
Return zero means collection completed, not that every numerical gate passed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time


def sha(path):
    result=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):result.update(block)
    return result.hexdigest()


def tree(root):
    result=[]
    for path in sorted(root.rglob("*")):
        if path.is_symlink():raise ValueError("compiled bundle symlink not accepted")
        if path.is_file():result.append({"path":str(path.relative_to(root)),"sha256":sha(path)})
    return result


def local(root,name):
    path=(root/name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("fixture asset escapes or is missing")
    return path


def metric(reference,actual,np):
    if reference.shape!=actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("shape/nonfinite diagnostic output")
    a,b=reference.astype(np.float64).ravel(),actual.astype(np.float64).ravel()
    energy=float(np.sum(a*a)); error=float(np.sum((a-b)**2))
    return {"relative_l2":math.sqrt(error/energy) if energy else None,"max_abs":float(np.max(np.abs(a-b))),"source_kind":"derived"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled",type=Path,required=True)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--point",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    compiled=args.compiled.resolve()
    report={"schema":"tc-coreml-same-bundle-policy-v1","status":"starting","production_qualified":False,
            "scope":"same compiled bundle and input, correctness diagnostic, not formal timing or physical ANE proof",
            "recorded_at_singapore":time.strftime("%Y-%m-%dT%H:%M:%S",time.gmtime(time.time()+8*3600))+"+08:00",
            "runner_sha256":sha(Path(__file__)),"calls":[],"errors":[]}
    shutil.copy2(Path(__file__),output/"policy_probe.py")
    report["source_snapshot"]={"path":"policy_probe.py","sha256":sha(output/"policy_probe.py")}

    def save():
        (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")

    save()
    try:
        import numpy as np
        import coremltools as ct
        assets=tree(compiled)
        if not assets or not (compiled/"model.mil").is_file():raise ValueError("explicit compiled ML Program required")
        report["compiled_assets_before"]=assets
        fixture=json.loads(args.fixture.read_text()); root=args.fixture.resolve().parent
        if fixture.get("schema")!="tc-real-ffn-fixture-v1" or fixture.get("loras")!=[]:
            raise ValueError("real base FFN fixture required")
        point=next(value for value in fixture["points"] if value["id"]==args.point)
        for name in (point["input"],fixture["gate_up"],fixture["down"]):
            if sha(local(root,name))!=fixture["files"][name]:raise ValueError("fixture content differs")
        x=np.load(local(root,point["input"]),allow_pickle=False)
        gu=np.load(local(root,fixture["gate_up"]),allow_pickle=False)
        down=np.load(local(root,fixture["down"]),allow_pickle=False)
        m,h,f=fixture["shape_mhf"]
        if x.shape!=(m,h) or gu.shape!=(2*f,h) or down.shape!=(h,f) or any(a.dtype!=np.float16 for a in (x,gu,down)):
            raise ValueError("original full FP16-source shape differs")
        rows=sorted({0,m//3,2*m//3,m-1}); cols=sorted({0,h//3,2*h//3,h-1})
        projected=np.einsum("mk,nk->mn",x[rows].astype(np.float64),gu.astype(np.float64),optimize=False)
        gate,up=np.split(projected,2,axis=1)
        decay=np.exp(-np.abs(gate)); sigmoid=np.where(gate>=0,1/(1+decay),decay/(1+decay))
        witness=np.einsum("mk,nk->mn",gate*sigmoid*up,down[cols].astype(np.float64),optimize=False)
        report.update(fixture_sha256=sha(args.fixture),point=args.point,input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                      shape_mhf=[m,h,f],coremltools=ct.__version__,witness_rows=rows,witness_cols=cols,
                      fixed_sample_relative_l2_gate=.02,order=["CPU_ONLY","CPU_AND_NE","CPU_AND_NE","CPU_ONLY"])
        models={name:ct.models.CompiledMLModel(str(compiled),compute_units=units)
                for name,units in (("CPU_ONLY",ct.ComputeUnit.CPU_ONLY),("CPU_AND_NE",ct.ComputeUnit.CPU_AND_NE))}
        outputs={}
        for ordinal,name in enumerate(report["order"]):
            started=time.perf_counter()
            actual=np.array(models[name].predict({"x":x})["y"],copy=True)
            wall=time.perf_counter()-started
            if actual.shape!=(m,h):raise ValueError("actual full output shape differs")
            score=metric(witness,actual[np.ix_(rows,cols)],np)
            outputs[name]=actual
            input_stable=hashlib.sha256(x.tobytes()).hexdigest()==report["input_sha256"]
            if not input_stable:raise ValueError("prediction changed the shared input")
            report["calls"].append({"ordinal":ordinal,"compute_units_policy":name,"host_call_seconds":wall,
                                     "input_unchanged_after_call":input_stable,
                                     "timing_qualified":False,"sampled_fp64":score,
                                     "component_sample_gate_pass":score["relative_l2"] is not None and score["relative_l2"]<=.02})
            save()
        report["full_output_policy_difference"]=metric(outputs["CPU_ONLY"],outputs["CPU_AND_NE"],np)
        report["compiled_assets_after"]=tree(compiled)
        if report["compiled_assets_after"]!=assets:raise ValueError("compiled bundle changed during diagnostic")
        report["status"]="collected"
        report["inference_limit"]="Policy/path-dependent numerical observation on this bundle; no isolated op or physical-device cause proved."
    except Exception as error:
        report.update(status="failed",errors=[{"type":type(error).__name__,"message":str(error)}])
    if sha(Path(__file__))!=report["runner_sha256"]:
        report.update(status="failed",errors=report["errors"]+[{"type":"ValueError","message":"runner source changed"}])
    save()
    print(json.dumps(report,indent=2))
    return int(report["status"]!="collected")


if __name__=="__main__":
    raise SystemExit(main())
