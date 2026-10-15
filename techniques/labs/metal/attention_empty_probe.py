#!/usr/bin/env python3
"""Small explicit mask-semantics diagnostic; return0 means collection, not qualification."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

from attention_source import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    report={"schema":"tc-attention-empty-diagnostic-v1","status":"starting","synthetic":True,
            "production_qualified":False,"formal_performance_score":False,"cases":[],"errors":[],
            "runner_sha256":digest(Path(__file__)),"physical_dispatch_not_verified":True,
            "contract":"all-false query row: explicit local zero convention, not a universal framework promise"}
    shutil.copy2(Path(__file__),output/Path(__file__).name)

    def save():
        (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    save();mx=None
    try:
        if platform.system()!="Darwin":raise ValueError("macOS GPU required")
        import numpy as np
        import mlx.core as mx

        mx.set_default_device(mx.gpu);info=mx.device_info()
        report["environment"]={"mlx":version("mlx"),"macos":platform.mac_ver()[0],
            "device":{key:info.get(key) for key in ("device_name","architecture","memory_size")},
            "core_binary_sha256":digest(Path(mx.__file__))}
        for dim in (64,96):
            for keys in (29,32):
                q=mx.full((1,4,17,dim),.125,dtype=mx.bfloat16)
                k=mx.full((1,4,keys,dim),.125,dtype=mx.bfloat16)
                v=mx.ones((1,4,keys,dim),dtype=mx.bfloat16)
                mask=np.ones((17,keys),dtype=np.bool_);mask[0]=False
                m=mx.array(mask);mx.eval(q,k,v,m)
                result=mx.fast.scaled_dot_product_attention(q,k,v,scale=dim**-.5,mask=m)
                mx.eval(result);actual=np.array(result.astype(mx.float32));finite=bool(np.isfinite(actual).all())
                report["cases"].append({"head_dim":dim,"key_rows":keys,"query_rows":17,"value_constant":1.,
                    "finite":finite,"masked_first_row_min":float(actual[:,:,0,:].min()) if finite else None,
                    "masked_first_row_max":float(actual[:,:,0,:].max()) if finite else None,
                    "nonempty_rows_max_abs_vs_one":float(np.max(np.abs(actual[:,:,1:,:]-1))) if finite else None,
                    "local_zero_row_gate_pass":finite and bool(np.max(np.abs(actual[:,:,0,:]))<=1e-6)})
                save()
        report["status"]="diagnostic-collected"
    except Exception as error:
        report["status"]="failed";report["errors"].append({"type":type(error).__name__,"message":str(error)})
    finally:
        if mx is not None:
            try:mx.synchronize()
            except Exception as error:report["status"]="failed";report["errors"].append({"type":type(error).__name__,"message":str(error)})
        save()
    print(json.dumps({"status":report["status"],"zero_row_gate_passes":sum(case["local_zero_row_gate_pass"] for case in report["cases"]),
                      "production_qualified":False}))
    return int(report["status"]!="diagnostic-collected")


if __name__=="__main__":
    raise SystemExit(main())
