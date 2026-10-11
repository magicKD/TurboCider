#!/usr/bin/env python3
"""Actual source-bound Core ML prefix versus dense GPU FFN; not full latency."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/coreml"))
from flux_checkpoint import Checkpoint, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "manifest", "library", "output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():parser.error("fresh probe output required")
    import mlx.core as mx
    config = geometry(json.loads((args.model / "transformer/config.json").read_text()))
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["shape"]["buckets"][0]
    width = manifest["shape"]["ane_mlp_end"]
    hidden = config["hidden"]
    if (hidden,config["single_layers"],rows,width)!=(4096,24,1536,4096):
        raise ValueError("exact source9B/prefix4096/rows1536 probe required")
    checkpoint = Checkpoint(args.model)
    lib = C.CDLL(str(args.library.resolve(strict=True)))
    ptr = C.c_void_p
    u16 = C.POINTER(C.c_uint16)
    lib.tc_coreml_ffn_create.argtypes = [C.c_char_p,C.c_char_p,C.c_int,C.c_int,C.POINTER(ptr),C.POINTER(ptr)]
    lib.tc_coreml_ffn_predict.argtypes = [ptr,C.c_int,u16,C.c_int,u16,C.POINTER(ptr)]
    lib.tc_coreml_ffn_metrics_json.argtypes = [ptr,C.POINTER(ptr),C.POINTER(ptr)]
    lib.tc_coreml_ffn_free.argtypes = [ptr]
    lib.tc_string_free.argtypes = [ptr]

    def take(pointer):
        value=C.string_at(pointer).decode() if pointer.value else ""
        if pointer.value:lib.tc_string_free(pointer)
        return value

    bridge,error=ptr(),ptr()
    start=time.perf_counter()
    if lib.tc_coreml_ffn_create(str(args.manifest.resolve()).encode(),str(checkpoint.path).encode(),rows,0,C.byref(bridge),C.byref(error)):
        raise RuntimeError(take(error))
    result=dict(schema="tc-flux9-coreml-prefix-probe-v1",status="running",qualification_passed=False,goal_complete=False,
        load_seconds=time.perf_counter()-start,geometry=config,rows=rows,ane_channels=width,trials=[],
        scope="original BF16 weights narrowed to FP16 source-prefix baseline; CoreML INT8 per-channel weight compression, FP16 I/O; no activation8/native integer MAC or whole-model speedup claim")
    try:
        with checkpoint as source:
            for block in (0,12,23):
                prefix=f"single_transformer_blocks.{block}.attn"
                gate=source.tensor(prefix+".to_qkv_mlp_proj.weight",(hidden*9,hidden),np,rows=(hidden*3,hidden*3+width))
                up=source.tensor(prefix+".to_qkv_mlp_proj.weight",(hidden*9,hidden),np,rows=(hidden*6,hidden*6+width))
                down=source.tensor(prefix+".to_out.weight",(hidden,hidden*4),np,columns=(hidden,hidden+width))
                wg,wu,wd=mx.array(gate),mx.array(up),mx.array(down)
                mx.eval(wg,wu,wd)
                for amplitude in (0.25,1.0):
                    x=np.ascontiguousarray(np.random.default_rng(42+block).normal(0,amplitude,(1,rows,hidden)).astype(np.float16))
                    xx=mx.array(x)
                    gx,ux=mx.matmul(xx,mx.transpose(wg)),mx.matmul(xx,mx.transpose(wu))
                    expected=mx.matmul(gx*mx.sigmoid(gx)*ux,mx.transpose(wd))
                    mx.eval(expected)
                    baseline=np.array(expected).astype(np.float64)
                    output=np.empty_like(x)
                    error=ptr()
                    start=time.perf_counter()
                    status=lib.tc_coreml_ffn_predict(bridge,block,x.view(np.uint16).ctypes.data_as(u16),rows,
                        output.view(np.uint16).ctypes.data_as(u16),C.byref(error))
                    elapsed=time.perf_counter()-start
                    if status:raise RuntimeError(take(error))
                    actual=output.astype(np.float64)
                    if not np.isfinite(actual).all() or not np.isfinite(baseline).all():raise ValueError("finite actual FFN output required")
                    difference=actual-baseline
                    relative=float(np.sqrt(np.sum(difference*difference)/max(np.sum(baseline*baseline),1e-30)))
                    cosine=float(np.sum(actual*baseline)/max(np.sqrt(np.sum(actual*actual)*np.sum(baseline*baseline)),1e-30))
                    result["trials"].append(dict(block=block,amplitude=amplitude,actual_coreml_wall=elapsed,relative_l2=relative,
                        cosine=cosine,max_abs=float(np.max(np.abs(difference))),mae=float(np.mean(np.abs(difference)))))
        metrics,error=ptr(),ptr()
        if lib.tc_coreml_ffn_metrics_json(bridge,C.byref(metrics),C.byref(error)):raise RuntimeError(take(error))
        result["metrics"]=json.loads(take(metrics))
        if result["metrics"].get("runtime_failed") is not False or result["metrics"].get("runtime_calls_session_total")!=6:
            raise ValueError("actual six successful source-bound CoreML calls required")
        result["status"]="complete_diagnostic"
    except BaseException as exception:
        result.update(status="failed",error=str(exception))
        raise
    finally:
        lib.tc_coreml_ffn_free(bridge)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as file:json.dump(result,file,indent=2,allow_nan=False);file.write("\n")
    print(json.dumps(dict(status=result["status"],trials=result["trials"])))


if __name__=="__main__":main()
