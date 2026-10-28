#!/usr/bin/env python3
"""GPU paired-GEMM typed stage/index/ownership qualification, no throughput claim."""
import argparse
import ctypes as C
import json
import math
from pathlib import Path
import shutil
import sys
import sysconfig

from collect import ROOT,sha256_file,write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

SOURCES=(tuple("tools/profiling/qwen_base/"+s for s in ("dual_gup.hpp","dual_gup_kernel_probe.mm","dual_gup_kernel_collect.py","collect.py"))+
    ("tools/native/build_qwen_dual_gup_kernel_probe.sh",))


def validate(report):
    if (report.get("status")!="complete" or report.get("synthetic_only") is not True or report.get("complete_request") is not False or
        report.get("image_export") is not False or report.get("extra_packed_weight_bytes")!=0 or report.get("rejected")!=20 or
        report.get("source_unchanged") is not True or report.get("owned_output_restoration_exact") is not True):
        raise ValueError("complete bounded source/illegal-input/owned-output kernel controls required")
    expected=[(f,m,k,w,first,count,shared,stage) for f,(m,k,w,first,count) in enumerate(((33,128,320,17,257),(1024,4096,12288,0,6144)))
              for shared in (False,True) for stage in range(5)]
    cases=report.get("cases",[])
    if [tuple(r[x] for x in ("fixture","m","k","f","first","count","shared","stage")) for r in cases]!=expected:
        raise ValueError("all twenty actual geometry/offset/tail/typed stage comparisons required")
    for r in cases:
        if (r.get("bm"),r.get("bn"))!=(32,64):raise ValueError("fixed initial kernel tile required")
        for name in ("quality_vs_mlx","quality_vs_mpp"):
            q=r.get(name) or {}
            if (q.get("bits_exact") is not True or q.get("changed")!=0 or q.get("relative_l2")!=0 or q.get("max_abs")!=0 or q.get("elements")!=r["m"]*r["count"]):
                raise ValueError("all full synthetic typed stages must match both ordinary MLX and unfused MPP before timing")
    return dict(full_stage_pairs=40,illegal_rejections=20,all_stages_bits_exact=True,
                scope="synthetic dyadic GPU kernel qualification; no checkpoint, ANE or full request speed")


def child(args):
    C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library))
    fn=lib.tc_dual_gup_kernel_probe;fn.argtypes=[C.c_char_p];fn.restype=C.c_int
    if fn(str(args.output).encode()):raise RuntimeError("dual GPU kernel control failed; inspect report/stderr")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("library","native-library","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--child",action="store_true");args=p.parse_args()
    for name in ("library","native_library"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        if any(sha256_file(ROOT/s["path"])!=s["sha256"] for s in manifest["inputs"]["sources"]):raise ValueError("native source identity changed")
    native_sources()
    if shutil.disk_usage(ROOT).free<(2<<30)+(512<<20):raise ValueError("2GiB reserve plus512MiB kernel arrays allowance required")
    sources={s:sha256_file(ROOT/s) for s in SOURCES};assets={s:sha256_file(getattr(args,s)) for s in ("library","native_library")}
    mlx_root=Path(sysconfig.get_path("purelib"))/"mlx";typed_header=mlx_root/"include/mlx/backend/metal/kernels/unary_ops.h"
    assets["mlx_library"]=sha256_file(mlx_root/"lib/libmlx.dylib");assets["mlx_unary_header"]=sha256_file(typed_header)
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    summary=dict(schema="tc-dual-gup-kernel-screen-v1",status="running",qualification_passed=False,source_sha256=sources,assets=assets,
                 runtime_build_id=manifest["runtime_build_id"],scope="synthetic GPU kernel qualification only")
    write_json(args.output/"summary.json",summary);write_json(args.output/"source.json",{s:(ROOT/s).read_text() for s in SOURCES})
    load=LoadObservation(args.output/"load.jsonl")
    try:
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--native-library",str(args.native_library),"--output",str(trial)]
        env={k:v for k,v in benchmark_environment().items() if not k.startswith("TURBOCIDER_") and k!="MLX_DISABLE_COMPILE"}
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=env,stdout=stdout,stderr=stderr,
                timeout=300,interval_ms=100,max_gap_ms=500,observer=load)
        report=json.loads((trial/"report.json").read_text())
        if (report["loaded_runtime_id"]!=manifest["runtime_build_id"] or report["compiled_native_id"]!=manifest["runtime_build_id"] or
            report["cpp_sha256"]!=sources["tools/profiling/qwen_base/dual_gup_kernel_probe.mm"] or report["header_sha256"]!=sources["tools/profiling/qwen_base/dual_gup.hpp"]):
            raise ValueError("actual compiled kernel/header/native ABI identity differs")
        summary["analysis"]=validate(report);summary["report_sha256"]=sha256_file(trial/"report.json")
        summary["arrays_sha256"]={p.name:sha256_file(p) for p in trial.glob("*.safetensors")};native_sources();summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=load.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(ROOT/s)!=v for s,v in sources.items()) or any(sha256_file(getattr(args,s))!=assets[s] for s in ("library","native_library")):
        summary.update(status="failed",error="bound kernel sources/assets changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
