#!/usr/bin/env python3
"""Real fixed-share FFN block replay and diagnostic restore encoder timestamps."""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

import collect
import ffn_projection_collect as block
from collect import ROOT,sha256_file,write_json,model_snapshot
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

SOURCES=tuple(dict.fromkeys(block.SOURCES+(
    "tools/profiling/qwen_base/handoff_collect.py",
    "tools/profiling/qwen_base/handoff_observer.mm",
    "tools/native/build_qwen_ffn_handoff_probe.sh",
    "tools/native/build_qwen_ffn_down_probe.sh")))
DOWN_POLICIES={f"m{bm}n{bn}{suffix}":(bm,bn,static) for bm in (16,32,64) for bn in (64,128)
               for suffix,static in (("d",False),("s",True))}


class HandoffObserver(collect.Observer):
    def __init__(self,path):
        self.lib=C.CDLL(str(path))
        self.lib.tc_qwen_profile_scope.argtypes=[C.c_int]
        self.lib.tc_qwen_profile_now_ns.restype=C.c_uint64
        self.lib.tc_qwen_profile_string_free.argtypes=[C.c_void_p]
        self.lib.tc_qwen_handoff_finish.restype=C.c_void_p
        self.lib.tc_qwen_profile_counters.restype=C.c_void_p
        if self.lib.tc_qwen_handoff_install():raise RuntimeError("handoff timestamp observer unavailable; no model started")
        if self.lib.tc_qwen_handoff_self_test():
            self.take("tc_qwen_profile_finish")
            raise RuntimeError("real event/encoder timestamp and data self-test failed; no model started")

    def take(self,name):
        return super().take("tc_qwen_handoff_finish" if name=="tc_qwen_profile_finish" else name)


def handoff_analysis(raw,report,repeats,*,delayed_self_test=True):
    """Keep GPU encoder elapsed separate from CB event-wait windows and host waits."""
    h=raw["handoff"]
    if (h.get("schema")!="tc-metal-event-encoder-timestamps-v1" or h.get("drained") is not True or
            any(type(h.get(key)) is not int or h[key]!=0 for key in ("pending","dropped","uncommitted_event_buffers"))):
        raise ValueError("complete non-dropped handoff event/counter evidence required")
    c=h["calibration"]
    names=("begin_host_before_ns","begin_host_after_ns","end_host_before_ns","end_host_after_ns","begin_gpu_ticks","end_gpu_ticks")
    if any(type(c.get(k)) is not int or c[k]<=0 for k in names):raise ValueError("actual GPU/host calibration anchors required")
    if not c["begin_host_before_ns"]<=c["begin_host_after_ns"]<c["end_host_before_ns"]<=c["end_host_after_ns"]:
        raise ValueError("GPU/host calibration anchors regressed")
    host_begin=(c["begin_host_before_ns"]+c["begin_host_after_ns"])/2
    host_end=(c["end_host_before_ns"]+c["end_host_after_ns"])/2
    if c["end_gpu_ticks"]<=c["begin_gpu_ticks"] or host_end-host_begin<1e6:raise ValueError("positive measured calibration interval required")
    uncertainty=c["begin_host_after_ns"]-c["begin_host_before_ns"]+c["end_host_after_ns"]-c["end_host_before_ns"]
    if uncertainty/(host_end-host_begin)>.001:raise ValueError("calibration bracket uncertainty too large")
    ns_per_tick=(host_end-host_begin)/(c["end_gpu_ticks"]-c["begin_gpu_ticks"])
    records=[]
    for row in h["records"]:
        if (row.get("ok") is not True or type(row.get("request")) is not int or
                any(type(row.get(k)) is not int for k in ("begin_ns","end_ns","gpu_begin_ns","gpu_end_ns")) or
                not 0<row["begin_ns"]<=row["end_ns"]):raise ValueError("failed/incomplete handoff command buffer")
        if row.get("encoder_sampled") is not True:continue
        a,b=row.get("encoder_begin_ticks"),row.get("encoder_end_ticks")
        if (row.get("resolved_bytes")!=16 or type(a) is not int or type(b) is not int or
                not c["begin_gpu_ticks"]<=a<b<=c["end_gpu_ticks"] or b==(1<<64)-1 or
                len(row["events"])!=1 or row["events"][0].get("wait") is not True):
            raise ValueError("actual resolved two-point event-wait encoder timestamps required")
        converted_a=host_begin+(a-c["begin_gpu_ticks"])*ns_per_tick
        converted_b=host_begin+(b-c["begin_gpu_ticks"])*ns_per_tick
        tolerance=max(1e6,uncertainty)
        if not row["begin_ns"]-tolerance<=converted_a<=converted_b<=row["end_ns"]+tolerance:
            raise ValueError("counter/host clock domains disagree with actual command completion")
        if not 0<row["gpu_begin_ns"]<=row["gpu_end_ns"]:raise ValueError("handoff CB GPU timestamps missing")
        records.append(dict(request=row["request"],event=row["events"][0]["event"],value=row["events"][0]["value"],
            encoder_gpu_seconds=(b-a)*ns_per_tick/1e9,
            cb_gpu_window_seconds=(row["gpu_end_ns"]-row["gpu_begin_ns"])/1e9,
            cb_host_lifetime_seconds=(row["end_ns"]-row["begin_ns"])/1e9))
    if sum(r["request"]==-4 for r in records)!=1:raise ValueError("real self-test encoder timestamp missing")
    delayed=[r for r in records if r["request"]==-5]
    if delayed_self_test and (len(delayed)!=1 or delayed[0]["cb_host_lifetime_seconds"]<.015 or
            delayed[0]["encoder_gpu_seconds"]>=delayed[0]["cb_host_lifetime_seconds"]*.25):
        raise ValueError("delayed-event self-test did not separate encoder elapsed from event wait")
    samples=[]
    expected={1,2,3,4,5}
    for offset,sample in enumerate(report["samples"]):
        if sample["arm"]!="private-restaged":continue
        request=6+offset;expected.add(request)
        found=[r for r in records if r["request"]==request]
        if len(found)!=1:raise ValueError("exactly one sampled restore per actual private block required")
        native=sample.get("completion_deltas")
        if (not native or set(native)!={"prediction_wait_seconds","output_wait_seconds","caller_wait_seconds"} or
                any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in native.values())):
            raise ValueError("actual nonnegative per-block native completion differences required")
        samples.append(dict(**found[0],rep=sample["rep"],warmup=sample["warmup"],native_completion_deltas=native))
    workload=[r for r in records if r["request"]>=1]
    if len(workload)!=repeats+7 or {r["request"] for r in workload}!=expected:
        raise ValueError("missing/extra encoder consumers or wrong scope attribution")
    if len({r["event"] for r in workload})!=1:raise ValueError("one actual FFN ready/done event identity required")
    ordered=sorted(workload,key=lambda r:r["request"])
    if any(b["value"]-a["value"]!=2 for a,b in zip(ordered,ordered[1:])):
        raise ValueError("actual fixed FFN done timeline reset or skipped a consumer")
    hot=[r for r in samples if not r["warmup"]]
    return dict(schema="tc-qwen-handoff-encoder-analysis-v1",calibrated_ns_per_gpu_tick=ns_per_tick,
        calibration_bracket_fraction=uncertainty/(host_end-host_begin),actual_workload_restore_encoders=len(workload),samples=samples,
        hot_encoder_gpu_median_seconds=statistics.median(r["encoder_gpu_seconds"] for r in hot),
        hot_cb_gpu_window_median_seconds=statistics.median(r["cb_gpu_window_seconds"] for r in hot),
        hot_output_wait_median_seconds=statistics.median(r["native_completion_deltas"]["output_wait_seconds"] for r in hot),
        delayed_event_self_test=delayed[0] if delayed else None,
        scope="Diagnostic encoder-boundary elapsed, not per-shader busy time or physical ANE compute. CB GPU windows can include event waits; overlapping spans must not be summed.")


def down_analysis(report,repeats,channels,policy):
    if policy not in DOWN_POLICIES:raise ValueError("explicit supported F32 down tile required")
    bm,bn,static=DOWN_POLICIES[policy]
    expected=dict(name=policy,bm=bm,bn=bn,static_tiles=static,physical_weight_pitch=12288,
                  input_width=12288-channels,output_dtype="float32",weight_copies=0)
    actual=report.get("gpu_down_policy")
    if actual!=expected or any(type(actual[k]) is not type(v) for k,v in expected.items()):
        raise ValueError("actual compiled tile/F32/original physical pitch/no-copy contract differs")
    for key in ("down_quality_a","down_quality_b"):
        q=report.get(key,{})
        if (type(q.get("relative_l2")) not in (int,float) or not math.isfinite(q["relative_l2"]) or
                not 0<=q["relative_l2"]<1e-4 or q.get("elements")!=4194304):
            raise ValueError("complete finite original F32 down numerical controls required")
    arms=("gate-up-alone","down-alone","control-down-alone")
    rows=report.get("gpu_down_component_samples",[])
    expected_order=[(arms[(j+rep+3)%3],rep) for rep in range(-2,repeats) for j in range(3)]
    if [(r.get("arm"),r.get("rep")) for r in rows]!=expected_order:
        raise ValueError("all rotated gate/up and F32 down component samples required")
    for row in rows:
        if (row.get("warmup") is not (row["rep"]<0) or row.get("input_variant") is not bool(row["rep"]&1) or
                type(row.get("seconds")) not in (int,float) or not math.isfinite(row["seconds"]) or row["seconds"]<=0):
            raise ValueError("actual finite component interval required")
    return {arm:statistics.median(r["seconds"] for r in rows if r["arm"]==arm and not r["warmup"]) for arm in arms}


def validate(report,repeats,channels,strategy="separate",*,down_policy="none",gate_up="bf16"):
    result=block.validate(report,repeats,"descriptor","sigmoid-complement",channels=channels)
    if (report.get("handoff_screen") is not True or report.get("ane_channels")!=channels or
            report.get("gpu_channels")!=12288-channels or strategy not in ("separate","packed") or
            report.get("gpu_complement_strategy")!=strategy):raise ValueError("actual explicitly compiled handoff share/strategy required")
    expected_bytes=2*2*(12288-channels)*4096*2 if strategy=="packed" else 0
    if (report.get("gpu_packed_weight_bytes")!=expected_bytes or type(report.get("gpu_packed_setup_seconds")) not in (int,float) or
            not math.isfinite(report["gpu_packed_setup_seconds"]) or report["gpu_packed_setup_seconds"]<0):
        raise ValueError("explicit two-layer packed setup cost/storage required")
    rows=report.get("gpu_complement_alone_samples",[])
    if len(rows)!=repeats+2 or [r["rep"] for r in rows]!=list(range(-2,repeats)):
        raise ValueError("all independent GPU complement warm/measured samples required")
    for row in rows:
        if (row.get("warmup") is not (row["rep"]<0) or row.get("input_variant") is not bool(row["rep"]&1) or
                type(row.get("seconds")) not in (int,float) or not math.isfinite(row["seconds"]) or row["seconds"]<=0):
            raise ValueError("finite actual GPU complement alone timing required")
    result["gpu_complement_alone_median_seconds"]=statistics.median(r["seconds"] for r in rows if not r["warmup"])
    if down_policy!="none":
        if strategy!="separate":raise ValueError("down tile screen cannot mix the rejected packed GPU strategy")
        result["gpu_down_component_median_seconds"]=down_analysis(report,repeats,channels,down_policy)
    elif "gpu_down_policy" in report:raise ValueError("down screen requires an explicit policy, not legacy default")
    if gate_up not in ("bf16","fp16") or report.get("gpu_gate_up_precision","bf16")!=gate_up:
        raise ValueError("actual explicit GPU gate/up precision differs")
    if gate_up=="fp16":
        if down_policy=="none" or report.get("gpu_gate_up_weight_binding")!="runtime-arguments-per-call-cast":
            raise ValueError("FP16 screen must include actual per-call conversions, no captured half bank")
        for key in ("gate_up_quality_a","gate_up_quality_b"):
            q=report.get(key,{})
            if (type(q.get("relative_l2")) not in (int,float) or not math.isfinite(q["relative_l2"]) or
                    not 0<=q["relative_l2"]<.01 or q.get("elements")!=1024*(12288-channels)):
                raise ValueError("complete explicit approximate GPU gate/up control required")
    return result


def child(args):
    os.environ["TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION"]="sigmoid-complement"
    observer=HandoffObserver(args.observer) if args.mode=="timeline" else None
    native=C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library))
    callback_type=C.CFUNCTYPE(None,C.c_int)
    callback=callback_type(lambda phase:observer.lib.tc_qwen_profile_scope(phase)) if observer else callback_type()
    fn=lib.tc_ffn_projection_probe
    fn.argtypes=[C.c_char_p,C.c_char_p,C.c_char_p,C.c_char_p,C.c_char_p,C.c_int,callback_type];fn.restype=C.c_int
    try:
        if fn(str(args.model).encode(),b"-",str(args.manifest).encode(),str(args.output).encode(),b"descriptor",args.repeats,callback):
            raise RuntimeError("real FFN handoff capture failed; inspect original report")
    finally:
        if observer:write_json(args.output/"observer.json",observer.take("tc_qwen_profile_finish"))
    _=(native,lib)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("library","native-library","observer","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--channels",type=int,choices=(5120,6144),required=True)
    p.add_argument("--mode",choices=("stock","timeline"),default="stock")
    p.add_argument("--gpu-complement",choices=("separate","packed"),default="separate")
    p.add_argument("--down-policy",choices=("none",*DOWN_POLICIES),default="none")
    p.add_argument("--gate-up",choices=("bf16","fp16"),default="bf16")
    p.add_argument("--repeats",type=int,choices=(8,16),default=8)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--manifest",type=Path,default=ROOT/"outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json")
    p.add_argument("--child",action="store_true");args=p.parse_args()
    for name in ("library","native_library","observer","model","manifest"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.resolve()
    if args.down_policy!="none" and args.gpu_complement!="separate":p.error("down policy screen requires separate GPU gate/up")
    if args.gate_up!="bf16" and args.down_policy=="none":p.error("FP16 GPU gate/up requires an explicit down screen")
    if args.child:child(args);return
    native=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        if any(sha256_file(ROOT/e["path"])!=e["sha256"] for e in native["inputs"]["sources"]):raise ValueError("native source changed")
    native_sources()
    if shutil.disk_usage(ROOT).free<(2304<<20):p.error("reserve2GiB plus256MiB output before model capture")
    hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={x:sha256_file(getattr(args,x)) for x in ("library","native_library","observer","manifest")}
    identity=model_snapshot(args.model);args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    summary=dict(schema="tc-ffn-handoff-block-screen-v1",status="running",qualification_passed=False,channels=args.channels,
        mode=args.mode,gpu_complement=args.gpu_complement,down_policy=args.down_policy,gate_up=args.gate_up,repeats=args.repeats,source_sha256=hashes,assets=assets,model_snapshot=identity,runtime_build_id=native["runtime_build_id"])
    write_json(args.output/"summary.json",summary);write_json(args.output/"source.json",{x:(ROOT/x).read_text() for x in SOURCES})
    load=LoadObservation(args.output/"load.jsonl")
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child"]
    for key in ("library","native_library","observer","model","manifest","channels","repeats","mode","gpu_complement","down_policy","gate_up"):
        command += ["--"+key.replace("_","-"),str(getattr(args,key))]
    command += ["--output",str(trial)]
    try:
        env={k:v for k,v in benchmark_environment().items() if not k.startswith("TURBOCIDER_") and k!="MLX_DISABLE_COMPILE"}
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=env,stdout=stdout,stderr=stderr,
                timeout=300,interval_ms=100,max_gap_ms=500,observer=load)
        report=json.loads((trial/"report.json").read_text())
        if report["cpp_sha256"]!=hashes["tools/profiling/qwen_base/ffn_projection_probe.mm"] or report["loaded_runtime_id"]!=native["runtime_build_id"]:
            raise ValueError("actual probe/native identity differs")
        summary["analysis"]=validate(report,args.repeats,args.channels,args.gpu_complement,down_policy=args.down_policy,gate_up=args.gate_up);summary["arrays_sha256"]=sha256_file(trial/"arrays.safetensors")
        if args.down_policy!="none":summary["down_arrays_sha256"]=sha256_file(trial/"down-arrays.safetensors")
        if args.mode=="timeline":
            raw=json.loads((trial/"observer.json").read_text())
            if raw["source_sha256"]!=hashes["tools/profiling/qwen_base/observer.mm"] or raw["handoff"]["source_sha256"]!=hashes["tools/profiling/qwen_base/handoff_observer.mm"]:
                raise ValueError("actual base/handoff observer source differs")
            summary["observer"]=block.observer_contract(raw,args.repeats+7)
            summary["handoff"]=handoff_analysis(raw,report,args.repeats)
        native_sources();summary["status"]="complete"
    except Exception as error:
        summary.update(status="failed",error=str(error))
        try:summary["probe_error"]=json.loads((trial/"report.json").read_text()).get("error")
        except (OSError,ValueError):pass
    try:summary["load"]=load.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if model_snapshot(args.model)!=identity or any(sha256_file(ROOT/x)!=d for x,d in hashes.items()) or any(sha256_file(getattr(args,x))!=d for x,d in assets.items()):
        summary.update(status="failed",error="bound source/model/asset changed")
    write_json(args.output/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("source_sha256","model_snapshot","handoff")},indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
