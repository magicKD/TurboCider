#!/usr/bin/env python3
"""Serial same-library Qwen edits: GPU, prefill-only, decode-only, all-phase FFN.

Local models/real adapters only. Every request misses conditioning; preserve
cold and fresh warm samples, actual phase calls, memory/load and all PNGs.
These host diagnostics do not establish physical overlap or visual acceptance.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics

from qwen_encoder_residency_screen import (make_request, model_snapshot, validate_rows,
    validate_shared_ranks, validate_b_epilogue, validate_joint_ab)
from runtime_ane_common import benchmark_environment, sha256_file, validate_deferred_channel_join
from qwen_lora_source_proof import validate_source_proof
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_owned, run_sampled

ROOT=Path(__file__).resolve().parents[2]
MODES=("gpu","prefill","decode","all")
LAYER_MODES=("gpu","prefill","prefill_first8","prefill_last8")
LAYER_POLICIES={"prefill_first8":tuple(range(8)),"prefill_last8":tuple(range(24,32))}

def reference_resize_flags(references,lora):
    return {"TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC":"1"} if references and lora else {}


def count(data,key):
    value=data.get(key)
    if type(value) is not int or value<0:raise ValueError("missing/invalid phase counter: "+key)
    return value


def validate_phases(rows,policy,lora=False,gpu_layers=()):
    if not rows or policy not in MODES:raise ValueError("missing/unknown phase requests")
    if not isinstance(gpu_layers,tuple) or len(gpu_layers)>=32 or any(type(i) is not int or not 0<=i<32 for i in gpu_layers) or \
        gpu_layers!=tuple(sorted(set(gpu_layers))) or (gpu_layers and policy!="prefill"):
        raise ValueError("GPU layers require canonical partial prefill-only ordinals0..31")
    calls_before=blocks_before=forced_before=0
    for row in rows:
        evidence=row.get("qwen_ffn_phases")
        if not isinstance(evidence,dict) or evidence.get("policy")!=policy:
            raise ValueError("actual FFN phase policy receipt missing/mismatched")
        steps=count(row,"actual_denoise_steps")
        if steps<2 or steps!=row.get("steps") or row.get("student_ffn_reuse") is not None:
            raise ValueError("phase screen requires complete steps, not temporal reuse")
        target=count(row,"height")//16*(count(row,"width")//16)
        if target!=1024:raise ValueError("phase screen requires the matched512 target geometry")
        prefill_rows=target+count(row,"text_tokens")+count(row,"reference_tokens")
        hybrid=row.get("hybrid") or {};runtime=hybrid.get("runtime_weight") or {}
        bucket=count(hybrid,"bucket") if policy!="gpu" else 1
        if bucket<=0:raise ValueError("missing actual ANE bucket")
        total_calls=total_blocks=0;elapsed=0
        for name,nsteps,nrows in (("prefill",1,prefill_rows),("decode",steps-1,target)):
            phase=evidence.get(name)
            if not isinstance(phase,dict):raise ValueError("missing actual phase execution")
            active=policy in (name,"all")
            blocks=(32-len(gpu_layers))*nsteps if active else 0
            calls=blocks*((nrows+bucket-1)//bucket)
            if (count(phase,"steps_this_request")!=nsteps or count(phase,"actual_rows")!=nrows or
                count(phase,"runtime_calls_this_request")!=calls or count(phase,"completed_channel_blocks_this_request")!=blocks):
                raise ValueError("phase actual rows/steps/calls/blocks do not match the selected split")
            value=phase.get("step_seconds")
            if type(value) not in (float,int) or not math.isfinite(value) or value<=0:
                raise ValueError("missing/nonfinite/nonpositive completed phase timing")
            elapsed+=value;total_calls+=calls;total_blocks+=blocks
        denoise=(row.get("timings_seconds") or {}).get("denoise")
        if type(denoise) not in (float,int) or not math.isfinite(denoise) or elapsed>denoise+.05:
            raise ValueError("phase timings cannot exceed their enclosing denoise span")
        if policy!="gpu":
            if (hybrid.get("runtime_failed") is not False or count(hybrid,"runtime_failures_session_total") or
                count(runtime,"fallback_blocks_session_total") or count(runtime,"overflow_retries_session_total") or
                runtime.get("headroom_scale")!=1 or runtime.get("executor_backend")!="private_ane" or
                runtime.get("partition_axis")!="intermediate_channels" or runtime.get("data_path")!="w8a8_hadamard" or
                count(runtime,"forced_gpu_blocks_session_total")!=forced_before+len(gpu_layers)):
                raise ValueError("phase screen has a failure/retry/wrong forced-layer work or representation")
            if gpu_layers:
                marker="experimental runtime prefill complete-GPU FFN blocks="+",".join(map(str,gpu_layers))
                if runtime.get("requested_gpu_layers")!=list(gpu_layers) or \
                    count(runtime,"unsplit_gpu_blocks_session_total")!=forced_before+len(gpu_layers) or \
                    marker not in (row.get("acceleration_selection") or ""):
                    raise ValueError("prefill layer policy must perform actual unsplit GPU work, not relabel ANE counters")
            elif runtime.get("requested_gpu_layers"):
                raise ValueError("unselected prefill layer policy leaked into the control")
            if (count(hybrid,"runtime_calls_session_total")!=calls_before+total_calls or
                count(runtime,"channel_blocks_session_total")!=blocks_before+total_blocks or
                count(runtime,"async_hybrid_blocks_session_total")!=blocks_before+total_blocks):
                raise ValueError("request-local phase counters disagree with cumulative actual async work")
            calls_before+=total_calls;blocks_before+=total_blocks
            forced_before+=len(gpu_layers)
            if lora:
                shared=row.get("shared_lora_ranks") or {}
                if count(shared,"completed_hybrid_blocks_this_request")!=total_blocks:
                    raise ValueError("shared ranks were counted in the disabled GPU phase")
        elif total_calls or total_blocks or hybrid:
            raise ValueError("complete GPU control cannot publish ANE phase work")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--dit-manifest",type=Path,required=True)
    parser.add_argument("--reference",type=Path,action="append",default=[])
    parser.add_argument("--generation",action="store_true",help="original base/LoRA generation with no reference inputs; compare actual prefill and KV-hit phases")
    parser.add_argument("--prompt",action="append",required=True)
    parser.add_argument("--lora",type=Path)
    parser.add_argument("--fused-b",action="store_true",help="same fused B precision on every arm; requires the real six-step adapter")
    parser.add_argument("--joint-ab",action="store_true",help="same existing joint BF16 A/B/F32-rank profile on every arm; excludes fused-B-only")
    parser.add_argument("--prefill-layer-screen",action="store_true",help="GPU/all32 parallel/first8 GPU/last8 GPU, remaining prefill FFNs parallel and all decode complete GPU")
    parser.add_argument("--defer-prefill-join",action="store_true",help="same existing owned deferred join on all prefill hybrid arms")
    parser.add_argument("--channels",type=int,default=5120)
    parser.add_argument("--modes",help="matched subset containing gpu plus at least one known phase/layer arm")
    parser.add_argument("--order")
    parser.add_argument("--sample-memory",action="store_true")
    parser.add_argument("--observe-load",action="store_true")
    parser.add_argument("--timeout",type=int,default=900)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    allowed=LAYER_MODES if args.prefill_layer_screen else MODES
    modes=args.modes.split(",") if args.modes else list(allowed);order=args.order.split(",") if args.order else modes
    if len(modes)<2 or len(set(modes))!=len(modes) or "gpu" not in modes or not set(modes)<=set(allowed):
        parser.error("need complete GPU plus distinct known phase arms")
    if len(order)!=len(modes) or set(order)!=set(modes):parser.error("order must include each selected mode exactly once")
    if (bool(args.reference)==args.generation or len(args.reference)>2 or (args.generation and args.prefill_layer_screen) or
        not 3<=len(args.prompt)<=9 or len(set(args.prompt))!=len(args.prompt) or any(not p.strip() for p in args.prompt)):
        parser.error("need generation without references or one/two edit references, and three..nine distinct fresh prompts; layer screen is edit-only")
    if not 0<args.channels<12288 or args.channels%512:parser.error("fixed Private channels must be aligned and partial")
    if args.fused_b and not args.lora:parser.error("fused B requires a real unmerged adapter")
    if args.joint_ab and (not args.lora or args.fused_b):parser.error("joint A/B requires a real adapter and excludes B-only")
    if args.defer_prefill_join and any(mode not in LAYER_MODES for mode in modes):parser.error("deferred join requires prefill-only arms")
    if not 1<=args.timeout<=3600:parser.error("timeout must be1..3600")
    if args.output.exists() or args.output.is_symlink():parser.error("choose a fresh output directory")
    cli=args.cli.resolve(strict=True);library=cli.parent/"libturbocider.dylib"
    if not library.is_file() or not os.access(cli,os.X_OK):parser.error("executable CLI and adjacent dylib required")
    model=args.model.resolve(strict=True);manifest=args.dit_manifest.resolve(strict=True)
    refs=[p.resolve(strict=True) for p in args.reference];lora=args.lora.resolve(strict=True) if args.lora else None
    before={str(p):sha256_file(p) for p in (cli,library,manifest,*refs,*([lora] if lora else []))}
    model_before=model_snapshot(model);args.output.mkdir(parents=True)
    summary=dict(schema="tc-qwen-ffn-phase-screen-v1",time_utc=datetime.now(timezone.utc).isoformat(),status="incomplete",
        qualification_passed=False,order=order,channels=args.channels,fused_b=args.fused_b,joint_ab=args.joint_ab,
        prefill_layer_screen=args.prefill_layer_screen,defer_prefill_join=args.defer_prefill_join,generation=args.generation,
        operation="image.generate" if args.generation else "image.edit",prompts=args.prompt,
        source_identities=before,model_snapshot=model_before,
        model_identity_scope="file-generation stamps and bounded headers, not immutable full-payload signatures",
        scope="serial same-library fresh-condition native request walls; explicit Private channel FFN and complete GPU control; host diagnostics, not physical overlap proof",trials=[])
    target=args.output/"summary.json";target.write_text(json.dumps(summary,indent=2)+"\n")
    for mode in order:
        policy="gpu" if mode=="gpu" else "prefill" if mode in LAYER_MODES else mode
        gpu_layers=LAYER_POLICIES.get(mode,())
        if model_snapshot(model)!=model_before:raise ValueError("model generation changed between arms")
        env=benchmark_environment();env.update(TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS="1",TURBOCIDER_QWEN21_PROFILE_STEPS="1",
            TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE="all" if mode=="gpu" else policy,
            TURBOCIDER_QWEN21_LORA_B_FUSED_EPILOGUE="1" if args.fused_b else "0",
            TURBOCIDER_QWEN21_LORA_BF16_AB="1" if args.joint_ab else "0")
        if gpu_layers:env["TURBOCIDER_QWEN21_PREFILL_GPU_FFN_BLOCKS"]=",".join(map(str,gpu_layers))
        env.update(reference_resize_flags(refs,lora))
        if mode!="gpu":
            env.update(TURBOCIDER_ANE_BACKEND="private",TURBOCIDER_ALLOW_PRIVATE_ANE="1",TURBOCIDER_PRIVATE_ANE_CHANNELS=str(args.channels),
                TURBOCIDER_PRIVATE_ANE_DATA_PATH="w8a8",TURBOCIDER_PRIVATE_ANE_GPU_IO="1",TURBOCIDER_RUNTIME_ANE_CHUNKS="1",
                TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1",TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE="1",TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE="1",
                TURBOCIDER_PRIVATE_ANE_PREFETCH="0",TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD="0",TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN="1")
            if args.defer_prefill_join:env["TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN"]="1"
            if lora:env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1"
        requests=[]
        for i,prompt in enumerate(args.prompt):
            path=args.output/f"{mode}-{i}.json"
            path.write_text(json.dumps(make_request(prompt,refs,(args.output/f"{mode}-{i}.png").resolve(),lora=lora,
                dit_manifest=manifest if mode!="gpu" else None),indent=2)+"\n");requests.append(str(path.resolve()))
        command=[str(cli),"batch",str(model),*requests]
        observer=LoadObservation(args.output/f"{mode}-load.jsonl") if args.observe_load else None
        print(json.dumps(dict(starting=mode)),flush=True)
        with (args.output/f"{mode}.stdout.jsonl").open("x") as stdout,(args.output/f"{mode}.stderr.txt").open("x") as stderr:
            memory=run_sampled(command,repo=ROOT,output=args.output,stem=mode,env=env,stdout=stdout,stderr=stderr,
                timeout=args.timeout,interval_ms=100,max_gap_ms=500,observer=observer) if args.sample_memory else None
            if not args.sample_memory:
                result=run_owned(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=args.timeout,observer=observer)
                if result.returncode:raise RuntimeError("phase request failed; raw evidence retained")
        load=observer.verify() if observer else None
        rows=[json.loads(line) for line in (args.output/f"{mode}.stdout.jsonl").read_text().splitlines()]
        validate_rows(rows,"gpu_weights" if mode=="gpu" else "dit_weights",len(args.prompt),True,None,args.channels)
        validate_phases(rows,policy,bool(lora),gpu_layers)
        if lora:
            if args.joint_ab:validate_joint_ab(rows,True)
            else:validate_b_epilogue(rows,args.fused_b)
            if mode!="gpu":validate_shared_ranks(rows,True)
            if mode!="gpu":validate_source_proof((args.output/f"{mode}.stderr.txt").read_text().splitlines(),lora.stat().st_size,len(rows))
        if mode!="gpu" and policy=="prefill":validate_deferred_channel_join(rows,args.defer_prefill_join)
        if any(sha256_file(Path(p))!=h for p,h in before.items()) or model_snapshot(model)!=model_before:
            raise ValueError("runtime/model/reference/adapter identity changed")
        times=[r["timings_seconds"]["request_wall"] for r in rows]
        trial=dict(mode=mode,phase_policy=policy,gpu_layers=list(gpu_layers),cold_request_seconds=times[0],warm_fresh_request_seconds=times[1:],
            warm_fresh_median_seconds=statistics.median(times[1:]),phase_receipts=[r["qwen_ffn_phases"] for r in rows],
            warm_prefill_median_seconds=statistics.median(r["qwen_ffn_phases"]["prefill"]["step_seconds"] for r in rows[1:]),
            warm_decode_total_median_seconds=statistics.median(r["qwen_ffn_phases"]["decode"]["step_seconds"] for r in rows[1:]),
            memory=memory,load=load,png_sha256=[sha256_file(args.output/f"{mode}-{i}.png") for i in range(len(rows))])
        summary["trials"].append(trial);target.write_text(json.dumps(summary,indent=2)+"\n")
    summary["status"]="complete_diagnostic";target.write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
