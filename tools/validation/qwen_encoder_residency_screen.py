#!/usr/bin/env python3
"""Matched fresh-condition Qwen requests: GPU / local / retained encoder.

Serial, explicit local fixtures only. Keep every prompt, PNG and observation;
hot requests must miss the conditioning cache. Never infer visual acceptance,
physical overlap, or production qualification from these timings.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_owned, run_sampled

ROOT=Path(__file__).resolve().parents[2]
MODES=("gpu","encoder_local","encoder_retained")
WEIGHT_MODES=("gpu","gpu_weights","encoder_retained","encoder_weights")
COMBINED_MODES=("gpu_weights","dit_weights","dit_encoder_weights")
LORA_RANK_MODES=("ranks_off","ranks_on")
WEIGHT_CODE_MODES=("code_cache_off","code_cache_on")


def validate_weight_code_cache(rows, budget, storage="copy"):
    previous_hits=0
    for row in rows:
        cache=((row.get("hybrid") or {}).get("runtime_weight") or {}).get("weight_code_cache")
        if not isinstance(cache,dict) or cache.get("enabled") is not (budget>0) or \
            type(cache.get("budget_bytes")) is not int or cache.get("budget_bytes")!=budget:
            raise ValueError("converted weight code cache policy receipt missing/mismatched")
        if cache.get("native_surface_storage") is not (storage=="surface"):
            raise ValueError("converted weight cache storage receipt mismatched")
        names=("hits_session_total","misses_session_total","fills_session_total","failed_fills_session_total",
               "entries","ready_entries","retained_bytes","live_capacity_bytes","peak_capacity_bytes",
               "evictions_session_total","declines_session_total","ineligible_session_total")
        values=[cache.get(key) for key in names]
        if any(type(value) is not int or value<0 for value in values):raise ValueError("invalid converted code cache counters")
        hits,misses,fills,failed,entries,ready,retained,live,peak,*_=values
        if budget:
            if hits<=previous_hits or not 0<ready<=entries<=128 or not 0<retained<=live<=peak<=budget or failed or fills>misses:
                raise ValueError("need real successful converted-code reuse within admitted capacity")
        elif any(values):raise ValueError("disabled converted-code cache executed/retained work")
        previous_hits=hits
        copy_hits=cache.get("copy_hits_session_total");bind_hits=cache.get("surface_bind_hits_session_total")
        if any(type(v) is not int or v<0 for v in (copy_hits,bind_hits)) or copy_hits+bind_hits!=hits or \
            (storage=="surface" and budget and (bind_hits<=0 or copy_hits)) or (storage=="copy" and bind_hits):
            raise ValueError("need actual separately counted copy/surface reuse, not storage intent")


def validate_shared_ranks(rows, enabled):
    """Require completed dual-consumer work, not a selection-label claim."""
    for row in rows:
        metrics=row.get("shared_lora_ranks")
        if not isinstance(metrics,dict) or metrics.get("enabled") is not enabled:
            raise ValueError("shared rank policy receipt missing/mismatched")
        keys=("prepared_sets_this_request","completed_hybrid_blocks_this_request",
              "completed_adapter_rank_arrays_this_request")
        counts=[metrics.get(key) for key in keys]
        if any(type(value) is not int or value<0 for value in counts):
            raise ValueError("invalid shared rank execution counters")
        prepared,completed,arrays=counts
        if enabled:
            if not 0<completed<=prepared or arrays<completed:
                raise ValueError("shared ranks were not consumed by successful hybrid blocks")
        elif any(counts):
            raise ValueError("disabled rank-sharing arm executed shared work")


def model_snapshot(model):
    """Bound the local file generation/layout, not a full payload signature."""
    result={}
    for name in ("diffusion_models/qwen_image_2.1_bf16.safetensors",
                 "text_encoders/qwen3vl_8b_bf16.safetensors","vae/qwen_image_2.1_vae_bf16.safetensors"):
        path=model/name;before=path.stat()
        with path.open("rb") as stream:
            prefix=stream.read(8);length=int.from_bytes(prefix,"little")
            if len(prefix)!=8 or not 2<=length<=16<<20:raise ValueError("unbounded/invalid model header")
            header=stream.read(length)
            if len(header)!=length:raise ValueError("truncated model header")
        after=path.stat()
        fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if fields(before)!=fields(after):raise ValueError("model generation changed while observing header")
        result[name]=dict(device=after.st_dev,inode=after.st_ino,bytes=after.st_size,
            mtime_ns=after.st_mtime_ns,ctime_ns=after.st_ctime_ns,header_sha256=hashlib.sha256(header).hexdigest())
    return result


def make_request(prompt, image_paths, output, manifest=None, lora=None,dit_manifest=None):
    request=dict(model="qwen-image-2.1",operation="image.edit",prompt=prompt,
        width=512,height=512,steps=6 if lora else 40,seed=29,audio=False,
        residency="resident",execution="gpu",allow_approximation=True,qwen21_reference_size=512,
        inputs=[dict(kind="image",role="reference",path=str(path)) for path in image_paths],output=str(output))
    if manifest:request["encoder_ane_manifest"]=str(manifest)
    if lora:request.update(lora_strategy="inference_time",loras=[dict(path=str(lora),role="transformer",strength=1.0)])
    if dit_manifest:request.update(execution="gpu_ane",hybrid_mlp_mode="runtime",ane_manifest=str(dit_manifest))
    return request


def validate_rows(rows, mode, count, private, channels=None,global_channels=None):
    if len(rows)!=count:raise ValueError("missing request receipts")
    cumulative=0;dit_cumulative=0
    for index,row in enumerate(rows):
        with_dit=mode in ("dit_weights","dit_encoder_weights")
        expected_backend="mlx_cpp_metal+private_ane_runtime_weight_experimental" if with_dit else "mlx_cpp_metal"
        if row.get("prompt_cache_hit") is not False or row.get("runtime_backend")!=expected_backend:
            raise ValueError("need fresh conditioning and unchanged GPU DiT on every request")
        if with_dit:
            h=row.get("hybrid") or {};runtime=h.get("runtime_weight") or {}
            calls=h.get("runtime_calls_session_total")
            if type(calls) is not int or calls<=dit_cumulative or h.get("runtime_failed") is not False or \
                runtime.get("executor_backend")!="private_ane" or runtime.get("data_path")!="w8a8_hadamard" or \
                runtime.get("fallback_blocks_session_total")!=0 or \
                (global_channels is not None and runtime.get("ane_channels")!=global_channels):
                raise ValueError("combined DiT requires actual separate successful Private W8 execution")
            dit_cumulative=calls
        if row.get("actual_denoise_steps")!=row.get("steps"):
            raise ValueError("requested/actual denoise steps mismatch")
        for field in ("request_wall","text_encode","denoise"):
            value=(row.get("timings_seconds") or {}).get(field)
            if type(value) not in (float,int) or not math.isfinite(value) or value<=0:
                raise ValueError("missing/nonfinite/nonpositive request timing")
        evidence=row.get("encoder_runtime_reuse")
        metrics=row.get("encoder_hybrid") or {}
        with_weights=mode in ("gpu_weights","encoder_weights","dit_weights","dit_encoder_weights")
        weights=row.get("encoder_weight_residency")
        if with_weights:
            if not isinstance(weights,dict) or weights.get("enabled") is not True or weights.get("weights_retained") is not True:
                raise ValueError("admitted retained encoder source evidence missing")
            if weights.get("weights_reused")!=(index>0) or weights.get("loads_session_total")!=1 or weights.get("decline_reason"):
                raise ValueError("fresh-condition weight retention did not reuse the same admitted source")
            if not 0<weights.get("retained_bytes",0)<=20<<30:raise ValueError("retained source exceeds its bound")
        if mode in ("gpu","gpu_weights","dit_weights"):
            if evidence is not None or metrics or row.get("encoder_execution")!="gpu":
                raise ValueError("GPU baseline unexpectedly used encoder executor")
            continue
        retained=mode in ("encoder_retained","encoder_weights","dit_encoder_weights")
        if not isinstance(evidence,dict) or evidence.get("enabled")!=retained:
            raise ValueError("retention policy receipt missing/mismatched")
        if evidence.get("executor_retained")!=retained or evidence.get("executor_reused")!=(retained and index>0):
            raise ValueError("executor did not follow the requested lifecycle")
        calls=evidence.get("actual_calls_this_request")
        if type(calls) is not int or calls<=0 or metrics.get("runtime_failed") is not False:
            raise ValueError("missing actual successful encoder work")
        total=metrics.get("runtime_calls_session_total")
        if total!=(cumulative+calls if retained else calls):
            raise ValueError("encoder request/cumulative counters disagree")
        cumulative=total if retained else 0
        if metrics.get("session_released_after_encoding")!=(not retained):
            raise ValueError("encoder release receipt mismatch")
        runtime=metrics.get("runtime_weight") or {}
        if runtime.get("executor_backend")!=("private_ane" if private else "public_coreml") or runtime.get("fallback_blocks_session_total")!=0:
            raise ValueError("unexpected encoder executor/fallback")
        if private and (runtime.get("data_path")!="w8a8_hadamard" or runtime.get("partition_axis")!="intermediate_channels" or
            (channels is not None and runtime.get("ane_channels")!=channels)):
            raise ValueError("Private encoder representation/partition mismatch")
        if retained and not 0<evidence.get("retained_estimated_bytes",0)<=1<<30:
            raise ValueError("retained executor estimate exceeds its bound")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--dit-manifest",type=Path,help="explicit combined Private DiT/encoder screen; same source retention on every route")
    parser.add_argument("--lora-ranks-screen",action="store_true",
        help="same-library shared rank off/on; identical DiT share and retained encoder sources, GPU encoder on both arms")
    parser.add_argument("--lora-ranks-gpu-control",action="store_true",
        help="also include the same-library complete GPU baseline in a shared ranks screen")
    parser.add_argument("--weight-code-cache-bytes",type=int,
        help="matched GPU / code cache off / code cache on; fixed Private DiT, identical source retention and shared LoRA ranks")
    parser.add_argument("--weight-code-cache-mode",choices=("copy","surface"),default="copy",
        help="on arm: compact copy control or direct immutable native surface binding")
    parser.add_argument("--reference",type=Path,action="append",required=True)
    parser.add_argument("--prompt",action="append",required=True)
    parser.add_argument("--lora",type=Path)
    parser.add_argument("--backend",choices=("private","public"),default="private")
    parser.add_argument("--channels",type=int,default=3072)
    parser.add_argument("--global-channels",type=int,default=5120,
        help="separate global/DiT share; encoder always gets the explicit --channels override")
    parser.add_argument("--weights",action="store_true",help="fair GPU/ANE source-retention experiment; no disk cache")
    parser.add_argument("--order")
    parser.add_argument("--sample-memory",action="store_true")
    parser.add_argument("--observe-load",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--timeout",type=int,default=900)
    args=parser.parse_args()
    if args.weight_code_cache_mode!="copy" and args.weight_code_cache_bytes is None:
        parser.error("surface mode requires an explicit weight code cache budget")
    modes=LORA_RANK_MODES if args.lora_ranks_screen else COMBINED_MODES if args.dit_manifest else WEIGHT_MODES if args.weights else MODES
    if args.lora_ranks_gpu_control:
        if not args.lora_ranks_screen:parser.error("GPU rank control requires --lora-ranks-screen")
        modes=("gpu_weights",*LORA_RANK_MODES)
    if args.weight_code_cache_bytes is not None:
        if args.lora_ranks_screen or args.lora_ranks_gpu_control or not args.dit_manifest or \
            args.backend!="private" or not 0<args.weight_code_cache_bytes<=2<<30 or args.global_channels==0:
            parser.error("weight code cache screen requires fixed Private DiT, bytes1..2147483648 and no separate ranks screen")
        modes=("gpu_weights",*WEIGHT_CODE_MODES)
    order=args.order.split(",") if args.order else list(modes)
    if len(order)!=len(modes) or set(order)!=set(modes):parser.error("order must include each matched mode once")
    if not 1<=len(args.reference)<=2 or not 3<=len(args.prompt)<=9 or len(set(args.prompt))!=len(args.prompt):
        parser.error("need 1..2 references and 3..9 distinct fresh prompts")
    if any(not prompt.strip() for prompt in args.prompt):parser.error("empty prompt")
    if not 1<=args.timeout<=3600:parser.error("timeout must be 1..3600 seconds per serial mode")
    if args.backend=="private" and not (0<args.channels<12288 and args.channels%512==0):parser.error("Private channels require aligned partial width")
    if not (0<=args.global_channels<12288 and args.global_channels%512==0):parser.error("invalid global channel share")
    if args.backend=="public" and args.channels!=0:parser.error("Public uses its row ABI, channels=0")
    if args.dit_manifest and args.backend!="private":parser.error("combined screen requires explicit Private backend")
    if args.lora_ranks_screen and (not args.dit_manifest or not args.lora or args.global_channels==0):
        parser.error("shared ranks screen requires real LoRA and a fixed partial Private DiT channel share")
    if args.output.exists() or args.output.is_symlink():parser.error("choose a new output directory")
    cli=args.cli.resolve(strict=True);library=cli.parent/"libturbocider.dylib"
    if not library.is_file() or not os.access(cli,os.X_OK):parser.error("CLI and adjacent native library required")
    model=args.model.resolve(strict=True);manifest=args.manifest.resolve(strict=True)
    references=[p.resolve(strict=True) for p in args.reference]
    lora=args.lora.resolve(strict=True) if args.lora else None
    dit_manifest=args.dit_manifest.resolve(strict=True) if args.dit_manifest else None
    before={str(path):sha256_file(path) for path in [cli,library,manifest,*references,*([lora] if lora else []),*([dit_manifest] if dit_manifest else [])]}
    model_before=model_snapshot(model)
    args.output.mkdir(parents=True)
    summary=dict(schema="tc-qwen-encoder-residency-screen-v1",time_utc=datetime.now(timezone.utc).isoformat(),
        status="incomplete",qualification_passed=False,order=order,prompts=args.prompt,source_identities=before,
        model_snapshot=model_before,model_identity_scope="regular-file generation stamps and bounded safetensors header hashes; not full payload hashes or immutable leases",
        backend=args.backend,channels=args.channels,scope="native fresh-condition request wall; first request separate; host/process-tree diagnostics, not physical overlap proof",trials=[])
    summary.update(weight_retention_screen=args.weights or bool(dit_manifest),global_channels=args.global_channels,
        combined_dit_encoder=bool(dit_manifest) and not args.lora_ranks_screen and args.weight_code_cache_bytes is None,
        lora_ranks_screen=args.lora_ranks_screen,weight_code_cache_bytes=args.weight_code_cache_bytes,
        weight_code_cache_mode=args.weight_code_cache_mode)
    summary["lora_ranks_gpu_control"]=args.lora_ranks_gpu_control
    target=args.output/"summary.json"
    target.write_text(json.dumps(summary,indent=2)+"\n")
    for mode in order:
        if model_snapshot(model)!=model_before:raise ValueError("model file generation/layout changed between modes")
        env=benchmark_environment()
        if lora:env["TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC"]="1"
        if mode in ("gpu_weights","encoder_weights","dit_weights","dit_encoder_weights",*LORA_RANK_MODES,*WEIGHT_CODE_MODES):env["TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS"]="1"
        uses_encoder=mode.startswith("encoder_") or mode=="dit_encoder_weights"
        uses_dit=mode in ("dit_weights","dit_encoder_weights",*LORA_RANK_MODES,*WEIGHT_CODE_MODES)
        if mode in LORA_RANK_MODES:
            env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1" if mode=="ranks_on" else "0"
        if mode in WEIGHT_CODE_MODES:
            env["TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES"]=str(args.weight_code_cache_bytes if mode=="code_cache_on" else 0)
            env["TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_MODE"]=args.weight_code_cache_mode if mode=="code_cache_on" else "copy"
            if lora:env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1"
        if uses_encoder or uses_dit:
            env.update(TURBOCIDER_ANE_BACKEND=args.backend,TURBOCIDER_RUNTIME_ANE_CHUNKS="1",
                TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1",TURBOCIDER_QWEN21_ENCODER_ANE_CHANNELS=str(args.channels),
                TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME="1" if mode in ("encoder_retained","encoder_weights","dit_encoder_weights") else "0")
            if args.backend=="private":env.update(TURBOCIDER_ALLOW_PRIVATE_ANE="1",TURBOCIDER_PRIVATE_ANE_DATA_PATH="w8a8",
                TURBOCIDER_PRIVATE_ANE_GPU_IO="1",TURBOCIDER_PRIVATE_ANE_CHANNELS=str(args.global_channels),
                TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE="1",TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE="1",
                TURBOCIDER_PRIVATE_ANE_PREFETCH="0",TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD="0",
                TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN="1")
        requests=[]
        for index,prompt in enumerate(args.prompt):
            path=args.output/f"{mode}-{index}.json"
            path.write_text(json.dumps(make_request(prompt,references,(args.output/f"{mode}-{index}.png").resolve(),
                manifest if uses_encoder else None,lora,dit_manifest if uses_dit else None),indent=2)+"\n")
            requests.append(str(path.resolve()))
        command=[str(cli),"batch",str(model),*requests]
        observer=LoadObservation(args.output/f"{mode}-load.jsonl") if args.observe_load else None
        print(json.dumps(dict(starting=mode)),flush=True)
        with (args.output/f"{mode}.stdout.jsonl").open("x") as stdout,(args.output/f"{mode}.stderr.txt").open("x") as stderr:
            memory=run_sampled(command,repo=ROOT,output=args.output,stem=mode,env=env,stdout=stdout,stderr=stderr,
                timeout=args.timeout,interval_ms=100,max_gap_ms=500,observer=observer) if args.sample_memory else None
            if not args.sample_memory:
                result=run_owned(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=args.timeout,observer=observer)
                if result.returncode:raise RuntimeError("request process failed; raw evidence retained")
        load=observer.verify() if observer else None
        rows=[json.loads(line) for line in (args.output/f"{mode}.stdout.jsonl").read_text().splitlines()]
        validate_rows(rows,"dit_weights" if mode in (*LORA_RANK_MODES,*WEIGHT_CODE_MODES) else mode,
            len(args.prompt),args.backend=="private",args.channels,args.global_channels)
        if mode in LORA_RANK_MODES:validate_shared_ranks(rows,mode=="ranks_on")
        if mode in WEIGHT_CODE_MODES:
            validate_weight_code_cache(rows,args.weight_code_cache_bytes if mode=="code_cache_on" else 0,
                args.weight_code_cache_mode if mode=="code_cache_on" else "copy")
            if lora:validate_shared_ranks(rows,True)
        if any(sha256_file(Path(path))!=digest for path,digest in before.items()):raise ValueError("input/runtime bytes changed")
        if model_snapshot(model)!=model_before:raise ValueError("model file generation/layout changed during mode")
        times=[row["timings_seconds"]["request_wall"] for row in rows]
        trial=dict(mode=mode,cold_request_seconds=times[0],warm_fresh_request_seconds=times[1:],
            warm_fresh_median_seconds=statistics.median(times[1:]),text_seconds=[row["timings_seconds"]["text_encode"] for row in rows],
            memory=memory,load=load,png_sha256=[sha256_file(args.output/f"{mode}-{i}.png") for i in range(len(rows))])
        summary["trials"].append(trial);target.write_text(json.dumps(summary,indent=2)+"\n")
    summary["status"]="complete_diagnostic";target.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
