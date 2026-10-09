#!/usr/bin/env python3
"""Same-library optimized GPU / original / MPP ConvRot F32 partial diagnostic.

Explicit local fixtures only, serial processes, one cold plus two same-prompt
hot requests. Keep all evidence; no automatic visual/strict-load qualification.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from runtime_ane_common import benchmark_environment, sha256_file, validate_overflow_events, gpu_layer_policy
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
MODES=("gpu","original","mpp")
NARROW_MODES=("gpu","mpp","bf16")
MATCHED_BUCKET_MODES=("gpu","bf16_fixed","bf16_matched")
MARKER="experimental MPP shared-word ConvRot GPU F32 partial (m64/n64/k64; tail-k32)"
BF16_MARKER="experimental BF16-rounded packed ConvRot GPU base-down partial widened to F32; ONE joined-hidden down-LoRA"
LORA_MARKER="experimental full ConvRot runtime LoRA; base-only W8 banks, pre-SiLU gate/up and ONE joined-hidden down"


def source_snapshot(path):
    before=path.stat()
    with path.open("rb") as source:
        prefix=source.read(8);size=int.from_bytes(prefix,"little")
        if len(prefix)!=8 or not 2<=size<=16<<20:raise ValueError("invalid bounded safetensors header")
        header=source.read(size)
        if len(header)!=size:raise ValueError("truncated safetensors header")
    after=path.stat()
    fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if fields(before)!=fields(after):raise ValueError("source changed while observing header")
    return dict(device=after.st_dev,inode=after.st_ino,bytes=after.st_size,mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,header_sha256=hashlib.sha256(header).hexdigest())


def validate_rows(rows,mode,steps,count=3,cold_retry_cap=0,has_lora=False,gpu_layers=(),template_rows=0,matched_rows=False):
    matched_rows=matched_rows or mode=="bf16_matched"
    if mode in ("bf16_fixed","bf16_matched"):mode="bf16"
    if mode not in (*MODES,"bf16") or len(rows)!=count:raise ValueError("missing matched ConvRot requests")
    previous=0;previous_calls=0;previous_retries=0;first_headroom=None;previous_forced=0;previous_lora=0
    for index,row in enumerate(rows):
        if row.get("actual_denoise_steps")!=steps or row.get("steps")!=steps:
            raise ValueError("requested/actual ConvRot steps differ")
        value=(row.get("timings_seconds") or {}).get("request_wall")
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            raise ValueError("missing valid ConvRot request time")
        if (MARKER in (row.get("acceleration_selection") or ""))!=(mode=="mpp"):
            raise ValueError("ConvRot GPU partial selection differs from requested arm")
        if (BF16_MARKER in (row.get("acceleration_selection") or ""))!=(mode=="bf16"):
            raise ValueError("BF16-rounded partial must be distinct from exact F32/GPU controls")
        if has_lora and (row.get("lora_strategy")!="inference_time" or type(row.get("lora_applied_projections")) is not int or row.get("lora_applied_projections")!=238):
            raise ValueError("need actual complete unmerged local Z distill LoRA bindings")
        if mode=="gpu":
            expected="mlx_cpp_metal_convrot_packed_q8" if has_lora else "mlx_cpp_metal_convrot_compiled_experimental"
            if row.get("runtime_backend")!=expected or row.get("hybrid"):
                raise ValueError("GPU control must use complete compiled ConvRot GPU, not a split/ANE route")
            continue
        h=row.get("hybrid") or {};r=h.get("runtime_weight") or {}
        blocks=r.get("channel_blocks_session_total");calls=h.get("runtime_calls_session_total")
        retries=r.get("overflow_retries_session_total")
        expected_blocks=steps*(32-len(gpu_layers))
        if row.get("runtime_backend")!="mlx_cpp_metal_convrot+private_ane_runtime_weight_experimental" or \
            h.get("runtime_failed") is not False or r.get("executor_backend")!="private_ane" or \
            r.get("data_path")!="w8a8_convrot" or r.get("partition_axis")!="intermediate_channels" or \
            r.get("ane_channels")!=4096 or r.get("fp32_channel_join_enabled") is not True or \
            r.get("fallback_blocks_session_total")!=0 or type(retries) is not int or retries<0 or \
            (retries>cold_retry_cap if index==0 else retries!=previous_retries) or \
            type(blocks) is not int or blocks-previous!=expected_blocks or type(calls) is not int or calls-previous_calls<expected_blocks:
            raise ValueError("need actual successful fixed ConvRot F32 channel work, not just a selection marker")
        if mode=="bf16" or matched_rows:
            bucket=h.get("bucket");tokens=row.get("text_tokens")
            if type(bucket) is not int or bucket<=0 or type(tokens) is not int or tokens<=0 or \
                row.get("height")!=512 or row.get("width")!=512:
                raise ValueError("narrow partial requires actual bucket/token/512 geometry")
            # Two noise-refiner layers have no caption rows. Main layers have
            # the original 32-row caption padding; a single extra group can
            # double calls when it crosses a fixed bucket's capacity.
            main_rows=1024+((tokens+31)//32)*32
            selected_bucket=template_rows if template_rows>=main_rows and template_rows%32==0 else ((main_rows+127)//128)*128
            if matched_rows and (bucket!=selected_bucket or
                    "experimental request-matched Private FFN rows="+str(bucket) not in (row.get("acceleration_selection") or "")):
                raise ValueError("matched bucket requires actual bounded geometry selection, not relabelled fixed calls")
            noise_blocks=2-sum(layer<2 for layer in gpu_layers)
            main_blocks=30-sum(layer>=2 for layer in gpu_layers)
            expected_calls=steps*(noise_blocks*((1024+bucket-1)//bucket)+main_blocks*((main_rows+bucket-1)//bucket))
            if calls-previous_calls!=expected_calls+retries-previous_retries:
                raise ValueError("actual driver calls disagree with bucket/padding/retry geometry")
        if cold_retry_cap:
            headroom=r.get("headroom_scale")
            if type(headroom) not in (int,float) or not math.isfinite(headroom) or headroom<1 or \
                (index and headroom!=first_headroom) or validate_overflow_events(r,calls) is None:
                raise ValueError("cold retry diagnostic needs complete overflow/headroom evidence and no warm recipe changes")
            if not index:first_headroom=headroom
        if gpu_layers:
            forced=r.get("forced_gpu_blocks_session_total")
            if r.get("requested_gpu_layers")!=list(gpu_layers) or type(forced) is not int or forced-previous_forced!=steps*len(gpu_layers):
                raise ValueError("missing actual complete-GPU sensitive layer policy")
            previous_forced=forced
        if has_lora:
            corrections=r.get("lora_channel_range_calls_session_total")
            if LORA_MARKER not in (row.get("acceleration_selection") or "") or type(corrections) is not int or \
                corrections-previous_lora!=expected_blocks or r.get("lora_channel_full_calls_session_total")!=0:
                raise ValueError("need full successful ConvRot LoRA channel corrections, not label/binding alone")
            previous_lora=corrections
        previous=blocks;previous_calls=calls;previous_retries=retries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("cli","model","checkpoint","manifest","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--order")
    parser.add_argument("--matched-bucket-screen",action="store_true",help="same bf16 partial: GPU / fixed template / request-matched Private native rows")
    parser.add_argument("--runtime-match-rows",action="store_true",help="BF16-partial screen: same request-matched Private bucket on both F32 MPP and BF16 hybrid arms")
    parser.add_argument("--bf16-partial-screen",action="store_true",help="matched optimized GPU / exact MPP F32 / approximate BF16 QMM partial widened to F32; same ANE/math/LoRA")
    parser.add_argument("--steps",type=int,default=4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--timeout",type=int,default=600)
    parser.add_argument("--lora",type=Path,help="explicit local Z distill-patch adapter; complete LoRA, never merge into codes")
    parser.add_argument("--lora-strength",type=float,default=1.0)
    parser.add_argument("--gpu-ffn-blocks",type=gpu_layer_policy,default=(),help="same complete-GPU sensitive ordinals on both hybrid arms, not an error fallback")
    parser.add_argument("--cold-retry-cap",type=int,default=0,
        help="explicit additional diagnostic policy0..8; only cold retries, complete events, unchanged warm headroom and matched original/MPP final headroom; never strict qualification")
    parser.add_argument("--prompt",default="A curious red fox sitting in falling snow beside pine trees, detailed fur, natural winter light, photorealistic.")
    args=parser.parse_args()
    if args.runtime_match_rows and (not args.bf16_partial_screen or args.matched_bucket_screen):
        parser.error("runtime row matching requires the BF16-partial screen, not a fixed/matched bucket ablation")
    if args.matched_bucket_screen and args.bf16_partial_screen:parser.error("choose one partial/bucket screen")
    modes=MATCHED_BUCKET_MODES if args.matched_bucket_screen else NARROW_MODES if args.bf16_partial_screen else MODES
    order=args.order.split(",") if args.order else list(modes)
    if len(order)!=3 or set(order)!=set(modes) or not 1<=args.steps<=16 or not 30<=args.timeout<=1800 or \
        not 0<=args.cold_retry_cap<=8 or not args.prompt.strip() or not math.isfinite(args.lora_strength) or not -8<=args.lora_strength<=8:
        parser.error("need each matched mode once, steps1..16, timeout30..1800 and a nonempty prompt")
    if args.output.exists() or args.output.is_symlink():parser.error("choose a new output directory")
    cli=args.cli.resolve(strict=True);library=cli.parent/"libturbocider.dylib"
    model=args.model.resolve(strict=True);checkpoint=args.checkpoint.resolve(strict=True);manifest=args.manifest.resolve(strict=True)
    template_rows=json.loads(manifest.read_text())["rows"]
    if not library.is_file() or not model.is_dir():parser.error("CLI, adjacent library and local model directory required")
    bindings={str(p):sha256_file(p) for p in (cli,library,manifest)}
    lora=args.lora.resolve(strict=True) if args.lora else None
    if lora:bindings[str(lora)]=sha256_file(lora)
    sources=[checkpoint,model/"split_files/text_encoders/qwen_3_4b.safetensors",model/"split_files/vae/ae.safetensors"]
    snapshot={str(p):source_snapshot(p) for p in sources}
    args.output.mkdir(parents=True)
    summary=dict(schema="tc-convrot-partial-mpp-model-screen-v1",time_utc=datetime.now(timezone.utc).isoformat(),
        status="incomplete",qualification_passed=False,order=order,steps=args.steps,seed=args.seed,prompt=args.prompt,
        source_identities=bindings,source_snapshots=snapshot,
        source_identity_scope="file generation stamps and bounded headers, not complete payload hashes/immutable source leases",
        scope="same-library resident512 base, first request separate then two same-prompt hot requests; optimized compiled-dense GPU with explicit retained3GiB allocator hint; process-tree memory, no strict-load/physical-overlap qualification",
        cold_retry_cap=args.cold_retry_cap,bf16_partial_screen=args.bf16_partial_screen,matched_bucket_screen=args.matched_bucket_screen,
        runtime_match_rows=args.runtime_match_rows,lora=str(lora) if lora else None,lora_strength=args.lora_strength if lora else None,
        gpu_ffn_blocks=list(args.gpu_ffn_blocks),trials=[])
    if lora:summary["scope"]="same-library resident512 real unmerged Z distill LoRA, shared-rotation complete GPU; first then two same-prompt hot requests; process-tree memory, no strict-load/physical-overlap qualification"
    target=args.output/"summary.json";target.write_text(json.dumps(summary,indent=2)+"\n")
    for mode in order:
        env=benchmark_environment();env["TURBOCIDER_Z_IMAGE_TRANSFORMER"]=str(checkpoint)
        if lora:env["TURBOCIDER_Z_CONVROT_SHARED_GATE_UP"]="1"
        if mode=="gpu" and not lora:
            env.update(TURBOCIDER_Z_CONVROT_GPU_RECIPE="compiled_dense",TURBOCIDER_Z_CONVROT_CACHE_BYTES=str(3<<30),
                TURBOCIDER_Z_CONVROT_CACHE_RETAIN="1")
        elif mode!="gpu":
            env.update(TURBOCIDER_Z_RUNTIME_CONVROT="1",TURBOCIDER_Z_CONVROT_FP32_MPP="1" if mode=="mpp" else "0",
                TURBOCIDER_Z_CONVROT_BF16_PARTIAL="1" if mode in ("bf16","bf16_fixed","bf16_matched") else "0",
                TURBOCIDER_ANE_BACKEND="private",TURBOCIDER_ALLOW_PRIVATE_ANE="1",TURBOCIDER_PRIVATE_ANE_GPU_IO="1",
                TURBOCIDER_PRIVATE_ANE_DATA_PATH="convrot_w8a8",TURBOCIDER_PRIVATE_ANE_CHANNELS="4096",
                TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE="1",TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE="1",
                TURBOCIDER_PRIVATE_ANE_PREFETCH="0",TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD="0",
                TURBOCIDER_RUNTIME_ANE_CHUNKS="1",TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1",
                TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN="1")
            if args.matched_bucket_screen:env["TURBOCIDER_Z_RUNTIME_MATCH_ROWS"]="1" if mode=="bf16_matched" else "0"
            if args.runtime_match_rows:env["TURBOCIDER_Z_RUNTIME_MATCH_ROWS"]="1"
            if lora:env["TURBOCIDER_Z_RUNTIME_CONVROT_LORA"]="1"
            if args.gpu_ffn_blocks:env["TURBOCIDER_Z_RUNTIME_GPU_FFN_BLOCKS"]=",".join(map(str,args.gpu_ffn_blocks))
        requests=[]
        for index in range(3):
            request=dict(model="z-image-turbo",operation="image.generate",prompt=args.prompt,width=512,height=512,
                steps=args.steps,seed=args.seed,audio=False,residency="resident",execution="gpu" if mode=="gpu" else "gpu_ane",
                allow_approximation=True,output=str((args.output/f"{mode}-{index}.png").resolve()))
            if lora:request.update(lora_strategy="inference_time",loras=[dict(path=str(lora),role="transformer",strength=args.lora_strength)])
            if mode=="gpu":request["compile_gpu"]=not bool(lora)
            else:request.update(hybrid_mlp_mode="runtime",ane_manifest=str(manifest))
            path=args.output/f"{mode}-{index}.json";path.write_text(json.dumps(request,indent=2)+"\n");requests.append(str(path.resolve()))
        print(json.dumps(dict(starting=mode)),flush=True)
        with (args.output/f"{mode}.stdout.jsonl").open("x") as stdout,(args.output/f"{mode}.stderr.txt").open("x") as stderr:
            memory=run_sampled([str(cli),"batch",str(model),*requests],repo=ROOT,output=args.output,stem=mode,
                env=env,stdout=stdout,stderr=stderr,timeout=args.timeout,interval_ms=100,max_gap_ms=500)
        rows=[json.loads(line) for line in (args.output/f"{mode}.stdout.jsonl").read_text().splitlines()]
        validate_rows(rows,mode,args.steps,cold_retry_cap=args.cold_retry_cap,has_lora=bool(lora),gpu_layers=args.gpu_ffn_blocks,
            template_rows=template_rows,matched_rows=args.runtime_match_rows)
        if any(sha256_file(Path(p))!=value for p,value in bindings.items()) or \
            any(source_snapshot(Path(p))!=value for p,value in snapshot.items()):raise ValueError("runtime/source identity changed")
        times=[row["timings_seconds"]["request_wall"] for row in rows]
        summary["trials"].append(dict(mode=mode,cold_request_seconds=times[0],warm_request_seconds=times[1:],
            warm_median_seconds=statistics.median(times[1:]),memory=memory,
            conditioning_cache_hits=[row.get("prompt_cache_hit") for row in rows],
            resolved_headroom=[((row.get("hybrid") or {}).get("runtime_weight") or {}).get("headroom_scale") for row in rows],
            overflow_retries_cumulative=[((row.get("hybrid") or {}).get("runtime_weight") or {}).get("overflow_retries_session_total") for row in rows],
            png_sha256=[sha256_file(args.output/f"{mode}-{i}.png") for i in range(3)]))
        target.write_text(json.dumps(summary,indent=2)+"\n")
    hybrid={trial["mode"]:trial for trial in summary["trials"] if trial["mode"]!="gpu"}
    if args.cold_retry_cap and len({tuple(t["resolved_headroom"]) for t in hybrid.values()})!=1:
        raise ValueError("hybrid headroom recipes differ; cannot compare this diagnostic")
    summary["status"]="complete_diagnostic";target.write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
