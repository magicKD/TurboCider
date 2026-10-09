#!/usr/bin/env python3
"""Actual original-LoRA short/long/long/short caption bucket switch, no source writes."""
import argparse
import json
from pathlib import Path

from convrot_partial_mpp_screen import source_snapshot
from runtime_ane_common import benchmark_environment,sha256_file,validate_z_matched_rows
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
FOX="A curious red fox sitting in falling snow beside pine trees, detailed fur, natural winter light, photorealistic."
LIGHTHOUSE="A weathered lighthouse on a rocky coastline at dawn, soft ocean mist, cinematic natural lighting, detailed stonework, photorealistic."


def validate(rows,hashes):
    if len(rows)!=4 or len(hashes)!=4:raise ValueError("need four actual bucket-switch requests")
    if any(not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value) for value in hashes):
        raise ValueError("need actual SHA256 output identities")
    for index,row in enumerate(rows):
        bucket=1056 if index in (0,3) else 1152
        validate_z_matched_rows(rows[1:3] if index==2 else [row],8,(2,),1056)
        hybrid=row.get("hybrid") or {};runtime=hybrid.get("runtime_weight") or {}
        if row.get("model")!="z-image-turbo" or row.get("text_tokens")!=(31 if index in (0,3) else 37) or \
            row.get("seed")!=(42 if index in (0,3) else 17) or \
            row.get("actual_denoise_steps")!=8 or row.get("lora_applied_projections")!=238 or row.get("lora_strategy")!="inference_time" or \
            hybrid.get("bucket")!=bucket or hybrid.get("runtime_calls_session_total")!=(496 if index==2 else 248) or \
            runtime.get("channel_blocks_session_total")!=(496 if index==2 else 248) or hybrid.get("runtime_failed") is not False or \
            runtime.get("fallback_blocks_session_total")!=0 or runtime.get("overflow_retries_session_total")!=0 or \
            "experimental request-matched Private FFN rows="+str(bucket) not in (row.get("acceleration_selection") or ""):
            raise ValueError("bucket switch must rebuild changed geometry, retain repeated geometry, and perform complete LoRA FFNs")
    if hashes[0]!=hashes[3] or hashes[1]!=hashes[2]:raise ValueError("restored/repeated prompt output changed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("cli","model","checkpoint","manifest","lora","output"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink():parser.error("choose fresh evidence directory")
    cli=args.cli.resolve(strict=True);model=args.model.resolve(strict=True);checkpoint=args.checkpoint.resolve(strict=True)
    manifest=args.manifest.resolve(strict=True);lora=args.lora.resolve(strict=True);library=cli.parent/"libturbocider.dylib"
    identity={str(p):sha256_file(p) for p in (cli,library,manifest,lora)}
    snapshot={str(p):source_snapshot(p) for p in (checkpoint,model/"split_files/text_encoders/qwen_3_4b.safetensors",model/"split_files/vae/ae.safetensors")}
    args.output.mkdir(parents=True)
    env=benchmark_environment();env.update(TURBOCIDER_Z_IMAGE_TRANSFORMER=str(checkpoint),TURBOCIDER_Z_RUNTIME_CONVROT="1",
        TURBOCIDER_Z_RUNTIME_CONVROT_LORA="1",TURBOCIDER_Z_CONVROT_BF16_PARTIAL="1",TURBOCIDER_Z_CONVROT_SHARED_GATE_UP="1",
        TURBOCIDER_Z_RUNTIME_MATCH_ROWS="1",TURBOCIDER_ANE_BACKEND="private",TURBOCIDER_ALLOW_PRIVATE_ANE="1",
        TURBOCIDER_PRIVATE_ANE_GPU_IO="1",TURBOCIDER_PRIVATE_ANE_DATA_PATH="convrot_w8a8",TURBOCIDER_PRIVATE_ANE_CHANNELS="4096",
        TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE="1",TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE="1",TURBOCIDER_PRIVATE_ANE_PREFETCH="0",
        TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD="0",TURBOCIDER_Z_RUNTIME_GPU_FFN_BLOCKS="2",TURBOCIDER_RUNTIME_ANE_CHUNKS="1",
        TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1",TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN="1")
    requests=[]
    for index,prompt in enumerate((FOX,LIGHTHOUSE,LIGHTHOUSE,FOX)):
        request=dict(model="z-image-turbo",operation="image.generate",prompt=prompt,width=512,height=512,steps=8,
            seed=42 if index in (0,3) else 17,audio=False,residency="resident",execution="gpu_ane",allow_approximation=True,
            hybrid_mlp_mode="runtime",ane_manifest=str(manifest),lora_strategy="inference_time",
            loras=[dict(path=str(lora),strength=1.0,role="transformer")],output=str((args.output/f"switch-{index}.png").resolve()))
        path=args.output/f"switch-{index}.json";path.write_text(json.dumps(request,indent=2)+"\n");requests.append(str(path.resolve()))
    summary=dict(schema="tc-z-runtime-bucket-switch-v1",status="incomplete",qualification_passed=False,source_identities=identity,
        source_snapshots=snapshot,scope="same original source/adapter and process; changed bucket rebuild, same bucket reuse, output restore; not matched speed/load/physical qualification")
    target=args.output/"summary.json";target.write_text(json.dumps(summary,indent=2)+"\n")
    with (args.output/"switch.stdout.jsonl").open("x") as stdout,(args.output/"switch.stderr.txt").open("x") as stderr:
        summary["memory"]=run_sampled([str(cli),"batch",str(model),*requests],repo=ROOT,output=args.output,stem="switch",
            env=env,stdout=stdout,stderr=stderr,timeout=600,interval_ms=100,max_gap_ms=500)
    rows=[json.loads(line) for line in (args.output/"switch.stdout.jsonl").read_text().splitlines()]
    hashes=[sha256_file(args.output/f"switch-{i}.png") for i in range(4)];validate(rows,hashes)
    if any(sha256_file(Path(p))!=digest for p,digest in identity.items()) or any(source_snapshot(Path(p))!=value for p,value in snapshot.items()):
        raise ValueError("source/runtime changed during bucket switch")
    summary.update(status="complete_diagnostic",buckets=[r["hybrid"]["bucket"] for r in rows],
        actual_calls_cumulative=[r["hybrid"]["runtime_calls_session_total"] for r in rows],png_sha256=hashes)
    target.write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
