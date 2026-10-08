#!/usr/bin/env python3
"""Real base -> local LoRA1 -> same LoRA.5 -> base on one ConvRot executor.

State/ownership diagnostic, not a speed or broader LoRA quality qualification.
No derived adapter/model files; all original fixtures remain read-only.
"""
import argparse
import json
from pathlib import Path

from convrot_partial_mpp_screen import source_snapshot,MARKER,LORA_MARKER
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]


def verify(rows,png_hashes):
    if len(rows)!=4 or len(png_hashes)!=4:raise ValueError("need four actual switch results")
    load=None;previous_calls=0;previous_blocks=0;previous_corrections=0
    for index,row in enumerate(rows):
        adapter=index in (1,2)
        if row.get("actual_denoise_steps")!=8 or row.get("lora_strategy")!=("inference_time" if adapter else "none") or \
            type(row.get("lora_applied_projections",0)) is not int or \
            row.get("lora_applied_projections",0)!=(238 if adapter else 0):raise ValueError("actual adapter bindings/steps differ")
        h=row.get("hybrid") or {};r=h.get("runtime_weight") or {}
        calls=h.get("runtime_calls_session_total");blocks=r.get("channel_blocks_session_total")
        corrections=r.get("lora_channel_range_calls_session_total")
        if row.get("runtime_backend")!="mlx_cpp_metal_convrot+private_ane_runtime_weight_experimental" or \
            h.get("runtime_failed") is not False or r.get("executor_backend")!="private_ane" or \
            r.get("data_path")!="w8a8_convrot" or r.get("fp32_channel_join_enabled") is not True or \
            r.get("partition_axis")!="intermediate_channels" or r.get("ane_channels")!=4096 or \
            r.get("lora_channel_full_calls_session_total")!=0 or \
            r.get("fallback_blocks_session_total")!=0 or r.get("overflow_retries_session_total")!=0 or \
            r.get("headroom_scale")!=1 or r.get("requested_gpu_layers")!=[2] or \
            r.get("forced_gpu_blocks_session_total")!=8*(index+1) or \
            type(calls) is not int or calls-previous_calls!=248 or type(blocks) is not int or blocks-previous_blocks!=248 or \
            type(corrections) is not int or corrections-previous_corrections!=(248 if adapter else 0):
            raise ValueError("switch did not complete the same no-retry executor/channel/correction policy")
        if MARKER not in (row.get("acceleration_selection") or "") or \
            (LORA_MARKER in (row.get("acceleration_selection") or ""))!=adapter:
            raise ValueError("request-local source/LoRA selection did not switch")
        if load is None:load=h.get("load_seconds")
        if type(load) not in (int,float) or load<=0 or h.get("load_seconds")!=load:
            raise ValueError("executor reloaded instead of using the same admitted graph")
        previous_calls=calls;previous_blocks=blocks;previous_corrections=corrections
    if png_hashes[0]!=png_hashes[3] or len(set(png_hashes))!=3:
        raise ValueError("base state was not restored or adapter strengths did not change real outputs")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("cli","model","checkpoint","manifest","lora","output"):p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.output.is_symlink():p.error("choose a new output directory")
    cli=args.cli.resolve(strict=True);model=args.model.resolve(strict=True);checkpoint=args.checkpoint.resolve(strict=True)
    manifest=args.manifest.resolve(strict=True);lora=args.lora.resolve(strict=True);library=cli.parent/"libturbocider.dylib"
    bindings={str(path):sha256_file(path) for path in (cli,library,manifest,lora)}
    sources={str(path):source_snapshot(path) for path in (checkpoint,model/"split_files/text_encoders/qwen_3_4b.safetensors",model/"split_files/vae/ae.safetensors")}
    args.output.mkdir(parents=True)
    env=benchmark_environment();env.update(TURBOCIDER_Z_IMAGE_TRANSFORMER=str(checkpoint),TURBOCIDER_Z_RUNTIME_CONVROT="1",
        TURBOCIDER_Z_RUNTIME_CONVROT_LORA="1",TURBOCIDER_Z_CONVROT_FP32_MPP="1",TURBOCIDER_Z_CONVROT_SHARED_GATE_UP="1",
        TURBOCIDER_ANE_BACKEND="private",TURBOCIDER_ALLOW_PRIVATE_ANE="1",TURBOCIDER_PRIVATE_ANE_GPU_IO="1",
        TURBOCIDER_PRIVATE_ANE_DATA_PATH="convrot_w8a8",TURBOCIDER_PRIVATE_ANE_CHANNELS="4096",
        TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE="1",TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE="1",
        TURBOCIDER_PRIVATE_ANE_PREFETCH="0",TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD="0",TURBOCIDER_Z_RUNTIME_GPU_FFN_BLOCKS="2",
        TURBOCIDER_RUNTIME_ANE_CHUNKS="1",TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1",TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN="1")
    requests=[]
    for index,strength in enumerate((None,1.0,.5,None)):
        request=dict(model="z-image-turbo",operation="image.generate",prompt="A curious red fox sitting in falling snow beside pine trees, detailed fur, natural winter light, photorealistic.",
            width=512,height=512,steps=8,seed=42,audio=False,residency="resident",execution="gpu_ane",allow_approximation=True,
            hybrid_mlp_mode="runtime",ane_manifest=str(manifest),output=str((args.output/f"switch-{index}.png").resolve()))
        if strength is not None:request.update(lora_strategy="inference_time",loras=[dict(path=str(lora),strength=strength,role="transformer")])
        path=args.output/f"switch-{index}.json";path.write_text(json.dumps(request,indent=2)+"\n");requests.append(str(path.resolve()))
    summary=dict(schema="tc-convrot-runtime-lora-state-switch-v1",status="incomplete",qualification_passed=False,
        scope="one process/equal graph policy/base-real adapter1-same adapter.5-base; no speed, second trained adapter, strict-load or physical-overlap qualification",
        source_identities=bindings,source_snapshots=sources)
    target=args.output/"summary.json";target.write_text(json.dumps(summary,indent=2)+"\n")
    with (args.output/"switch.stdout.jsonl").open("x") as stdout,(args.output/"switch.stderr.txt").open("x") as stderr:
        summary["memory"]=run_sampled([str(cli),"batch",str(model),*requests],repo=ROOT,output=args.output,stem="switch",
            env=env,stdout=stdout,stderr=stderr,timeout=600,interval_ms=100,max_gap_ms=500)
    rows=[json.loads(line) for line in (args.output/"switch.stdout.jsonl").read_text().splitlines()]
    hashes=[sha256_file(args.output/f"switch-{i}.png") for i in range(4)]
    verify(rows,hashes)
    if any(sha256_file(Path(path))!=value for path,value in bindings.items()) or \
        any(source_snapshot(Path(path))!=value for path,value in sources.items()):raise ValueError("source/runtime changed during switch")
    summary.update(status="complete_diagnostic",state_restore_passed=True,png_sha256=hashes,
        actual_calls_cumulative=[row["hybrid"]["runtime_calls_session_total"] for row in rows])
    target.write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
