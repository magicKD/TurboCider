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

from runtime_ane_common import benchmark_environment, sha256_file, validate_deferred_channel_join, validate_channel_gpu_first
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_owned, run_sampled

ROOT=Path(__file__).resolve().parents[2]
MODES=("gpu","encoder_local","encoder_retained")
WEIGHT_MODES=("gpu","gpu_weights","encoder_retained","encoder_weights")
COMBINED_MODES=("gpu_weights","dit_weights","dit_encoder_weights")
LORA_RANK_MODES=("ranks_off","ranks_on")
WEIGHT_CODE_MODES=("code_cache_off","code_cache_on")
DOWN_RANK_MODES=("down_ranks_off","down_ranks_on")
BF16_RANK_MODES=("gpu_operand_off","gpu_operand_on","hybrid_operand_off","hybrid_operand_on")
BF16_RANK_MARKER="experimental original BF16 LoRA A operands with FP32 ranks and B/delta arithmetic"
STUDENT_REUSE_MODES=("gpu_student_off","gpu_student_on","hybrid_student_off","hybrid_student_on")
STUDENT_REUSE_MARKER="experimental six-step student final FFN reuse, includes full LoRA FFN output"
B_EPILOGUE_MODES=("gpu_b_off","gpu_b_on","hybrid_b_off","hybrid_b_on")
B_EPILOGUE_MARKER="experimental BF16 LoRA B operands with fused F32 scale/base epilogue, original F32 A ranks"
JOINT_AB_MODES=("gpu_joint_off","gpu_joint_on","hybrid_joint_off","hybrid_joint_on")
JOINT_AB_MARKER="experimental joint BF16 LoRA A/B operands, F32 ranks and fused B epilogue"
PRECISION_MODES=("gpu_joint","gpu_fp16","hybrid_joint","hybrid_fp16")
FP16_RANK_MARKER="experimental FP16 low-rank LoRA matmuls"
DEFER_PREFILL_MODES=("gpu_defer_control","hybrid_defer_off","hybrid_defer_on")
GPU_FIRST_MODES=("gpu_launch_control","hybrid_launch_ane_first","hybrid_launch_gpu_first")
COMPILED_ENCODER_MODES=("gpu_encoder_eager","gpu_encoder_compiled","encoder_eager","encoder_compiled")
COMPILED_ENCODER_MARKER="experimental compiled Qwen3-VL GPU blocks with dynamic original source arrays"


def validate_compiled_encoder(rows,lines,enabled,with_ane,require_graph_reuse=False):
    if not rows or type(enabled) is not bool or type(with_ane) is not bool:
        raise ValueError("need explicit compiled encoder policy and actual requests")
    records=[json.loads(line)["qwen_encoder_compiled_gpu"] for line in lines if "qwen_encoder_compiled_gpu" in line]
    if len(records)!=(len(rows) if enabled else 0):raise ValueError("missing/duplicate compiled encoder execution records")
    for row in rows:
        if (COMPILED_ENCODER_MARKER in (row.get("acceleration_selection") or ""))!=enabled:
            raise ValueError("compiled encoder selection differs from actual profile")
    for record in records:
        if not isinstance(record,dict):raise ValueError("malformed compiled encoder execution record")
        keys=("rows","layers","full_blocks","attention_segments","channel_ffn_segments","full_ffn_segments")
        if any(type(record.get(k)) is not int or record[k]<0 for k in keys) or record["rows"]<=0 or record["layers"]!=36 or \
            record["full_blocks"]!=(0 if with_ane else 36) or record["attention_segments"]!=(36 if with_ane else 0) or \
            record["channel_ffn_segments"]!=(36 if with_ane else 0) or record["full_ffn_segments"]!=0 or \
            record.get("scope")!="evaluated language hidden; dynamic original array arguments; graph invocations, not physical kernels":
            raise ValueError("compiled encoder needs successful whole GPU blocks or Private channel segments, not policy-only/fallback telemetry")
        if require_graph_reuse and (type(record.get("block_graphs")) is not int or record["block_graphs"]!=1 or
                type(record.get("ffn_graphs")) is not int or record["ffn_graphs"]!=(2 if with_ane else 0) or
                record.get("native_gqa_attention") is not True):
            raise ValueError("canonical encoder screen requires one declared block body and bounded request-local FFN bodies")


def validate_precision_policy(rows,policy):
    if policy=="joint":
        validate_joint_ab(rows,True)
        return
    if policy!="fp16" or not rows:raise ValueError("missing/unknown rank precision policy")
    for row in rows:
        selection=row.get("acceleration_selection") or ""
        if (FP16_RANK_MARKER not in selection or any(marker in selection for marker in
                (JOINT_AB_MARKER,B_EPILOGUE_MARKER,BF16_RANK_MARKER,STUDENT_REUSE_MARKER)) or
            row.get("student_ffn_reuse") is not None or row.get("lora_strategy")!="inference_time" or
            type(row.get("lora_applied_projections")) is not int or row["lora_applied_projections"]!=227):
            raise ValueError("FP16 screen needs the existing complete FP16-rank route, not joint/F32 policies")


def validate_joint_ab(rows,enabled):
    if not rows:raise ValueError("missing joint A/B requests")
    for row in rows:
        selection=row.get("acceleration_selection") or ""
        if ((JOINT_AB_MARKER in selection)!=enabled or (B_EPILOGUE_MARKER in selection)==enabled or
            BF16_RANK_MARKER in selection or STUDENT_REUSE_MARKER in selection or
            "experimental FP16 low-rank LoRA matmuls" in selection or row.get("student_ffn_reuse") is not None or
            row.get("lora_strategy")!="inference_time" or type(row.get("lora_applied_projections")) is not int or
            row["lora_applied_projections"]!=227):
            raise ValueError("joint screen requires distinct B-only and joint F32-rank policies with complete original adapter")


def validate_b_epilogue(rows,enabled):
    """Actual adapter bindings and policy selection, not physical B kernel counts."""
    if not rows:raise ValueError("missing fused B request receipts")
    for row in rows:
        selection=row.get("acceleration_selection") or ""
        if (B_EPILOGUE_MARKER in selection)!=enabled or BF16_RANK_MARKER in selection or STUDENT_REUSE_MARKER in selection or \
            "experimental FP16 low-rank LoRA matmuls" in selection or row.get("student_ffn_reuse") is not None or \
            row.get("lora_strategy")!="inference_time" or type(row.get("lora_applied_projections")) is not int or row["lora_applied_projections"]!=227:
            raise ValueError("fused B screen requires original F32 A ranks/full adapter, not another arithmetic/temporal policy")


def validate_student_reuse(rows,enabled,layers=32):
    if not rows:raise ValueError("missing student FFN reuse receipts")
    for row in rows:
        metrics=row.get("student_ffn_reuse")
        if (STUDENT_REUSE_MARKER in (row.get("acceleration_selection") or ""))!=enabled:
            raise ValueError("student FFN selection mismatch")
        if row.get("lora_strategy")!="inference_time" or type(row.get("lora_applied_projections")) is not int or row["lora_applied_projections"]!=227:
            raise ValueError("student screen requires actual complete original LoRA binding")
        if not enabled:
            if metrics is not None:raise ValueError("disabled student arm published reuse")
            continue
        if not isinstance(metrics,dict) or metrics.get("enabled") is not True or metrics.get("released_before_vae") is not True:
            raise ValueError("student cache lifecycle receipt missing")
        if any(type(metrics.get(k)) is not int for k in ("captured_blocks_this_request","reused_blocks_this_request","peak_logical_cache_bytes")) or \
            metrics["captured_blocks_this_request"]!=32 or metrics["reused_blocks_this_request"]!=layers or \
            not 0<metrics["peak_logical_cache_bytes"]<=256<<20:
            raise ValueError("student reuse did not complete a bounded full32-layer capture/reuse")


def validate_bf16_rank_operands(rows,enabled):
    """Check selection and real adapter bindings, not physical kernel counts."""
    if not rows:raise ValueError("missing BF16 operand request receipts")
    for row in rows:
        if (BF16_RANK_MARKER in (row.get("acceleration_selection") or ""))!=enabled or \
            "experimental FP16 low-rank LoRA matmuls" in (row.get("acceleration_selection") or "") or \
            row.get("lora_strategy")!="inference_time" or type(row.get("lora_applied_projections")) is not int or \
            row.get("lora_applied_projections")!=227:
            raise ValueError("need actual original LoRA bindings and BF16 operand/F32 rank selection, not FP16 ranks")


def validate_down_ranks(rows, enabled):
    previous_blocks=0;previous_arrays=0
    for row in rows:
        runtime=(row.get("hybrid") or {}).get("runtime_weight") or {}
        blocks=runtime.get("split_down_rank_blocks_session_total")
        arrays=runtime.get("split_down_rank_arrays_session_total")
        if any(type(v) is not int or v<0 for v in (blocks,arrays)):raise ValueError("missing split down-rank execution counters")
        if enabled:
            if blocks<=previous_blocks or arrays<=previous_arrays or arrays<blocks:
                raise ValueError("need actual successful split down-rank blocks, not selection intent")
        elif blocks or arrays:raise ValueError("disabled split down-rank arm executed sharded corrections")
        previous_blocks=blocks;previous_arrays=arrays


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
    request=dict(model="qwen-image-2.1",operation="image.edit" if image_paths else "image.generate",prompt=prompt,
        width=512,height=512,steps=6 if lora else 40,seed=29,audio=False,
        residency="resident",execution="gpu",allow_approximation=True,qwen21_reference_size=512 if image_paths else 1024,
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
    parser.add_argument("--down-ranks-screen",action="store_true",
        help="GPU / original joined-hidden down / split FP32 input ranks; no weight code cache, shared gate/up ranks on both hybrid arms")
    parser.add_argument("--bf16-rank-operands-screen",action="store_true",
        help="matched complete GPU off/on and Private hybrid off/on; original BF16 A operands with F32 ranks, no rank narrowing or down-rank split/cache")
    parser.add_argument("--student-final-ffn-reuse-screen",action="store_true",
        help="matched GPU/hybrid off/on; full LoRA FFN outputs reused on final step, not exact arithmetic or a faster kernel")
    parser.add_argument("--student-reuse-layers",type=int,choices=(16,32),default=32)
    parser.add_argument("--b-epilogue-screen",action="store_true",help="matched GPU/hybrid off/on for BF16 B operands and fused epilogue, original F32 A ranks")
    parser.add_argument("--joint-ab-screen",action="store_true",help="GPU and prefill-parallel B-only vs joint BF16 A/B; complete GPU decode on both hybrid arms")
    parser.add_argument("--lora-precision-screen",action="store_true",help="matched best joint BF16/F32 ranks vs existing FP16 ranks on GPU and prefill-parallel runtime")
    parser.add_argument("--defer-prefill-screen",action="store_true",help="complete joint GPU / prefill-only eager / prefill-only owned deferred join; actual deferred counters required")
    parser.add_argument("--gpu-first-prefill-screen",action="store_true",help="joint GPU / prefill ANE-first / prefill GPU-first; same share, precision and join policy, GPU decode")
    parser.add_argument("--gpu-first-deferred",action="store_true",help="use the same owned deferred join on BOTH launch-order hybrid arms")
    parser.add_argument("--compiled-encoder-screen",action="store_true",help="same-library GPU/Private encoder eager/compiled, GPU DiT, same original source/executor retention")
    parser.add_argument("--reference",type=Path,action="append",default=[])
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
    if args.gpu_first_deferred and not args.gpu_first_prefill_screen:
        parser.error("GPU-first deferred policy requires --gpu-first-prefill-screen")
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
    if args.down_ranks_screen:
        if args.lora_ranks_screen or args.lora_ranks_gpu_control or args.weight_code_cache_bytes is not None or \
            not args.lora or not args.dit_manifest or args.backend!="private" or args.global_channels==0:
            parser.error("down rank screen requires real LoRA, fixed Private DiT and no unrelated rank/cache screen")
        modes=("gpu_weights",*DOWN_RANK_MODES)
    if args.bf16_rank_operands_screen:
        if args.lora_ranks_screen or args.lora_ranks_gpu_control or args.down_ranks_screen or args.weight_code_cache_bytes is not None or \
            not args.lora or not args.dit_manifest or args.backend!="private" or args.global_channels==0:
            parser.error("BF16 rank operand screen requires real LoRA/fixed Private DiT and no unrelated ranks/cache screen")
        modes=BF16_RANK_MODES
    if args.student_final_ffn_reuse_screen:
        if args.lora_ranks_screen or args.lora_ranks_gpu_control or args.down_ranks_screen or args.bf16_rank_operands_screen or args.weight_code_cache_bytes is not None or \
            not args.lora or not args.dit_manifest or args.backend!="private" or args.global_channels==0:
            parser.error("student reuse screen requires real LoRA/fixed Private DiT and no unrelated rank/cache screen")
        modes=STUDENT_REUSE_MODES
    if args.b_epilogue_screen:
        if args.lora_ranks_screen or args.lora_ranks_gpu_control or args.down_ranks_screen or args.bf16_rank_operands_screen or args.student_final_ffn_reuse_screen or args.weight_code_cache_bytes is not None or \
            not args.lora or not args.dit_manifest or args.backend!="private" or args.global_channels==0:
            parser.error("fused B screen requires real LoRA/fixed Private DiT and no unrelated rank/cache screen")
        modes=B_EPILOGUE_MODES
    if args.joint_ab_screen:
        if args.lora_ranks_screen or args.lora_ranks_gpu_control or args.down_ranks_screen or args.bf16_rank_operands_screen or args.student_final_ffn_reuse_screen or args.b_epilogue_screen or args.weight_code_cache_bytes is not None or \
            not args.lora or not args.dit_manifest or args.backend!="private" or args.global_channels==0:
            parser.error("joint A/B screen requires real LoRA/fixed Private DiT and no other precision/cache screen")
        modes=JOINT_AB_MODES
    if args.lora_precision_screen:
        if args.lora_ranks_screen or args.lora_ranks_gpu_control or args.down_ranks_screen or args.bf16_rank_operands_screen or args.student_final_ffn_reuse_screen or args.b_epilogue_screen or args.joint_ab_screen or args.weight_code_cache_bytes is not None or \
            not args.lora or not args.dit_manifest or args.backend!="private" or args.global_channels==0:
            parser.error("precision screen requires real LoRA/fixed Private DiT without another precision/cache screen")
        modes=PRECISION_MODES
    if args.defer_prefill_screen:
        if args.lora_ranks_screen or args.lora_ranks_gpu_control or args.down_ranks_screen or args.bf16_rank_operands_screen or args.student_final_ffn_reuse_screen or args.b_epilogue_screen or args.joint_ab_screen or args.lora_precision_screen or args.weight_code_cache_bytes is not None or \
            not args.lora or not args.dit_manifest or args.backend!="private" or args.global_channels==0:
            parser.error("deferred prefill screen requires real LoRA/fixed Private DiT without another cache/precision screen")
        modes=DEFER_PREFILL_MODES
    if args.gpu_first_prefill_screen:
        if args.lora_ranks_screen or args.lora_ranks_gpu_control or args.down_ranks_screen or args.bf16_rank_operands_screen or args.student_final_ffn_reuse_screen or args.b_epilogue_screen or args.joint_ab_screen or args.lora_precision_screen or args.defer_prefill_screen or args.weight_code_cache_bytes is not None or \
            not args.lora or not args.dit_manifest or args.backend!="private" or args.global_channels==0:
            parser.error("GPU-first prefill screen requires original LoRA/fixed Private DiT without another screen")
        modes=GPU_FIRST_MODES
    if args.compiled_encoder_screen:
        if args.lora_ranks_screen or args.lora_ranks_gpu_control or args.down_ranks_screen or args.bf16_rank_operands_screen or args.student_final_ffn_reuse_screen or args.b_epilogue_screen or args.joint_ab_screen or args.lora_precision_screen or args.defer_prefill_screen or args.gpu_first_prefill_screen or args.weight_code_cache_bytes is not None or args.dit_manifest or args.backend!="private":
            parser.error("compiled encoder screen requires separate fixed Private encoder with GPU DiT and no other screen")
        modes=COMPILED_ENCODER_MODES
    order=args.order.split(",") if args.order else list(modes)
    if len(order)!=len(modes) or set(order)!=set(modes):parser.error("order must include each matched mode once")
    if not (0 if args.compiled_encoder_screen else 1)<=len(args.reference)<=2 or not 3<=len(args.prompt)<=9 or len(set(args.prompt))!=len(args.prompt):
        parser.error("need 1..2 references (compiled encoder also supports generation) and 3..9 distinct fresh prompts")
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
    summary.update(weight_retention_screen=args.weights or bool(dit_manifest) or args.compiled_encoder_screen,global_channels=args.global_channels,
        combined_dit_encoder=bool(dit_manifest) and not args.lora_ranks_screen and not args.down_ranks_screen and not args.bf16_rank_operands_screen and not args.student_final_ffn_reuse_screen and not args.b_epilogue_screen and not args.joint_ab_screen and not args.lora_precision_screen and not args.defer_prefill_screen and not args.gpu_first_prefill_screen and args.weight_code_cache_bytes is None,
        lora_ranks_screen=args.lora_ranks_screen,weight_code_cache_bytes=args.weight_code_cache_bytes,
        weight_code_cache_mode=args.weight_code_cache_mode)
    summary["lora_ranks_gpu_control"]=args.lora_ranks_gpu_control
    summary["down_ranks_screen"]=args.down_ranks_screen
    summary["bf16_rank_operands_screen"]=args.bf16_rank_operands_screen
    summary["student_final_ffn_reuse_screen"]=args.student_final_ffn_reuse_screen
    summary["student_reuse_layers"]=args.student_reuse_layers if args.student_final_ffn_reuse_screen else 0
    summary["b_epilogue_screen"]=args.b_epilogue_screen
    summary["joint_ab_screen"]=args.joint_ab_screen
    summary["lora_precision_screen"]=args.lora_precision_screen
    summary["defer_prefill_screen"]=args.defer_prefill_screen
    summary["gpu_first_prefill_screen"]=args.gpu_first_prefill_screen
    summary["gpu_first_deferred"]=args.gpu_first_deferred
    summary["compiled_encoder_screen"]=args.compiled_encoder_screen
    target=args.output/"summary.json"
    target.write_text(json.dumps(summary,indent=2)+"\n")
    for mode in order:
        if model_snapshot(model)!=model_before:raise ValueError("model file generation/layout changed between modes")
        env=benchmark_environment()
        if lora:env["TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC"]="1"
        if mode in ("gpu_weights","encoder_weights","dit_weights","dit_encoder_weights",*LORA_RANK_MODES,*WEIGHT_CODE_MODES,*DOWN_RANK_MODES,*BF16_RANK_MODES,*STUDENT_REUSE_MODES,*B_EPILOGUE_MODES,*JOINT_AB_MODES,*PRECISION_MODES,*DEFER_PREFILL_MODES,*GPU_FIRST_MODES,*COMPILED_ENCODER_MODES):env["TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS"]="1"
        uses_encoder=mode.startswith("encoder_") or mode=="dit_encoder_weights"
        uses_dit=mode in ("dit_weights","dit_encoder_weights",*LORA_RANK_MODES,*WEIGHT_CODE_MODES,*DOWN_RANK_MODES,"hybrid_operand_off","hybrid_operand_on","hybrid_student_off","hybrid_student_on","hybrid_b_off","hybrid_b_on","hybrid_joint_off","hybrid_joint_on","hybrid_joint","hybrid_fp16","hybrid_defer_off","hybrid_defer_on","hybrid_launch_ane_first","hybrid_launch_gpu_first")
        if mode in COMPILED_ENCODER_MODES:
            env["TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU"]="1" if mode.endswith("_compiled") else "0"
            if lora:env["TURBOCIDER_QWEN21_LORA_BF16_AB"]="1"
        if mode in GPU_FIRST_MODES:
            env.update(TURBOCIDER_QWEN21_LORA_BF16_AB="1",TURBOCIDER_QWEN21_PROFILE_STEPS="1",
                TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE="prefill",
                TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN="1" if uses_dit and args.gpu_first_deferred else "0")
            if uses_dit:
                env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1"
                env["TURBOCIDER_RUNTIME_ANE_CHANNEL_GPU_FIRST"]="1" if mode=="hybrid_launch_gpu_first" else "0"
        if mode in DEFER_PREFILL_MODES:
            env.update(TURBOCIDER_QWEN21_LORA_BF16_AB="1",TURBOCIDER_QWEN21_PROFILE_STEPS="1",TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE="prefill",
                TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN="1" if mode=="hybrid_defer_on" else "0")
            if uses_dit:env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1"
        if mode in PRECISION_MODES:
            env.update(TURBOCIDER_QWEN21_LORA_BF16_AB="1" if mode.endswith("_joint") else "0",
                TURBOCIDER_QWEN21_VIGGLE_LORA_FP16="1" if mode.endswith("_fp16") else "0",
                TURBOCIDER_QWEN21_PROFILE_STEPS="1",TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE="prefill")
            if uses_dit:env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1"
        if mode in JOINT_AB_MODES:
            env.update(TURBOCIDER_QWEN21_LORA_BF16_AB="1" if mode.endswith("_on") else "0",
                TURBOCIDER_QWEN21_LORA_B_FUSED_EPILOGUE="0" if mode.endswith("_on") else "1",
                TURBOCIDER_QWEN21_PROFILE_STEPS="1",TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE="prefill")
            if uses_dit:env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1"
        if mode in B_EPILOGUE_MODES:
            env["TURBOCIDER_QWEN21_LORA_B_FUSED_EPILOGUE"]="1" if mode.endswith("_on") else "0"
            if uses_dit:env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1"
        if mode in STUDENT_REUSE_MODES:
            env["TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE"]=str(args.student_reuse_layers) if mode.endswith("_on") else "0"
            if uses_dit:env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1"
        if mode in BF16_RANK_MODES:
            env["TURBOCIDER_QWEN21_LORA_BF16_OPERANDS_FP32_RANKS"]="1" if mode.endswith("_on") else "0"
            if uses_dit:env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1"
        if mode in LORA_RANK_MODES:
            env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1" if mode=="ranks_on" else "0"
        if mode in WEIGHT_CODE_MODES:
            env["TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES"]=str(args.weight_code_cache_bytes if mode=="code_cache_on" else 0)
            env["TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_MODE"]=args.weight_code_cache_mode if mode=="code_cache_on" else "copy"
            if lora:env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1"
        if mode in DOWN_RANK_MODES:
            env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="1"
            env["TURBOCIDER_QWEN21_RUNTIME_SPLIT_DOWN_RANKS"]="1" if mode=="down_ranks_on" else "0"
        if uses_encoder or uses_dit:
            env.update(TURBOCIDER_ANE_BACKEND=args.backend,TURBOCIDER_RUNTIME_ANE_CHUNKS="1",
                TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1",TURBOCIDER_QWEN21_ENCODER_ANE_CHANNELS=str(args.channels),
                TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME="1" if mode in ("encoder_retained","encoder_weights","dit_encoder_weights","encoder_eager","encoder_compiled") else "0")
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
        baseline_mode="gpu_weights" if mode.startswith(("gpu_operand_","gpu_student_","gpu_b_","gpu_joint_","gpu_defer_","gpu_launch_")) or mode in ("gpu_joint","gpu_fp16") else "dit_weights" if mode.startswith(("hybrid_operand_","hybrid_student_","hybrid_b_","hybrid_joint_","hybrid_defer_","hybrid_launch_")) or mode in ("hybrid_joint","hybrid_fp16") else mode
        if mode in COMPILED_ENCODER_MODES:baseline_mode="encoder_weights" if uses_encoder else "gpu_weights"
        validate_rows(rows,"dit_weights" if mode in (*LORA_RANK_MODES,*WEIGHT_CODE_MODES,*DOWN_RANK_MODES) else baseline_mode,
            len(args.prompt),args.backend=="private",args.channels,args.global_channels)
        if mode in LORA_RANK_MODES:validate_shared_ranks(rows,mode=="ranks_on")
        if mode in WEIGHT_CODE_MODES:
            validate_weight_code_cache(rows,args.weight_code_cache_bytes if mode=="code_cache_on" else 0,
                args.weight_code_cache_mode if mode=="code_cache_on" else "copy")
            if lora:validate_shared_ranks(rows,True)
        if mode in DOWN_RANK_MODES:
            validate_down_ranks(rows,mode=="down_ranks_on");validate_shared_ranks(rows,True)
        if mode in BF16_RANK_MODES:
            validate_bf16_rank_operands(rows,mode.endswith("_on"))
            if uses_dit:validate_shared_ranks(rows,True)
        if mode in STUDENT_REUSE_MODES:
            validate_student_reuse(rows,mode.endswith("_on"),args.student_reuse_layers)
            if uses_dit:validate_shared_ranks(rows,True)
        if mode in B_EPILOGUE_MODES:
            validate_b_epilogue(rows,mode.endswith("_on"))
            if uses_dit:validate_shared_ranks(rows,True)
        if mode in JOINT_AB_MODES:
            from qwen_ffn_phase_screen import validate_phases
            validate_joint_ab(rows,mode.endswith("_on"));validate_phases(rows,"prefill" if uses_dit else "gpu",True)
            if uses_dit:validate_shared_ranks(rows,True)
        if mode in PRECISION_MODES:
            from qwen_ffn_phase_screen import validate_phases
            validate_precision_policy(rows,"joint" if mode.endswith("_joint") else "fp16")
            validate_phases(rows,"prefill" if uses_dit else "gpu",True)
            if uses_dit:validate_shared_ranks(rows,True)
        if mode in DEFER_PREFILL_MODES:
            from qwen_ffn_phase_screen import validate_phases
            validate_joint_ab(rows,True);validate_phases(rows,"prefill" if uses_dit else "gpu",True)
            if uses_dit:
                validate_shared_ranks(rows,True);validate_deferred_channel_join(rows,mode=="hybrid_defer_on")
        if mode in GPU_FIRST_MODES:
            from qwen_ffn_phase_screen import validate_phases
            validate_joint_ab(rows,True);validate_phases(rows,"prefill" if uses_dit else "gpu",True)
            if uses_dit:
                validate_shared_ranks(rows,True)
                validate_deferred_channel_join(rows,args.gpu_first_deferred)
                validate_channel_gpu_first(rows,mode=="hybrid_launch_gpu_first")
        if mode in COMPILED_ENCODER_MODES:
            validate_compiled_encoder(rows,(args.output/f"{mode}.stderr.txt").read_text().splitlines(),mode.endswith("_compiled"),uses_encoder,True)
            if lora:validate_joint_ab(rows,True)
        if any(sha256_file(Path(path))!=digest for path,digest in before.items()):raise ValueError("input/runtime bytes changed")
        if model_snapshot(model)!=model_before:raise ValueError("model file generation/layout changed during mode")
        times=[row["timings_seconds"]["request_wall"] for row in rows]
        trial=dict(mode=mode,cold_request_seconds=times[0],warm_fresh_request_seconds=times[1:],
            warm_fresh_median_seconds=statistics.median(times[1:]),text_seconds=[row["timings_seconds"]["text_encode"] for row in rows],
            memory=memory,load=load,png_sha256=[sha256_file(args.output/f"{mode}-{i}.png") for i in range(len(rows))])
        if mode in (*JOINT_AB_MODES,*PRECISION_MODES,*DEFER_PREFILL_MODES,*GPU_FIRST_MODES):
            trial.update(phase_receipts=[row["qwen_ffn_phases"] for row in rows],
                warm_prefill_median_seconds=statistics.median(row["qwen_ffn_phases"]["prefill"]["step_seconds"] for row in rows[1:]),
                warm_decode_total_median_seconds=statistics.median(row["qwen_ffn_phases"]["decode"]["step_seconds"] for row in rows[1:]))
        summary["trials"].append(trial);target.write_text(json.dumps(summary,indent=2)+"\n")
    summary["status"]="complete_diagnostic";target.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
