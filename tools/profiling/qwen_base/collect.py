#!/usr/bin/env python3
"""Base-only complete-request profiler; stock and intrusive diagnostics separate."""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux9_bf16_framework_screen import write_json
import qwen_base_runtime_screen as base
from qwen_encoder_residency_screen import validate_weight_code_cache, model_snapshot
from runtime_ane_common import benchmark_environment, sha256_file, validate_results, validate_fixed_async, validate_fp32_channel_join
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

CACHE_BYTES = 1600 << 20
FFN_PROJECTIONS = ("descriptor", "k4096", "single", "up-single", "down-single")
FFN_ACTIVATIONS = ("exp", "sigmoid", "silu", "sigmoid-complement")
SOURCES = tuple("tools/profiling/qwen_base/" + name for name in ("CMakeLists.txt", "observer.mm", "collect.py", "analyze.py")) + base.SOURCES


class Observer:
    def __init__(self, path):
        self.lib = C.CDLL(str(path))
        self.lib.tc_qwen_profile_install.argtypes = [C.c_int]
        self.lib.tc_qwen_profile_scope.argtypes = [C.c_int]
        self.lib.tc_qwen_profile_now_ns.restype = C.c_uint64
        self.lib.tc_qwen_profile_wait.argtypes = [C.c_uint]
        self.lib.tc_qwen_profile_string_free.argtypes = [C.c_void_p]
        for name in ("tc_qwen_profile_finish", "tc_qwen_profile_counters"):
            getattr(self.lib, name).restype = C.c_void_p
        if self.lib.tc_qwen_profile_install(1):
            raise RuntimeError("native selector observer could not install; no model started")
        if self.lib.tc_qwen_profile_self_test():
            self.take("tc_qwen_profile_finish")
            raise RuntimeError("64-byte Metal completion/clock calibration failed")

    def take(self, name):
        pointer = getattr(self.lib, name)()
        if not pointer:
            raise RuntimeError("native observer report unavailable")
        try:
            return json.loads(C.string_at(pointer))
        finally:
            self.lib.tc_qwen_profile_string_free(pointer)

    def now(self):
        return self.lib.tc_qwen_profile_now_ns()


def environment(route, mode, channels=4096, runtime_prefix=False, resident_prefix=False, private_qkv=False, fixed_program=False, ffn_projection="descriptor", ffn_activation="exp", *,runtime_text_prefix=None,text_prefix_qkv_step0=None,private_qkv_prefetch=None,gpu_dual_swiglu=None,function_rows=None,edit_prefix_qkv_step0=None,compiled_complement=None):
    if route not in ("gpu", "runtime", "cache") or channels not in (4096, 5120, 6144) or (channels != 4096 and route != "runtime"):
        raise ValueError("explicit supported route/fixed partition required")
    if ((runtime_prefix or resident_prefix) and (route != "runtime" or mode not in ("stock", "timeline"))) or (runtime_prefix and resident_prefix):
        raise ValueError("runtime prefix pilot requires cache-off stock/timeline runtime")
    if private_qkv and (route != "runtime" or mode not in ("stock", "timeline") or resident_prefix):
        raise ValueError("joint Private QKV requires explicit cache-off stock/timeline runtime without retained Transformer")
    if fixed_program and (route != "runtime" or mode not in ("stock", "timeline") or resident_prefix):
        raise ValueError("fixed program requires cache-off stock/timeline runtime without retained Transformer")
    if ffn_projection not in FFN_PROJECTIONS or (ffn_projection!="descriptor" and
            (not fixed_program or route!="runtime" or mode not in ("stock","timeline") or resident_prefix)):
        raise ValueError("explicit FFN projection requires the fixed cache-off runtime recipe")
    if ffn_activation not in FFN_ACTIVATIONS or (ffn_activation!="exp" and
            (ffn_projection!="descriptor" or not fixed_program or route!="runtime" or mode not in ("stock","timeline") or resident_prefix)):
        raise ValueError("explicit FFN activation requires fixed cache-off runtime and descriptor projection")
    if runtime_text_prefix is not None and type(runtime_text_prefix) is not bool:
        raise ValueError("explicit text prefix boolean required")
    if runtime_text_prefix and (route!="runtime" or mode not in ("stock","timeline") or not fixed_program or runtime_prefix or resident_prefix):
        raise ValueError("text prefix requires separate fixed-program cache-off runtime stock/timeline")
    if text_prefix_qkv_step0 not in (None,"ane","gpu") or (text_prefix_qkv_step0=="gpu" and (not runtime_text_prefix or not private_qkv)):
        raise ValueError("explicit GPU first-step QKV requires joint runtime text prefix")
    if private_qkv_prefetch is not None and type(private_qkv_prefetch) is not bool:
        raise ValueError("explicit QKV prefetch boolean required")
    if private_qkv_prefetch and (not private_qkv or not fixed_program or route!="runtime" or mode not in ("stock","timeline") or resident_prefix):
        raise ValueError("QKV prefetch requires fixed joint cache-off runtime stock/timeline")
    if gpu_dual_swiglu is not None and type(gpu_dual_swiglu) is not bool:
        raise ValueError("explicit GPU dual SwiGLU boolean required")
    if gpu_dual_swiglu and (route!="runtime" or mode not in ("stock","timeline") or channels!=6144 or not private_qkv or not fixed_program or
        resident_prefix or ffn_activation!="sigmoid-complement" or ffn_projection!="descriptor"):
        raise ValueError("GPU dual SwiGLU requires fixed6144 joint sigmoid-complement runtime")
    flags = base.environment("gpu" if route == "gpu" else "runtime-off" if route == "runtime" else "runtime-surface")
    flags["TURBOCIDER_QWEN21_PRIVATE_QKV"] = "1" if private_qkv else "0"
    flags["TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM"] = "1" if fixed_program else "0"
    flags["TURBOCIDER_PRIVATE_ANE_W8_PROJECTION"] = ffn_projection
    flags["TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION"] = ffn_activation
    if route == "cache":
        flags["TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES"] = str(CACHE_BYTES)
    if route != "gpu":
        flags["TURBOCIDER_PRIVATE_ANE_CHANNELS"] = str(channels)
        flags["TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT"] = "1" if runtime_prefix else "0"
    if runtime_text_prefix is not None:
        flags["TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_SNAPSHOT"]="1" if runtime_text_prefix else "0"
    if text_prefix_qkv_step0 is not None:
        flags["TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_QKV_STEP0"]=text_prefix_qkv_step0
    if private_qkv_prefetch is not None:
        flags["TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH"]="1" if private_qkv_prefetch else "0"
    if gpu_dual_swiglu is not None:
        flags["TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU"]="1" if gpu_dual_swiglu else "0"
    if resident_prefix:
        flags["TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV"] = "1"
    if mode != "stock":
        flags["TURBOCIDER_QWEN21_PROFILE_STEPS"] = "1"
    if mode == "gpu-ops":
        flags["TURBOCIDER_QWEN21_PROFILE_GPU_OPS"] = "1"
    if mode == "runtime-layers":
        flags["TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC"] = "0"
        flags["TURBOCIDER_RUNTIME_ANE_PROFILE"] = "1"
    if function_rows is not None:
        if (function_rows not in ('512,1024','512,1024,1536','512,1024,1536,2048') or route!='runtime' or
                mode not in ('stock','timeline') or not fixed_program or resident_prefix or ffn_projection!='descriptor'):
            raise ValueError('function family requires an explicit fixed base runtime stock/timeline configuration')
        flags['TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS']=function_rows
    if edit_prefix_qkv_step0 is not None:
        if (edit_prefix_qkv_step0 not in ('ane','gpu') or route!='runtime' or mode not in ('stock','timeline') or
                not runtime_prefix or resident_prefix or not private_qkv or not fixed_program or runtime_text_prefix):
            raise ValueError('edit first-step policy requires an explicit fixed joint edit-prefix stock/timeline')
        flags['TURBOCIDER_QWEN21_RUNTIME_EDIT_PREFIX_QKV_STEP0']=edit_prefix_qkv_step0
    if compiled_complement is not None:
        if type(compiled_complement) is not bool:
            raise ValueError('explicit compiled complement boolean required')
        if compiled_complement and (route!='runtime' or mode not in ('stock','timeline') or channels!=6144 or not private_qkv or
            not fixed_program or resident_prefix or ffn_projection!='descriptor' or ffn_activation!='sigmoid-complement' or gpu_dual_swiglu or function_rows):
            raise ValueError('compiled complement requires fixed6144 joint sigmoid-complement runtime without GPU-dual/family')
        flags['TURBOCIDER_QWEN21_COMPILED_COMPLEMENT']='1' if compiled_complement else '0'
    return flags


def validate(metrics, route, mode, channels=4096, runtime_prefix=False, resident_prefix=False, private_qkv=False, fixed_program=False, ffn_projection="descriptor", ffn_activation="exp", *,runtime_text_prefix=False,text_prefix_matrix=False,text_prefix_qkv_step0="ane",private_qkv_prefetch=None,gpu_dual_swiglu=None,function_rows=None,edit_prefix_qkv_step0=None,compiled_complement=None):
    if text_prefix_qkv_step0 not in ("ane","gpu") or (text_prefix_qkv_step0=="gpu" and (not runtime_text_prefix or not private_qkv)):
        raise ValueError("explicit selected joint text-prefix QKV policy required")
    if not metrics:
        raise ValueError("complete base results required")
    for row in metrics:
        if ((row.get("model"), row.get("width"), row.get("height"), row.get("steps"), row.get("actual_denoise_steps")) !=
                ("qwen-image-2.1", 512, 512, 40, 40) or row.get("lora_applied_projections", 0) != 0 or row.get("qwen21_dbcache")):
            raise ValueError("base-only full forty-step geometry required")
    validate_results(metrics, "gpu" if route == "gpu" else "runtime", len(metrics), model_id="qwen-image-2.1",
                     runtime_backend="public" if route == "gpu" else "private", expect_device_io=route != "gpu",
                     expected_data_path=None if route == "gpu" else "w8a8_hadamard")
    if route == "gpu":
        if private_qkv or private_qkv_prefetch or gpu_dual_swiglu or compiled_complement or ffn_activation!="exp" or any(row.get("hybrid") or row.get("qwen_private_qkv") or row.get("qwen_private_qkv_prefetch") or row.get("qwen_gpu_dual_swiglu") or row.get('qwen_compiled_complement') for row in metrics):
            raise ValueError("GPU control cannot retain a hybrid backend")
        return
    validate_fp32_channel_join(metrics, True)
    validate_compiled_complement(metrics,bool(compiled_complement))
    validate_fixed_async(metrics, mode != "runtime-layers")
    validate_weight_code_cache(metrics, CACHE_BYTES if route == "cache" else 0, "surface" if route == "cache" else "copy")
    for index, row in enumerate(metrics):
        runtime = row["hybrid"]["runtime_weight"]
        functions=[] if function_rows is None else list(map(int,function_rows.split(',')))
        if runtime.get('function_rows',[])!=functions:raise ValueError('actual function list differs from explicit configuration')
        if functions and (row['hybrid'].get('bucket')!=functions[-1] or runtime.get('row_alignment')!=functions[0] or
                runtime.get('capability_calls_session_total')!=3*len(functions)):
            raise ValueError('every actual fixed function must be prewarmed and use its minimum row alignment')
        if (runtime["ane_channels"] != channels or runtime["gpu_channels"] != 12288-channels or
                runtime["hybrid_blocks_session_total"] != 1280 * (index + 1) or runtime["overflow_retries_session_total"] != 0 or
                runtime["headroom_scale"] != 1 or runtime["lora_channel_range_calls_session_total"] or runtime["lora_channel_full_calls_session_total"]):
            raise ValueError("one fixed base runtime partition, no recipe rebuild/fallback/adapter work required")
    if runtime_prefix or resident_prefix:
        validate_runtime_prefix(metrics, resident_prefix)
    if edit_prefix_qkv_step0 is not None and (edit_prefix_qkv_step0 not in ('ane','gpu') or not runtime_prefix or not private_qkv or not fixed_program):
        raise ValueError('explicit selected edit first-step policy requires its joint prefix route')
    for row in metrics:
        marker='; runtime edit prefix preserves first-step GPU QKV' in row.get('acceleration_selection','')
        if marker!=(edit_prefix_qkv_step0=='gpu'):raise ValueError('actual edit first-step QKV policy marker differs')
    if runtime_text_prefix:
        if runtime_prefix or resident_prefix or not fixed_program or route!="runtime" or mode not in ("stock","timeline"):
            raise ValueError("explicit separate fixed runtime text prefix contract required")
        validate_runtime_text_prefix(metrics,matrix=text_prefix_matrix,qkv_step0=text_prefix_qkv_step0)
    elif any(row.get("operation")=="image.generate" and row.get("qwen_runtime_prefix_snapshot") for row in metrics):
        raise ValueError("unrequested generation text prefix receipt")
    if private_qkv:
        hits=([False]*len(metrics) if text_prefix_qkv_step0=="gpu" else
              [False,True,False,False] if text_prefix_matrix else [i>0 for i in range(len(metrics))]) if runtime_text_prefix else None
        if edit_prefix_qkv_step0=='gpu':hits=[False]*len(metrics) # prefix still hits; QKV step0 is deliberately not ANE-eligible.
        validate_private_qkv(metrics, runtime_prefix,hit_pattern=hits)
    elif any(row.get("qwen_private_qkv") for row in metrics):
        raise ValueError("QKV-off control unexpectedly retained a Private QKV owner")
    if private_qkv_prefetch and (not private_qkv or not fixed_program or route!="runtime" or mode not in ("stock","timeline") or resident_prefix):
        raise ValueError("explicit fixed joint runtime required for lookahead receipts")
    if private_qkv and (private_qkv_prefetch is not None or any(row.get("qwen_private_qkv_prefetch") for row in metrics)):
        validate_private_qkv_prefetch(metrics,bool(private_qkv_prefetch))
    elif not private_qkv and any(row.get("qwen_private_qkv_prefetch") for row in metrics):
        raise ValueError("QKV-off control retained lookahead resources")
    if gpu_dual_swiglu and (not private_qkv or not fixed_program or channels!=6144 or ffn_activation!="sigmoid-complement" or ffn_projection!="descriptor"):
        raise ValueError("actual fixed6144 joint recipe required for GPU dual receipts")
    if private_qkv and (gpu_dual_swiglu is not None or any(row.get("qwen_gpu_dual_swiglu") for row in metrics)):
        validate_gpu_dual_swiglu(metrics,bool(gpu_dual_swiglu))
    elif not private_qkv and any(row.get("qwen_gpu_dual_swiglu") for row in metrics):
        raise ValueError("GPU dual receipt without joint runtime owner")
    if fixed_program:
        validate_fixed_program(metrics)
    elif any(row["hybrid"]["runtime_weight"].get("fixed_program_requested",False) for row in metrics):
        raise ValueError("legacy control unexpectedly selected fixed-program policy")
    validate_projection_receipt(metrics,ffn_projection)
    validate_activation_receipt(metrics,ffn_activation)


def validate_activation_receipt(metrics,activation):
    if activation not in FFN_ACTIVATIONS:raise ValueError("explicit supported FFN activation required")
    for row in metrics:
        recipe=row["hybrid"]["runtime_weight"].get("source_recipe","")
        tag=":w8-activation-"
        if ((activation=="exp" and tag in recipe) or
                (activation!="exp" and (recipe.count(tag)!=1 or tag+activation+"-v1" not in recipe or ":w8-projection-" in recipe))):
            raise ValueError("actual captured FFN activation recipe differs from requested policy")


def validate_projection_receipt(metrics,projection):
    if projection not in FFN_PROJECTIONS:raise ValueError("explicit supported FFN projection required")
    for row in metrics:
        recipe=row["hybrid"]["runtime_weight"].get("source_recipe","")
        tag=":w8-projection-"
        if (projection=="descriptor" and tag in recipe) or (projection!="descriptor" and tag+projection+"-v1" not in recipe):
            raise ValueError("actual captured FFN projection recipe differs from the requested policy")


def validate_fixed_program(metrics):
    for row in metrics:
        m=row["hybrid"]["runtime_weight"]
        if (m.get("fixed_program_requested") is not True or m.get("fixed_program_enabled") is not True or
                m.get("program_disabled") is not False or type(m.get("program_instances_session_total")) is not int or
                m["program_instances_session_total"]!=1 or m.get("blocked_rebuilds_session_total")!=0 or
                m.get("overflow_retries_session_total")!=0 or m.get("headroom_scale")!=1 or
                m.get("fallback_blocks_session_total")!=0):
            raise ValueError("one actual fixed Program instance, no blocked rebuilds/fallback/headroom change required")


def validate_private_qkv(metrics, runtime_prefix=False,*,hit_pattern=None):
    """Successful cumulative receipts; never accept a silent GPU fallback."""
    total = gpu = 0
    key = None
    previous_times = dict.fromkeys(("stage_lifetime_seconds_session_total", "activation_seconds_session_total", "join_seconds_session_total"), 0)
    for index, row in enumerate(metrics):
        hit = hit_pattern[index] if hit_pattern is not None else runtime_prefix and index > 0
        total += 1280 if hit else 1248
        gpu += 0 if hit else 32
        m = row.get("qwen_private_qkv") or {}
        if (m.get("failed") is not False or m.get("failure_reason") != "" or m.get("validation_passed") is not True or
                m.get("validation_calls_session_total") != 1 or type(m.get("validation_relative_l2")) not in (int, float) or
                not math.isfinite(m["validation_relative_l2"]) or not 0 <= m["validation_relative_l2"] < .02 or
                any(type(m.get(k)) is not int or m[k] != total for k in ("predictions_session_total", "completed_blocks_session_total", "weight_stages_session_total")) or
                m.get("fallback_blocks_session_total") != 0 or m.get("gpu_blocks_session_total") != gpu or
                m.get("slot_bytes") != 65814528 or m.get("estimated_bytes") != 288 << 20 or
                m["estimated_bytes"] + row["hybrid"]["runtime_weight"]["estimated_bytes"] > 2 << 30 or
                not isinstance(m.get("program_key"), str) or not m["program_key"] or (key is not None and m["program_key"] != key)):
            raise ValueError("actual healthy fixed QKV graph, numeric validation and complete successful block counts required")
        for name, previous in previous_times.items():
            value = m.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < previous:
                raise ValueError("Private QKV cumulative timing regressed/nonfinite")
            previous_times[name] = value
        key = m["program_key"]


def validate_compiled_complement(metrics,enabled):
    if type(enabled) is not bool or not metrics:raise ValueError('explicit compiled complement policy and complete results required')
    validation_seconds=None
    for row in metrics:
        m=row.get('qwen_compiled_complement')
        marker='; experimental shared compiled GPU complement with explicit runtime W' in row.get('acceleration_selection','')
        if not enabled:
            if m or marker:raise ValueError('unrequested compiled complement retained actual work/marker')
            continue
        if not m or not marker:raise ValueError('actual compiled complement receipt and marker required')
        phase=row.get('qwen_ffn_phases') or {};prefill=phase.get('prefill') or {};decode=phase.get('decode') or {}
        steps=prefill.get('steps_this_request');decode_steps=decode.get('steps_this_request')
        if (steps not in (0,1) or decode_steps!=40-steps or decode.get('actual_rows')!=1024 or
            prefill.get('completed_channel_blocks_this_request')!=32*steps or decode.get('completed_channel_blocks_this_request')!=32*decode_steps):
            raise ValueError('actual complete target/prefill channel blocks required')
        expected=32*decode_steps
        if (m.get('requested') is not True or m.get('enabled') is not True or m.get('validation_passed') is not True or
            m.get('disabled') is not False or m.get('failure_reason')!='' or
            m.get('recipe')!='gpu-compiled-complement-explicit-w-target1024-v1' or m.get('extra_owned_weight_bytes')!=0 or
            m.get('validation_calls_session_total')!=1 or m.get('trace_count_session_total')!=1):
            raise ValueError('actual source-bound shared compiled complement identity/validation/trace differs')
        for name,want in (('attempts_this_request',expected),('completed_blocks_this_request',expected),
            ('ordinary_blocks_this_request',1280-expected),('fallback_blocks_this_request',0)):
            if type(m.get(name)) is not int or m[name]!=want:raise ValueError('actual compiled/ordinary route count differs: '+name)
        seconds=m.get('validation_seconds_session_total')
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0 or (validation_seconds is not None and seconds!=validation_seconds):
            raise ValueError('actual once-only source validation time differs')
        validation_seconds=seconds


def validate_gpu_dual_swiglu(metrics,enabled):
    if type(enabled) is not bool or not metrics:raise ValueError("explicit GPU dual policy and complete results required")
    previous_seconds=0
    for row in metrics:
        phase=row.get("qwen_ffn_phases") or {};prefill=phase.get("prefill") or {};decode=phase.get("decode") or {}
        steps=prefill.get("steps_this_request");decode_steps=decode.get("steps_this_request")
        if (steps not in (0,1) or decode_steps!=40-steps or decode.get("actual_rows")!=1024 or
            prefill.get("completed_channel_blocks_this_request")!=32*steps or decode.get("completed_channel_blocks_this_request")!=32*decode_steps):
            raise ValueError("actual complete target/prefill channel blocks required")
        expected=(40 if steps==0 else 39)*32 if enabled else 0
        m=row.get("qwen_gpu_dual_swiglu") or {}
        if (m.get("requested") is not enabled or m.get("enabled") is not enabled or m.get("validation_passed") is not enabled or
            m.get("disabled") is not False or m.get("failure_reason")!="" or m.get("recipe")!="gpu-dual-swiglu-register-m32n64-target1024-v1" or
            m.get("extra_owned_weight_bytes")!=0 or m.get("validation_calls_session_total")!=(1 if enabled else 0)):
            raise ValueError("actual healthy source-bound fixed GPU dual mode/validation differs")
        for name,want in (("attempts_this_request",expected),("completed_blocks_this_request",expected),
            ("ordinary_blocks_this_request",1280-expected),("fallback_blocks_this_request",0)):
            if type(m.get(name)) is not int or m[name]!=want:raise ValueError("actual GPU dual/ordinary completed block count differs: "+name)
        seconds=m.get("validation_seconds_session_total")
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<previous_seconds or (enabled and seconds<=0) or (not enabled and seconds!=0):
            raise ValueError("actual source-bound validation span regressed/nonfinite or missing")
        previous_seconds=seconds


def validate_private_qkv_prefetch(metrics, enabled):
    """Exact successful-window balance, not a hit-rate-based speed claim."""
    if type(enabled) is not bool or not metrics:
        raise ValueError("explicit lookahead policy and complete receipts required")
    previous_qkv=expected=0
    times=dict.fromkeys(("weight_join_wait_seconds_session_total","producer_lifetime_seconds_session_total"),0)
    for row in metrics:
        current=(row.get("qwen_private_qkv") or {}).get("predictions_session_total")
        if type(current) is not int or current-previous_qkv not in (1248,1280):
            raise ValueError("complete forty-step QKV prediction coverage required")
        # Target-only step0 starts without an earlier FFN. Otherwise the last
        # prefill FFN stages step1/layer0, including GPU-step0 text-prefix hits.
        if enabled:expected+=current-previous_qkv-(current-previous_qkv==1280)
        previous_qkv=current
        m=row.get("qwen_private_qkv_prefetch") or {}
        if m.get("requested") is not enabled or m.get("enabled") is not enabled or m.get("future_pending") is not False:
            raise ValueError("actual lookahead mode or successful end-of-request drain differs")
        for name in ("attempts","submissions","claims","weight_completions","completed_hits","producer_submissions","discards","retired_claims","failures"):
            value=m.get(name+"_session_total")
            want=(3*expected if name=="producer_submissions" else 0 if name in ("discards","retired_claims","failures") else expected)
            if type(value) is not int or value!=want:
                raise ValueError("lookahead successful producer/claim/completion balance differs: "+name)
        for name,previous in times.items():
            value=m.get(name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<previous or (not enabled and name.startswith("producer_") and value!=0):
                raise ValueError("lookahead cumulative host span regressed/nonfinite or off retained producers")
            times[name]=value


def function_calls(row,rows):
    w=(row.get('hybrid') or {}).get('runtime_weight') or {};functions=w.get('function_rows',[])
    if not functions:return (rows+1023)//1024
    if functions not in ([512,1024],[512,1024,1536],[512,1024,1536,2048]) or w.get('row_alignment')!=512:
        raise ValueError('explicit healthy base function geometry required')
    left=(rows+511)//512*512;count=0
    while left:
        n=max(n for n in functions if n<=left);left-=n;count+=1
    return count


def validate_runtime_prefix(metrics, resident=False):
    previous_calls = 0
    recipe = None
    for index, row in enumerate(metrics):
        receipt = row.get("qwen_runtime_prefix_snapshot") or {}
        hit = index > 0
        if row.get("operation") != "image.edit" or row.get("reference_tokens") != 4096:
            raise ValueError("matched single-reference edits required")
        if resident:
            label = "; experimental resident prefix KV " + ("hit" if hit else "miss")
            if receipt or label not in row.get("acceleration_selection", ""):
                raise ValueError("actual retained Transformer prefix miss/hit required")
        elif (
                receipt.get("admitted") is not True or receipt.get("hit") is not hit or
                receipt.get("published") is not (not hit) or receipt.get("decline_reason") != "" or
                type(receipt.get("bytes")) is not int or not 0 < receipt["bytes"] <= 8 << 30 or
                receipt["bytes"] != (row.get("text_tokens", 0) + row["reference_tokens"]) * 524288 or
                not isinstance(receipt.get("recipe_sha256"), str) or len(receipt["recipe_sha256"]) != 64):
            raise ValueError("actual admitted runtime prefix miss/publication then hits required")
        if not resident and recipe is not None and recipe != receipt["recipe_sha256"]:
            raise ValueError("runtime prefix arithmetic recipe changed during matched repeats")
        recipe = receipt["recipe_sha256"] if not resident else None
        phases = row.get("qwen_ffn_phases") or {}
        prefill, decode = phases.get("prefill", {}), phases.get("decode", {})
        if (phases.get("policy") != "all" or prefill.get("steps_this_request") != (0 if hit else 1) or
                decode.get("steps_this_request") != (40 if hit else 39) or decode.get("actual_rows") != 1024 or
                prefill.get("completed_channel_blocks_this_request") != (0 if hit else 32) or
                decode.get("completed_channel_blocks_this_request") != (1280 if hit else 1248)):
            raise ValueError("prefix hit must execute forty real target-only steps, not stale/full-prefill work")
        calls = row["hybrid"]["runtime_calls_session_total"]
        functions=(row.get('hybrid') or {}).get('runtime_weight',{}).get('function_rows',[])
        expected=1280 if hit else 32*(39+function_calls(row,1024+row['reference_tokens']+row['text_tokens'])) if functions else 1440
        if calls - previous_calls != expected:
            raise ValueError("actual runtime predictions differ from full miss/target-only hit geometry")
        previous_calls = calls


def validate_runtime_text_prefix(metrics,*,matrix=False,qkv_step0="ane"):
    if qkv_step0 not in ("ane","gpu"):raise ValueError("explicit text-prefix first-step QKV policy required")
    previous=0;recipe=None
    for index,row in enumerate(metrics):
        hit=([False,True,False,False][index] if matrix else index>0)
        receipt=row.get("qwen_runtime_prefix_snapshot") or {};text=row.get("text_tokens")
        selection=row.get("acceleration_selection","")
        if ("; runtime text prefix preserves first-step GPU QKV" in selection)!=(qkv_step0=="gpu"):
            raise ValueError("actual native first-step QKV policy marker differs")
        if (row.get("operation")!="image.generate" or row.get("reference_tokens")!=0 or type(text) is not int or text<=0 or
                receipt.get("admitted") is not True or receipt.get("hit") is not hit or receipt.get("published") is not (not hit) or
                receipt.get("decline_reason")!="" or receipt.get("bytes")!=text*524288 or
                not 0<receipt["bytes"]<=8<<30 or not isinstance(receipt.get("recipe_sha256"),str) or len(receipt["recipe_sha256"])!=64 or
                "; runtime text prefix KV snapshot "+("hit" if hit else "miss") not in selection or "; edit prefix KV snapshot" in selection):
            raise ValueError("actual text-only prefix miss/publication, expected hit/miss pattern and bounded K/V required")
        if recipe is not None and recipe!=receipt["recipe_sha256"]:raise ValueError("runtime text prefix arithmetic recipe changed")
        recipe=receipt["recipe_sha256"]
        phase=row.get("qwen_ffn_phases") or {};prefill=phase.get("prefill",{});decode=phase.get("decode",{})
        if (phase.get("policy")!="all" or prefill.get("steps_this_request")!=(0 if hit else 1) or
                decode.get("steps_this_request")!=(40 if hit else 39) or decode.get("actual_rows")!=1024 or
                prefill.get("completed_channel_blocks_this_request")!=(0 if hit else 32) or
                decode.get("completed_channel_blocks_this_request")!=(1280 if hit else 1248) or
                (not hit and prefill.get("actual_rows")!=1024+text)):
            raise ValueError("text prefix hit must execute forty real target-only steps")
        calls=row["hybrid"]["runtime_calls_session_total"]
        if calls-previous!=(1280 if hit else 32*(39+function_calls(row,1024+text))):
            raise ValueError("actual FFN chunks disagree with text-only prefix geometry")
        previous=calls


def requests(args):
    request_args = argparse.Namespace(**vars(args))
    request_args.route = "gpu" if args.route == "gpu" else "runtime-off"
    values = base.requests(request_args)[:args.requests]
    if getattr(args,"scenario","repeat")=="text-prefix-matrix":
        if args.requests!=4 or args.mode!="timeline" or args.route!="runtime" or not args.runtime_text_prefix_snapshot:
            raise ValueError("text-prefix-matrix requires four explicit runtime text-prefix timeline requests")
        first=dict(values[0]);alternate=dict(first);alternate["seed"]=74
        changed={**first,"prompt":"A studio photograph of three objects arranged left to right: a red wooden cube, a polished steel sphere, and a transparent glass bottle. Neutral gray backdrop, soft even lighting, realistic materials."}
        values=[first,alternate,changed,dict(first)]
        for index,value in enumerate(values):value["output"]=str(args.output/f"{index}.png")
    if getattr(args, "scenario", "repeat") == "sequence-matrix":
        if args.requests != 4 or args.mode != "timeline" or args.route != "runtime":
            raise ValueError("sequence-matrix requires four timeline requests on the explicit runtime route")
        generated = base.requests(argparse.Namespace(**{**vars(request_args), "case": "generate"}))[0]
        edited = base.requests(argparse.Namespace(**{**vars(request_args), "case": "edit"}))[0]
        values = [generated, edited, {**edited, "inputs": edited["inputs"] * 2}, dict(generated)]
        # All geometry transitions run real complete requests at the SAME seed.
        # Duplicate references test longer layout, not semantic two-image quality.
        for index, value in enumerate(values):
            value["output"] = str(args.output / f"{index}.png")
    if getattr(args, "prompt", None) is not None:
        for value in values:
            value["prompt"] = args.prompt
    return values


def child(args):
    if not hasattr(args,'function_rows'):args.function_rows=None
    if not hasattr(args,'edit_prefix_qkv_step0'):args.edit_prefix_qkv_step0=None
    if not hasattr(args,'compiled_complement'):args.compiled_complement=False
    if args.function_rows is None:os.environ.pop('TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS',None)
    if args.edit_prefix_qkv_step0 is None:os.environ.pop('TURBOCIDER_QWEN21_RUNTIME_EDIT_PREFIX_QKV_STEP0',None)
    os.environ.update(environment(args.route, args.mode, args.channels, args.runtime_prefix_snapshot, args.resident_prefix, args.private_qkv, args.fixed_program, args.ffn_projection,args.ffn_activation,runtime_text_prefix=args.runtime_text_prefix_snapshot,text_prefix_qkv_step0=args.text_prefix_qkv_step0,private_qkv_prefetch=args.private_qkv_prefetch,gpu_dual_swiglu=args.gpu_dual_swiglu,function_rows=args.function_rows,edit_prefix_qkv_step0=args.edit_prefix_qkv_step0,compiled_complement=args.compiled_complement))
    observer = Observer(args.observer) if args.mode != "stock" else None
    if observer:
        observer.lib.tc_qwen_profile_scope(-1)
    lib = C.CDLL(str(args.library))
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_runtime_build_identity.restype = C.c_void_p
    lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    callback_type = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)
    lib.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, callback_type, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_free.argtypes = [C.c_void_p]

    def consume(pointer):
        if not pointer:
            return ""
        try:
            return C.string_at(pointer).decode()
        finally:
            lib.tc_string_free(pointer)

    progress, callback_errors, current = [], [], -1
    @callback_type
    def callback(raw, _context):
        if observer:
            try:
                progress.append(dict(at_ns=observer.now(), request=current, event=json.loads(raw)))
            except Exception as error:
                callback_errors.append(str(error))

    engine, error = C.c_void_p(), C.c_void_p()
    report = dict(schema="tc-qwen-base-profile-report-v4", mode=args.mode, route=args.route, channels=args.channels,
                  scenario=args.scenario, private_qkv=args.private_qkv, private_qkv_prefetch=args.private_qkv_prefetch, gpu_dual_swiglu=args.gpu_dual_swiglu, function_rows=args.function_rows, edit_prefix_qkv_step0=args.edit_prefix_qkv_step0,compiled_complement=args.compiled_complement, fixed_program=args.fixed_program, ffn_projection=args.ffn_projection,ffn_activation=args.ffn_activation,runtime_text_prefix_snapshot=args.runtime_text_prefix_snapshot,text_prefix_qkv_step0=args.text_prefix_qkv_step0,
                  loaded_runtime_id=consume(lib.tc_runtime_build_identity()), runs=[], progress=progress)
    try:
        started = time.perf_counter()
        if lib.tc_engine_create_model(b"qwen-image-2.1", str(args.model).encode(), C.byref(engine), C.byref(error)):
            raise RuntimeError(consume(error.value))
        report["constructor_seconds"] = time.perf_counter() - started
        for index, request in enumerate(requests(args)):
            current = index
            if "loras" in request:
                raise ValueError("profiling base model must not carry LoRA")
            write_json(args.output / f"request-{index}.json", request)
            if observer:
                observer.lib.tc_qwen_profile_scope(index)
                before = observer.take("tc_qwen_profile_counters")
            begin_ns = observer.now() if observer else None
            result, error = C.c_void_p(), C.c_void_p()
            start = time.perf_counter()
            code = lib.tc_engine_generate(engine, json.dumps(request).encode(), callback, None, C.byref(result), C.byref(error))
            wall = time.perf_counter() - start
            end_ns = observer.now() if observer else None
            text, failure = consume(result.value), consume(error.value)
            if code:
                raise RuntimeError(failure)
            if callback_errors:
                raise RuntimeError("invalid progress callback payload: " + callback_errors[0])
            row = dict(index=index, request_wall_including_export=wall, begin_ns=begin_ns, end_ns=end_ns, metrics=json.loads(text))
            if observer:
                if observer.lib.tc_qwen_profile_wait(2000):
                    raise RuntimeError("observer completion callbacks failed to drain")
                row.update(observer_before=before, observer_after=observer.take("tc_qwen_profile_counters"))
            report["runs"].append(row)
            write_json(args.output / "report.json", report)
            # Reject a silent all-GPU fallback immediately, before spending
            # two further complete requests under a false runtime label.
            validate([item["metrics"] for item in report["runs"]], args.route, args.mode, args.channels, args.runtime_prefix_snapshot, args.resident_prefix, args.private_qkv, args.fixed_program, args.ffn_projection,args.ffn_activation,runtime_text_prefix=args.runtime_text_prefix_snapshot,text_prefix_matrix=args.scenario=="text-prefix-matrix",text_prefix_qkv_step0=args.text_prefix_qkv_step0,private_qkv_prefetch=args.private_qkv_prefetch,gpu_dual_swiglu=args.gpu_dual_swiglu,function_rows=args.function_rows,edit_prefix_qkv_step0=args.edit_prefix_qkv_step0,compiled_complement=args.compiled_complement)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        if observer:
            observer.lib.tc_qwen_profile_scope(-3)
        if engine.value:
            lib.tc_engine_free(engine)
        if observer:
            write_json(args.output / "observer.json", observer.take("tc_qwen_profile_finish"))
        write_json(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--observer", type=Path, default=ROOT / "build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/qwen21/teapot-blue-edit-native-512.png")
    parser.add_argument("--route", choices=("gpu", "runtime", "cache"), required=True)
    parser.add_argument("--case", choices=("generate", "edit"), default="generate")
    parser.add_argument("--scenario", choices=("repeat", "sequence-matrix","text-prefix-matrix"), default="repeat")
    parser.add_argument("--runtime-prefix-snapshot", action="store_true", help="explicit base-only same-conditioning edit reuse pilot")
    parser.add_argument('--function-rows',choices=('512,1024','512,1024,1536','512,1024,1536,2048'),help='default-off fixed-procedure family; descriptor rows must equal its maximum')
    parser.add_argument('--edit-prefix-qkv-step0',choices=('ane','gpu'),help='separate default-off base40 edit prefix-hit first-step QKV route')
    parser.add_argument("--runtime-text-prefix-snapshot",action="store_true",help="explicit fixed runtime generation text-only prefix reuse")
    parser.add_argument("--text-prefix-qkv-step0",choices=("ane","gpu"),default="ane",help="preserve GPU QKV at step0 on a text-prefix hit; remaining39 stay Private")
    parser.add_argument("--resident-prefix", action="store_true", help="compare existing retained-Transformer prefix route")
    parser.add_argument("--private-qkv", action="store_true", help="explicit fixed512-row Private QKV combined with the healthy runtime FFN")
    parser.add_argument("--private-qkv-prefetch", action="store_true", help="one request-local QKV weight group staged after current FFN submissions; fixed joint runtime only")
    parser.add_argument("--gpu-dual-swiglu",action="store_true",help="explicit source-validated fixed6144 register dual-GEMM GPU complement for target1024 rows")
    parser.add_argument('--compiled-complement',action='store_true',help='explicit shared compiled GPU complement with original runtime X/W; fixed6144 joint base40 only')
    parser.add_argument("--fixed-program", action="store_true", help="forbid FFN Program replacement; sticky full-GPU fallback on failure")
    parser.add_argument("--ffn-projection",choices=FFN_PROJECTIONS,default="descriptor",help="one fixed FFN projection recipe for this entire process")
    parser.add_argument("--ffn-activation",choices=FFN_ACTIVATIONS,default="exp",help="one fixed FFN activation expression; non-default requires fixed descriptor runtime")
    parser.add_argument("--prompt", help="one fixed custom prompt for this process; defaults to the recorded teapot task")
    parser.add_argument("--mode", choices=("stock", "timeline", "gpu-ops", "runtime-layers"), default="timeline")
    parser.add_argument("--requests", type=int, choices=(1, 3, 4), default=3)
    parser.add_argument("--channels", type=int, choices=(4096, 5120, 6144), default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    environment(args.route,args.mode,args.channels,args.runtime_prefix_snapshot,args.resident_prefix,args.private_qkv,args.fixed_program,args.ffn_projection,args.ffn_activation,runtime_text_prefix=args.runtime_text_prefix_snapshot,text_prefix_qkv_step0=args.text_prefix_qkv_step0,private_qkv_prefetch=args.private_qkv_prefetch,gpu_dual_swiglu=args.gpu_dual_swiglu,function_rows=args.function_rows,edit_prefix_qkv_step0=args.edit_prefix_qkv_step0,compiled_complement=args.compiled_complement)
    if args.prompt is not None and (not args.prompt.strip() or len(args.prompt.encode()) > 32768):
        parser.error("prompt requires nonempty UTF-8 up to32768 bytes")
    if args.private_qkv and (args.route != "runtime" or args.mode not in ("stock", "timeline") or args.resident_prefix):
        parser.error("Private QKV requires cache-off stock/timeline runtime without retained Transformer")
    if args.fixed_program and (args.route != "runtime" or args.mode not in ("stock", "timeline") or args.resident_prefix):
        parser.error("fixed program requires cache-off stock/timeline runtime without retained Transformer")
    if args.ffn_projection!="descriptor" and (not args.fixed_program or args.route!="runtime" or args.mode not in ("stock","timeline") or args.resident_prefix):
        parser.error("experimental FFN projection requires fixed cache-off stock/timeline runtime")
    if (args.runtime_prefix_snapshot and args.resident_prefix) or ((args.runtime_prefix_snapshot or args.resident_prefix) and (args.route != "runtime" or args.case != "edit" or args.scenario != "repeat" or args.mode not in ("stock", "timeline"))):
        parser.error("runtime prefix requires repeat single-reference edits, cache-off runtime, stock/timeline")
    if args.runtime_text_prefix_snapshot and (args.case!="generate" or args.scenario not in ("repeat","text-prefix-matrix")):
        parser.error("text prefix requires generation repeat or its explicit prompt-change matrix")
    if args.scenario=="text-prefix-matrix" and (not args.runtime_text_prefix_snapshot or args.requests!=4 or args.mode!="timeline" or args.route!="runtime" or args.prompt is not None):
        parser.error("text-prefix-matrix requires four timeline runtime text-prefix requests and its fixed prompts")
    if args.scenario == "sequence-matrix":
        if args.requests != 4 or args.mode != "timeline" or args.route != "runtime":
            parser.error("sequence-matrix requires --requests4 --mode timeline --route runtime")
    elif args.requests == 4 and args.scenario!="text-prefix-matrix":
        parser.error("four requests are reserved for the heterogeneous sequence-matrix diagnostic")
    if args.channels != 4096 and args.route != "runtime":
        parser.error("larger fixed partitions require the explicit cache-off runtime route; GPU/cache controls retain4096")
    if (args.mode == "gpu-ops" and args.route != "gpu") or (args.mode == "runtime-layers" and args.route == "gpu"):
        parser.error("gpu-ops requires GPU; runtime-layers requires a runtime route")
    for name in ("library", "model", "manifest", "reference", "observer"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    if args.function_rows and json.loads(args.manifest.read_text())['rows']!=int(args.function_rows.split(',')[-1]):
        parser.error('descriptor physical rows must equal the function family maximum')
    if args.child:
        child(args)
        return
    manifest = json.loads((args.library.parent / "runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        for entry in manifest["inputs"]["sources"]:
            if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("linked native source changed")
    native_sources()
    hashes = {name: sha256_file(ROOT / name) for name in SOURCES}
    assets = {name: sha256_file(getattr(args, name)) for name in ("library", "manifest", "reference", "observer")}
    model_identity = model_snapshot(args.model)
    if shutil.disk_usage(ROOT).free < (2304 << 20):
        parser.error("reserve2GiB free plus256MiB output allowance before starting models")
    args.output.mkdir(parents=True, exist_ok=False)
    trial = args.output / "trial"
    trial.mkdir()
    summary = dict(schema="tc-qwen-base-profile-screen-v6", status="running", qualification_passed=False,
                   mode=args.mode, route=args.route, case=args.case, scenario=args.scenario, requests=args.requests, channels=args.channels, source_sha256=hashes, assets=assets,
                   runtime_prefix_snapshot=args.runtime_prefix_snapshot,
                   runtime_text_prefix_snapshot=args.runtime_text_prefix_snapshot,
                   text_prefix_qkv_step0=args.text_prefix_qkv_step0,
                   resident_prefix=args.resident_prefix,
                   private_qkv=args.private_qkv,
                   private_qkv_prefetch=args.private_qkv_prefetch,
                   gpu_dual_swiglu=args.gpu_dual_swiglu,
                   compiled_complement=args.compiled_complement,
                   function_rows=args.function_rows,
                   edit_prefix_qkv_step0=args.edit_prefix_qkv_step0,
                   fixed_program=args.fixed_program,
                   ffn_projection=args.ffn_projection,
                   ffn_activation=args.ffn_activation,
                   environment=environment(args.route, args.mode, args.channels, args.runtime_prefix_snapshot, args.resident_prefix, args.private_qkv, args.fixed_program, args.ffn_projection,args.ffn_activation,runtime_text_prefix=args.runtime_text_prefix_snapshot,text_prefix_qkv_step0=args.text_prefix_qkv_step0,private_qkv_prefetch=args.private_qkv_prefetch,gpu_dual_swiglu=args.gpu_dual_swiglu,function_rows=args.function_rows,edit_prefix_qkv_step0=args.edit_prefix_qkv_step0,compiled_complement=args.compiled_complement), runtime_build_id=manifest["runtime_build_id"],
                   prompt=args.prompt or base.PROMPTS[args.case],
                   model_snapshot=model_identity, model_identity_scope="regular-file generations and bounded header hashes, not full payload hashes or immutable leases")
    write_json(args.output / "source.json", {name: (ROOT / name).read_text() for name in SOURCES})
    write_json(args.output / "summary.json", summary)
    observer = LoadObservation(args.output / "load.jsonl")
    try:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--library", str(args.library), "--observer", str(args.observer),
                   "--model", str(args.model), "--manifest", str(args.manifest), "--reference", str(args.reference), "--route", args.route,
                   "--case", args.case, "--mode", args.mode, "--requests", str(args.requests), "--channels", str(args.channels), "--output", str(trial)]
        command.extend(["--scenario", args.scenario])
        if args.function_rows:command.extend(['--function-rows',args.function_rows])
        if args.edit_prefix_qkv_step0:command.extend(['--edit-prefix-qkv-step0',args.edit_prefix_qkv_step0])
        if args.runtime_prefix_snapshot:
            command.append("--runtime-prefix-snapshot")
        if args.runtime_text_prefix_snapshot:command.append("--runtime-text-prefix-snapshot")
        command.extend(["--text-prefix-qkv-step0",args.text_prefix_qkv_step0])
        if args.resident_prefix:
            command.append("--resident-prefix")
        if args.private_qkv:
            command.append("--private-qkv")
        if args.private_qkv_prefetch:
            command.append("--private-qkv-prefetch")
        if args.gpu_dual_swiglu:
            command.append("--gpu-dual-swiglu")
        if args.compiled_complement:
            command.append('--compiled-complement')
        if args.fixed_program:
            command.append("--fixed-program")
        command.extend(["--ffn-projection",args.ffn_projection])
        command.extend(["--ffn-activation",args.ffn_activation])
        if args.prompt is not None:
            command.extend(["--prompt", args.prompt])
        with (args.output / "stdout.log").open("x") as stdout, (args.output / "stderr.log").open("x") as stderr:
            summary["memory"] = run_sampled(command, repo=ROOT, output=args.output, stem="trial", env=benchmark_environment(),
                                            stdout=stdout, stderr=stderr, timeout=1800, interval_ms=100, max_gap_ms=500, observer=observer)
        report = json.loads((trial / "report.json").read_text())
        validate([row["metrics"] for row in report["runs"]], args.route, args.mode, args.channels, args.runtime_prefix_snapshot, args.resident_prefix, args.private_qkv, args.fixed_program, args.ffn_projection,args.ffn_activation,runtime_text_prefix=args.runtime_text_prefix_snapshot,text_prefix_matrix=args.scenario=="text-prefix-matrix",text_prefix_qkv_step0=args.text_prefix_qkv_step0,private_qkv_prefetch=args.private_qkv_prefetch,gpu_dual_swiglu=args.gpu_dual_swiglu,function_rows=args.function_rows,edit_prefix_qkv_step0=args.edit_prefix_qkv_step0,compiled_complement=args.compiled_complement)
        if report["loaded_runtime_id"] != manifest["runtime_build_id"]:
            raise ValueError("actual linked runtime differs")
        if args.mode != "stock" and json.loads((trial / "observer.json").read_text())["source_sha256"] != hashes["tools/profiling/qwen_base/observer.mm"]:
            raise ValueError("actual observer binary differs from bound source")
        import analyze
        summary["analysis"] = analyze.analyze_directory(args.output)
        summary['png_sha256']={p.name:sha256_file(p) for p in sorted(trial.glob('*.png'))}
        if len(summary['png_sha256'])!=args.requests:raise ValueError('all complete original PNG files required')
        native_sources()
        summary["status"] = "complete"
    except Exception as error:
        summary.update(status="failed", error=str(error))
    try:
        summary["load"] = observer.verify()
    except ValueError as error:
        summary["load_error"] = str(error)
    if (model_snapshot(args.model) != model_identity or any(sha256_file(ROOT / name) != digest for name, digest in hashes.items())
            or any(sha256_file(getattr(args, name)) != digest for name, digest in assets.items())):
        summary.update(status="failed", error="bound source or asset changed")
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
