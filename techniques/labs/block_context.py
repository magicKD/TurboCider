"""Separate actual block execution, numerical gates, export and promotion axes.

Pure recorded-data checks. No runtime/model/candidate imports, hardware replay,
or qualified speed ratio after incomplete artifact delivery.
"""
import math
import statistics

from common import report

ARMS=("gpu-original","gpu-staged","copy-hybrid","shared-serial","shared-hybrid")
IDS={"request_input_matches","request_output_matches","metal_input_matches","metal_output_matches","input_output_distinct"}
RAW_REFUSAL="post-execution artifact reserve insufficient; no overwrite"
ARCHIVE_REFUSAL="lossless archive actual-size reserve insufficient; no data overwritten"


def require(value,message):
    if not value:raise ValueError(message)


def metric(value,limit):
    require(value.get("finite") is True,"finite observed block/component required")
    for key in ("relative_l2","cosine","max_abs"):
        require(type(value.get(key)) in (int,float) and math.isfinite(value[key]),"typed finite observed metric required")
    require(0<=value["relative_l2"]<=limit and .999<=value["cosine"]<=1+1e-10 and value["max_abs"]>=0,"fixed block/component gate rejected")


def identity(value):
    require(set(value)==IDS and all(x is True for x in value.values()),"actual block branch backing proof incomplete")


def verify(document):
    require(document.get("schema")=="tc-qwen-ordinary-block-wrapper-v1" and document.get("status")=="native-failed"
            and document.get("device_requested") is True and type(document.get("native_returncode")) is int and document["native_returncode"]==3
            and document.get("errors")==[] and document.get("comparisons") is None,"retain failed protocol, not a relabelled measured/promoted run")
    require(type(document["seed"]) is int and document["seed"]!=document["baseline_seed"]
            and document["prompt_sha256"]!=document["baseline_prompt_sha256"],"different actual requested seed AND prompt required")
    for key in ("source_inputs_unchanged","native_library_unchanged","mlx_assets_unchanged","original_model_stats_unchanged","private_bank_unchanged"):
        require(document.get(key) is True,"source/library/bank generation changed")
    native=document["native"]
    require(native.get("schema")=="tc-qwen-ordinary-block-shared-v1" and native.get("status")=="failed"
            and native.get("case_id")==document["case_id"] and native.get("seed")==document["seed"]
            and native.get("prompt_sha256")==document["prompt_sha256"] and native.get("mlx_version")=="0.32.0",
            "actual native case/version/request binding differs")
    for value in (document,native):
        require(value.get("production_qualified") is False and value.get("physical_overlap_qualified") is False
                and value.get("runtime_internal_copy_behavior")=="unknown","block replay does not grant product or physical qualification")
    require(native.get("shape_mhf")==[1024,4096,12288] and native.get("share")==2048 and type(native.get("source_step")) is int and native["source_step"]==1
            and type(native.get("source_layer")) is int and native["source_layer"]==0 and native.get("block_gate")=={"relative_l2_max":.001,"cosine_min":.999}
            and native.get("ffn_gate")=={"relative_l2_max":.03,"cosine_min":.999},"fixed ordinary steady block and frozen FFN budgets required")
    metric(native["staged_gpu_replay_gate"],.001)
    require(native["staged_gpu_replay_gate"]["max_abs"]==0,"staged source replay must be exactly observed here")
    metric(native["independent_full_reference_gate"],.03)
    controls=native["runtime_argument_controls"]
    for key in ("zero_ffn_yields_original_residual","zero_private_yields_original_tail"):
        metric(controls[key],.001);require(controls[key]["max_abs"]==0,"runtime argument zero control did not bind actual input")
    require(type(controls["nonzero_full_ffn_changes_block_elements"]) is int and 0<controls["nonzero_full_ffn_changes_block_elements"]<=4194304
            and controls["used_for_speed_ratio"] is False,"whole FFN cannot be omitted or counted as a performance candidate")
    require(len(native["ordinary_argument_sha256"])==9,"actual original input/modulation/RoPE/KV identities required")
    errors=native["errors"]
    if errors==["real prefix MIL load failed; no fallback"]:
        require(native.get("samples",[])==[] and native.get("compile_selector_called") is False and native.get("compile_wall_ns") is None
                and native.get("mapping_quarantined_until_child_exit") is True,"load rejection cannot manufacture execution or zero compile cost")
        return {"execution_axis":"original request reached observer; candidate load failed","numeric_axis":"staged replay/reference/runtime-argument prerequisites passed",
                "candidate_timed_blocks":0,"artifact_axis":"not reached","qualified_performance_ratio_available":False,
                "cause_qualified":False,"hardware_unsupported":False,"physical_overlap_qualified":False}
    require(errors in ([RAW_REFUSAL],[ARCHIVE_REFUSAL]),"specific post-execution artifact refusal required")
    require(native.get("private_unloaded") is True and native.get("restore_adoption_releases")==2 and type(native["restore_adoption_releases"]) is int
            and native.get("owned_mapping_release_pass") is True and native.get("borrowed_inputs_outputs_unchanged") is True
            and native.get("loaded_runtime_id")==document["runtime_build_id"],"completed original source and normal branch lifecycle required")
    require(native.get("completed_full_dit_steps")==1 and native.get("stopped_step")==1 and native.get("stopped_layer")==0
            and native.get("vae_decode") is False and native.get("image_export") is False,"complete block replay is not a generated request/image")
    repeats=native["measured_groups"];require(type(repeats) is int and repeats in (10,30),"declared bounded balanced groups required")
    order=[]
    for group in range(repeats+3):
        phase="first" if group==0 else "warmup" if group<3 else "measured";trial=0 if group==0 else group-1 if group<3 else group-3
        row=group if group<3 else trial
        for p in range(5):order.append((phase,trial,ARMS[(row%5+(4-p if row//5%2 else p))%5]))
    samples=native["samples"]
    require([(s["phase"],s["trial"],s["arm"]) for s in samples]==order,"all ordinary/staged/copy/shared calls and balanced order required")
    joins=0;intersections=[]
    for sample in samples:
        require(type(sample["trial"]) is int and type(sample["wall_ns"]) is int and sample["wall_ns"]>0 and sample["output_dtype"]=="bfloat16"
                and sample["numeric_pass"] is True and sample["guards_pass"] is True and sample["physical_overlap_qualified"] is False,"complete typed output/gate record required")
        arm=sample["arm"];original=arm=="gpu-original";staged=arm=="gpu-staged";copy=arm=="copy-hybrid";private=not original and not staged
        expected={"whole_ordinary_block"}|(set() if original else {"full_ffn","independent_full_ffn"})|( {"prefix","independent_prefix"} if private else set())
        require(set(sample["metrics"])==expected,"whole block AND complete FFN/prefix/FP64 gates required")
        for key,value in sample["metrics"].items():metric(value,.001 if key=="whole_ordinary_block" else .03)
        if original:require(sample["metrics"]["whole_ordinary_block"]["max_abs"]==0 and sample["metrics"]["whole_ordinary_block"]["relative_l2"]==0
                            and sample["host_events"]==[],"borrowed original callable must match actual original output exactly")
        require(type(sample["external_feature_copy_count"]) is int and sample["external_feature_copy_count"]==(2 if copy else 0)
                and type(sample["external_feature_copy_bytes"]) is int and sample["external_feature_copy_bytes"]==(16777216 if copy else 0),"actual same-client feature-copy accounting differs")
        if original:continue
        names={"attention-norm-producer-ready","modulation-residual-output-ready"}|({"full-gpu-ffn-ready"} if staged else
            {"gpu-pack-ready","gpu-remainder-ready","private-evaluate-returned","gpu-ffn-restore-join-ready"})
        if copy:names|={"cpu-input-feature-copy","cpu-output-feature-copy"}
        require(len(sample["host_events"])==len(names) and {e["name"] for e in sample["host_events"]}==names,"complete producer and downstream consumer boundary required")
        table={};last=-1
        for event in sample["host_events"]:
            start,duration=event["start_ns"],event["duration_ns"]
            require(type(start) is int and type(duration) is int and 0<=start and start>=last and duration>0
                    and start+duration<=sample["wall_ns"] and event["clock_domain"]=="same-process-uptime-raw-host","associated host-only interval required")
            table[event["name"]]=(start,start+duration);last=start
        producer,finish=(table[k] for k in ("attention-norm-producer-ready","modulation-residual-output-ready"))
        if staged:
            require(producer[1]<=table["full-gpu-ffn-ready"][0] and table["full-gpu-ffn-ready"][1]<=finish[0],"staged GPU dependency violated")
            continue
        pack,ane,gpu,join=(table[k] for k in ("gpu-pack-ready","private-evaluate-returned","gpu-remainder-ready","gpu-ffn-restore-join-ready"))
        supplied=table["cpu-input-feature-copy"][1] if copy else pack[1]
        require(producer[1]<=pack[0] and supplied<=min(ane[0],gpu[0]) and max(ane[1],gpu[1])<=join[0] and join[1]<=finish[0]
                and (arm!="shared-serial" or ane[1]<=gpu[0]),"actual front/serial/private/join/downstream dependency violated")
        if copy:require(pack[1]<=table["cpu-input-feature-copy"][0] and ane[1]<=table["cpu-output-feature-copy"][0] and table["cpu-output-feature-copy"][1]<=join[0],"feature copy cannot bypass the producer")
        identity(sample["identity_before"]);identity(sample["identity_after"])
        exact=sample["exact_runtime_join"]
        require(type(exact["elements"]) is int and exact["elements"]==4194304 and type(exact["exact_mismatches"]) is int and exact["exact_mismatches"]==0
                and exact["actual_bf16_sha256"]==exact["expected_bf16_sha256"] and len(exact["actual_bf16_sha256"])==64,"full actual-component exact FFN join required")
        joins+=1;intersections.append({"arm":arm,"phase":sample["phase"],"trial":sample["trial"],"api_host_intersection_ns":max(0,min(ane[1],gpu[1])-max(ane[0],gpu[0])),"physical_overlap_qualified":False})
    return {"execution_axis":"all original/staged/private block calls completed before export refusal","numeric_axis":"all fixed whole-block/FFN/prefix/FP64/join gates passed",
            "candidate_timed_blocks":len(samples),"exact_component_joins":joins,"artifact_axis":"post-execution refused; no raw source persisted by this run",
            "qualified_performance_ratio_available":False,"diagnostic_only_median_host_wall_ns":{arm:statistics.median(s["wall_ns"] for s in samples if s["phase"]=="measured" and s["arm"]==arm) for arm in ARMS},
            "host_intersections":intersections,"original_full_request_candidate_executed":False,"physical_overlap_qualified":False}


def status_contract(seed=17):
    return report("block-evidence-axes",{"execution_complete_is_not_artifact_complete":True,"nil_error_is_not_load_success":True,
        "stored_same_input_sha_is_not_full_raw_replay":True,"qualified_performance_ratio_available":False,"actual_device_executed":False,"seed":seed},
        scope="CPU evidence-axis contracts; no model/device execution or performance promotion")
