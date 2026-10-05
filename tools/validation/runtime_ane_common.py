"""Shared host-only contracts for GPU/ANE validation tools.

No model loading, inference or filesystem writes occur on import. Keep this
module independent of runner entry points and optional MLX/Core ML packages.
"""
import argparse
import hashlib
import json
import json
import math
import os
import subprocess
import time
from runtime_ane_calibration import validate_channel_calibration


def sha256_file(path):
    """Hash evidence in bounded memory; missing/unreadable files fail closed.

    This is a byte identity check, not an immutable artifact lease. Keep
    pre/post validation at the call sites that require mutation detection.
    """
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def session_counter(data, name):
    """JSON booleans/floats and missing counters are not successful telemetry."""
    value = data.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid or missing {name} telemetry")
    return value


def validate_edit_results(rows, edit):
    """Shared edit receipt contract for timing screens and graph-switch tests.

    Pass an empty mapping for generation. Workload/route eligibility belongs
    to each caller; a valid receipt does not establish visual quality.
    """
    if not edit:
        return None
    counts = []
    for row in rows:
        plan = row.get("plan") or {}
        if (row.get("operation") != "image.edit" or plan.get("operation") != "image.edit" or
                type(plan.get("qwen21_reference_size")) is not int or
                plan.get("qwen21_reference_size") != edit["qwen21_reference_size"]):
            raise ValueError("editing operation/reference size was not reported")
        counts.append(session_counter(row, "reference_tokens"))
    if not counts or counts[0] == 0 or len(set(counts)) != 1:
        raise ValueError("missing or inconsistent reference tokens in editing results")
    return counts[0]


def chunk_policy(value):
    """Match native scheduling options before creating any benchmark output."""
    if value == "auto":
        return value
    if not value.isascii() or not value.isdigit() or len(value) > 3 or int(value) > 128:
        raise argparse.ArgumentTypeError("chunks must be auto or an integer 0...128")
    return str(int(value))


def validate_results(rows, route, expected_count, model_id="z-image-turbo", expect_lora=False,
                     runtime_backend="public", expect_device_io=False, expected_data_path=None, channel_auto=False):
    """Do not report a failed/degraded route as a successful acceleration run.

    Raw JSONL/PNG evidence is already saved by the caller. Adaptive GPU probes
    and the explicit chunks=0 ablation are valid, unlike an error fallback.
    Backend labels alone do not establish that Core ML predictions succeeded.
    """
    if len(rows) != expected_count:
        raise ValueError("wrong result count")
    if runtime_backend not in ("public", "private", "auto"):
        raise ValueError("unknown runtime executor backend")
    if expected_data_path not in (None, "fp16", "w8a8_hadamard"):
        raise ValueError("unknown runtime data path")
    base_backend = {"z-image-turbo": "mlx_cpp_metal",
                    "qwen-image-2.1": "mlx_cpp_metal",
                    "z-image-turbo-gguf": "mlx_cpp_metal_gguf"}[model_id]
    backend = base_backend + {"gpu": "", "frozen": "+coreml",
                              "runtime": "+coreml_runtime_weight",
                              "qkv": "+coreml_runtime_qkv"}[route]
    private_backend = base_backend + "+private_ane_runtime_weight_experimental"
    allowed_backends = {backend}
    if route == "runtime" and runtime_backend != "public":
        allowed_backends = {private_backend} if runtime_backend == "private" else {backend, private_backend}
    if channel_auto:
        if route != "runtime" or runtime_backend == "public":
            raise ValueError("native automatic channels require an authorized runtime route")
        allowed_backends.add(base_backend)
    session_backend = None
    previous_calls = 0
    previous_qkv_gpu = 0
    previous_full_probes = 0
    previous_untimed_hybrid = 0
    previous_async_hybrid = 0
    previous_async_wait = 0.0
    async_receipt = None
    previous_serial = (0.0, 0.0)
    previous_device_calls = 0
    previous_ready = (0.0, 0.0)
    previous_a8 = (0, 0.0)
    a8_policy = None
    stage_policy = None
    previous_stage_variants = 0
    previous_lora_channels = (0, 0)
    previous_deferred_join = 0
    deferred_join_policy = None
    lora_channel_receipt_seen = False
    calibration_signature = None
    previous_declined_gpu = 0
    for row in rows:
        actual_backend = row.get("runtime_backend")
        if actual_backend not in allowed_backends:
            raise ValueError(f"requested {route} backend was not reported")
        if session_backend is not None and actual_backend != session_backend:
            raise ValueError("runtime backend changed within resident session")
        session_backend = actual_backend
        if expect_lora and (row.get("lora_strategy") != "inference_time" or
                            session_counter(row, "lora_applied_projections") == 0):
            raise ValueError("runtime LoRA was not bound without merging")
        timing = row.get("timings_seconds") or {}
        for name in ("request_wall", "denoise"):
            value = timing.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError(f"invalid {name} timing")
        if route == "gpu":
            continue
        if route == "qkv":
            if model_id != "qwen-image-2.1" or expect_lora or row.get("hybrid"):
                raise ValueError("QKV benchmark must have GPU FFN and base Qwen weights")
            receipt = row.get("qkv") or {}
            if (receipt.get("mode") != "runtime_weight_qkv" or
                    receipt.get("failed") is not False or
                    receipt.get("failure_reason") != "" or
                    session_counter(receipt, "failures_session_total") != 0 or
                    session_counter(receipt, "fallback_blocks_session_total") != 0):
                raise ValueError("QKV runtime failed, fell back or lacks separate telemetry")
            calls = session_counter(receipt, "calls_session_total")
            blocks = session_counter(receipt, "hybrid_blocks_session_total")
            gpu = (session_counter(receipt, "gpu_blocks_session_total")
                   if receipt.get("auto_scheduling") is True or "gpu_blocks_session_total" in receipt
                   else 0)
            idle_auto = (receipt.get("auto_scheduling") is True and
                         calls == previous_calls and gpu > previous_qkv_gpu and
                         previous_calls > 0)
            if ((calls <= previous_calls and not idle_auto) or gpu < previous_qkv_gpu or
                    blocks <= 0 or blocks > calls or
                    receipt.get("observed_ane_residency") != "unknown"):
                raise ValueError("QKV prediction/ownership receipt is incomplete")
            if "gpu_probe_blocks_session_total" in receipt:
                probes = session_counter(receipt, "gpu_probe_blocks_session_total")
                measured = session_counter(receipt, "measured_hybrid_blocks_session_total")
                gpu = session_counter(receipt, "gpu_blocks_session_total")
                probe_seconds = receipt.get("gpu_probe_seconds_session_total")
                hybrid_seconds = receipt.get("measured_hybrid_seconds_session_total")
                if (probes > gpu or measured > blocks or
                        any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                            not math.isfinite(value) or value < 0
                            for value in (probe_seconds, hybrid_seconds)) or
                        (probes == 0) != (probe_seconds == 0) or
                        (measured == 0) != (hybrid_seconds == 0)):
                    raise ValueError("QKV full-block probe telemetry is inconsistent")
            previous_calls, previous_qkv_gpu = calls, gpu
            continue
        hybrid = row.get("hybrid") or {}
        if channel_auto:
            runtime = hybrid.get("runtime_weight") or {}
            report = runtime.get("channel_calibration")
            full_width, hidden = (12288,4096) if model_id == "qwen-image-2.1" else (10240,3840)
            selected = validate_channel_calibration(report, full_width, hidden)
            signature = json.dumps({k:v for k,v in report.items() if k != "cache_hit"}, sort_keys=True, allow_nan=False)
            if calibration_signature is not None and signature != calibration_signature:
                raise ValueError("native automatic calibration evidence changed within resident session")
            calibration_signature = signature
            if selected == 0:
                gpu_blocks = session_counter(runtime,"gpu_blocks_session_total")
                if (actual_backend != base_backend or hybrid.get("runtime_failed") is not False or
                        session_counter(hybrid,"runtime_failures_session_total") or
                        session_counter(hybrid,"runtime_calls_session_total") or
                        session_counter(runtime,"fallback_blocks_session_total") or
                        session_counter(runtime,"overflow_retries_session_total") or
                        session_counter(runtime,"device_io_calls_session_total") or
                        gpu_blocks <= previous_declined_gpu or runtime.get("executor_backend") not in (None, "")):
                    raise ValueError("declined native calibration was not a clean whole-GPU route")
                previous_declined_gpu = gpu_blocks
                continue
            if (actual_backend != private_backend or runtime.get("partition_axis") != "intermediate_channels" or
                    runtime.get("ane_channels") != selected or runtime.get("gpu_channels") != full_width-selected):
                raise ValueError("native automatic calibration and adopted runtime geometry disagree")
        if expect_lora and hybrid.get("mlp_output_kind") != (
                "runtime_weight_swiglu_lora_inputs" if route == "runtime" else "fused_lora"):
            raise ValueError("LoRA benchmark requires a complete activation-correction graph")
        if (hybrid.get("runtime_failed") is not False or
                session_counter(hybrid, "runtime_failures_session_total") != 0):
            raise ValueError(f"{route} failed or lacks failure telemetry; inspect saved results")
        calls = session_counter(hybrid, "runtime_calls_session_total")
        if calls < previous_calls:
            raise ValueError("session prediction counter decreased; inspect saved results")
        previous_calls = calls
        if route == "runtime":
            runtime = hybrid.get("runtime_weight") or {}
            defer_keys=("deferred_channel_join_enabled","deferred_channel_join_blocks_session_total","post_join_scope")
            has_defer=any(name in runtime for name in defer_keys)
            if deferred_join_policy is not None and not has_defer:
                raise ValueError("deferred join receipt disappeared")
            if has_defer:
                enabled=runtime.get("deferred_channel_join_enabled")
                count=session_counter(runtime,"deferred_channel_join_blocks_session_total")
                channels=session_counter(runtime,"channel_blocks_session_total")
                asynchronous=session_counter(runtime,"async_hybrid_blocks_session_total")
                expected_scope=("evaluated_join_host_span" if count==0 else
                    "host_graph_construction_deferred_gpu_consumption" if count==channels else
                    "mixed_evaluated_and_deferred_join_spans")
                if (type(enabled) is not bool or count<previous_deferred_join or
                        count>channels or count>asynchronous or (not enabled and count) or
                        (deferred_join_policy is not None and enabled is not deferred_join_policy) or
                        runtime.get("post_join_scope")!=expected_scope):
                    raise ValueError("invalid deferred join policy/count/timing scope")
                previous_deferred_join,deferred_join_policy=count,enabled
            lora_keys=("lora_channel_range_calls_session_total","lora_channel_full_calls_session_total")
            has_lora_channels=any(name in runtime for name in lora_keys)
            if lora_channel_receipt_seen and not has_lora_channels:
                raise ValueError("LoRA channel correction receipt disappeared within session")
            if has_lora_channels:
                counts=tuple(session_counter(runtime,name) for name in lora_keys)
                if any(value<prior for value,prior in zip(counts,previous_lora_channels)):
                    raise ValueError("LoRA channel correction counters reset within session")
                previous_lora_channels,lora_channel_receipt_seen=counts,True
            has_stage = any(n in runtime for n in ("stage_specialized","stage_pipeline_variants"))
            if stage_policy is not None and not has_stage:
                raise ValueError("stage specialization receipt disappeared within session")
            if has_stage:
                specialized=runtime.get("stage_specialized")
                variants=session_counter(runtime,"stage_pipeline_variants")
                if (type(specialized) is not bool or variants>(18 if specialized else 1) or
                        variants<previous_stage_variants or (stage_policy is not None and specialized is not stage_policy)):
                    raise ValueError("invalid bounded stage specialization receipt")
                stage_policy,previous_stage_variants = specialized,variants
            has_a8 = any(name in runtime for name in ("a8_lookahead_enabled", "a8_prefetches_session_total", "a8_wait_seconds_session_total"))
            if a8_policy is not None and not has_a8:
                raise ValueError("bounded A8 lookahead receipt disappeared within session")
            if has_a8:
                enabled = runtime.get("a8_lookahead_enabled")
                prefetches = session_counter(runtime, "a8_prefetches_session_total")
                waiting = runtime.get("a8_wait_seconds_session_total")
                if (type(enabled) is not bool or prefetches > calls or (not enabled and prefetches) or
                        isinstance(waiting, bool) or not isinstance(waiting, (int, float)) or
                        not math.isfinite(waiting) or waiting < previous_a8[1] or prefetches < previous_a8[0] or
                        (a8_policy is not None and a8_policy is not enabled)):
                    raise ValueError("invalid bounded A8 lookahead receipt")
                previous_a8, a8_policy = (prefetches, waiting), enabled
            if "scale_cache_enabled" in runtime:
                enabled=runtime.get("scale_cache_enabled")
                counts=tuple(session_counter(runtime,n) for n in ("scale_cache_hits_session_total","scale_cache_misses_session_total",
                    "scale_cache_entries","scale_cache_bytes","scale_cache_evictions_session_total"))
                if(type(enabled) is not bool or counts[2]>128 or counts[3]>(4<<20) or
                   (not enabled and any(counts)) or (counts[2]==0)!=(counts[3]==0)):
                    raise ValueError("invalid bounded weight scale-cache receipt")
            if "prefetch_enabled" in runtime:
                enabled=runtime.get("prefetch_enabled")
                names=("prefetch_submissions_session_total","prefetch_hits_session_total",
                       "prefetch_discards_session_total","prefetch_failures_session_total")
                counts=tuple(session_counter(runtime,name) for name in names)
                waiting=runtime.get("prefetch_wait_seconds_session_total")
                if (type(enabled) is not bool or counts[1]+counts[2]>counts[0] or
                    isinstance(waiting,bool) or not isinstance(waiting,(int,float)) or not math.isfinite(waiting) or waiting<0 or
                    (not enabled and (any(counts) or waiting))):
                    raise ValueError("invalid future-bank prefetch receipt")
            if expect_device_io:
                device_calls = session_counter(runtime, "device_io_calls_session_total")
                if (runtime.get("io_path") != "gpu_iosurface" or runtime.get("executor_backend") != "private_ane" or
                        not previous_device_calls <= device_calls == calls):
                    raise ValueError("private GPU I/O was not reported consistently for every prediction")
                previous_device_calls = device_calls
            if expected_data_path is not None and runtime.get("data_path") != expected_data_path:
                raise ValueError("requested runtime data path was not reported")
            if runtime_backend != "public" or "executor_backend" in runtime:
                expected_executor = "private_ane" if actual_backend == private_backend else "public_coreml"
                if runtime.get("executor_backend") != expected_executor:
                    raise ValueError("executor telemetry disagrees with runtime backend")
            if "lora_input_ready_seconds_session_total" in runtime:
                ready = (runtime.get("lora_input_ready_seconds_session_total"),
                         runtime.get("pre_ffn_seconds_session_total"))
                if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                        not math.isfinite(value) or value < 0 for value in ready) or
                        any(value < prior for value, prior in zip(ready, previous_ready)) or
                        ready[0] > ready[1] + 1e-9):
                    raise ValueError("invalid combined LoRA input readiness telemetry")
                previous_ready = ready
            serial_names = ("lora_gate_up_seconds_session_total", "post_join_seconds_session_total")
            if any(name in runtime for name in serial_names):
                serial = tuple(runtime.get(name) for name in serial_names)
                total = runtime.get("hybrid_ffn_seconds_session_total")
                if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                        not math.isfinite(value) or value < 0 for value in (*serial, total)) or
                        any(value < prior for value, prior in zip(serial, previous_serial)) or
                        sum(serial) > total + 1e-9):
                    raise ValueError("invalid serialized FFN timing telemetry")
                previous_serial = serial
            if "untimed_hybrid_blocks_session_total" in runtime:
                untimed = session_counter(runtime, "untimed_hybrid_blocks_session_total")
                total = session_counter(runtime, "hybrid_blocks_session_total")
                if not previous_untimed_hybrid <= untimed <= total:
                    raise ValueError("invalid untimed hybrid telemetry")
                previous_untimed_hybrid = untimed
            # New receipts distinguish a wait that OVERLAPS GPU work from
            # the old exposed join after GPU completion. Old builds need not
            # provide these fields; partial/new invalid receipts must fail.
            has_async_receipt = ("async_hybrid_blocks_session_total" in runtime or
                                 "async_ane_wait_seconds_session_total" in runtime)
            if async_receipt is not None and has_async_receipt != async_receipt:
                raise ValueError("invalid async hybrid telemetry availability changed within session")
            async_receipt = has_async_receipt
            if has_async_receipt:
                asynchronous = session_counter(runtime, "async_hybrid_blocks_session_total")
                untimed = session_counter(runtime, "untimed_hybrid_blocks_session_total")
                waiting = runtime.get("async_ane_wait_seconds_session_total")
                wall = runtime.get("hybrid_ffn_seconds_session_total")
                if (not previous_async_hybrid <= asynchronous <= untimed or
                        any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                            not math.isfinite(value) or value < 0 for value in (waiting, wall)) or
                        waiting < previous_async_wait or waiting > wall + 1e-9 or
                        (asynchronous == 0) != (waiting == 0)):
                    raise ValueError("invalid async hybrid wait telemetry")
                previous_async_hybrid, previous_async_wait = asynchronous, waiting
            if (session_counter(runtime, "fallback_blocks_session_total") != 0 or
                    runtime.get("failure_reason") != ""):
                raise ValueError("runtime fallback or missing telemetry; inspect saved results")
            # Optional on old receipts, complete and consistent on builds
            # that report full-block GPU probes. They are a measured subset
            # of unsplit GPU blocks, never additional model executions.
            if ("full_gpu_probe_blocks_session_total" in runtime or
                    "full_gpu_probe_seconds_session_total" in runtime):
                probes = session_counter(runtime, "full_gpu_probe_blocks_session_total")
                unsplit = session_counter(runtime, "unsplit_gpu_blocks_session_total")
                gpu = session_counter(runtime, "gpu_blocks_session_total")
                elapsed = runtime.get("full_gpu_probe_seconds_session_total")
                if (not previous_full_probes <= probes <= unsplit <= gpu or
                        isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or
                        not math.isfinite(elapsed) or elapsed < 0 or
                        (probes == 0) != (elapsed == 0)):
                    raise ValueError("invalid full GPU probe telemetry")
                previous_full_probes = probes
        elif calls <= 0:
            raise ValueError("frozen route did not execute Core ML predictions")


def check_load():
    """Heuristic only: Python's executable name does not identify its workload.

    Inspect arguments for matching, but retain only executable names in the
    evidence snapshot so prompts, tokens and other arguments are not recorded.
    """
    snapshot = subprocess.check_output(["ps", "-Ao", "pid,pcpu,comm"], text=True)
    arguments = subprocess.check_output(["ps", "-Ao", "pid=,args="], text=True)
    commands = {}
    for line in arguments.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2:
            commands[fields[0]] = fields[1]
    busy = []
    for line in snapshot.splitlines()[1:]:
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, cpu, command = fields
        if int(pid) == os.getpid():
            continue
        workload = (command + " " + commands.get(pid, "")).lower()
        if float(cpu) > 5 and any(token in workload for token in
                                  ("comfyui", "turbocider", "ane-runtime-probe", "llama-server", "mlx_lm",
                                   "download_ltx25.py")):
            busy.append(line.strip())
    return snapshot, busy


def wait_for_idle():
    """Shared bounded preflight; never terminate another user's workload."""
    for attempt in range(13):
        snapshot, busy = check_load()
        if not busy:
            return snapshot
        print(json.dumps({"waiting_for_inference": busy, "attempt": attempt}), flush=True)
        if attempt == 12:
            raise RuntimeError("other inference stayed busy; no benchmark started")
        time.sleep(10)


def qwen_qk_environment(model_id, size, enabled, *, base_generation=False):
    """Explicit kernel selection shared by screen and graph-switch tools."""
    if not enabled:
        return {}
    if model_id != "qwen-image-2.1" or not (size == 512 or (size == 1024 and base_generation is True)):
        raise ValueError("Q/K norm-RoPE requires Qwen 512px output or explicit 1024px base generation")
    return {"TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE": "1"}


def validate_qwen_qk_receipts(rows, enabled):
    """Require agreement between the planned and actual optional GPU kernel.

    Older disabled receipts may omit labels; enabled receipts must include
    both. This verifies selection, not numerical or visual qualification.
    """
    if not rows:
        raise ValueError("missing Q/K norm-RoPE receipts")
    for row in rows:
        selection = row.get("acceleration_selection", "")
        plan = row.get("plan", {})
        labels = plan.get("algorithm_approximations", []) if isinstance(plan, dict) else None
        if (not isinstance(selection, str) or not isinstance(labels, list) or
                ("experimental fused Metal Q/K norm-RoPE" in selection) != enabled or
                ("qwen21_metal_qk_norm_rope" in labels) != enabled):
            raise ValueError("Q/K norm-RoPE selection does not match the requested experiment")


def qwen_lora_1024_environment(model_id, size, steps, enabled, *, has_lora,
                               references=False, fp16=False, routes=("gpu", "runtime")):
    """One opt-in on EVERY route; retain original rank precision and workload."""
    if not enabled:
        return {}
    if (model_id != "qwen-image-2.1" or size != 1024 or steps != 6 or
            not has_lora or references or fp16 or
            not routes or any(route not in ("gpu", "runtime") for route in routes)):
        raise ValueError("1024 LoRA diagnostic requires Qwen 1024px six-step GPU/runtime generation with an adapter and FP32 rank")
    return {"TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC": "1"}


def validate_qwen_lora_1024_receipts(rows, enabled):
    """An env/self-test is not an actual model route or precision receipt."""
    if not rows:
        raise ValueError("missing 1024 LoRA receipts")
    marker = "experimental 1024px six-step runtime LoRA generation, FP32 rank"
    label = "qwen21_lora_1024_generation_fp32_diagnostic"
    for row in rows:
        selection = row.get("acceleration_selection", "")
        plan = row.get("plan") or {}
        labels = plan.get("algorithm_approximations", []) if isinstance(plan, dict) else None
        if (not isinstance(selection, str) or not isinstance(labels, list) or
                (marker in selection) != enabled or (label in labels) != enabled):
            raise ValueError("1024 LoRA selection/plan does not match the requested diagnostic")
        if enabled and (any(type(row.get(key)) is not int or row.get(key) != expected
                               for key,expected in (("width",1024),("height",1024),("actual_denoise_steps",6))) or
                        row.get("lora_strategy") != "inference_time" or
                        "experimental FP16 low-rank LoRA matmuls" in selection or
                        "qwen21_viggle_lora_fp16_matmuls" in labels):
            raise ValueError("1024 LoRA workload or FP32 rank receipt is inconsistent")


def validate_fixed_async(rows, enabled):
    """Require actual successful model-block receipts for this ablation.

    These host scheduling counters do not prove physical device overlap.
    A self-test, omitted executor or timed fallback cannot stand in for a
    fixed async head actually consumed by the model.
    """
    if not rows:
        raise ValueError("missing fixed async results")
    for row in rows:
        runtime=(row.get("hybrid") or {}).get("runtime_weight") or {}
        blocks,untimed,asynchronous=(session_counter(runtime,name) for name in
            ("hybrid_blocks_session_total","untimed_hybrid_blocks_session_total","async_hybrid_blocks_session_total"))
        if blocks<=0 or (enabled and (untimed!=blocks or asynchronous!=blocks)) or (
                not enabled and (untimed!=0 or asynchronous!=0)):
            raise ValueError("requested fixed async head policy was not executed")


def validate_deferred_channel_join(rows, enabled):
    """Actual owned channel joins, not flag intent or a component self-test."""
    if not rows:
        raise ValueError("missing deferred channel join results")
    for row in rows:
        runtime=(row.get("hybrid") or {}).get("runtime_weight") or {}
        blocks=session_counter(runtime,"channel_blocks_session_total")
        asynchronous=session_counter(runtime,"async_hybrid_blocks_session_total")
        deferred=session_counter(runtime,"deferred_channel_join_blocks_session_total")
        if (runtime.get("executor_backend")!="private_ane" or
                runtime.get("partition_axis")!="intermediate_channels" or blocks<=0 or
                runtime.get("deferred_channel_join_enabled") is not enabled or
                (enabled and asynchronous!=blocks) or deferred>asynchronous or
                deferred!=(blocks if enabled else 0) or
                runtime.get("post_join_scope")!=("host_graph_construction_deferred_gpu_consumption"
                    if enabled else "evaluated_join_host_span")):
            raise ValueError("requested deferred channel join policy was not executed")


def validate_lora_channel_range(rows, enabled):
    """Verify an explicit correction ablation actually executed that path.

    Counters are per-session callback executions, not prediction counts or
    environment intent. Auto may stop using ANE in later requests; it cannot
    invent a narrow/full callback receipt from the executor self-test.
    """
    if not rows:
        raise ValueError("missing LoRA channel correction results")
    for row in rows:
        runtime=(row.get("hybrid") or {}).get("runtime_weight") or {}
        counts=tuple(session_counter(runtime,name) for name in
                     ("lora_channel_range_calls_session_total","lora_channel_full_calls_session_total"))
        if (row.get("lora_strategy")!="inference_time" or
                runtime.get("executor_backend")!="private_ane" or
                runtime.get("partition_axis")!="intermediate_channels" or
                counts[0 if enabled else 1]<=0 or counts[1 if enabled else 0]!=0):
            raise ValueError("requested LoRA channel correction path was not executed")


def benchmark_environment():
    """Do not inherit unrelated model experiments into a matched comparison."""
    return {key: value for key, value in os.environ.items()
            if key not in ("TURBOCIDER_ANE_BACKEND", "TURBOCIDER_ALLOW_PRIVATE_ANE") and
            not key.startswith("TURBOCIDER_PRIVATE_ANE_") and
            not key.startswith(("TURBOCIDER_QWEN21_", "TURBOCIDER_Z_", "TURBOCIDER_RUNTIME_ANE_"))}


def system_memory():
    # System-wide snapshots, not attribution to this process. Retain raw units
    # and page size so later analysis cannot confuse bytes with page counts.
    return {"swapusage": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip(),
            "vm_stat": subprocess.check_output(["vm_stat"], text=True)}
