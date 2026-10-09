#include "bridge.hpp"
#include "../../core/quantized_execution_profiles.hpp"
#include "../../models/qwen21/diagnostic_options.hpp"
#include <cstdlib>
#include <string_view>
#include "../../runtime/memory_execution.hpp"
#include "../../runtime/streaming/actual_receipt.hpp"
namespace tc {
static NSString *gpu_graph_label(const Request &r) {
    if (r.hybrid_mlp_mode == "runtime") return @"runtime_weight_token_row_ffn";
    if (r.hybrid_mlp_mode == "runtime_qkv") return @"runtime_weight_token_row_qkv";
    if (r.model == "qwen-image-2.1")
        return r.execution == "gpu_ane" ? @"qwen21_decode_mlp_complement" : @"qwen21_compiled_prefix_blocks";
    if (r.model == "minimax-h3-vdn")
        return @"h3_vdn_int6_window_delta";
    if (r.model.starts_with("minimax-h3-fasth3-mlx-int6"))
        return r.model.ends_with("-vsa") ? @"fasth3_int6_vsa" : @"fasth3_int6_qmm";
    if (r.model == "wan2.1-1.3b-qad")
        return r.execution == "gpu_ane" ? @"compiled_mlp_complement" :
            (r.compile_gpu ? @"compiled_whole_dit" : @"eager_blocks");
    if (r.model == "ltx-2.5-distilled")
        return r.ltx_backend == "cpp_mlx" ? @"ltx_cpp_mlx_blocks" :
            (r.ltx_sol_stage2 ?
                (r.ltx_stage2_text_rows ?
                    @"ltx_c_metal_fast_av_sol_text_pruned" :
                    @"ltx_c_metal_fast_av_sol") :
                (r.ltx_fast_av ? @"ltx_c_metal_fast_av" :
                                     @"ltx_c_metal_baseline"));
    if (r.model == "z-image-turbo-gguf") {
        if (r.execution == "gpu_ane")
            return @"compiled_mlp_complement";
        return @"native_quantized_blocks";
    }
    if (r.model == "z-image-turbo" && r.execution == "gpu_ane")
        return @"compiled_mlp_complement";
    if (r.model.starts_with("flux2-klein-") && r.execution == "gpu_ane")
        return @"compiled_hybrid_complement";
    if (!r.compile_gpu)
        return @"eager_blocks";
    return r.model == "z-image-turbo" ? @"compiled_fused_blocks"
                                       : @"compiled_single_blocks";
}
static NSString *gpu_graph_label(const RunResult &result) {
    if(result.backend=="mlx_cpp_metal_dense_split_gpu_control")
        return @"compiled_split_gpu_ffn_control";
    if (result.request.hybrid_mlp_mode == "runtime" && result.hybrid) {
        const auto &m = *result.hybrid;
        if (m.runtime_weight_backend.empty()) {
            Request gpu = result.request;gpu.execution="gpu";gpu.hybrid_mlp_mode="auto";
            return gpu_graph_label(gpu);
        }
        if (m.runtime_weight_partition_axis == "intermediate_channels")
            return @"runtime_weight_intermediate_channel_ffn";
    }
    if (result.request.model == "minimax-h3-vdn")
        return @"h3_vdn_int6_window_delta";
    if (result.backend == "mlx_cpp_metal" &&
        result.request.model.starts_with("minimax-h3-fasth3-mlx-int6"))
        return result.request.model.ends_with("-vsa") ? @"fasth3_int6_vsa" : @"fasth3_int6_qmm";
    if (result.backend == "mlx_cpp_metal_gguf+coreml")
        return @"compiled_mlp_complement";
    if (result.backend == "mlx_cpp_metal_gguf")
        return @"native_quantized_blocks";
    return gpu_graph_label(result.request);
}
static bool h3_qwen3_vl_encoder(const Request &request) {
    return request.model == "minimax-h3-vdn" ||
           request.model.starts_with("minimax-h3-fasth3-mlx-int6");
}
static bool ltx_gemma4_encoder(const Request &request) {
    return request.model == "ltx-2.5-distilled";
}
static NSString *encoder_backend_label(const Request &request, bool hybrid) {
    if (ltx_gemma4_encoder(request))
        return hybrid ? @"metal_mps+coreml" : @"metal_mps";
    return hybrid ? @"mlx_cpp_metal+coreml" : @"mlx_cpp_metal";
}
static NSString *encoder_gpu_graph_label(const Request &request, bool hybrid) {
    if (request.model == "qwen-image-2.1")
        return hybrid ? @"qwen3_vl_language_runtime_ffn_gpu_complement" : @"qwen3_vl_deepstack_gpu_only";
    if (ltx_gemma4_encoder(request))
        return hybrid ? @"gemma4_encoder_mlp_complement"
                      : @"gemma4_gpu_only";
    if (h3_qwen3_vl_encoder(request))
        return hybrid ? @"qwen3_vl_encoder_mlp_complement"
                      : @"qwen3_vl_gpu_only";
    return hybrid ? @"qwen3_encoder_mlp_complement" : @"qwen3_gpu_only";
}
static bool encoder_executed(const RunResult &result) {
    // Keep a cold attempt's failure/decline diagnostics, but never label a
    // zero-call Qwen encoder or a conditioning cache hit as model ANE work.
    return result.encoder_hybrid && (result.request.model != "qwen-image-2.1" ||
        (result.encoder_runtime_reuse ? result.encoder_runtime_reuse->calls_this_request :
                                      result.encoder_hybrid->runtime_calls) > 0);
}
static id encoder_reuse_dictionary(const RunResult &result) {
    if(!result.encoder_runtime_reuse)return NSNull.null;
    const auto &m=*result.encoder_runtime_reuse;
    return @{@"enabled":@(m.enabled),@"executor_reused":@(m.reused),@"executor_retained":@(m.retained),
        @"actual_calls_this_request":@(m.calls_this_request),@"retained_estimated_bytes":@(m.retained_estimated_bytes),
        @"weight_residency":result.encoder_weight_residency && result.encoder_weight_residency->retained ? @"resident_source_arrays" : @"request_local",
        @"scope":@"request-local execution evidence; HybridMetrics remain cumulative; estimate is not a RAM cap"};
}
static id encoder_weight_dictionary(const RunResult &result) {
    if(!result.encoder_weight_residency)return NSNull.null;
    const auto &m=*result.encoder_weight_residency;
    return @{@"enabled":@(m.enabled),@"weights_reused":@(m.reused),@"weights_retained":@(m.retained),
        @"source_bytes":@(m.source_bytes),@"retained_bytes":@(m.retained_bytes),@"loads_session_total":@(m.loads_session_total),
        @"decline_reason":@(m.decline_reason.c_str()),@"logical_capacity_upper_bytes":@(uint64_t(20)<<30),
        @"scope":@"admitted existing encoder source arrays; no extra weight copy, precision change or disk cache; generation stamps are not payload signatures"};
}
static id shared_lora_rank_dictionary(const RunResult &result) {
    if (!result.shared_lora_ranks) return NSNull.null;
    const auto &m = *result.shared_lora_ranks;
    return @{
        @"enabled" : @(m.enabled),
        @"prepared_sets_this_request" : @(m.prepared_sets),
        @"completed_hybrid_blocks_this_request" : @(m.completed_hybrid_blocks),
        @"completed_adapter_rank_arrays_this_request" : @(m.completed_adapter_rank_arrays),
        @"scope" : @"operation-local rank graph outputs consumed by both GPU channel and ANE correction paths in successful hybrid blocks; not physical kernel counts"
    };
}
static id student_ffn_reuse_dictionary(const RunResult &result) {
    return @{@"enabled":@(result.student_ffn_reuse_enabled),
        @"requested_final_layers":@(result.student_ffn_requested_layers),
        @"captured_blocks_this_request":@(result.student_ffn_captured_blocks),
        @"reused_blocks_this_request":@(result.student_ffn_reused_blocks),
        @"peak_logical_cache_bytes":@(result.student_ffn_peak_logical_bytes),
        @"released_before_vae":@YES,
        @"scope":@"successful native request graph outputs; complete FFN including LoRA, attention/modulation still recomputed; logical bytes not physical RAM/overlap proof"};
}
static id qwen_ffn_phase_dictionary(const RunResult &result) {
    const auto &m=*result.qwen_ffn_phases;
    auto phase=[](const QwenFfnPhaseCounters &p) {
        return @{@"steps_this_request":@(p.steps),@"actual_rows":@(p.rows),
            @"step_seconds":@(p.step_seconds),@"runtime_calls_this_request":@(p.runtime_calls),
            @"completed_channel_blocks_this_request":@(p.completed_channel_blocks)};
    };
    return @{@"policy":@(m.policy.c_str()),@"prefill":phase(m.prefill),@"decode":phase(m.decode),
        @"scope":@"completed finite native denoise steps, classified by actual prefix reuse; request-local driver calls and successful channel blocks; host step spans, not physical kernel/overlap proof"};
}
static NSString *encoder_backend_label(const RunResult &result) {
    if (encoder_executed(result) && result.request.model == "qwen-image-2.1" &&
        !result.encoder_hybrid->runtime_weight_backend.empty())
        return result.encoder_hybrid->runtime_weight_backend == "private_ane"
            ? @"mlx_cpp_metal+private_ane_runtime_weight_experimental"
            : @"mlx_cpp_metal+coreml_runtime_weight";
    return encoder_backend_label(result.request, encoder_executed(result));
}
static NSString *encoder_precision_label(const RunResult &result) {
    if (!encoder_executed(result)) return @"bf16";
    const auto &metrics = *result.encoder_hybrid;
    if (result.request.model == "qwen-image-2.1" &&
        !metrics.runtime_weight_backend.empty()) {
        const auto &path = metrics.runtime_weight_data_path;
        return path == "w8a8_convrot" ? @"bf16_gpu+runtime_convrot_w8a8_ffn_bf16_io" :
            path == "w8a8_hadamard" ? @"bf16_gpu+runtime_w8a8_ffn_bf16_io" :
            @"bf16_gpu+runtime_fp16_ffn_bf16_io";
    }
    return @(hybrid_precision_label(metrics).c_str());
}
static NSString *encoder_weight_validation_label(const Request &request,
                                                  bool hybrid,
                                                  bool executed) {
    if (request.model == "qwen-image-2.1")
        return @"native Qwen3-VL language/vision/DeepStack checkpoint loaded directly";
    if (ltx_gemma4_encoder(request)) {
        if (hybrid)
            return executed
                ? @"Gemma4 checkpoint and Core ML bank verified at execution"
                : @"Gemma4 checkpoint and Core ML bank must be verified at execution";
        return @"native Gemma4 checkpoint loaded directly";
    }
    if (h3_qwen3_vl_encoder(request)) {
        if (hybrid)
            return executed
                ? @"Qwen3-VL shard set and Core ML manifest verified at execution"
                : @"Qwen3-VL shard set and Core ML manifest must be verified at execution";
        return @"native Qwen3-VL shard set loaded directly";
    }
    if (hybrid)
        return executed
            ? @"Qwen3 checkpoint and Core ML manifest verified at execution"
            : @"Qwen3 checkpoint and Core ML manifest must be verified at execution";
    return @"native Qwen3 checkpoint loaded directly";
}
static NSString *encoder_approximation_label(const Request &request) {
    if (request.model == "qwen-image-2.1")
        return @"qwen3_vl_language_runtime_ffn_approximation";
    if (ltx_gemma4_encoder(request))
        return @"gemma4_encoder_mlp_coreml_approximation";
    return h3_qwen3_vl_encoder(request)
        ? @"qwen3_vl_encoder_mlp_coreml_approximation"
        : @"qwen3_encoder_mlp_coreml_approximation";
}
static NSArray *strings(const std::vector<std::string> &values) {
    NSMutableArray *array = [NSMutableArray array];
    for (auto &value : values)
        [array addObject:@(value.c_str())];
    return array;
}
static NSDictionary *quantized_config(const QuantizedExecutionConfig &c) {
    if (!c.active()) return @{@"enabled": @NO, @"schema_version": @(c.schema_version.value_or(1))};
    return @{@"enabled": @YES, @"schema_version": @1,
        @"mode": @(c.mode.value_or(gguf_raw_gpu_profile(c.precision_profile.value_or("")) ? "bounded_raw_packed" : c.precision_profile == "z-source-native-affine-v1" || c.precision_profile == "z-mlx-compat-affine-v1" ? "bounded_packed" : "bounded_dequant").c_str()),
        @"source_residency": @(c.source_residency.value_or("packed_resident").c_str()),
        @"decode_backend": @(c.decode_backend.value_or(gguf_raw_gpu_profile(c.precision_profile.value_or("")) ? "cpu_io_gpu_affine" : "cpu_simd").c_str()),
        @"precision_profile": @(c.precision_profile.value_or("z-source-mixed-v1").c_str()),
        @"granularity": @"layer", @"prefetch_layers": @(c.prefetch_layers.value_or(1)),
        @"persistent_dense_layers": @0, @"oversized_layer_policy": @"reject",
        @"ane_compute": @"off", @"allow_requantization": @NO};
}
NSDictionary *to_dictionary(const ModelDescriptor &d) {
    NSMutableDictionary *result = [@{
        @"id" : @(d.id.c_str()),
        @"name" : @(d.name.c_str()),
        @"executor" : @(d.executable),
        @"operations" : strings(d.operations),
        @"inputs" : strings(d.inputs),
        @"roles" : strings(d.roles),
        @"output" : @(d.output.c_str()),
        @"default_steps" : @(d.steps),
        @"default_frames" : @(d.frames),
        @"default_width" : @(d.width),
        @"default_height" : @(d.height),
        @"default_audio" : @(d.default_audio),
        @"default_residency" : @(d.default_residency.c_str())
    } mutableCopy];
    if (d.max_images)
        result[@"max_images"] = @(d.max_images);
    if (d.weight_validation_pending)
        result[@"weight_validation"] = @"pending";
    result[@"supports_lora"] = @(d.supports_lora);
    result[@"runtime_lora"] = @(d.runtime_lora);
    result[@"supports_gpu_ane"] = @(d.supports_gpu_ane);
    result[@"default_execution"] = @"gpu";
    result[@"gpu_ane_policy"] = d.supports_gpu_ane
        ? @"optional_manifest_gated" : @"unsupported";
    result[@"supports_encoder_gpu_ane"] = @(d.supports_encoder_gpu_ane);
    result[@"encoder_gpu_ane_policy"] = d.supports_encoder_gpu_ane
        ? @"optional_explicit_manifest" : @"unsupported";
    result[@"native_gemma4_candidate"] = @(d.native_gemma4_candidate);
    result[@"native_conditioning_connector"] = @(d.native_conditioning_connector);
    result[@"native_i2v_clean_prefix"] = @(d.native_i2v_clean_prefix);
    result[@"native_gpu_ane_profile"] = @(d.native_gpu_ane_profile);
    result[@"native_audio_output_candidate"] = @(d.native_audio_output_candidate);
    result[@"native_audio_vae_candidate"] = @(d.native_audio_vae_candidate);
    result[@"native_base_vocoder_candidate"] = @(d.native_base_vocoder_candidate);
    result[@"audio_output"] = @(d.audio_output);
    result[@"request_lora_identity_validation"] = @(d.request_lora_identity_validation);
    result[@"lora_strategies"] = strings(d.lora_strategies);
    if (!d.default_lora_strategy.empty())
        result[@"default_lora_strategy"] = @(d.default_lora_strategy.c_str());
    if (!d.backend.empty()) result[@"backend"] = @(d.backend.c_str());
    if (!d.lora_mode.empty()) result[@"lora_mode"] = @(d.lora_mode.c_str());
    if (!d.runtime_dependency.empty()) result[@"runtime_dependency"] = @(d.runtime_dependency.c_str());
    if (!d.parallel_strategy.empty()) result[@"parallel_strategy"] = @(d.parallel_strategy.c_str());
    if (!d.audio_capability.empty()) result[@"audio_capability"] = @(d.audio_capability.c_str());
    if (!d.executor_operations.empty()) result[@"executor_operations"] = strings(d.executor_operations);
    if (!d.candidate_limitations.empty()) result[@"candidate_limitations"] = strings(d.candidate_limitations);
    if (d.fps) result[@"default_fps"] = @(d.fps);
    return result;
}
NSDictionary *to_dictionary(const std::vector<ModelDescriptor> &descriptors) {
    NSMutableArray *models = [NSMutableArray array];
    for (auto &d : descriptors)
        [models addObject:to_dictionary(d)];
    return @{@"schema_version" : @2, @"models" : models};
}
NSDictionary *to_dictionary(const ExecutionPlan &plan) {
    auto &r = plan.request;
    auto &recipe = plan.recipe;
    bool hybrid = r.execution == "gpu_ane";
    bool encoder_hybrid = !r.encoder_ane_manifest.empty();
    const auto lora_strategy = effective_lora_strategy(r);
    auto validation = r.model == "z-image-turbo-gguf" ? @"native_gguf_candidate" :
                      r.model.starts_with("flux2-klein-") ? @"native_candidate" :
                      r.model == "minimax-h3-turbo" ? @"manifest_verified_native" :
                      r.model == "wan2.1-1.3b-qad" ? @"manifest_verified_native" :
                      r.model == "ltx-2.5-distilled" ?
                          (recipe.executable ? @"native_video_executor" : @"native_capability_gated") :
                      r.model == "minimax-h3-vdn" ? @"modelscope_vdn_stage_dmd_candidate" :
                      r.model.starts_with("minimax-h3-fasth3-mlx-int6") ? @"modelscope_int6_parity_candidate" :
                      r.model == "z-image-turbo" ? @"native_candidate" :
                      r.model == "llada-image-turbo" ? @"native_llada_candidate" :
                      r.model == "qwen-image-2.1" ? @"native_qwen21_experimental" :
                      @"weights_pending";
    auto weight_validation = (r.hybrid_mlp_mode == "runtime" || r.hybrid_mlp_mode == "runtime_qkv")
        ? @"runtime graph geometry, receipt and sparse weight-switch self-test checked at load; full-model quality experimental"
        : r.model == "z-image-turbo-gguf" ?
            @"GGUF header checked at load; paired output parity pending" :
        r.model.starts_with("flux2-klein-") ? @"see parity evidence" :
        r.model == "minimax-h3-turbo" ?
            (r.loras.empty() ? @"manifest-verified" : @"premerged-sidecar-verified-at-execution") :
        r.model == "wan2.1-1.3b-qad" ?
            (r.loras.empty() ? @"checkpoint-and-ane-identity-verified-at-load" : @"premerged-manifest-verified-at-execution") :
        r.model == "ltx-2.5-distilled" ?
            (r.loras.empty() ? @"checkpoint-validated-at-load" : @"premerged-sidecar-verified-at-execution") :
        r.model == "minimax-h3-vdn" ?
            @"ModelScope VDN stage, FL2VA base, Turbo adapter, and prepared artifact identity required at execution" :
        r.model.starts_with("minimax-h3-fasth3-mlx-int6") ?
            @"ModelScope FastH3 manifest and component checks at execution" :
        r.model == "z-image-turbo" ?
            (r.loras.empty() ? @"comfy-oracle-validated; m4max-a4096-hybrid-qualified" :
                               @"in-memory-lora; comfy-oracle-validated") :
        r.model == "llada-image-turbo" ?
            @"native checkpoint loaded directly; see recorded parity evidence" :
        r.model == "qwen-image-2.1" ?
            @"native Comfy BF16 components; component parity recorded; edit quality pending" :
        @"pending";
    auto lora_fusion = lora_strategy == "none" ? @"none" :
        (lora_strategy == "inference_time" ?
            @"inference_time_low_rank" :
         lora_strategy == "in_memory_merge" ?
            (r.model.starts_with("flux2-klein-") ? @"load_time_baked" : @"in_memory_delta") :
         r.model == "wan2.1-1.3b-qad" ? @"premerged_manifest_verified" :
         @"premerged_manifest_verified");
    auto backend = r.hybrid_mlp_mode == "runtime_qkv" ? @"mlx_cpp_metal+coreml_runtime_qkv" :
        r.hybrid_mlp_mode == "runtime" ?
        (r.model == "z-image-turbo-gguf" ? @"mlx_cpp_metal_gguf+coreml_runtime_weight"
                                       : @"mlx_cpp_metal+coreml_runtime_weight") : hybrid ?
        (r.model == "wan2.1-1.3b-qad" ? @"wan-mlx+coreml" :
         r.model == "ltx-2.5-distilled" ? @"ltx-gpu+ane" :
         r.model == "z-image-turbo-gguf" ? @"mlx_cpp_metal_gguf+coreml" : @"mlx_cpp_metal+coreml") :
        r.model == "z-image-turbo-gguf" ?
            @"mlx_cpp_metal_gguf" :
        (r.model == "minimax-h3-vdn" ||
         r.model.starts_with("minimax-h3-fasth3-mlx-int6")) ? @"mlx_cpp_metal" :
        r.model == "minimax-h3-turbo" ? @"h3-metal-mps" :
        r.model == "ltx-2.5-distilled" ?
            (r.ltx_backend == "cpp_mlx" ? @"mlx_cpp_metal" : @"ltx-metal-mps") :
        r.model == "wan2.1-1.3b-qad" ? @"wan-mlx" :
        r.model == "llada-image-turbo" ? @"mlx_cpp_metal" :
            @"mlx_cpp_metal";
    int dw = r.width, dh = r.height;
    NSMutableArray *stages = [NSMutableArray array];
    for (auto &s : recipe.stages) {
        NSMutableArray *deps = [NSMutableArray array];
        for (auto &d : s.dependencies)
            [deps addObject:@(d.c_str())];
        [stages addObject:@{
            @"id" : @(s.id.c_str()),
            @"dependencies" : deps,
            @"iterations" : @(s.id == "denoise" ? r.steps : s.iterations)
        }];
    }
    NSMutableArray *algorithm_approximations = [NSMutableArray array];
    if(r.model=="qwen-image-2.1" && r.hybrid_mlp_mode=="runtime") {
        const auto phase=qwen21::runtime_ffn_phase(std::getenv("TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE"));
        if(phase!=qwen21::RuntimeFfnPhase::All)
            [algorithm_approximations addObject:[NSString stringWithFormat:@"qwen21_runtime_ffn_phase_%s",qwen21::runtime_ffn_phase_name(phase)]];
    }
    if (r.model == "qwen-image-2.1" && qwen21::option_enabled(
            std::getenv("TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC")))
        [algorithm_approximations addObject:@"qwen21_decode_dbcache_diagnostic"];
    if (r.model == "qwen-image-2.1" && qwen21::option_enabled(
            std::getenv("TURBOCIDER_QWEN21_RECT_W8A8_DIAGNOSTIC")))
        [algorithm_approximations addObject:@"qwen21_rectangular_decode_w8a8_tiled_diagnostic"];
    if (r.model == "qwen-image-2.1" && qwen21::gate_up_ane(r))
        [algorithm_approximations addObject:@"qwen21_w8a8_gate_up_gpu_silu_down_diagnostic"];
    if (r.model == "qwen-image-2.1" && qwen21::fused_lora_ane(r))
        [algorithm_approximations addObject:@"qwen21_w8a8_fused_lora_pre_silu_diagnostic"];
    if (r.model == "qwen-image-2.1" && qwen21::lora_base_ane(r) &&
            !qwen21::gate_up_ane(r) && !qwen21::fused_lora_ane(r))
        [algorithm_approximations addObject:@"qwen21_runtime_lora_base_ane_suffix_only_diagnostic"];
    if (r.model == "z-image-turbo" && r.hybrid_mlp_mode == "lora_suffix")
        [algorithm_approximations addObject:@"z_image_runtime_lora_base_ane_suffix_only_diagnostic"];
    if (r.model == "z-image-turbo" && r.hybrid_mlp_mode == "lora_fused")
        [algorithm_approximations addObject:@"z_image_runtime_lora_base_fused_pre_silu_experimental"];
    if (r.model == "qwen-image-2.1" && qwen21::option_enabled(
            std::getenv("TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC")))
        [algorithm_approximations addObject:@"qwen21_viggle_reference_resize_512_diagnostic"];
    const char *last_target = std::getenv("TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC");
    if (r.model == "qwen-image-2.1" && last_target && std::string_view(last_target) == "1")
        [algorithm_approximations addObject:@"qwen21_prefill_last_target_only_diagnostic"];
    const char *fused_qkv = std::getenv("TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC");
    if (r.model == "qwen-image-2.1" && fused_qkv && std::string_view(fused_qkv) == "1")
        [algorithm_approximations addObject:@"qwen21_metal_fused_qkv_diagnostic"];
    const char *tiled_prefill = std::getenv("TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC");
    const int tiled_layers = qwen21::tiled_prefill_layer_count(tiled_prefill ? tiled_prefill : "0");
    if (r.model == "qwen-image-2.1" && tiled_layers > 0)
        [algorithm_approximations addObject:tiled_layers == 8 ?
            @"qwen21_tiled_prefill_last8_w8a8_diagnostic" :
            tiled_layers == 16 ?
            @"qwen21_tiled_prefill_last16_w8a8_diagnostic" :
            tiled_layers == 20 ?
            @"qwen21_tiled_prefill_last20_w8a8_diagnostic" :
            tiled_layers == 24 ?
            @"qwen21_tiled_prefill_last24_w8a8_diagnostic" :
            @"qwen21_tiled_prefill_w8a8_diagnostic"];
    const char *tiled_prefix_reuse = std::getenv("TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC");
    if (r.model == "qwen-image-2.1" && tiled_prefix_reuse && std::string_view(tiled_prefix_reuse) == "1")
        [algorithm_approximations addObject:@"qwen21_tiled_prefill_prefix_kv_diagnostic"];
    const char *prefix_target_only = std::getenv("TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC");
    if (r.model == "qwen-image-2.1" && prefix_target_only && std::string_view(prefix_target_only) == "1")
        [algorithm_approximations addObject:@"qwen21_tiled_prefix_target_only_diagnostic"];
    if(qwen21::student_final_ffn_reuse(r))
        [algorithm_approximations addObject:@"qwen21_student_final_ffn_reuse"];
    const char *hybrid_reuse = std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC");
    if (r.model == "qwen-image-2.1" && hybrid_reuse && std::string_view(hybrid_reuse) == "1")
        [algorithm_approximations addObject:@"qwen21_hybrid_reuse_final_ffn_diagnostic"];
    const char *hybrid_last16 =
        std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC");
    if (r.model == "qwen-image-2.1" && hybrid_last16 && std::string_view(hybrid_last16) == "1")
        [algorithm_approximations addObject:@"qwen21_hybrid_reuse_final_last16_ffn_diagnostic"];
    const char *hybrid_half = std::getenv(
        "TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC");
    if (r.model == "qwen-image-2.1" && hybrid_half && std::string_view(hybrid_half) == "1")
        [algorithm_approximations addObject:@"qwen21_hybrid_reuse_penultimate_even_ffn_diagnostic"];
    if (r.hybrid_mlp_mode == "runtime")
        [algorithm_approximations addObject:@"runtime_weight_fp16_token_row_ffn"];
    if (r.hybrid_mlp_mode == "runtime_qkv")
        [algorithm_approximations addObject:@"runtime_weight_fp16_token_row_qkv"];
    if (r.model == "minimax-h3-vdn")
        [algorithm_approximations addObject:
            @"affine_int6_g64_base_weight_quantization"];
    else if (r.model == "z-image-turbo-gguf")
        [algorithm_approximations addObject:
            @"checkpoint_defined_gguf_weight_quantization"];
    else if (!r.quantized_cache.empty())
        [algorithm_approximations addObject:
            @"row_symmetric_int8_weight_quantization"];
    else if (hybrid && r.hybrid_mlp_mode != "runtime" && r.hybrid_mlp_mode != "runtime_qkv")
        [algorithm_approximations addObject:
            r.model == "qwen-image-2.1"
                ? (r.qwen21_w8a8 ? @"qwen21_decode_mlp_w8a8_per_tensor" : @"qwen21_decode_mlp_fp16_partition")
                : @"single_block_mlp_coreml_approximation"];
    if (r.model == "qwen-image-2.1" && r.qwen21_reference_size != 1024)
        [algorithm_approximations addObject:[NSString stringWithFormat:
            @"qwen21_reference_resize_%d", r.qwen21_reference_size]];
    if (r.model == "qwen-image-2.1" && hybrid && r.operation == "image.edit" &&
        r.qwen21_w8a8 && r.qwen21_reference_size == 1024)
        [algorithm_approximations addObject:@"qwen21_w8a8_full_reference_diagnostic"];
    if (r.model == "qwen-image-2.1" && !r.loras.empty())
        [algorithm_approximations addObject:@"qwen21_viggle_v021_r256_6step_distillation"];
    if (qwen21::lora_1024_generation(r))
        [algorithm_approximations addObject:@"qwen21_lora_1024_generation_fp32_diagnostic"];
    if (r.model == "qwen-image-2.1" && !r.loras.empty()) {
        const char *fp16_lora = std::getenv("TURBOCIDER_QWEN21_VIGGLE_LORA_FP16");
        if (fp16_lora && std::string_view(fp16_lora) == "1")
            [algorithm_approximations addObject:@"qwen21_viggle_lora_fp16_matmuls"];
    }
    if (r.model == "qwen-image-2.1" && r.steps >= 3) {
        const char *reuse_flag = std::getenv("TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN");
        if (reuse_flag && std::string_view(reuse_flag) == "1")
            [algorithm_approximations addObject:@"qwen21_gpu_reuse_final_ffn"];
    }
    if (r.model == "qwen-image-2.1" && r.steps >= 4) {
        const char *half_reuse = std::getenv("TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN");
        if (half_reuse && std::string_view(half_reuse) == "1")
            [algorithm_approximations addObject:@"qwen21_gpu_reuse_penultimate_even_ffn"];
    }
    if (r.model == "qwen-image-2.1") {
        const char *local_references = std::getenv("TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION");
        if (local_references && std::string_view(local_references) == "1")
            [algorithm_approximations addObject:@"qwen21_reference_local_attention"];
        if (local_references && std::string_view(local_references) == "2")
            [algorithm_approximations addObject:@"qwen21_last_reference_local_attention"];
        if (local_references && std::string_view(local_references) == "3")
            [algorithm_approximations addObject:@"qwen21_last16_reference_local_attention_diagnostic"];
        const char *norm_rope = std::getenv("TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE");
        if (norm_rope && std::string_view(norm_rope) == "1")
            [algorithm_approximations addObject:@"qwen21_metal_qk_norm_rope"];
    }
    if (encoder_hybrid)
        [algorithm_approximations addObject:encoder_approximation_label(r)];
    if (r.model == "ltx-2.5-distilled") {
        if (r.ltx_sol_stage1)
            [algorithm_approximations addObject:@"ltx_sol_stage1"];
        if (r.ltx_sol_stage2)
            [algorithm_approximations addObject:@"ltx_sol_stage2"];
        if (r.ltx_stage2_text_rows)
            [algorithm_approximations addObject:@"ltx_stage2_text_context_pruning"];
    }
    NSMutableDictionary *report = [@{
        @"selection_pending" : @(r.execution == "auto"),
        @"requested_execution" : @(r.execution.c_str()),
        @"prompt_enhance" : @(r.prompt_enhance),
        @"prompt_enhance_edit_experimental" : @(r.prompt_enhance_edit_experimental),
        @"prompt_enhancer_path" : @(r.prompt_enhancer_path.c_str()),
        @"schema_version" : @1,
        @"model" : @(r.model.c_str()),
        @"executable" : @(recipe.executable),
        @"validation" : validation,
        @"backend" : backend,
        @"execution" : hybrid ? @"gpu_ane_experimental" : @"gpu",
        @"encoder_execution" : encoder_hybrid ? @"gpu_ane_experimental" : @"gpu",
        @"encoder_backend" : encoder_backend_label(r, encoder_hybrid),
        @"encoder_gpu_graph" : encoder_gpu_graph_label(r, encoder_hybrid),
        @"encoder_precision" : encoder_hybrid ? @"bf16_gpu+coreml_mlp_fp16_io"
                                                : @"bf16",
        @"encoder_weight_validation" :
            encoder_weight_validation_label(r, encoder_hybrid, false),
        @"gpu_graph" : gpu_graph_label(r),
        @"precision" : r.hybrid_mlp_mode == "runtime_qkv" ? @"bf16_gpu+runtime_fp16_qkv_bf16_io" :
                      r.hybrid_mlp_mode == "runtime" ?
                          (r.model == "z-image-turbo-gguf" ? @"gguf_native_gpu+runtime_fp16_ffn"
                                                          : @"bf16_gpu+runtime_fp16_ffn_bf16_io") :
                      r.model == "minimax-h3-vdn" ? @"int6_g64_base+bf16_vdn+fp32_solve" :
                      r.model.starts_with("minimax-h3-fasth3-mlx-int6") ? @"int6_g64_bf16_activation" :
                      r.model == "wan2.1-1.3b-qad" ? @"fp16-int8-affine-dit+bf16-umt5+fp32-taehv" :
                      r.model == "z-image-turbo-gguf" ? @"checkpoint_defined_gguf" :
            (!r.quantized_cache.empty() ? @"int8_weight_bf16_activation_streamed" :
             (hybrid ? (r.model == "qwen-image-2.1"
                            ? (r.qwen21_w8a8
                                ? (r.qwen21_gpu_w8a16 ? @"w8a16_gpu+w8a8_mlp_fp16_io" : @"bf16_gpu+w8a8_mlp_fp16_io")
                                : @"bf16_gpu+fp16_mlp_fp16_io")
                            : @"bf16_gpu+coreml_mlp_fp16_io") : @"bf16")),
        @"algorithm_approximations" : algorithm_approximations,
        @"requested_shape" : @[ @(r.width), @(r.height), @(r.frames) ],
        @"decoded_shape" : @[ @(dw), @(dh), @(r.frames) ],
        @"stages" : stages,
        @"memory_estimate_bytes" : plan.memory_estimate_bytes ? @(*plan.memory_estimate_bytes)
                                                              : [NSNull null],
        @"memory_estimate_kind" : @"conservative_heuristic_not_hard_limit",
        @"memory_budget_bytes" : r.memory_budget_bytes ? @(r.memory_budget_bytes)
                                                        : [NSNull null],
        @"memory_budget_scope" :
            (r.model == "minimax-h3-vdn" &&
                   r.residency == "streamed" && r.memory_budget_bytes)
                    ? @"vdn_base_branch_working_set_target_not_process_cap"
                : (r.model == "minimax-h3-turbo" &&
                   r.residency == "streamed" && r.memory_budget_bytes)
                    ? @"h3_dit_working_set_target_not_process_cap"
                : (r.model == "ltx-2.5-distilled" &&
                   r.residency == "streamed" && r.memory_budget_bytes)
                    ? @"ltx_denoiser_working_set_target_not_process_cap"
                : (r.memory_budget_bytes ? @"runtime_request_budget" : @"unset"),
        @"operation" : @(r.operation.c_str()),
        @"residency" : @(r.residency.c_str()),
        @"streaming_offload" : @(r.streaming_offload ||
                                  r.residency == "streaming" ||
                                  r.residency == "streamed"),
        @"quantized_cache" : r.quantized_cache.empty()
            ? (id)[NSNull null] : @(r.quantized_cache.c_str()),
        @"profile_identity" : @(r.profile_identity.c_str()),
        @"weight_validation" : weight_validation,
        @"lora_count" : @(r.loras.size()),
        @"lora_strategy" : @(lora_strategy.c_str()),
        @"hybrid_mlp_mode" : @(r.hybrid_mlp_mode.c_str()),
        @"lora_fusion" : lora_fusion,
        @"audio" : @(r.audio),
        @"ltx_backend" : @(r.ltx_backend.c_str()),
        @"ltx_fast_av" : @(r.ltx_fast_av),
        @"ltx_video_attention_batch" : @(r.ltx_video_attention_batch),
        @"ltx_sol_stage1" : @(r.ltx_sol_stage1),
        @"ltx_sol_stage2" : @(r.ltx_sol_stage2),
        @"ltx_sol_tau" : @(r.ltx_sol_tau),
        @"ltx_sparse_mode" : @(r.ltx_sparse_mode),
        @"ltx_sparse_radius" : @(r.ltx_sparse_radius),
        @"ltx_sparse_anchor_stride" : @(r.ltx_sparse_anchor_stride),
        @"ltx_sparse_tokens_per_frame" : @(r.ltx_sparse_tokens_per_frame),
        @"ltx_sparse_keep_blocks" : @(r.ltx_sparse_keep_blocks),
        @"ltx_sol_dense_edge_blocks" : @(r.ltx_sol_dense_edge_blocks),
        @"ltx_sol_dense_edge_steps" : @(r.ltx_sol_dense_edge_steps),
        @"ltx_stage2_text_rows" : @(r.ltx_stage2_text_rows),
        @"audio_capability" : (r.model == "minimax-h3-vdn" ||
                                  r.model.starts_with("minimax-h3-fasth3-mlx-int6")) ?
            (r.audio ? @"full_h3_audio_vae_32khz_stereo" : @"video_only_native") :
            (r.model == "ltx-2.5-distilled" ?
             (r.audio ? @"latent_to_48khz_aac_candidate" : @"video_only_native") : @"not_applicable"),
        @"executor_operations" : (r.model == "minimax-h3-vdn" ||
                                      r.model.starts_with("minimax-h3-fasth3-mlx-int6")) ? @[ @"video.generate" ] :
            r.model == "ltx-2.5-distilled" ? @[ @"video.generate", @"video.image" ] :
            (r.model == "wan2.1-1.3b-qad" ? @[ @"video.generate" ] : [NSNull null]),
        @"limitation" : recipe.executable
            ? @"capabilities depend on model artifacts and configured hardware"
            : @"native executor migration incomplete"
    } mutableCopy];
    if (r.model == "qwen-image-2.1") {
        NSMutableArray *blocks = [NSMutableArray array];
        for (int block : r.qwen21_gpu_full_ffn_blocks) [blocks addObject:@(block)];
        report[@"qwen21_w8a8"] = @(r.qwen21_w8a8);
        report[@"qwen21_gpu_w8a16"] = @(r.qwen21_gpu_w8a16);
        report[@"qwen21_reference_size"] = @(r.qwen21_reference_size);
        report[@"qwen21_gpu_full_ffn_blocks"] = blocks;
        if (r.qwen21_w8a8)
            report[@"planned_w8a8_ffn_layer_coverage"] = @((32. - blocks.count) / 32.);
    }
    if (r.quantized_execution.specified()) report[@"quantized_execution"] = quantized_config(r.quantized_execution);
    if (r.quantized_execution.active()) {
        report[@"executable"] = @NO;
        report[@"quantized_execution_qualification"] = @"experimental-unqualified";
        report[@"memory_estimate_kind"] = @"requires_gguf_layout_and_framework_envelope";
    }
    if (r.streaming.active()) {
        // Quantized execution remains an explicit experimental candidate;
        // its typed request is reported separately below.
        report[@"executable"] = @NO;
        report[@"memory_estimate_kind"] = @"requires_layout_metadata";
        NSMutableDictionary *origins = [NSMutableDictionary dictionary];
        for (const auto &[key, origin] : r.streaming.provenance)
            origins[@(key.c_str())] = @(origin.c_str());
        report[@"streaming"] = @{
            @"requested_layout": r.streaming_requested
                ? streaming_config_dictionary(*r.streaming_requested) : (id)NSNull.null,
            @"merged_intent": streaming_config_dictionary(r.streaming),
            @"field_provenance": origins,
            @"eligibility": @"plan_only",
            @"execution_supported": @NO,
            @"rejection_code": @"streaming_layout_not_certified",
            @"resolution_state": @"requires_checkpoint_metadata",
            @"resolved_layout": NSNull.null,
            @"actual_layout": NSNull.null,
            @"enforcement": @"none"
        };
    }
    if (r.streaming_selector && r.streaming_selector->active()) {
        report[@"executable"] = @NO;
        report[@"memory_estimate_kind"] = @"requires_preset_resolution";
        NSMutableDictionary *origins = [NSMutableDictionary dictionary];
        for (const auto &[key, origin] : r.streaming_selector->provenance)
            origins[@(key.c_str())] = @(origin.c_str());
        report[@"streaming"] = @{
            @"requested_selector": r.streaming_selector_requested
                ? streaming_selector_dictionary(*r.streaming_selector_requested)
                : (id)NSNull.null,
            @"merged_selector": streaming_selector_dictionary(
                *r.streaming_selector),
            @"field_provenance": origins,
            @"eligibility": @"plan_only",
            @"execution_supported": @NO,
            @"rejection_code": @"streaming_preset_resolution_required",
            @"resolution_state": @"requires_engine_artifact_and_catalog",
            @"resolved_layout": NSNull.null,
            @"actual_layout": NSNull.null,
            @"enforcement": @"none"
        };
    }
    if (plan.memory_policy) {
        const auto &policy = *plan.memory_policy;
        report[@"memory_policy"] = @{
            @"enabled" : @(policy.enabled),
            @"user_limit_bytes" : @(policy.user_limit_bytes),
            @"effective_budget_bytes" : @(policy.effective_budget_bytes),
            @"system_reserve_bytes" : @(policy.system_reserve_bytes),
            @"buffer_percent" : @(policy.buffer_percent),
            @"max_refill_slots" : @(policy.max_refill_slots),
            @"allow_quality_preserving_tiling" :
                @(policy.allow_quality_preserving_tiling),
            @"estimate_fits" : @(policy.estimate_fits),
            @"route_available" : @(policy.route_available),
            @"execution_supported" : @(policy.execution_supported),
            @"capability_level" :
                @(memory_capability_level_name(policy.capability_level)),
            @"certification_state" :
                @(memory_certification_state_name(policy.certification_state)),
            @"release_stable" : @(policy.release_stable),
            @"manifest_digest" : @(policy.manifest_digest.c_str()),
            @"evidence_digest" : @(policy.evidence_digest.c_str()),
            @"planned_upper_bytes" : @(policy.planned_upper_bytes),
            @"framework_upper_bytes" : @(policy.framework_upper_bytes),
            @"non_denoiser_reserve_bytes" :
                @(policy.non_denoiser_reserve_bytes),
            @"denoiser_budget_bytes" : @(policy.denoiser_budget_bytes),
            @"refill_slots" : @(policy.refill_slots),
            @"adapter_candidate" : @(policy.adapter_candidate.c_str()),
            @"candidate_backend" : @(policy.candidate_backend.c_str()),
            @"candidate_dtype" : @(policy.candidate_dtype.c_str()),
            @"candidate_model_variant" :
                @(policy.candidate_model_variant.c_str()),
            @"candidate_sampler_mode" :
                @(policy.candidate_sampler_mode.c_str()),
            @"candidate_tiling_mode" :
                @(policy.candidate_tiling_mode.c_str()),
            @"effective_residency" : @(policy.effective_residency.c_str()),
            @"estimate_provenance" : @(policy.estimate_provenance.c_str()),
            @"admission_state" : @(policy.admission_state.c_str()),
            @"enforcement_scope" : @(policy.enforcement_scope.c_str()),
            @"reason" : @(policy.reason.c_str()),
            @"digest" : @(policy.digest.c_str())
        };
    }
    return report;
}
static NSDictionary *actual_streaming_layout(
        const StreamingRuntimeMetrics &m) {
    NSMutableDictionary *value = [@{
        @"stage" : @(m.stage.c_str()),
        @"resident_prefix_blocks" : @(m.resident_prefix_blocks),
        @"block_group_size" : @(m.block_group_size),
        @"slot_count" : @(m.slot_count),
        @"prefetch_distance" : @(m.prefetch_distance),
        @"io_workers" : @(m.io_workers),
        @"group_count" : @(m.group_count),
        @"pass_count" : @(m.pass_count),
        @"startup_policy" : @(m.startup_policy.c_str()),
        @"pass_transition" : @(m.pass_transition.c_str()),
        @"retention" : @(m.retention.c_str()),
        @"reader_revision" : @(m.reader_revision),
        @"weight_format" : @(m.weight_format.c_str()),
        @"kernel_revision" : @(m.kernel_revision.c_str()),
        @"conditioning_recipe" : @(m.conditioning_recipe.c_str()),
        @"upsample_boundary" : @(m.upsample_boundary.c_str()),
        @"component_policy_revision" :
            @(m.component_policy_revision.c_str()),
        @"multi_pool_policy" : @(m.multi_pool_policy.c_str()),
        @"pool_count" : @(m.pool_count),
        @"slot_bundle_count" : @(m.slot_bundle_count),
        @"refill_worker_count" : @(m.refill_worker_count),
        @"source_lease_verified" : @(m.source_lease_verified),
        @"drained" : @(m.drained),
    } mutableCopy];
    if (!m.layout_digest.empty())
        value[@"digest"] = @(m.layout_digest.c_str());
    if (m.receipt_schema_version) {
        value[@"receipt"] = @{
            @"schema_version" : @(m.receipt_schema_version),
            @"fills" : @(m.receipt_fills),
            @"groups_submitted" : @(m.receipt_groups_submitted),
            @"logical_read_bytes" : @(m.receipt_logical_read_bytes),
            @"reader_fences_issued" :
                @(m.receipt_reader_fences_issued),
            @"reader_fences_completed" :
                @(m.receipt_reader_fences_completed),
            @"source_generation" : @(m.receipt_source_generation),
            @"event_digest" : @(m.receipt_event_digest.c_str()),
            @"canonical_digest" : @(m.receipt_digest.c_str()),
            @"verifier_revision" :
                @(m.receipt_verifier_revision.c_str()),
        };
    }
    return value;
}

static NSDictionary *actual_streaming_stage(
        const StreamingStageRuntimeMetrics &stage) {
    return @{
        @"stage_index" : @(stage.stage_index),
        @"runtime" : actual_streaming_layout(stage.runtime),
    };
}

static NSDictionary *actual_streaming_boundary(
        const StreamingBoundaryRuntimeMetrics &boundary) {
    return @{
        @"boundary_index" : @(boundary.boundary_index),
        @"id" : @(boundary.id.c_str()),
        @"from_stage" : @(boundary.from_stage.c_str()),
        @"to_stage" : @(boundary.to_stage.c_str()),
        @"source_stage_drained" : @(boundary.source_stage_drained),
        @"source_stage_backing_released" :
            @(boundary.source_stage_backing_released),
        @"live_slot_bytes_before" : @(boundary.live_slot_bytes_before),
        @"live_slot_bytes_after" : @(boundary.live_slot_bytes_after),
        @"pending_readers_before" : @(boundary.pending_readers_before),
        @"pending_readers_after" : @(boundary.pending_readers_after),
        @"released_slot_bytes" : @(boundary.released_slot_bytes),
        @"event_digest" : @(boundary.event_digest.c_str()),
    };
}

static NSDictionary *actual_stage_receipt(
        const streaming::ActualStageReceipt &stage) {
    return @{
        @"stage_index" : @(stage.stage_index),
        @"stage_id" : @(stage.stage_id.c_str()),
        @"schema_version" : @(stage.schema_version),
        @"implementation" : @(stage.implementation.c_str()),
        @"layout_digest" : @(stage.layout_digest.c_str()),
        @"completed_passes" : @(stage.completed_passes),
        @"completed_groups" : @(stage.completed_groups),
        @"fills" : @(stage.fills),
        @"groups_submitted" : @(stage.groups_submitted),
        @"logical_read_bytes" : @(stage.logical_read_bytes),
        @"reader_fences_issued" : @(stage.reader_fences_issued),
        @"reader_fences_completed" : @(stage.reader_fences_completed),
        @"source_generation" : @(stage.source_generation),
        @"drain_completed" : @(stage.drain_completed),
        @"event_digest" : @(stage.event_digest.c_str()),
        @"canonical_digest" : @(stage.canonical_digest.c_str()),
    };
}

static NSDictionary *actual_boundary_receipt(
        const streaming::ActualBoundaryReceipt &boundary) {
    return @{
        @"boundary_index" : @(boundary.boundary_index),
        @"id" : @(boundary.id.c_str()),
        @"from_stage_index" : @(boundary.from_stage_index),
        @"to_stage_index" : @(boundary.to_stage_index),
        @"source_generation" : @(boundary.source_generation),
        @"last_reader_sequence" : @(boundary.last_reader_sequence),
        @"completed_reader_sequence" : @(boundary.completed_reader_sequence),
        @"live_slot_bytes_before" : @(boundary.live_slot_bytes_before),
        @"live_slot_bytes_after" : @(boundary.live_slot_bytes_after),
        @"released_slot_bytes" : @(boundary.released_slot_bytes),
        @"pending_readers_before" : @(boundary.pending_readers_before),
        @"pending_readers_after" : @(boundary.pending_readers_after),
        @"source_stage_drained" : @(boundary.source_stage_drained),
        @"source_stage_backing_released" :
            @(boundary.source_stage_backing_released),
        @"next_stage_started" : @(boundary.next_stage_started),
        @"event_digest" : @(boundary.event_digest.c_str()),
        @"canonical_digest" : @(boundary.canonical_digest.c_str()),
    };
}

static NSDictionary *actual_streaming_receipt(
        const streaming::ActualExecutionReceipt &receipt) {
    NSMutableArray *stages =
        [NSMutableArray arrayWithCapacity:receipt.stages.size()];
    for (const auto &stage : receipt.stages)
        [stages addObject:actual_stage_receipt(stage)];
    NSMutableArray *boundaries =
        [NSMutableArray arrayWithCapacity:receipt.boundaries.size()];
    for (const auto &boundary : receipt.boundaries)
        [boundaries addObject:actual_boundary_receipt(boundary)];
    return @{
        @"schema_version" : @(receipt.schema_version),
        @"implementation" : @(receipt.implementation.c_str()),
        @"layout_digest" : @(receipt.layout_digest.c_str()),
        @"component_policy_revision" :
            @(receipt.component_policy_revision.c_str()),
        @"stages" : stages,
        @"boundaries" : boundaries,
        @"canonical_digest" : @(receipt.canonical_digest.c_str()),
    };
}

static void attach_streaming_details(
        NSMutableDictionary *value, const RunResult &result) {
    if (!result.streaming_stages.empty()) {
        NSMutableArray *stages = [NSMutableArray
            arrayWithCapacity:result.streaming_stages.size()];
        for (const auto &stage : result.streaming_stages)
            [stages addObject:actual_streaming_stage(stage)];
        value[@"streaming_stages"] = stages;
    }
    if (!result.streaming_boundaries.empty()) {
        NSMutableArray *boundaries = [NSMutableArray
            arrayWithCapacity:result.streaming_boundaries.size()];
        for (const auto &boundary : result.streaming_boundaries)
            [boundaries addObject:actual_streaming_boundary(boundary)];
        value[@"streaming_boundaries"] = boundaries;
    }
    if (result.streaming_receipt)
        value[@"streaming_receipt"] = actual_streaming_receipt(
            *result.streaming_receipt);
}

static NSDictionary *public_streaming_result(
        const PublicStreamingSelectionMetrics &m) {
    return @{
        @"schema_version" : @1,
        @"target_request_memory_bytes" :
            @(m.target_request_memory_bytes),
        @"calibrated_request_bytes" : @(m.calibrated_request_bytes),
        @"preset_id" : @(m.preset_id.c_str()),
        @"preset_revision" : @(m.preset_revision),
        @"catalog_revision" : @(m.catalog_revision.c_str()),
        @"record_digest" : @(m.record_digest.c_str()),
        @"resolution_digest" : @(m.resolution_digest.c_str()),
        @"source_digest" : @(m.source_digest.c_str()),
        @"workload_digest" : @(m.workload_digest.c_str()),
        @"runtime_digest" : @(m.runtime_digest.c_str()),
        @"device_digest" : @(m.device_digest.c_str()),
        @"authorized_layout_digest" :
            @(m.authorized_layout_digest.c_str()),
        @"actual_layout_digest" : @(m.actual_layout_digest.c_str()),
        @"component_policy_revision" :
            @(m.component_policy_revision.c_str()),
        @"execution_container" : @(m.execution_container.c_str()),
        @"memory_scope" : @(m.memory_scope.c_str()),
        @"receipt_schema_version" : @(m.receipt_schema_version),
        @"receipt_source_generation" :
            @(m.receipt_source_generation),
        @"receipt_digest" : @(m.receipt_digest.c_str()),
        @"receipt_verifier_revision" :
            @(m.receipt_verifier_revision.c_str()),
        @"actual_plan_verified" : @(m.actual_plan_verified),
    };
}

static NSDictionary *to_dictionary(const BlockResidencyMetrics &);

NSDictionary *streaming_result_envelope(const RunResult &result) {
    require(result.public_streaming.has_value() &&
                result.public_streaming->actual_plan_verified,
            "public streaming finalizer envelope is unverified");
    require(result.streaming_receipt != nullptr &&
                !result.streaming_stages.empty(),
            "public streaming finalizer envelope is incomplete");
    NSMutableDictionary *value = [@{
        @"format" : @"turbocider-ltx-public-finalizer-envelope-v1",
        @"schema_version" : @1,
    } mutableCopy];
    value[@"public_streaming"] = public_streaming_result(
        *result.public_streaming);
    if (result.block_residency) {
        NSDictionary *residency = to_dictionary(*result.block_residency);
        value[@"block_residency"] = residency;
        NSMutableDictionary *block = [residency mutableCopy];
        const auto &first = result.streaming_stages.front().runtime;
        block[@"implementation"] = @(first.implementation.c_str());
        block[@"layout_digest"] = @(first.layout_digest.c_str());
        NSMutableArray *stages = [NSMutableArray
            arrayWithCapacity:result.streaming_stages.size()];
        for (const auto &stage : result.streaming_stages)
            [stages addObject:actual_streaming_stage(stage)];
        block[@"actual_layout"] = @{
            @"schema_version" : @3,
            @"digest" : @(first.layout_digest.c_str()),
            @"stage_count" : @(result.streaming_stages.size()),
            @"stages" : stages,
        };
        value[@"block_streaming"] = block;
    }
    attach_streaming_details(value, result);
    return value;
}
static NSDictionary *runtime_plan(const RunResult &result) {
    NSMutableDictionary *plan = [to_dictionary(result.plan) mutableCopy];
    if (!result.backend.empty())
        plan[@"backend"] = @(result.backend.c_str());
    if (!result.precision.empty())
        plan[@"precision"] = @(result.precision.c_str());
    plan[@"gpu_graph"] = gpu_graph_label(result);
    const bool runtime_unselected = result.request.hybrid_mlp_mode == "runtime" && result.hybrid &&
        result.hybrid->runtime_weight_backend.empty();
    const bool hybrid = result.request.execution == "gpu_ane" && !runtime_unselected;
    const bool encoder_hybrid = encoder_executed(result);
    plan[@"execution"] = hybrid ? @"gpu_ane_experimental" : @"gpu";
    plan[@"encoder_execution"] = encoder_hybrid ? @"gpu_ane_experimental" : @"gpu";
    plan[@"encoder_backend"] = encoder_backend_label(result);
    plan[@"encoder_gpu_graph"] =
        encoder_gpu_graph_label(result.request, encoder_hybrid);
    plan[@"encoder_precision"] = encoder_precision_label(result);
    if (result.hybrid && result.request.model != "z-image-turbo-gguf")
        plan[@"precision"] = @((result.precision.empty()
            ? hybrid_precision_label(*result.hybrid) : result.precision).c_str());
    plan[@"encoder_weight_validation"] = encoder_weight_validation_label(
        result.request, encoder_hybrid, true);
    if (result.request.model == "z-image-turbo-gguf") {
        plan[@"validation"] = hybrid ? @"checkpoint_bound_native_gguf_hybrid_candidate"
                                     : @"native_mlx_gguf_candidate";
        plan[@"weight_validation"] = hybrid
            ? @"GGUF checkpoint and Core ML manifest verified at execution"
            : @"GGUF loaded directly by native MLX";
        if (!result.request.loras.empty()) {
            const auto strategy = effective_lora_strategy(result.request);
            plan[@"lora_strategy"] = @(strategy.c_str());
            plan[@"lora_fusion"] = strategy == "inference_time"
                ? @"inference_time_low_rank"
                : @"in_memory_delta";
        }
        NSMutableArray *approximations = [NSMutableArray arrayWithObject:
            @"checkpoint_defined_gguf_weight_quantization"];
        if (hybrid)
            [approximations addObject:@"single_block_mlp_coreml_approximation"];
        if (encoder_hybrid)
            [approximations addObject:encoder_approximation_label(result.request)];
        plan[@"algorithm_approximations"] = approximations;
    }
    if (result.request.hybrid_mlp_mode == "runtime" && result.hybrid) {
        const auto &m = *result.hybrid;
        NSMutableArray *approximations = [plan[@"algorithm_approximations"] mutableCopy];
        [approximations removeObject:@"runtime_weight_fp16_token_row_ffn"];
        // The runtime graph's actual representation/axis, not environment
        // intent. Auto->Core ML fallback must not retain a W8/channel label.
        [approximations removeObject:@"single_block_mlp_coreml_approximation"];
        if (!m.runtime_weight_backend.empty()) {
            const bool channels = m.runtime_weight_partition_axis == "intermediate_channels";
            const bool w8 = m.runtime_weight_data_path == "w8a8_hadamard";
            const bool comfy = m.runtime_weight_data_path == "w8a8_convrot";
            [approximations addObject:comfy ?
                (channels ? @"runtime_weight_w8a8_convrot_channel_ffn" : @"runtime_weight_w8a8_convrot_token_row_ffn") : w8 ?
                (channels ? @"runtime_weight_w8a8_hadamard_channel_ffn" : @"runtime_weight_w8a8_hadamard_token_row_ffn") :
                (channels ? @"runtime_weight_fp16_channel_ffn" : @"runtime_weight_fp16_token_row_ffn")];
        }
        plan[@"algorithm_approximations"] = approximations;
        plan[@"runtime_weight_contract"] = @{
            @"executor_backend": m.runtime_weight_backend.empty() ? (id)NSNull.null : @(m.runtime_weight_backend.c_str()),
            @"data_path": @(m.runtime_weight_data_path.c_str()),
            @"partition_axis": @(m.runtime_weight_partition_axis.c_str()),
            @"source": @"selected_executor_receipt_not_environment_or_physical_placement",
        };
    }
    if (result.streaming_runtime) {
        NSDictionary *actual = actual_streaming_layout(
            *result.streaming_runtime);
        NSMutableDictionary *streaming =
            [plan[@"streaming"] isKindOfClass:NSDictionary.class]
                ? [plan[@"streaming"] mutableCopy]
                : [NSMutableDictionary dictionary];
        const bool public_execution = result.public_streaming.has_value();
        streaming[@"eligibility"] = public_execution
            ? @"public_reviewed_preset" : @"experimental_candidate";
        streaming[@"execution_supported"] = @YES;
        streaming[@"rejection_code"] = NSNull.null;
        streaming[@"resolution_state"] = @"executed_exact_layout";
        streaming[@"resolved_layout"] = actual;
        streaming[@"actual_layout"] = actual;
        streaming[@"enforcement"] = @"exact_layout";
        streaming[@"authority"] = public_execution
            ? @"public_preset_authority"
            : @"private_candidate_constructor";
        plan[@"streaming"] = streaming;
        plan[@"executable"] = @YES;
    }
    return plan;
}
NSDictionary *to_dictionary(const LoadResult &r) {
    return @{
        @"scope" : @"image_weights; text encoder loads on demand",
        @"weight_bytes" : @(r.weight_bytes),
        @"mlx_active_bytes" : @(r.active_bytes)
    };
}
static id calibration_number(double value) {
    return std::isfinite(value) ? (id)@(value) : (id)NSNull.null;
}
static NSArray *calibration_values(const std::vector<double> &values) {
    NSMutableArray *result=[NSMutableArray arrayWithCapacity:values.size()];
    for(double value:values)[result addObject:calibration_number(value)];
    return result;
}
static NSDictionary *calibration_dictionary(const ane::ChannelCalibrationReport &r) {
    id identity=NSNull.null,baseline=NSNull.null,trial=NSNull.null;
    if(r.identity) {
        const auto &k=*r.identity;
        identity=@{@"model_sha256":@(k.model_sha256.c_str()),@"adapter":@(k.adapter.c_str()),
            @"encoding":@(k.encoding.c_str()),@"precision":@(k.precision.c_str()),@"backend":@(k.backend.c_str()),
            @"recipe":@(k.recipe.c_str()),@"soc":@(k.soc.c_str()),@"os_build":@(k.os_build.c_str()),
            @"runtime_build":@(k.runtime_build.c_str()),@"metal_abi":@(k.metal_abi.c_str()),@"graph_abi":@(k.graph_abi.c_str()),
            @"source_generation":@(k.source_generation.c_str()),@"rows":@(k.rows),@"hidden":@(k.hidden),
            @"width":@(k.width),@"tile_k":@(k.tile_k),@"tile_n":@(k.tile_n),@"prefetch":@(k.prefetch)};
    }
    if(r.baseline)baseline=@{@"layer_seconds":calibration_number(r.baseline->layer_seconds),
        @"raw_seconds":@[calibration_values(r.baseline->seconds[0]),calibration_values(r.baseline->seconds[1])]};
    NSMutableArray *points=[NSMutableArray arrayWithCapacity:r.points.size()];
    for(const auto &sample:r.points) {
        NSMutableArray *raw=[NSMutableArray arrayWithCapacity:3];
        for(const auto &part:sample.seconds)[raw addObject:@[calibration_values(part[0]),calibration_values(part[1])]];
        [points addObject:@{@"share":calibration_number(sample.point.share),@"gpu":calibration_number(sample.point.gpu),
            @"ane":calibration_number(sample.point.ane),@"both":calibration_number(sample.point.both),
            @"prefetch":@(sample.prefetch),@"ane_calls":@(sample.ane_calls),@"raw_seconds":raw}];
        NSMutableDictionary *point=[points.lastObject mutableCopy];
        point[@"correction_computations"]=@(sample.correction_computations);
        point[@"correction_uploads"]=@(sample.correction_uploads);
        points[points.count-1]=point;
    }
    if(r.trial) {
        const auto &t=*r.trial;
        trial=@{@"gpu_seconds":calibration_values(t.gpu_seconds),@"candidate_seconds":calibration_values(t.candidate_seconds),
            @"calls":@(t.calls),@"fallbacks":@(t.fallbacks),@"retries":@(t.retries),
            @"relative_l2":calibration_number(t.relative_l2),@"cosine":calibration_number(t.cosine),
            @"completed":@(t.completed),@"accepted":@(t.accepts())};
    }
    NSMutableArray *depth=[NSMutableArray arrayWithCapacity:r.sampled_depths.size()];
    for(int value:r.sampled_depths)[depth addObject:@(value)];
    NSMutableArray *admissions=[NSMutableArray arrayWithCapacity:r.memory_admissions.size()];
    for(const auto &m:r.memory_admissions) [admissions addObject:@{
        @"channels":@(m.channels),@"surface_bytes":@(m.surface_bytes),
        @"gpu_scratch_upper_bytes":@(m.gpu_scratch_upper_bytes),@"gpu_restore_bytes":@(m.gpu_restore_bytes),
        @"input_bytes":@(m.input_bytes),@"internal_allowance_bytes":@(m.internal_allowance_bytes),
        @"estimated_bytes":@(m.estimated_bytes),@"optional_limit_bytes":@(m.optional_limit_bytes),
        @"headroom_bytes":@(m.headroom_bytes),@"admitted":@(m.admitted),@"reason":@(m.reason.c_str())}];
    return @{@"schema_version":@(r.schema_version),@"enabled":@(r.enabled),@"cache_hit":@(r.cache_hit),
        @"gpu_retention_layers":@(r.gpu_retention_layers),@"memory_limited_points":@(r.memory_limited_points),
        @"memory_admission_scope":@"complete calibration payload estimate; preflight observation; not physical RAM cap",
        @"memory_admissions":admissions,
        @"trial_passed":@(r.trial_passed),@"complete":@(r.complete),@"lora":@(r.lora),@"selected_channels":@(r.selected_channels),
        @"proposed_channels":@(r.proposed_channels),@"bucket_rows":@(r.bucket_rows),@"actual_rows":@(r.actual_rows),
        @"hidden":@(r.hidden),@"width":@(r.width),@"layer_count":@(r.layer_count),@"warmups":@(r.warmups),
        @"repeats":@(r.repeats),@"status":@(r.status.c_str()),@"reason":@(r.reason.c_str()),@"scope":@(r.scope.c_str()),
        @"input_recipe":@(r.input_recipe.c_str()),@"sampled_depths":depth,@"identity":identity,
        @"baseline":baseline,@"points":points,@"trial":trial,@"predicted_layer_seconds":calibration_number(r.predicted_layer_seconds)};
}
NSDictionary *to_dictionary(const HybridMetrics &m) {
    // Runtime provenance/counters belong to the graph contract, not a
    // particular precision label. W8A8 and FP16 share the same receipt.
    const bool runtime_weight = m.mlp_output_kind == "runtime_weight_swiglu" ||
                                m.mlp_output_kind == "runtime_weight_swiglu_lora_inputs";
    NSMutableArray *overflow_events=[NSMutableArray arrayWithCapacity:m.runtime_weight_overflow_events.size];
    NSMutableArray *gpu_layers=[NSMutableArray arrayWithCapacity:m.runtime_weight_gpu_layers.size()];
    for(int layer:m.runtime_weight_gpu_layers)[gpu_layers addObject:@(layer)];
    for(size_t i=0;i<m.runtime_weight_overflow_events.size;++i) {
        const auto &e=m.runtime_weight_overflow_events.events[i];
        [overflow_events addObject:@{@"layer":@(e.layer),@"rows":@(e.rows),
            @"runtime_call_begin":@(e.runtime_call_begin),@"runtime_call_count":@(e.runtime_call_count),
            @"retries":@(e.retries),@"headroom_before":calibration_number(e.headroom_before),
            @"headroom_after":calibration_number(e.headroom_after),@"completed":@(e.completed)}];
    }
    return @{
        @"runtime_weight" : runtime_weight ? @{
            @"channel_calibration" : m.runtime_weight_calibration ? (id)calibration_dictionary(*m.runtime_weight_calibration) : (id)NSNull.null,
            @"executor_backend" : m.runtime_weight_backend.empty() ? [NSNull null] : @(m.runtime_weight_backend.c_str()),
            @"backend_fallback_reason" : @(m.runtime_weight_backend_fallback_reason.c_str()),
            @"io_path" : @(m.runtime_weight_io_path.c_str()),
            @"data_path" : @(m.runtime_weight_data_path.c_str()),
            @"partition_axis" : @(m.runtime_weight_partition_axis.c_str()),
            @"row_placement" : @(m.runtime_weight_row_placement.c_str()),
            @"row_suffix_blocks_session_total" : @(m.runtime_weight_row_suffix_blocks),
            @"row_prefix_blocks_session_total" : @(m.runtime_weight_row_prefix_blocks),
            @"row_image_tail_blocks_session_total" : @(m.runtime_weight_row_image_tail_blocks),
            @"row_protected_rows_session_total" : @(m.runtime_weight_row_protected_rows),
            @"row_pack_peak_logical_bytes" : @(m.runtime_weight_row_pack_peak_bytes),
            @"ane_channels" : @(m.runtime_weight_ane_channels),
            @"gpu_channels" : @(m.runtime_weight_gpu_channels),
            @"channel_blocks_session_total" : @(m.runtime_weight_channel_blocks),
            @"prefetch_enabled" : @(m.runtime_weight_prefetch_enabled),
            @"prefetch_after_gpu" : @(m.runtime_weight_prefetch_after_gpu),
            @"prefetch_submissions_session_total" : @(m.runtime_weight_prefetch_submissions),
            @"prefetch_hits_session_total" : @(m.runtime_weight_prefetch_hits),
            @"prefetch_discards_session_total" : @(m.runtime_weight_prefetch_discards),
            @"prefetch_failures_session_total" : @(m.runtime_weight_prefetch_failures),
            @"prefetch_wait_seconds_session_total" : @(m.runtime_weight_prefetch_wait_seconds),
            @"scale_cache_enabled" : @(m.runtime_weight_scale_cache_enabled),
            @"stage_specialized" : @(m.runtime_weight_stage_specialized),
            @"stage_pipeline_variants" : @(m.runtime_weight_stage_pipeline_variants),
            @"launch_fence_enabled" : @(m.runtime_weight_launch_fence_enabled),
            @"a8_lookahead_enabled" : @(m.runtime_weight_a8_lookahead_enabled),
            @"fp32_channel_join_enabled" : @(m.runtime_weight_fp32_channel_join_enabled),
            @"a8_group_size" : @(m.runtime_weight_a8_group_size),
            @"hidden_a8_group_size" : @(m.runtime_weight_hidden_a8_group_size),
            @"a8_prefetches_session_total" : @(m.runtime_weight_a8_prefetches),
            @"a8_wait_seconds_session_total" : @(m.runtime_weight_a8_wait_seconds),
            @"scale_cache_hits_session_total" : @(m.runtime_weight_scale_cache_hits),
            @"scale_cache_misses_session_total" : @(m.runtime_weight_scale_cache_misses),
            @"scale_cache_entries" : @(m.runtime_weight_scale_cache_entries),
            @"scale_cache_bytes" : @(m.runtime_weight_scale_cache_bytes),
            @"scale_cache_evictions_session_total" : @(m.runtime_weight_scale_cache_evictions),
            @"weight_code_cache" : @{
                @"enabled" : @(m.runtime_weight_code_cache.enabled),
                @"native_surface_storage" : @(m.runtime_weight_code_cache.native_surface_storage),
                @"budget_bytes" : @(m.runtime_weight_code_cache.budget_bytes),
                @"hits_session_total" : @(m.runtime_weight_code_cache.hits),
                @"copy_hits_session_total" : @(m.runtime_weight_code_cache.copy_hits),
                @"surface_bind_hits_session_total" : @(m.runtime_weight_code_cache.surface_bind_hits),
                @"misses_session_total" : @(m.runtime_weight_code_cache.misses),
                @"fills_session_total" : @(m.runtime_weight_code_cache.fills),
                @"failed_fills_session_total" : @(m.runtime_weight_code_cache.failed_fills),
                @"entries" : @(m.runtime_weight_code_cache.entries),
                @"ready_entries" : @(m.runtime_weight_code_cache.ready_entries),
                @"retained_bytes" : @(m.runtime_weight_code_cache.retained_bytes),
                @"live_capacity_bytes" : @(m.runtime_weight_code_cache.live_capacity_bytes),
                @"peak_capacity_bytes" : @(m.runtime_weight_code_cache.peak_capacity_bytes),
                @"evictions_session_total" : @(m.runtime_weight_code_cache.evictions),
                @"declines_session_total" : @(m.runtime_weight_code_cache.declines),
                @"ineligible_session_total" : @(m.runtime_weight_code_cache.ineligible),
                @"policy" : @"first-admitted-live-generations-v1",
                @"scope" : @"completed converted-code copies or validated immutable ready-surface bindings, separately counted; weak generations; capacity follows producer and surface readers; not physical overlap or whole-process RAM"
            },
            @"device_io_calls_session_total" : @(m.runtime_weight_device_io_calls),
            @"lora_channel_range_calls_session_total" : @(m.runtime_weight_lora_channel_range_calls),
            @"lora_channel_full_calls_session_total" : @(m.runtime_weight_lora_channel_full_calls),
            @"split_down_rank_blocks_session_total" : @(m.runtime_weight_down_rank_blocks),
            @"split_down_rank_arrays_session_total" : @(m.runtime_weight_down_rank_arrays),
            @"slot_bytes" : @(m.runtime_weight_slot_bytes),
            @"estimated_bytes" : @(m.runtime_weight_estimated_bytes),
            @"hybrid_blocks_session_total" : @(m.runtime_weight_hybrid_blocks),
            @"gpu_blocks_session_total" : @(m.runtime_weight_gpu_blocks),
            @"unsplit_gpu_blocks_session_total" : @(m.runtime_weight_unsplit_gpu_blocks),
            @"full_gpu_probe_blocks_session_total" : @(m.runtime_weight_full_gpu_probe_blocks),
            @"untimed_hybrid_blocks_session_total" : @(m.runtime_weight_untimed_hybrid_blocks),
            @"async_hybrid_blocks_session_total" : @(m.runtime_weight_async_hybrid_blocks),
            @"deferred_channel_join_enabled" : @(m.runtime_weight_deferred_join_enabled),
            @"deferred_channel_join_blocks_session_total" : @(m.runtime_weight_deferred_join_blocks),
            @"post_join_scope" : !m.runtime_weight_deferred_join_blocks ? @"evaluated_join_host_span"
                : m.runtime_weight_deferred_join_blocks == m.runtime_weight_channel_blocks
                    ? @"host_graph_construction_deferred_gpu_consumption" : @"mixed_evaluated_and_deferred_join_spans",
            @"full_gpu_probe_seconds_session_total" : @(m.runtime_weight_full_gpu_probe_seconds),
            @"fallback_blocks_session_total" : @(m.runtime_weight_fallback_blocks),
            @"ane_rows_session_total" : @(m.runtime_weight_ane_rows),
            @"overflow_retries_session_total" : @(m.runtime_weight_overflow_retries),
            @"overflow_events" : overflow_events,
            @"overflow_events_dropped_session_total" : @(m.runtime_weight_overflow_events.dropped),
            @"overflow_event_scope" : @"aggregated per-FFN launch host telemetry; no chunk/physical-engine trace",
            @"requested_gpu_layers" : gpu_layers,
            @"forced_gpu_blocks_session_total" : @(m.runtime_weight_forced_gpu_blocks),
            @"headroom_scale" : @(m.runtime_weight_headroom),
            @"source_recipe" : @(m.runtime_weight_source_recipe.c_str()),
            @"convrot_stage_submissions_session_total" : @(m.runtime_weight_convrot_stage_submissions),
            @"stage_seconds_session_total" : @(m.runtime_weight_stage_seconds),
            @"stage_wait_seconds_session_total" : @(m.runtime_weight_stage_wait_seconds),
            @"join_seconds_session_total" : @(m.runtime_weight_join_seconds),
            @"gpu_ffn_seconds_session_total" : @(m.runtime_weight_gpu_seconds),
            @"async_ane_wait_seconds_session_total" : @(m.runtime_weight_async_ane_wait_seconds),
            @"lora_gate_up_seconds_session_total" : @(m.runtime_weight_lora_gate_up_seconds),
            @"lora_input_ready_seconds_session_total" : @(m.runtime_weight_lora_input_ready_seconds),
            @"post_join_seconds_session_total" : @(m.runtime_weight_post_join_seconds),
            @"hybrid_ffn_seconds_session_total" : @(m.runtime_weight_wall_seconds),
            @"pre_ffn_seconds_session_total" : @(m.runtime_weight_pre_seconds),
            @"failure_reason" : @(m.prefill_plan_reason.c_str())
        } : (id)[NSNull null],
        @"weight_variant" : @(m.weight_variant.c_str()),
        @"load_seconds" : @(m.load_seconds),
        @"manifest_validation_seconds" : @(m.manifest_validation_seconds),
        @"output_backing_setup_seconds" : @(m.output_backing_setup_seconds),
        @"model_load_seconds" : @(m.model_load_seconds),
        @"model_interface_setup_seconds" : @(m.model_interface_setup_seconds),
        @"zero_input_warmup_seconds" : @(m.zero_input_warmup_seconds),
        @"prediction_seconds_session_total" : @(m.prediction_seconds),
        @"feature_binding_seconds_session_total" : @(m.feature_binding_seconds),
        @"model_prediction_seconds_session_total" : @(m.model_prediction_seconds),
        @"output_handling_seconds_session_total" : @(m.output_handling_seconds),
        @"first_runtime_prediction_seconds_session_total" :
            @(m.first_runtime_prediction_seconds),
        @"subsequent_runtime_prediction_seconds_session_total" :
            @(m.subsequent_runtime_prediction_seconds),
        @"calls_session_total" : @(m.calls),
        @"warmup_calls_session_total" : @(m.warmup_calls),
        @"runtime_calls_session_total" : @(m.runtime_calls),
        @"first_runtime_prediction_calls_session_total" :
            @(m.first_runtime_prediction_calls),
        @"subsequent_runtime_prediction_calls_session_total" :
            @(m.subsequent_runtime_prediction_calls),
        @"runtime_failures_session_total" : @(m.runtime_failures),
        @"runtime_failed" : @(m.runtime_failed),
        @"runtime_failure_block" : @(m.runtime_failure_block),
        @"bucket" : @(m.bucket),
        @"minimum_profitable_rows" : @(m.minimum_profitable_rows),
        @"hidden" : @(m.hidden),
        @"output_channels" : @(m.output_channels),
        @"mlp_output_kind" : @(m.mlp_output_kind.c_str()),
        @"block_count" : @(m.block_count),
        @"mlp_width" : @(m.mlp_width),
        @"ane_mlp_range" : @[ @(m.ane_mlp_start), @(m.ane_mlp_end) ],
        @"output_scale" : @(m.output_scale),
        @"qualified_flexible_backing" : @(m.qualified_flexible_backing),
        @"compute_units" : m.runtime_weight_backend == "private_ane" ? @"not_applicable_private_ane_client" : @"cpuAndNeuralEngine",
        @"observed_ane_residency" : @"unknown",
        @"output_copy_bytes_session_total" : @(m.copied_bytes),
        @"checkpoint_sha256_verified" : @(m.checkpoint_sha_verified),
        @"lora_identity_verified" : @(m.lora_identity_verified),
        @"session_released_after_encoding" : @(m.session_released_after_encoding),
        @"quality_validation_enabled" : @(m.quality_validation_calls > 0),
        @"quality_validation_calls_session_total" : @(m.quality_validation_calls),
        @"quality_max_relative_l2_session" : @(m.quality_max_relative_l2),
        @"quality_min_cosine_session" : @(m.quality_min_cosine),
        @"quality_max_abs_session" : @(m.quality_max_abs),
        @"quality_max_relative_abs_session" : @(m.quality_max_relative_abs),
        @"quality_validation_passed" : @(m.quality_validation_passed),
        @"prefill_actual_tokens" : @(m.prefill_actual_tokens),
        @"prefill_selected_bucket" : @(m.prefill_selected_bucket),
        @"prefill_compute_tokens" : @(m.prefill_compute_tokens),
        @"prefill_padding_tokens" : @(m.prefill_padding_tokens),
        @"prefill_fixed_shape" : @(m.prefill_fixed_shape),
        @"prefill_plan_reason" : @(m.prefill_plan_reason.c_str()),
        @"provenance" : runtime_weight
            ? @"checkpoint-independent graph receipt; weights supplied by the loaded model at runtime"
            : m.checkpoint_sha_verified
            ? @"local checkpoint path, size and SHA-256 verified"
            : @"local checkpoint path+size; source SHA absent in legacy artifact; experimental only"
    };
}
static NSDictionary *qkv_dictionary(const QkvMetrics &m) {
    return @{
        @"mode" : @"runtime_weight_qkv",
        @"auto_scheduling" : @(m.auto_scheduling),
        @"calls_session_total" : @(m.calls),
        @"hybrid_blocks_session_total" : @(m.hybrid_blocks),
        @"gpu_blocks_session_total" : @(m.gpu_blocks),
        @"gpu_probe_blocks_session_total" : @(m.gpu_probe_blocks),
        @"measured_hybrid_blocks_session_total" : @(m.measured_hybrid_blocks),
        @"gpu_probe_seconds_session_total" : @(m.gpu_probe_seconds),
        @"measured_hybrid_seconds_session_total" : @(m.measured_hybrid_seconds),
        @"fallback_blocks_session_total" : @(m.fallback_blocks),
        @"ane_rows_session_total" : @(m.ane_rows),
        @"failures_session_total" : @(m.failures),
        @"failure_block" : @(m.failure_block),
        @"failed" : @(m.failed),
        @"failure_reason" : @(m.failure_reason.c_str()),
        @"chunk_rows" : @(m.chunk_rows),
        @"chunks" : @(m.chunks),
        @"slot_bytes" : @(m.slot_bytes),
        @"estimated_bytes" : @(m.estimated_bytes),
        @"load_seconds" : @(m.load_seconds),
        @"stage_seconds_session_total" : @(m.stage_seconds),
        @"stage_wait_seconds_session_total" : @(m.stage_wait_seconds),
        @"prediction_seconds_session_total" : @(m.prediction_seconds),
        @"output_seconds_session_total" : @(m.output_seconds),
        // Includes lazy upstream input readiness; not a pure QKV kernel timer.
        @"bridge_wall_seconds_session_total" : @(m.wall_seconds),
        @"compute_units" : @"cpuAndNeuralEngine",
        @"observed_ane_residency" : @"unknown"
    };
}
static NSDictionary *to_dictionary(const BlockResidencyMetrics &m) {
    return @{
        @"policy" : @"shared_block_residency_v1",
        @"enabled" : @(m.enabled),
        @"fully_resident" : @(m.fully_resident),
        @"quantized" : @(m.quantized),
        @"active_blocks" : @(m.active_blocks),
        @"pinned_blocks" : @(m.pinned_blocks),
        @"streamed_blocks" : @(m.streamed_blocks),
        @"refill_slots" : @(m.refill_slots),
        @"memory_budget_bytes" : @(m.memory_budget_bytes),
        @"activation_reserve_bytes" : @(m.activation_reserve_bytes),
        @"block_bytes" : @(m.block_bytes),
        @"estimated_working_set_bytes" : @(m.estimated_working_set_bytes),
        @"request_bytes_loaded" : @(m.request_bytes_loaded),
        @"request_slot_allocations" : @(m.request_slot_allocations),
        @"request_slot_refills" : @(m.request_slot_refills),
        @"request_slot_fills" : @(m.request_slot_fills),
        @"request_load_seconds" : @(m.request_load_seconds),
        @"request_wait_seconds" : @(m.request_wait_seconds),
        @"request_refill_load_seconds" : @(m.request_refill_load_seconds),
        @"request_max_refill_seconds" : @(m.request_max_refill_seconds),
        @"request_max_refill_block" : @(m.request_max_refill_block),
        @"mlp_prefix_channels" : @(m.mlp_prefix_channels),
        @"suffix_pack_bytes" : @(m.suffix_pack_bytes),
        @"request_pack_read_bytes" : @(m.request_pack_read_bytes),
        @"request_pack_write_bytes" : @(m.request_pack_write_bytes),
        @"request_pack_seconds" : @(m.request_pack_seconds),
    };
}
static NSDictionary *to_dictionary(const MemoryAdmissionMetrics &m) {
    return @{
        @"budget_bytes" : @(m.budget_bytes),
        @"planned_increment_bytes" : @(m.planned_increment_bytes),
        @"framework_upper_bytes" : @(m.framework_upper_bytes),
        @"planned_process_upper_bytes" :
            @(m.planned_process_upper_bytes),
        @"allocation_ceiling_bytes" : @(m.allocation_ceiling_bytes),
        @"ledger_budget_bytes" : @(m.ledger_budget_bytes),
        @"ledger_process_baseline_bytes" :
            @(m.ledger_process_baseline_bytes),
        @"ledger_storage_bytes" : @(m.ledger_storage_bytes),
        @"ledger_known_bytes" : @(m.ledger_known_bytes),
        @"ledger_committed_bytes" : @(m.ledger_committed_bytes),
        @"ledger_active_bytes" : @(m.ledger_active_bytes),
        @"ledger_reserved_bytes" : @(m.ledger_reserved_bytes),
        @"ledger_pending_release_bytes" :
            @(m.ledger_pending_release_bytes),
        @"ledger_cached_bytes" : @(m.ledger_cached_bytes),
        @"ledger_unknown_bytes" : @(m.ledger_unknown_bytes),
        @"ledger_peak_unknown_bytes" : @(m.ledger_peak_unknown_bytes),
        @"initial_process_footprint_bytes" :
            @(m.initial_process_footprint_bytes),
        @"peak_process_footprint_bytes" : @(m.peak_process_footprint_bytes),
        @"final_process_footprint_bytes" : @(m.final_process_footprint_bytes),
        @"unattributed_process_footprint_bytes" :
            @(m.unattributed_process_footprint_bytes),
        @"peak_unattributed_process_footprint_bytes" :
            @(m.peak_unattributed_process_footprint_bytes),
        @"peak_observed_over_budget_bytes" :
            @(m.peak_observed_over_budget_bytes),
        @"system_available_bytes_at_admission" :
            @(m.system_available_bytes_at_admission),
        @"final_system_available_bytes" :
            @(m.final_system_available_bytes),
        @"minimum_system_available_bytes" :
            @(m.minimum_system_available_bytes),
        @"ledger_peak_committed_bytes" : @(m.ledger_peak_committed_bytes),
        @"ledger_storage_count" : @(m.ledger_storage_count),
        @"ledger_site_allocation_count" :
            @(m.ledger_site_allocation_count),
        @"ledger_site_reserved_count" : @(m.ledger_site_reserved_count),
        @"ledger_site_active_count" : @(m.ledger_site_active_count),
        @"ledger_site_pending_count" : @(m.ledger_site_pending_count),
        @"ledger_site_cached_count" : @(m.ledger_site_cached_count),
        @"observation_count" : @(m.observation_count),
        @"observed_within_budget" : @(m.observed_within_budget),
        @"tainted" : @(m.tainted),
        @"worker_quarantined" : @(m.worker_quarantined),
        @"pressure_transitions" : @(m.pressure_transitions),
        @"pressure_state" : @(m.pressure_state.c_str()),
        @"execution_state" : @(m.execution_state.c_str()),
        @"watchdog_sample_count" : @(m.watchdog_sample_count),
        @"watchdog_dropped_samples" : @(m.watchdog_dropped_samples),
        @"watchdog_critical" : @(m.watchdog_critical),
        @"watchdog_failure_reason" : m.watchdog_failure_reason.empty()
            ? (id)[NSNull null] : @(m.watchdog_failure_reason.c_str()),
        @"trace_event_count" : @(m.trace_event_count),
        @"trace_dropped_events" : @(m.trace_dropped_events),
        @"trace_overflowed" : @(m.trace_overflowed),
        @"explicit_epoch_transition_count" :
            @(m.explicit_epoch_transition_count),
        @"automatic_epoch_transition_count" :
            @(m.automatic_epoch_transition_count),
        @"current_epoch" : @(m.current_epoch),
        @"planned_peak_epoch" : @(m.planned_peak_epoch),
        @"schedule_event_count" : @(m.schedule_event_count),
        @"schedule_event_attempted_count" :
            @(m.schedule_event_attempted_count),
        @"schedule_event_rejected_count" :
            @(m.schedule_event_rejected_count),
        @"schedule_expected_event_count" :
            @(m.schedule_expected_event_count),
        @"schedule_mismatch_count" : @(m.schedule_mismatch_count),
        @"schedule_next_sequence" : @(m.schedule_next_sequence),
        @"schedule_cursor_state" : @(m.schedule_cursor_state.c_str()),
        @"schedule_first_failure" : m.schedule_first_failure.empty()
            ? (id)[NSNull null] : @(m.schedule_first_failure.c_str()),
        @"schedule_cursor_complete" : @(m.schedule_cursor_complete),
        @"swap_observation_available" : @(m.swap_observation_available),
        @"swap_counter_invalid" : @(m.swap_counter_invalid),
        @"swap_activity_detected" : @(m.swap_activity_detected),
        @"swap_sample_count" : @(m.swap_sample_count),
        @"swapins_begin" : @(m.swapins_begin),
        @"swapins_end" : @(m.swapins_end),
        @"swapins_delta" : @(m.swapins_delta),
        @"swapouts_begin" : @(m.swapouts_begin),
        @"swapouts_end" : @(m.swapouts_end),
        @"swapouts_delta" : @(m.swapouts_delta),
        @"compressed_pages_begin" : @(m.compressed_pages_begin),
        @"compressed_pages_end" : @(m.compressed_pages_end),
        @"swap_first_observed_phase" :
            m.swap_first_observed_phase.empty()
                ? (id)[NSNull null]
                : @(m.swap_first_observed_phase.c_str()),
        @"swap_observation_source" :
            m.swap_observation_source.empty()
                ? (id)[NSNull null]
                : @(m.swap_observation_source.c_str()),
        @"failure_disposition" : @(m.failure_disposition.c_str()),
        @"failure_reason" : m.failure_reason.empty()
            ? (id)[NSNull null] : @(m.failure_reason.c_str()),
        @"cleanup_failure" : m.cleanup_failure.empty()
            ? (id)[NSNull null] : @(m.cleanup_failure.c_str()),
        @"last_phase" : @(m.last_phase.c_str()),
        @"observation_source" : @(m.observation_source.c_str()),
    };
}
static NSDictionary *to_dictionary(const MemoryTraceEvent &event) {
    return @{
        @"sequence" : @(event.sequence),
        @"monotonic_ns" : @(event.monotonic_ns),
        @"kind" : @(memory_trace_event_kind_name(event.kind)),
        @"phase_id" : @(event.phase_id),
        @"site_id" : @(event.site_id),
        @"epoch" : @(event.epoch),
        @"upper_bytes" : @(event.upper_bytes),
        @"actual_bytes" : @(event.actual_bytes),
        @"committed_bytes" : @(event.committed_bytes),
        @"reserved_bytes" : @(event.reserved_bytes),
        @"pending_bytes" : @(event.pending_bytes),
        @"stage_id" : @(event.stage_id),
        @"slot_id" : @(event.slot_id),
        @"status" : @(event.status),
    };
}
static NSArray *to_array(const std::vector<MemoryTraceEvent> &events) {
    NSMutableArray *result = [NSMutableArray arrayWithCapacity:events.size()];
    for (const auto &event : events)
        [result addObject:to_dictionary(event)];
    return result;
}
NSDictionary *to_dictionary(const MemoryExecutionReport &report) {
    return @{
        @"memory_admission" : to_dictionary(report.metrics),
        @"memory_trace" : to_array(report.trace),
    };
}
RunResult native_run_result(NSDictionary *value, const Request &request,
                            const ExecutionPlan &plan) {
    require([value isKindOfClass:NSDictionary.class], "native session returned an invalid result");
    RunResult result;
    result.request = request;
    result.plan = plan;
    result.native_json = json(value);
    result.selection = request.execution;
    result.actual_steps = request.steps;
    id timings = value[@"timings_seconds"];
    if ([timings isKindOfClass:NSDictionary.class])
        result.timings.wall = [timings[@"request_wall"] doubleValue];
    if (!result.timings.wall && [value[@"seconds"] isKindOfClass:NSNumber.class])
        result.timings.wall = [value[@"seconds"] doubleValue];
    result.active_bytes = [value[@"mlx_active_bytes"] unsignedLongLongValue];
    result.peak_bytes = [value[@"mlx_peak_bytes"] unsignedLongLongValue];
    result.warmup = [value[@"warmup"] boolValue];
    result.prepared = [value[@"prepared"] boolValue];
    result.prompt_cache_hit = [value[@"prompt_cache_hit"] boolValue] ||
                              [value[@"conditioning_cache_hit"] boolValue];
    result.text_tokens = [value[@"text_tokens"] intValue];
    result.valid_text_tokens = [value[@"valid_text_tokens"] intValue];
    result.total_tokens = [value[@"total_tokens"] intValue];
    result.reference_tokens = [value[@"reference_tokens"] intValue];
    return result;
}
NSDictionary *to_dictionary(const RunResult &result) {
    if (!result.native_json.empty()) {
        auto value = parse_json(result.native_json.c_str());
        NSMutableDictionary *copy = [value mutableCopy];
        if (!copy[@"model"]) copy[@"model"] = @(result.request.model.c_str());
        if (!copy[@"plan"]) copy[@"plan"] = to_dictionary(result.plan);
        if (!copy[@"execution"]) copy[@"execution"] = @(result.request.execution.c_str());
        if (!copy[@"lora_strategy"])
            copy[@"lora_strategy"] = @(effective_lora_strategy(result.request).c_str());
        if (result.block_residency)
            copy[@"block_residency"] = to_dictionary(*result.block_residency);
        if (result.memory_admission)
            copy[@"memory_admission"] = to_dictionary(*result.memory_admission);
        if (!result.memory_trace.empty())
            copy[@"memory_trace"] = to_array(result.memory_trace);
        if (result.memory_admission)
            copy[@"plan"] = runtime_plan(result);
        if (result.encoder_hybrid) {
            copy[@"encoder_execution"] = encoder_executed(result) ? @"gpu_ane_experimental" : @"gpu";
            copy[@"encoder_runtime_backend"] = encoder_backend_label(result);
            copy[@"encoder_gpu_graph"] =
                encoder_gpu_graph_label(result.request, encoder_executed(result));
            copy[@"encoder_runtime_precision"] = encoder_precision_label(result);
            copy[@"encoder_hybrid"] = to_dictionary(*result.encoder_hybrid);
            copy[@"encoder_runtime_reuse"] = encoder_reuse_dictionary(result);
            copy[@"encoder_weight_residency"] = encoder_weight_dictionary(result);
            copy[@"plan"] = runtime_plan(result);
        }
        if (result.streaming_runtime) {
            copy[@"plan"] = runtime_plan(result);
            const auto &runtime = *result.streaming_runtime;
            NSMutableDictionary *block = result.block_residency
                ? [to_dictionary(*result.block_residency) mutableCopy]
                : [NSMutableDictionary dictionary];
            block[@"implementation"] = @(runtime.implementation.c_str());
            block[@"layout_digest"] = runtime.layout_digest.empty()
                ? (id)NSNull.null : (id)@(runtime.layout_digest.c_str());
            block[@"actual_layout"] = actual_streaming_layout(runtime);
            copy[@"block_streaming"] = block;
        }
        if (result.public_streaming)
            copy[@"public_streaming"] = public_streaming_result(
                *result.public_streaming);
        attach_streaming_details(copy, result);
        if (result.shared_lora_ranks)
            copy[@"shared_lora_ranks"] = shared_lora_rank_dictionary(result);
        return copy;
    }
    const auto &r = result.request;
    auto hybrid = result.hybrid ? to_dictionary(*result.hybrid) : @{};
    id qkv = result.qkv ? qkv_dictionary(*result.qkv) : (id)NSNull.null;
    auto encoder_hybrid = result.encoder_hybrid
                              ? to_dictionary(*result.encoder_hybrid)
                              : @{};
    auto encoder_execution = result.encoder_quantized_execution ? @"gguf_bounded_gpu_experimental" : encoder_executed(result) ? @"gpu_ane_experimental" : @"gpu";
    auto encoder_backend = result.encoder_quantized_execution ? @"mlx_cpp_metal_gguf_bounded" : encoder_backend_label(result);
    auto encoder_gpu_graph =
        result.encoder_quantized_execution ? @"each-layer-eager-submission" : encoder_gpu_graph_label(r, encoder_executed(result));
    auto encoder_precision = result.encoder_quantized_execution ? @"source_mixed_weights+fp32_residual_rope+bf16_conditioning" : encoder_precision_label(result);
    if (result.prepared) {
        NSMutableDictionary *prepared = [@{
            @"acceleration_selection" : @(result.selection.c_str()),
            @"prepared" : @YES,
            @"warmup" : @(result.warmup),
            @"prompt_cache_hit" : @(result.prompt_cache_hit),
            @"execution" : @(r.execution.c_str()),
            @"lora_strategy" : @(effective_lora_strategy(r).c_str()),
            @"runtime_backend" : result.backend.empty() ? [NSNull null]
                                                         : @(result.backend.c_str()),
            @"runtime_precision" : result.precision.empty() ? [NSNull null]
                                                             : @(result.precision.c_str()),
            @"encoder_execution" : encoder_execution,
            @"encoder_runtime_backend" : encoder_backend,
            @"encoder_gpu_graph" : encoder_gpu_graph,
            @"encoder_runtime_precision" : encoder_precision,
            @"checkpoint" : result.checkpoint.empty() ? [NSNull null]
                                                       : @(result.checkpoint.c_str()),
            @"plan" : runtime_plan(result),
            @"text_tokens" : @(result.text_tokens),
            @"total_tokens" : @(result.total_tokens),
            @"seconds" : @(result.timings.wall),
            @"mlx_active_bytes" : @(result.active_bytes),
            @"block_residency" : result.block_residency ? to_dictionary(*result.block_residency) : (id)[NSNull null],
            @"hybrid" : hybrid,
            @"qkv" : qkv,
            @"encoder_hybrid" : encoder_hybrid
        } mutableCopy];
        prepared[@"encoder_runtime_reuse"]=encoder_reuse_dictionary(result);
        prepared[@"encoder_weight_residency"]=encoder_weight_dictionary(result);
        if (result.shared_lora_ranks)
            prepared[@"shared_lora_ranks"] = shared_lora_rank_dictionary(result);
        if (result.memory_admission)
            prepared[@"memory_admission"] =
                to_dictionary(*result.memory_admission);
        if (!result.memory_trace.empty())
            prepared[@"memory_trace"] = to_array(result.memory_trace);
        if (result.public_streaming)
            prepared[@"public_streaming"] = public_streaming_result(
                *result.public_streaming);
        attach_streaming_details(prepared, result);
        return prepared;
    }
    NSMutableDictionary *memory = [@{
              @"mlx_peak_bytes" : @(result.peak_bytes),
              @"mlx_active_bytes" : @(result.active_bytes),
              @"scope" : @"MLX allocator; excludes Core ML/OS/file cache"
          } mutableCopy];
    if (result.memory_admission)
        memory[@"admission"] = to_dictionary(*result.memory_admission);
    NSMutableDictionary *value = [@{
        @"acceleration_selection" : @(result.selection.c_str()),
        @"schema_version" : @1,
        @"warmup" : @(result.warmup),
        @"model" : @(r.model.c_str()),
        @"output" : result.warmup ? [NSNull null] : @(r.output.c_str()),
        @"width" : @(r.width),
        @"height" : @(r.height),
        @"seed" : @(r.seed),
        @"steps" : @(r.steps),
        @"operation" : @(r.operation.c_str()),
        @"lora_strategy" : @(effective_lora_strategy(r).c_str()),
        @"reference_tokens" : @(result.reference_tokens),
        @"gpu_graph" : gpu_graph_label(result),
        @"runtime_backend" : result.backend.empty() ? [NSNull null]
                                                     : @(result.backend.c_str()),
        @"runtime_precision" : result.precision.empty() ? [NSNull null]
                                                         : @(result.precision.c_str()),
        @"encoder_execution" : encoder_execution,
        @"encoder_runtime_backend" : encoder_backend,
        @"encoder_gpu_graph" : encoder_gpu_graph,
        @"encoder_runtime_precision" : encoder_precision,
        @"checkpoint" : result.checkpoint.empty() ? [NSNull null]
                                                   : @(result.checkpoint.c_str()),
        @"actual_denoise_steps" : @(result.actual_steps),
        @"text_tokens" : @(result.text_tokens),
        @"valid_text_tokens" : @(result.valid_text_tokens),
        @"prompt_cache_hit" : @(result.prompt_cache_hit),
        @"plan" : runtime_plan(result),
        @"timings_seconds" : @{
            @"request_wall" : @(result.timings.wall),
            @"text_encode" : @(result.timings.text),
            @"image_encode" : @(result.timings.image),
            @"hybrid_setup" : @(result.timings.hybrid),
            @"denoise" : @(result.timings.denoise),
            @"vae_decode" : @(result.timings.decode)
        },
        @"memory" : memory,
        @"hybrid" : hybrid,
        @"qkv" : qkv,
        @"encoder_hybrid" : encoder_hybrid,
        @"validation" : @"candidate; consult recorded parity suite"
    } mutableCopy];
    value[@"encoder_runtime_reuse"]=encoder_reuse_dictionary(result);
    value[@"encoder_weight_residency"]=encoder_weight_dictionary(result);
    if(result.student_ffn_reuse_enabled)
        value[@"student_ffn_reuse"]=student_ffn_reuse_dictionary(result);
    if (result.shared_lora_ranks)
        value[@"shared_lora_ranks"] = shared_lora_rank_dictionary(result);
    if (result.qwen_ffn_phases)
        value[@"qwen_ffn_phases"] = qwen_ffn_phase_dictionary(result);
    if (result.gguf_import) {
        const auto &m=*result.gguf_import;
        value[@"gguf_import"]=@{
            @"experimental":@YES,@"whole_request_bounded_certified":@NO,
            @"recipe":@"gguf-mlx-compat-affine-packed-bank-v1",@"allocator_cache_limit_bytes":@(m.allocator_cache_limit_bytes),
            @"affine_packing_recipe":@(m.affine_packing_recipe.c_str()),@"affine_packing_backend":@(m.affine_packing_backend.c_str()),
            @"float_import_recipe":@(m.float_import_recipe.c_str()),
            @"consumer_revision":m.session_packed_retention ? @"z-session-packed-experimental-v1" : @"z-serial-refiners-release-before-vae-v1",
            @"session_packed_retention":@(m.session_packed_retention),@"reused_packed_bank":@(m.reused_packed_bank),
            @"gpu_graph_recipe":m.ref_mpp_dynamic ? @"z-gpu-affine-qmm-f16-refmpp-dynamic-v1" : m.f16_refiners ? @"z-gpu-affine-qmm-f16-ref16-fp32-io-down64-v1" : m.qmm_f16_compute ? @"z-gpu-affine-qmm-f16-fp32-io-down64-v1" : m.gpu_f16_mpp ? @"z-gpu-affine-f16-mpp-fp32-io-down64-v1" : m.gpu_f16_compute ? @"z-gpu-affine-f16-fp32-io-down64-v1" : m.compiled_packed_blocks ? @"z-parameterized-affine-block-v1" : @"native-compat-eager-v1",
            @"dense_weight_capacity_upper_bytes":@(m.dense_weight_capacity_upper),
            @"dense_scope":m.gpu_f16_compute && !m.qmm_f16_compute ? @"current-main-block-only-eval-v1" : @"none",
            @"main_eval_policy":m.qmm_f16_compute ? @"whole-pass-immutable-packed-v1" : m.gpu_f16_compute ? @"each-main-block-v1" : @"native-default",
            @"released_before_vae":@(m.released_before_vae),@"serial_refiner_eval":@(m.serial_refiner_eval),
            @"source_sha256":@(m.source_sha256.c_str()),@"plan_digest":@(m.plan_digest.c_str()),
            @"planned_packed_capacity_bytes":@(m.planned_packed_capacity_bytes),
            @"packed_capacity_bytes":@(m.packed_capacity_bytes),@"read_buffer_capacity_bytes":@(m.read_buffer_capacity_bytes),
            @"managed_peak_bytes":@(m.managed_peak_bytes),@"output_bytes":@(m.output_bytes),
            @"source_read_bytes":@(m.source_read_bytes),@"logical_source_bytes":@(m.logical_source_bytes),
            @"verification_bytes":@(m.verification_bytes),@"tensor_count":@(m.tensor_count),@"field_count":@(m.field_count),
            @"load_seconds":@(m.load_seconds),@"read_seconds":@(m.read_seconds),@"decode_seconds":@(m.decode_seconds),
            @"affine_decode_seconds":@(m.affine_decode_seconds),@"float_decode_seconds":@(m.float_decode_seconds),
            @"request_load_seconds":@(m.reused_packed_bank ? 0 : m.load_seconds),
            @"request_source_read_bytes":@(m.reused_packed_bank ? 0 : m.source_read_bytes),
            @"ane_weight_source":m.raw_window_budget_bytes ? @"bounded-raw-ggml-window-v1" : @"mlx-affine-import",
            @"raw_window_budget_bytes":@(m.raw_window_budget_bytes),@"raw_window_live_bytes":@(m.raw_window_live_bytes),
            @"raw_window_peak_bytes":@(m.raw_window_peak_bytes),@"raw_window_entries":@(m.raw_window_entries),
            @"raw_window_hits_session_total":@(m.raw_window_hits),@"raw_window_misses_session_total":@(m.raw_window_misses),
            @"raw_window_evictions_session_total":@(m.raw_window_evictions),
            @"raw_window_source_read_bytes_session_total":@(m.raw_window_source_read_bytes),
            @"raw_window_read_seconds_session_total":@(m.raw_window_read_seconds),
            @"scope":@"managed immutable packed bank and import buffer; excludes encoder/VAE/activations/cache/framework/OS"
        };
        NSMutableDictionary *private_plan=[value[@"plan"] mutableCopy];
        private_plan[@"executable"]=@NO;
        private_plan[@"gguf_import_qualification"]=@"experimental-unqualified";
        if (m.compiled_packed_blocks) private_plan[@"gpu_graph"]=m.qmm_f16_compute ? @"compiled_affine_qmm_fp16" : m.gpu_f16_compute ? @"compiled_affine_gpu_decode_fp16" : @"compiled_affine_blocks";
        value[@"plan"]=private_plan;
        value[@"validation"]=@"experimental CPU direct packed import; not a production capability";
    }
    if (result.quantized_execution) {
        const auto &m = *result.quantized_execution;
        value[@"quantized_execution"] = @{
            @"experimental": @YES, @"whole_request_bounded_certified": @NO,
            @"source_sha256": @(m.source_sha256.c_str()), @"layout_digest": @(m.layout_digest.c_str()),
            @"packed_source_bytes": @(m.packed_bytes), @"packed_capacity_bytes": @(m.packed_capacity_bytes),
            @"source_float_bytes": @(m.source_float_bytes), @"max_dense_pool_capacity_bytes": @(m.dense_capacity_bytes),
            @"managed_peak_bytes": @(m.managed_peak_bytes), @"slot_count": @(m.slots), @"prefetch_layers": @(m.prefetch),
            @"fill_count": @(m.fills), @"decoded_bytes": @(m.decoded_bytes),
            @"slot_filled_bytes":@(m.decoded_bytes),
            @"source_load_seconds": @(m.source_load_seconds), @"decode_active_seconds": @(m.decode_seconds),
            @"exposed_ready_wait_seconds": @(m.exposed_wait_seconds),
            @"scope": @"managed GGUF source/slot and current GPU affine output buffers; excludes encoder/VAE/activations/framework/OS",
            @"source_residency": @(m.source_residency.c_str()), @"source_logical_bytes": @(m.source_logical_bytes),
            @"packed_read_buffer_capacity_bytes": @(m.read_buffer_bytes), @"source_read_bytes": @(m.source_read_bytes),
            @"streamed_read_seconds": @(m.streamed_read_seconds),
            @"refiner_fill_count": @(m.refiner_fills), @"refiner_slot_count": @(m.refiner_slots),
            @"refiner_decoded_bytes": @(m.refiner_decoded_bytes),
            @"refiner_pool_capacity_bytes": @(m.refiner_capacity_bytes),
            @"decode_backend":@(m.decode_backend.c_str()),
            @"gpu_affine_preparations":@(m.gpu_affine_preparations),@"gpu_affine_output_bytes":@(m.gpu_affine_output_bytes),
            @"gpu_prepare_capacity_upper_bytes":@(m.gpu_prepare_capacity_upper),@"gpu_prepare_wall_seconds":@(m.gpu_prepare_seconds),
            @"gpu_prepare_timing_scope":gguf_dependency_gpu_profile(m.precision_profile) ? @"host-dependency-construction-only-v1" : @"host-wall-through-packing-completion-v1",
            @"gpu_consumer_ready":gguf_dependency_gpu_profile(m.precision_profile) ? @"dependency-ready-not-status-validated-v1" : @"packing-complete-status-validated-v1",
            @"allocator_cache_limit_bytes":@(m.allocator_cache_limit_bytes),
            @"gpu_fixed_output_banks_created":@(m.gpu_fixed_output_banks),@"gpu_fixed_output_bank_capacity_bytes":@(m.gpu_fixed_output_bank_bytes),
            @"precision_profile":@(m.precision_profile.c_str()),@"ready_representation":gguf_raw_gpu_profile(m.precision_profile) ? @"raw-gguf-v1" : @"cpu-materialized-v1"
        };
        value[@"validation"] = @"experimental source-mixed GGUF execution; not a production capability";
    }
    if (!result.quantized_source_comparisons.empty()) {
        NSMutableArray *rows=[NSMutableArray array];bool layers_pass=true,final_pass=true;
        for (const auto &row:result.quantized_source_comparisons) {
            const auto &m=row.metrics;const bool pass=row.final_latent ? m.n1_final() : m.n1_layer();
            if (row.final_latent) final_pass &= pass;else layers_pass &= pass;
            [rows addObject:@{@"name":@(row.name.c_str()),@"step":@(row.step),@"rel_l2":@(m.rel_l2),@"cosine":@(m.cosine),
                @"max_abs":@(m.max_abs),@"max_norm_error":@(m.max_norm_error),@"n1_pass":@(pass),@"final_latent":@(row.final_latent)}];
        }
        value[@"quantized_source_validation"]=@{@"source_profile":@"z-mlx-compat-affine-v1",
            @"scope":@"diagnostic dual-forward independent source trajectory; FP64 valid-row metrics excluding padding; not timing/media/qualification",
            @"all_block_n1_pass":@(layers_pass),@"final_latent_n1_pass":@(final_pass),@"comparisons":rows};
    }
    if (result.encoder_quantized_execution) {
        const auto &m = *result.encoder_quantized_execution;
        value[@"encoder_quantized_execution"] = @{
            @"experimental": @YES, @"whole_request_bounded_certified": @NO,
            @"precision_profile": @"qwen3-z-source-mixed-v1", @"submit_policy": @"each-layer-eval-v1",
            @"decode_backend": @(m.decode_backend.c_str()),
            @"source_metadata_policy":@(m.source_metadata_policy.c_str()),
            @"source_metadata_reused":@(m.source_metadata_reused),
            @"source_metadata_preparations":@(m.source_metadata_preparations),
            @"conditioning_producer_generation":@(m.conditioning_producer_generation),
            @"source_residency": @(m.source_residency.c_str()), @"source_logical_bytes": @(m.source_logical_bytes),
            @"packed_read_buffer_capacity_bytes": @(m.read_buffer_bytes), @"source_read_bytes": @(m.source_read_bytes),
            @"streamed_read_seconds": @(m.streamed_read_seconds),
            @"hidden_tap": @"block34-post-residual-no-final-norm", @"source_sha256": @(m.source_sha256.c_str()),
            @"layout_digest": @(m.layout_digest.c_str()), @"packed_source_bytes": @(m.packed_bytes),
            @"packed_capacity_bytes": @(m.packed_capacity_bytes), @"source_float_bytes": @(m.source_float_bytes),
            @"max_dense_pool_capacity_bytes": @(m.dense_capacity_bytes), @"managed_peak_bytes": @(m.managed_peak_bytes),
            @"slot_count": @(m.slots), @"prefetch_layers": @(m.prefetch), @"fill_count": @(m.fills),
            @"decoded_bytes": @(m.decoded_bytes), @"source_load_seconds": @(m.source_load_seconds),
            @"decode_active_seconds": @(m.decode_seconds), @"exposed_ready_wait_seconds": @(m.exposed_wait_seconds),
            @"scope": @"managed encoder packed weights, refill slots and gathered embedding; excludes other activations/framework/OS"
        };
    }
    if (result.db_cache_enabled)
        value[@"qwen21_dbcache"] = @{
            @"front_blocks": @8, @"back_blocks": @0, @"warmup_steps": @8,
            @"threshold": @(result.db_cache_threshold),
            @"max_consecutive": @(result.db_cache_max_consecutive),
            @"cached_steps": @(result.db_cache_steps),
            @"saved_middle_blocks": @(result.db_cache_steps * 24)
        };
    if (!r.loras.empty() && result.lora_applied_projections)
        value[@"lora_applied_projections"] = @(result.lora_applied_projections);
    if (result.block_residency)
        value[@"block_residency"] = to_dictionary(*result.block_residency);
    if (result.streaming_runtime) {
        const auto &runtime = *result.streaming_runtime;
        NSMutableDictionary *block = result.block_residency
            ? [to_dictionary(*result.block_residency) mutableCopy]
            : [NSMutableDictionary dictionary];
        block[@"implementation"] = @(runtime.implementation.c_str());
        block[@"layout_digest"] = runtime.layout_digest.empty()
            ? (id)NSNull.null : (id)@(runtime.layout_digest.c_str());
        block[@"actual_layout"] = actual_streaming_layout(runtime);
        value[@"block_streaming"] = block;
    }
    if (result.public_streaming)
        value[@"public_streaming"] = public_streaming_result(
            *result.public_streaming);
    attach_streaming_details(value, result);
    if (!result.memory_trace.empty())
        value[@"memory_trace"] = to_array(result.memory_trace);
    if (!result.enhanced_prompt.empty())
        value[@"prompt_enhancement"] = @{
            @"backend": r.prompt_enhance_edit_experimental ? @"native_qwen35_pe_i2i_experimental" : @"native_qwen35_pe_t2i", @"complete": @YES,
            @"experimental_edit": @(r.prompt_enhance_edit_experimental),
            @"visual_precision": r.prompt_enhance_edit_experimental ? @"float32" : @"not_used",
            @"quality_accepted": @NO,
            @"original_prompt": @(result.original_prompt.c_str()),
            @"positive_prompt": @(result.enhanced_prompt.c_str()),
            @"wh_ratio": @(result.enhanced_wh_ratio.c_str()),
            @"ratio_follow": @(result.enhanced_ratio_follow.c_str()),
            @"applied_ratio": @NO, // request dimensions remain explicit
            @"generated_tokens": @(result.prompt_enhance_tokens),
            @"chunked_prefill": @(result.prompt_enhance_chunked_prefill),
            @"seconds": @(result.prompt_enhance_seconds)
        };
    return value;
}
} // namespace tc
