#include "bridge.hpp"
#include "../../runtime/memory_execution.hpp"
namespace tc {
static NSString *gpu_graph_label(const Request &r) {
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
    if (ltx_gemma4_encoder(request))
        return hybrid ? @"gemma4_encoder_mlp_complement"
                      : @"gemma4_gpu_only";
    if (h3_qwen3_vl_encoder(request))
        return hybrid ? @"qwen3_vl_encoder_mlp_complement"
                      : @"qwen3_vl_gpu_only";
    return hybrid ? @"qwen3_encoder_mlp_complement" : @"qwen3_gpu_only";
}
static NSString *encoder_weight_validation_label(const Request &request,
                                                  bool hybrid,
                                                  bool executed) {
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
    if (ltx_gemma4_encoder(request))
        return @"gemma4_encoder_mlp_int8_per_channel";
    return h3_qwen3_vl_encoder(request)
        ? @"qwen3_vl_encoder_mlp_int8_per_channel"
        : @"qwen3_encoder_mlp_int8_per_channel";
}
static NSArray *strings(const std::vector<std::string> &values) {
    NSMutableArray *array = [NSMutableArray array];
    for (auto &value : values)
        [array addObject:@(value.c_str())];
    return array;
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
                      @"weights_pending";
    auto weight_validation = r.model == "z-image-turbo-gguf" ?
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
        @"pending";
    auto lora_fusion = lora_strategy == "none" ? @"none" :
        (lora_strategy == "inference_time" ?
            @"inference_time_low_rank" :
         lora_strategy == "in_memory_merge" ?
            (r.model.starts_with("flux2-klein-") ? @"load_time_baked" : @"in_memory_delta") :
         r.model == "wan2.1-1.3b-qad" ? @"premerged_manifest_verified" :
         @"premerged_manifest_verified");
    auto backend = hybrid ?
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
    if (r.model == "minimax-h3-vdn")
        [algorithm_approximations addObject:
            @"affine_int6_g64_base_weight_quantization"];
    else if (r.model == "z-image-turbo-gguf")
        [algorithm_approximations addObject:
            @"checkpoint_defined_gguf_weight_quantization"];
    else if (!r.quantized_cache.empty())
        [algorithm_approximations addObject:
            @"row_symmetric_int8_weight_quantization"];
    else if (hybrid)
        [algorithm_approximations addObject:
            @"single_block_mlp_int8_per_channel"];
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
    NSMutableDictionary *result = [@{
        @"selection_pending" : @(r.execution == "auto"),
        @"requested_execution" : @(r.execution.c_str()),
        @"schema_version" : @1,
        @"model" : @(r.model.c_str()),
        @"executable" : @(recipe.executable),
        @"validation" : validation,
        @"backend" : backend,
        @"execution" : hybrid ? @"gpu_ane_experimental" : @"gpu",
        @"encoder_execution" : encoder_hybrid ? @"gpu_ane_experimental" : @"gpu",
        @"encoder_backend" : encoder_backend_label(r, encoder_hybrid),
        @"encoder_gpu_graph" : encoder_gpu_graph_label(r, encoder_hybrid),
        @"encoder_precision" : encoder_hybrid ? @"bf16_gpu+int8_mlp_fp16_io"
                                                : @"bf16",
        @"encoder_weight_validation" :
            encoder_weight_validation_label(r, encoder_hybrid, false),
        @"gpu_graph" : gpu_graph_label(r),
        @"precision" : r.model == "minimax-h3-vdn" ? @"int6_g64_base+bf16_vdn+fp32_solve" :
                      r.model.starts_with("minimax-h3-fasth3-mlx-int6") ? @"int6_g64_bf16_activation" :
                      r.model == "wan2.1-1.3b-qad" ? @"fp16-int8-affine-dit+bf16-umt5+fp32-taehv" :
                      r.model == "z-image-turbo-gguf" ? @"checkpoint_defined_gguf" :
            (!r.quantized_cache.empty() ? @"int8_weight_bf16_activation_streamed" :
             (hybrid ? @"bf16_gpu+int8_mlp_fp16_io" : @"bf16")),
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
    if (r.streaming.active()) {
        result[@"executable"] = @NO;
        result[@"memory_estimate_kind"] = @"requires_layout_metadata";
        NSMutableDictionary *origins = [NSMutableDictionary dictionary];
        for (const auto &[key, origin] : r.streaming.provenance)
            origins[@(key.c_str())] = @(origin.c_str());
        result[@"streaming"] = @{
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
        result[@"executable"] = @NO;
        result[@"memory_estimate_kind"] = @"requires_preset_resolution";
        NSMutableDictionary *origins = [NSMutableDictionary dictionary];
        for (const auto &[key, origin] : r.streaming_selector->provenance)
            origins[@(key.c_str())] = @(origin.c_str());
        result[@"streaming"] = @{
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
        result[@"memory_policy"] = @{
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
    return result;
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
static NSDictionary *runtime_plan(const RunResult &result) {
    NSMutableDictionary *plan = [to_dictionary(result.plan) mutableCopy];
    if (!result.backend.empty())
        plan[@"backend"] = @(result.backend.c_str());
    if (!result.precision.empty())
        plan[@"precision"] = @(result.precision.c_str());
    plan[@"gpu_graph"] = gpu_graph_label(result);
    const bool hybrid = result.request.execution == "gpu_ane";
    const bool encoder_hybrid = result.encoder_hybrid.has_value();
    plan[@"execution"] = hybrid ? @"gpu_ane_experimental" : @"gpu";
    plan[@"encoder_execution"] = encoder_hybrid ? @"gpu_ane_experimental" : @"gpu";
    plan[@"encoder_backend"] = encoder_backend_label(
        result.request, encoder_hybrid);
    plan[@"encoder_gpu_graph"] =
        encoder_gpu_graph_label(result.request, encoder_hybrid);
    plan[@"encoder_precision"] = encoder_hybrid ? @"bf16_gpu+int8_mlp_fp16_io"
                                                  : @"bf16";
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
            [approximations addObject:@"single_block_mlp_int8_per_channel"];
        if (encoder_hybrid)
            [approximations addObject:encoder_approximation_label(result.request)];
        plan[@"algorithm_approximations"] = approximations;
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
NSDictionary *to_dictionary(const HybridMetrics &m) {
    return @{
        @"load_seconds" : @(m.load_seconds),
        @"manifest_validation_seconds" : @(m.manifest_validation_seconds),
        @"output_backing_setup_seconds" : @(m.output_backing_setup_seconds),
        @"model_load_seconds" : @(m.model_load_seconds),
        @"model_interface_setup_seconds" : @(m.model_interface_setup_seconds),
        @"zero_input_warmup_seconds" : @(m.zero_input_warmup_seconds),
        @"prediction_seconds_session_total" : @(m.prediction_seconds),
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
        @"block_count" : @(m.block_count),
        @"mlp_width" : @(m.mlp_width),
        @"ane_mlp_range" : @[ @(m.ane_mlp_start), @(m.ane_mlp_end) ],
        @"output_scale" : @(m.output_scale),
        @"qualified_flexible_backing" : @(m.qualified_flexible_backing),
        @"compute_units" : @"cpuAndNeuralEngine",
        @"observed_ane_residency" : @"unknown",
        @"output_copy_bytes_session_total" : @(m.copied_bytes),
        @"checkpoint_sha256_verified" : @(m.checkpoint_sha_verified),
        @"lora_identity_verified" : @(m.lora_identity_verified),
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
        @"provenance" : m.checkpoint_sha_verified
            ? @"local checkpoint path, size and SHA-256 verified"
            : @"local checkpoint path+size; source SHA absent in legacy artifact; experimental only"
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
            copy[@"encoder_execution"] = @"gpu_ane_experimental";
            copy[@"encoder_runtime_backend"] = encoder_backend_label(
                result.request, true);
            copy[@"encoder_gpu_graph"] =
                encoder_gpu_graph_label(result.request, true);
            copy[@"encoder_runtime_precision"] = @"bf16_gpu+int8_mlp_fp16_io";
            copy[@"encoder_hybrid"] = to_dictionary(*result.encoder_hybrid);
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
        return copy;
    }
    const auto &r = result.request;
    auto hybrid = result.hybrid ? to_dictionary(*result.hybrid) : @{};
    auto encoder_hybrid = result.encoder_hybrid
                              ? to_dictionary(*result.encoder_hybrid)
                              : @{};
    auto encoder_execution = result.encoder_hybrid ? @"gpu_ane_experimental" : @"gpu";
    auto encoder_backend = encoder_backend_label(
        r, result.encoder_hybrid.has_value());
    auto encoder_gpu_graph =
        encoder_gpu_graph_label(r, result.encoder_hybrid.has_value());
    auto encoder_precision = result.encoder_hybrid ? @"bf16_gpu+int8_mlp_fp16_io"
                                                    : @"bf16";
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
            @"encoder_hybrid" : encoder_hybrid
        } mutableCopy];
        if (result.memory_admission)
            prepared[@"memory_admission"] =
                to_dictionary(*result.memory_admission);
        if (!result.memory_trace.empty())
            prepared[@"memory_trace"] = to_array(result.memory_trace);
        if (result.public_streaming)
            prepared[@"public_streaming"] = public_streaming_result(
                *result.public_streaming);
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
        @"encoder_hybrid" : encoder_hybrid,
        @"validation" : @"candidate; consult recorded parity suite"
    } mutableCopy];
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
    if (!result.memory_trace.empty())
        value[@"memory_trace"] = to_array(result.memory_trace);
    return value;
}
} // namespace tc
