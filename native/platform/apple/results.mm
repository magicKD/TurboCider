#include "bridge.hpp"
namespace tc {
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
    result[@"native_gemma4_candidate"] = @(d.native_gemma4_candidate);
    result[@"native_conditioning_connector"] = @(d.native_conditioning_connector);
    result[@"native_i2v_clean_prefix"] = @(d.native_i2v_clean_prefix);
    result[@"native_gpu_ane_profile"] = @(d.native_gpu_ane_profile);
    result[@"native_audio_output_candidate"] = @(d.native_audio_output_candidate);
    result[@"native_audio_vae_candidate"] = @(d.native_audio_vae_candidate);
    result[@"native_base_vocoder_candidate"] = @(d.native_base_vocoder_candidate);
    result[@"audio_output"] = @(d.audio_output);
    result[@"request_lora_identity_validation"] = @(d.request_lora_identity_validation);
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
    auto validation = r.model.starts_with("flux2-klein-") ? @"native_candidate" :
                      r.model == "minimax-h3-turbo" ? @"manifest_verified_native" :
                      r.model == "fastmetal-1.3b-qad" ? @"manifest_verified_python_runtime" :
                      r.model == "ltx-2.5-distilled" ?
                          (recipe.executable ? @"native_video_executor" : @"native_capability_gated") :
                      r.model == "z-image-turbo" ? @"native_candidate" :
                      @"weights_pending";
    auto weight_validation = r.model.starts_with("flux2-klein-") ? @"see parity evidence" :
        r.model == "minimax-h3-turbo" ?
            (r.loras.empty() ? @"manifest-verified" : @"runtime-cache-or-sidecar-verified-at-execution") :
        r.model == "fastmetal-1.3b-qad" ?
            (r.loras.empty() ? @"checkpoint-and-ane-identity-verified-at-load" : @"premerged-manifest-verified-at-execution") :
        r.model == "ltx-2.5-distilled" ?
            (r.loras.empty() ? @"checkpoint-validated-at-load" : @"runtime-cache-or-sidecar-verified-at-execution") :
        r.model == "z-image-turbo" ?
            (r.loras.empty() ? @"comfy-oracle-validated; gpu_ane-pending" :
                               @"in-memory-lora; comfy-oracle-validated") :
        @"pending";
    auto lora_fusion = r.loras.empty() ? @"none" :
        (r.model.starts_with("flux2-klein-") ? @"load_time_baked" :
         r.model == "fastmetal-1.3b-qad" ? @"premerged_manifest_verified" :
         r.model == "z-image-turbo" ? @"in_memory_delta" : @"runtime_bake_cache");
    auto backend = hybrid ?
        (r.model == "fastmetal-1.3b-qad" ? @"fastmetal-mlx+ane_parallel" :
         r.model == "ltx-2.5-distilled" ? @"ltx-gpu+ane" : @"mlx_cpp_metal+coreml") :
        r.model == "minimax-h3-turbo" ? @"h3-metal-mps" :
        r.model == "ltx-2.5-distilled" ? @"ltx-metal-mps" :
        r.model == "fastmetal-1.3b-qad" ? @"fastmetal-mlx" : @"mlx_cpp_metal";
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
    return @{
        @"selection_pending" : @(r.execution == "auto"),
        @"requested_execution" : @(r.execution.c_str()),
        @"schema_version" : @1,
        @"model" : @(r.model.c_str()),
        @"executable" : @(recipe.executable),
        @"validation" : validation,
        @"backend" : backend,
        @"execution" : hybrid ? @"gpu_ane_experimental" : @"gpu",
        @"gpu_graph" : r.compile_gpu ? @"compiled_single_blocks" : @"eager_blocks",
        @"precision" : hybrid ? @"bf16_gpu+int8_mlp_fp16_io" : @"bf16",
        @"algorithm_approximations" : hybrid ? @[ @"single_block_mlp_int8_per_channel" ] : @[],
        @"requested_shape" : @[ @(r.width), @(r.height), @(r.frames) ],
        @"decoded_shape" : @[ @(dw), @(dh), @(r.frames) ],
        @"stages" : stages,
        @"memory_estimate_bytes" : plan.memory_estimate_bytes ? @(*plan.memory_estimate_bytes)
                                                              : [NSNull null],
        @"memory_estimate_kind" : @"conservative_heuristic_not_hard_limit",
        @"operation" : @(r.operation.c_str()),
        @"residency" : @(r.residency.c_str()),
        @"profile_identity" : @(r.profile_identity.c_str()),
        @"weight_validation" : weight_validation,
        @"lora_count" : @(r.loras.size()),
        @"lora_fusion" : lora_fusion,
        @"audio" : @(r.audio),
        @"audio_capability" : r.model == "ltx-2.5-distilled" ?
            (r.audio ? @"latent_to_48khz_aac_candidate" : @"video_only_native") : @"not_applicable",
        @"executor_operations" : r.model == "ltx-2.5-distilled" ? @[ @"video.generate" ] :
            (r.model == "fastmetal-1.3b-qad" ? @[ @"video.generate" ] : [NSNull null]),
        @"limitation" : recipe.executable
            ? @"capabilities depend on model artifacts and configured hardware"
            : @"native executor migration incomplete"
    };
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
        @"prediction_seconds_session_total" : @(m.prediction_seconds),
        @"calls_session_total" : @(m.calls),
        @"bucket" : @(m.bucket),
        @"hidden" : @(m.hidden),
        @"block_count" : @(m.block_count),
        @"mlp_width" : @(m.mlp_width),
        @"ane_mlp_range" : @[ @(m.ane_mlp_start), @(m.ane_mlp_end) ],
        @"output_scale" : @(m.output_scale),
        @"compute_units" : @"cpuAndNeuralEngine",
        @"observed_ane_residency" : @"unknown",
        @"output_copy_bytes_session_total" : @(m.copied_bytes),
        @"checkpoint_sha256_verified" : @(m.checkpoint_sha_verified),
        @"lora_identity_verified" : @(m.lora_identity_verified),
        @"provenance" : m.checkpoint_sha_verified
            ? @"local checkpoint path, size and SHA-256 verified"
            : @"local checkpoint path+size; source SHA absent in legacy artifact; experimental only"
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
        return copy;
    }
    const auto &r = result.request;
    auto hybrid = result.hybrid ? to_dictionary(*result.hybrid) : @{};
    if (result.prepared)
        return @{
            @"acceleration_selection" : @(result.selection.c_str()),
            @"prepared" : @YES,
            @"warmup" : @NO,
            @"prompt_cache_hit" : @(result.prompt_cache_hit),
            @"execution" : @(r.execution.c_str()),
            @"text_tokens" : @(result.text_tokens),
            @"total_tokens" : @(result.total_tokens),
            @"seconds" : @(result.timings.wall),
            @"mlx_active_bytes" : @(result.active_bytes),
            @"hybrid" : hybrid
        };
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
        @"reference_tokens" : @(result.reference_tokens),
        @"gpu_graph" : r.compile_gpu ? @"compiled_single_blocks" : @"eager_blocks",
        @"actual_denoise_steps" : @(result.actual_steps),
        @"text_tokens" : @(result.text_tokens),
        @"valid_text_tokens" : @(result.valid_text_tokens),
        @"prompt_cache_hit" : @(result.prompt_cache_hit),
        @"plan" : to_dictionary(result.plan),
        @"timings_seconds" : @{
            @"request_wall" : @(result.timings.wall),
            @"text_encode" : @(result.timings.text),
            @"image_encode" : @(result.timings.image),
            @"hybrid_setup" : @(result.timings.hybrid),
            @"denoise" : @(result.timings.denoise),
            @"vae_decode" : @(result.timings.decode)
        },
        @"memory" : @{
            @"mlx_peak_bytes" : @(result.peak_bytes),
            @"mlx_active_bytes" : @(result.active_bytes),
            @"scope" : @"MLX allocator; excludes Core ML/OS/file cache"
        },
        @"hybrid" : hybrid,
        @"validation" : @"candidate; consult recorded parity suite"
    } mutableCopy];
    if (!r.loras.empty() && result.lora_applied_projections)
        value[@"lora_applied_projections"] = @(result.lora_applied_projections);
    return value;
}
} // namespace tc
