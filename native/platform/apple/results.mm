#include "bridge.hpp"
namespace tc {
static NSString *gpu_graph_label(const Request &r) {
    if (r.model == "z-image-turbo-gguf") {
        if (r.execution == "gpu_ane")
            return @"compiled_mlp_complement";
        if (r.streaming_offload || r.residency == "streaming")
            return @"streamed_sd_cpp_gguf";
        return effective_lora_strategy(r) == "in_memory_merge"
            ? @"native_quantized_blocks" : @"resident_sd_cpp_gguf";
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
    if (result.backend.starts_with("stable-diffusion-cpp-metal"))
        return result.request.streaming_offload || result.request.residency == "streaming"
            ? @"streamed_sd_cpp_gguf" : @"resident_sd_cpp_gguf";
    if (result.backend == "mlx_cpp_metal_gguf+coreml")
        return @"compiled_mlp_complement";
    if (result.backend == "mlx_cpp_metal_gguf")
        return @"native_quantized_blocks";
    return gpu_graph_label(result.request);
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
    const auto lora_strategy = effective_lora_strategy(r);
    auto validation = r.model == "z-image-turbo-gguf" ? @"native_gguf_candidate" :
                      r.model.starts_with("flux2-klein-") ? @"native_candidate" :
                      r.model == "minimax-h3-turbo" ? @"manifest_verified_native" :
                      r.model == "fastmetal-1.3b-qad" ? @"manifest_verified_python_runtime" :
                      r.model == "ltx-2.5-distilled" ?
                          (recipe.executable ? @"native_video_executor" : @"native_capability_gated") :
                      r.model == "z-image-turbo" ? @"native_candidate" :
                      r.model == "llada-image-turbo" ? @"native_llada_candidate" :
                      @"weights_pending";
    auto weight_validation = r.model == "z-image-turbo-gguf" ?
            @"GGUF header checked at load; paired output parity pending" :
        r.model.starts_with("flux2-klein-") ? @"see parity evidence" :
        r.model == "minimax-h3-turbo" ?
            (r.loras.empty() ? @"manifest-verified" : @"runtime-cache-or-sidecar-verified-at-execution") :
        r.model == "fastmetal-1.3b-qad" ?
            (r.loras.empty() ? @"checkpoint-and-ane-identity-verified-at-load" : @"premerged-manifest-verified-at-execution") :
        r.model == "ltx-2.5-distilled" ?
            (r.loras.empty() ? @"checkpoint-validated-at-load" : @"runtime-cache-or-sidecar-verified-at-execution") :
        r.model == "z-image-turbo" ?
            (r.loras.empty() ? @"comfy-oracle-validated; m4max-a4096-hybrid-qualified" :
                               @"in-memory-lora; comfy-oracle-validated") :
        r.model == "llada-image-turbo" ?
            @"native checkpoint loaded directly; see recorded parity evidence" :
        @"pending";
    auto lora_fusion = lora_strategy == "none" ? @"none" :
        (lora_strategy == "inference_time" ?
            (r.model == "z-image-turbo-gguf" ? @"sd_cpp_request_time"
                                               : @"inference_time_low_rank") :
         lora_strategy == "in_memory_merge" ?
            (r.model.starts_with("flux2-klein-") ? @"load_time_baked" : @"in_memory_delta") :
         r.model == "fastmetal-1.3b-qad" ? @"premerged_manifest_verified" :
         @"runtime_bake_cache");
    auto backend = hybrid ?
        (r.model == "fastmetal-1.3b-qad" ? @"fastmetal-mlx+ane_parallel" :
         r.model == "ltx-2.5-distilled" ? @"ltx-gpu+ane" : @"mlx_cpp_metal+coreml") :
        r.model == "z-image-turbo-gguf" ?
            (lora_strategy == "in_memory_merge" ? @"mlx_cpp_metal_gguf"
                                                  : @"stable_diffusion_cpp_metal") :
        r.model == "minimax-h3-turbo" ? @"h3-metal-mps" :
        r.model == "ltx-2.5-distilled" ? @"ltx-metal-mps" :
        r.model == "fastmetal-1.3b-qad" ? @"fastmetal-mlx" :
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
    return @{
        @"selection_pending" : @(r.execution == "auto"),
        @"requested_execution" : @(r.execution.c_str()),
        @"schema_version" : @1,
        @"model" : @(r.model.c_str()),
        @"executable" : @(recipe.executable),
        @"validation" : validation,
        @"backend" : backend,
        @"execution" : hybrid ? @"gpu_ane_experimental" : @"gpu",
        @"gpu_graph" : gpu_graph_label(r),
        @"precision" : r.model == "z-image-turbo-gguf" ? @"checkpoint_defined_gguf" :
            (hybrid ? @"bf16_gpu+int8_mlp_fp16_io" : @"bf16"),
        @"algorithm_approximations" : r.model == "z-image-turbo-gguf" ?
            @[ @"checkpoint_defined_gguf_weight_quantization" ] :
            (hybrid ? @[ @"single_block_mlp_int8_per_channel" ] : @[]),
        @"requested_shape" : @[ @(r.width), @(r.height), @(r.frames) ],
        @"decoded_shape" : @[ @(dw), @(dh), @(r.frames) ],
        @"stages" : stages,
        @"memory_estimate_bytes" : plan.memory_estimate_bytes ? @(*plan.memory_estimate_bytes)
                                                              : [NSNull null],
        @"memory_estimate_kind" : @"conservative_heuristic_not_hard_limit",
        @"memory_budget_bytes" : r.memory_budget_bytes ? @(r.memory_budget_bytes)
                                                        : [NSNull null],
        @"memory_budget_scope" :
            (r.model == "z-image-turbo-gguf" &&
             (r.streaming_offload || r.residency == "streaming"))
                ? @"sd_cpp_max_vram_hint_not_process_cap"
                : (r.model == "minimax-h3-turbo" &&
                   r.residency == "streamed" && r.memory_budget_bytes)
                    ? @"h3_dit_working_set_target_not_process_cap"
                : (r.memory_budget_bytes ? @"runtime_request_budget" : @"unset"),
        @"operation" : @(r.operation.c_str()),
        @"residency" : @(r.residency.c_str()),
        @"streaming_offload" : @(r.streaming_offload ||
                                  r.residency == "streaming" ||
                                  r.residency == "streamed"),
        @"profile_identity" : @(r.profile_identity.c_str()),
        @"weight_validation" : weight_validation,
        @"lora_count" : @(r.loras.size()),
        @"lora_strategy" : @(lora_strategy.c_str()),
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
static NSDictionary *runtime_plan(const RunResult &result) {
    NSMutableDictionary *plan = [to_dictionary(result.plan) mutableCopy];
    if (!result.backend.empty())
        plan[@"backend"] = @(result.backend.c_str());
    if (!result.precision.empty())
        plan[@"precision"] = @(result.precision.c_str());
    plan[@"gpu_graph"] = gpu_graph_label(result);
    const bool hybrid = result.request.execution == "gpu_ane";
    plan[@"execution"] = hybrid ? @"gpu_ane_experimental" : @"gpu";
    if (result.request.model == "z-image-turbo-gguf") {
        const bool native = result.backend.starts_with("mlx_cpp_metal_gguf");
        const bool streaming = result.request.streaming_offload ||
                               result.request.residency == "streaming";
        plan[@"validation"] = native
            ? (hybrid ? @"checkpoint_bound_native_gguf_hybrid_candidate"
                      : @"native_mlx_gguf_candidate")
            : (streaming ? @"streamed_sd_cpp_gguf_candidate"
                         : @"resident_sd_cpp_gguf");
        plan[@"weight_validation"] = native
            ? (hybrid ? @"GGUF checkpoint and Core ML manifest verified at execution"
                      : @"GGUF loaded directly by native MLX")
            : (streaming
                   ? @"GGUF header and CPU-staged sd.cpp layer-streaming runtime verified at load"
                   : @"GGUF header and resident sd.cpp runtime verified at load");
        if (!result.request.loras.empty()) {
            const auto strategy = effective_lora_strategy(result.request);
            plan[@"lora_strategy"] = @(strategy.c_str());
            plan[@"lora_fusion"] = strategy == "inference_time"
                ? (native ? @"inference_time_low_rank" : @"sd_cpp_request_time")
                : @"in_memory_delta";
        }
        plan[@"algorithm_approximations"] = hybrid
            ? @[ @"checkpoint_defined_gguf_weight_quantization",
                 @"single_block_mlp_int8_per_channel" ]
            : @[ @"checkpoint_defined_gguf_weight_quantization" ];
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
        if (!copy[@"lora_strategy"])
            copy[@"lora_strategy"] = @(effective_lora_strategy(result.request).c_str());
        return copy;
    }
    const auto &r = result.request;
    auto hybrid = result.hybrid ? to_dictionary(*result.hybrid) : @{};
    if (result.prepared)
        return @{
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
            @"checkpoint" : result.checkpoint.empty() ? [NSNull null]
                                                       : @(result.checkpoint.c_str()),
            @"plan" : runtime_plan(result),
            @"text_tokens" : @(result.text_tokens),
            @"total_tokens" : @(result.total_tokens),
            @"seconds" : @(result.timings.wall),
            @"mlx_active_bytes" : @(result.active_bytes),
            @"hybrid" : hybrid
        };
    NSDictionary *memory = result.external_physical_footprint_bytes
        ? @{
              @"resident_bytes" : @(result.external_resident_bytes),
              @"sampled_peak_resident_bytes" : @(result.external_peak_resident_bytes),
              @"physical_footprint_bytes" : @(result.external_physical_footprint_bytes),
              @"peak_physical_footprint_bytes" :
                  @(result.external_peak_physical_footprint_bytes),
              @"scope" : @"sd-server child process; peak resident sampled at job polling cadence"
          }
        : @{
              @"mlx_peak_bytes" : @(result.peak_bytes),
              @"mlx_active_bytes" : @(result.active_bytes),
              @"scope" : @"MLX allocator; excludes Core ML/OS/file cache"
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
        @"lora_strategy" : @(effective_lora_strategy(r).c_str()),
        @"reference_tokens" : @(result.reference_tokens),
        @"gpu_graph" : gpu_graph_label(result),
        @"runtime_backend" : result.backend.empty() ? [NSNull null]
                                                     : @(result.backend.c_str()),
        @"runtime_precision" : result.precision.empty() ? [NSNull null]
                                                         : @(result.precision.c_str()),
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
        @"validation" : @"candidate; consult recorded parity suite"
    } mutableCopy];
    if (!r.loras.empty() && result.lora_applied_projections)
        value[@"lora_applied_projections"] = @(result.lora_applied_projections);
    return value;
}
} // namespace tc
