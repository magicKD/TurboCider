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
        @"default_height" : @(d.height)
    } mutableCopy];
    if (d.max_images)
        result[@"max_images"] = @(d.max_images);
    if (d.weight_validation_pending)
        result[@"weight_validation"] = @"pending";
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
        @"validation" : r.model == "flux2-klein-4b" ? @"native_candidate" : @"weights_pending",
        @"backend" : hybrid             ? @"mlx_cpp_metal+coreml"
        : r.model == "minimax-h3-turbo" ? @"metal_mps_native"
                                        : @"mlx_cpp_metal",
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
        @"weight_validation" : r.model == "flux2-klein-4b" ? @"see parity evidence" : @"pending",
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
        @"compute_units" : @"cpuAndNeuralEngine",
        @"observed_ane_residency" : @"unknown",
        @"output_copy_bytes_session_total" : @(m.copied_bytes),
        @"provenance" :
            @"local checkpoint path+size; source SHA absent in legacy artifact; experimental only"
    };
}
NSDictionary *to_dictionary(const RunResult &result) {
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
    return @{
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
    };
}
} // namespace tc
