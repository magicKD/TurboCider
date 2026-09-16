#include "bridge.hpp"
#include "../../core/json_keys.hpp"
#include <cmath>
namespace tc {
std::string json(id object) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:object
                                                   options:NSJSONWritingSortedKeys
                                                     error:&error];
    if (!data)
        throw std::runtime_error(error.localizedDescription.UTF8String);
    return std::string((const char *)data.bytes, data.length);
}
static NSDictionary *dictionary(id value, const char *context) {
    require([value isKindOfClass:NSDictionary.class], std::string(context) + " must be an object");
    return value;
}
NSDictionary *parse_json(const char *s) {
    require(s, "missing JSON");
    NSData *data = [NSData dataWithBytes:s length:strlen(s)];
    auto result = dictionary([NSJSONSerialization JSONObjectWithData:data options:0 error:nil], "JSON");
    reject_duplicate_json_keys(std::string_view((const char *)data.bytes, data.length));
    return result;
}
NSDictionary *read_json(const std::filesystem::path &path) {
    NSData *data = [NSData dataWithContentsOfFile:@(path.c_str())];
    require(data != nil, "missing file: " + path.string());
    return dictionary([NSJSONSerialization JSONObjectWithData:data options:0 error:nil],
                      path.c_str());
}
NSDictionary *read_config_json(const std::filesystem::path &path) {
    NSData *data = [NSData dataWithContentsOfFile:@(path.c_str())];
    require(data != nil, "missing file: " + path.string());
    auto result = dictionary([NSJSONSerialization JSONObjectWithData:data options:0 error:nil], path.c_str());
    reject_duplicate_json_keys(std::string_view((const char *)data.bytes, data.length));
    return result;
}
std::string string_value(NSDictionary *d, NSString *k, const std::string &fallback) {
    dictionary(d, "container");
    id v = d[k];
    if (!v)
        return fallback;
    require([v isKindOfClass:NSString.class], std::string(k.UTF8String) + " must be string");
    const char *s = [v UTF8String];
    require(s, "invalid UTF-8");
    require(strlen(s) == [v lengthOfBytesUsingEncoding:NSUTF8StringEncoding],
            "embedded NUL is not allowed");
    return s;
}
static double numeric(NSDictionary *d, NSString *k, double fallback) {
    id v = d[k];
    if (!v)
        return fallback;
    require([v isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)v) != CFBooleanGetTypeID(),
            std::string(k.UTF8String) + " must be numeric");
    double x = [v doubleValue];
    require(std::isfinite(x), "nonfinite number");
    return x;
}
static int number(NSDictionary *d, NSString *k, int fallback) {
    double x = numeric(d, k, fallback);
    require(x == std::floor(x) && x >= 0 && x <= 2147483647, "invalid integer");
    return int(x);
}
static uint64_t byte_count(NSDictionary *d, NSString *k, uint64_t fallback) {
    double x = numeric(d, k, double(fallback));
    constexpr double maximum = double(1ull << 50);
    require(x == std::floor(x) && x >= 0 && x <= maximum,
            "invalid byte count: " + std::string(k.UTF8String));
    return uint64_t(x);
}
static bool boolean(NSDictionary *d, NSString *k, bool fallback) {
    id v = d[k];
    if (!v)
        return fallback;
    require(CFGetTypeID((__bridge CFTypeRef)v) == CFBooleanGetTypeID(),
            std::string(k.UTF8String) + " must be bool");
    return [v boolValue];
}
static void keys(NSDictionary *d, NSArray *allowed) {
    NSSet *set = [NSSet setWithArray:allowed];
    for (NSString *k in d)
        require([set containsObject:k], "unknown field: " + std::string(k.UTF8String));
}
static NSArray *ltx_option_keys() {
    return @[
        @"ltx_backend", @"ltx_fast_av", @"ltx_video_attention_batch",
        @"ltx_sol_stage1", @"ltx_sol_stage2", @"ltx_sol_tau",
        @"ltx_sol_dense_edge_blocks", @"ltx_sol_dense_edge_steps",
        @"ltx_stage2_text_rows", @"ltx_sparse_mode", @"ltx_sparse_radius",
        @"ltx_sparse_anchor_stride", @"ltx_sparse_tokens_per_frame",
        @"ltx_sparse_keep_blocks"
    ];
}
static void keys_with_ltx_options(NSDictionary *d, NSArray *allowed) {
    NSMutableArray *combined = [allowed mutableCopy];
    [combined addObjectsFromArray:ltx_option_keys()];
    keys(d, combined);
}
static void parse_ltx_options(NSDictionary *d, Request &r) {
    r.ltx_backend = string_value(d, @"ltx_backend", r.ltx_backend);
    r.ltx_fast_av = boolean(d, @"ltx_fast_av", r.ltx_fast_av);
    r.ltx_video_attention_batch = boolean(
        d, @"ltx_video_attention_batch", r.ltx_video_attention_batch);
    r.ltx_sol_stage1 = boolean(d, @"ltx_sol_stage1", r.ltx_sol_stage1);
    r.ltx_sol_stage2 = boolean(d, @"ltx_sol_stage2", r.ltx_sol_stage2);
    r.ltx_sol_tau = numeric(d, @"ltx_sol_tau", r.ltx_sol_tau);
    r.ltx_sol_dense_edge_blocks = number(
        d, @"ltx_sol_dense_edge_blocks", r.ltx_sol_dense_edge_blocks);
    r.ltx_sol_dense_edge_steps = number(
        d, @"ltx_sol_dense_edge_steps", r.ltx_sol_dense_edge_steps);
    r.ltx_stage2_text_rows = number(
        d, @"ltx_stage2_text_rows", r.ltx_stage2_text_rows);
    r.ltx_sparse_mode = number(d, @"ltx_sparse_mode", r.ltx_sparse_mode);
    r.ltx_sparse_radius = number(d, @"ltx_sparse_radius", r.ltx_sparse_radius);
    r.ltx_sparse_anchor_stride = number(
        d, @"ltx_sparse_anchor_stride", r.ltx_sparse_anchor_stride);
    r.ltx_sparse_tokens_per_frame = number(
        d, @"ltx_sparse_tokens_per_frame", r.ltx_sparse_tokens_per_frame);
    r.ltx_sparse_keep_blocks = number(
        d, @"ltx_sparse_keep_blocks", r.ltx_sparse_keep_blocks);
}
void parse_memory_constrained(NSDictionary *d, MemoryConstrainedConfig &config) {
    require([d isKindOfClass:NSDictionary.class],
            "memory_constrained must be an object");
    keys(d, @[
        @"enabled", @"limit_bytes", @"buffer_percent", @"min_free_bytes",
        @"max_refill_slots", @"allow_quality_preserving_tiling"
    ]);
    if (d[@"enabled"]) {
        config.enabled = boolean(d, @"enabled", false);
        config.specified_fields |= MemoryFieldEnabled;
    }
    if (d[@"limit_bytes"]) {
        config.limit_bytes = byte_count(d, @"limit_bytes", 0);
        config.specified_fields |= MemoryFieldLimit;
    }
    if (d[@"buffer_percent"]) {
        config.buffer_percent =
            static_cast<unsigned>(number(d, @"buffer_percent", 15));
        config.specified_fields |= MemoryFieldBufferPercent;
    }
    if (d[@"min_free_bytes"]) {
        config.min_free_bytes = byte_count(d, @"min_free_bytes", 1ull << 30);
        config.specified_fields |= MemoryFieldMinFree;
    }
    if (d[@"max_refill_slots"]) {
        config.max_refill_slots =
            static_cast<unsigned>(number(d, @"max_refill_slots", 3));
        config.specified_fields |= MemoryFieldMaxRefillSlots;
    }
    if (d[@"allow_quality_preserving_tiling"]) {
        config.allow_quality_preserving_tiling = boolean(
            d, @"allow_quality_preserving_tiling", true);
        config.specified_fields |= MemoryFieldAllowTiling;
    }
}
Request request_from_json(NSDictionary *d) {
    int version = number(d, @"schema_version", 1);
    require(version == 1 || version == 2, "unsupported schema_version");
    Request r;
    if (version == 1) {
        keys_with_ltx_options(d, @[
            @"compile_gpu",  @"schema_version", @"model",
            @"prompt",       @"output",         @"execution",
            @"width",        @"height",         @"steps",
            @"seed",         @"frames",         @"dynamic_text",
            @"dump_tensors", @"ane_manifest", @"encoder_ane_manifest",
            @"allow_approximation",
            @"operation",    @"inputs",         @"fps",
            @"residency",    @"profile",        @"model_variant",
            @"loras",        @"audio",          @"noise_path",
            @"vsa",          @"vsa_sparsity",   @"vsa_tile_size",
            @"vsa_prefix_mode", @"vsa_dense_first_n_steps",
            @"vsa_dense_layers", @"vsa_impl",
            @"lora_strategy", @"streaming_offload", @"memory_budget_bytes",
            @"quantized_cache", @"warmup_iterations", @"memory_constrained"
        ]);
        r.model = string_value(d, @"model", r.model);
        r.model_variant = string_value(d, @"model_variant", r.model_variant);
        auto model_descriptor = module_for(r.model).describe();
        auto descriptor = to_dictionary(model_descriptor);
        r.operation = string_value(d, @"operation",
                                   [descriptor[@"output"] isEqual:@"image"] ? "image.generate"
                                                                            : "video.generate");
        r.prompt = string_value(d, @"prompt");
        r.output = string_value(d, @"output");
        r.execution = string_value(d, @"execution", "gpu");
        r.ane_manifest = string_value(d, @"ane_manifest");
        r.encoder_ane_manifest = string_value(d, @"encoder_ane_manifest");
        r.width = number(d, @"width", model_descriptor.width);
        r.height = number(d, @"height", model_descriptor.height);
        r.frames = number(d, @"frames", model_descriptor.frames);
        r.steps = number(d, @"steps", model_descriptor.steps);
        r.seed = number(d, @"seed", 42);
        r.fps = number(d, @"fps", model_descriptor.fps ? model_descriptor.fps : 24);
        r.compile_gpu = boolean(d, @"compile_gpu", false);
        r.dynamic_text = boolean(d, @"dynamic_text", true);
        r.allow_approximation = boolean(d, @"allow_approximation", false);
        r.audio = boolean(d, @"audio", model_descriptor.default_audio);
        r.residency = string_value(d, @"residency", model_descriptor.default_residency);
        r.residency_specified = d[@"residency"] != nil;
        r.memory_budget_specified = d[@"memory_budget_bytes"] != nil;
        r.streaming_offload_specified = d[@"streaming_offload"] != nil;
        r.streaming_offload = boolean(d, @"streaming_offload", false);
        r.memory_budget_bytes = byte_count(d, @"memory_budget_bytes", 0);
        if (d[@"memory_constrained"])
            parse_memory_constrained(d[@"memory_constrained"],
                                     r.memory_constrained);
        r.quantized_cache = string_value(d, @"quantized_cache");
        parse_ltx_options(d, r);
        r.warmup_iterations = number(d, @"warmup_iterations", 0);
        require(r.warmup_iterations >= 0 && r.warmup_iterations <= 8,
                "warmup_iterations must be 0...8");
        r.profile = string_value(d, @"profile");
        r.noise_path = string_value(d, @"noise_path");
        r.vsa = boolean(d, @"vsa", false);
        r.vsa_sparsity = numeric(d, @"vsa_sparsity", r.vsa_sparsity);
        r.vsa_tile_size = number(d, @"vsa_tile_size", r.vsa_tile_size);
        r.vsa_prefix_mode = string_value(d, @"vsa_prefix_mode", r.vsa_prefix_mode);
        r.vsa_dense_first_n_steps = number(d, @"vsa_dense_first_n_steps", 0);
        r.vsa_impl = string_value(d, @"vsa_impl", r.vsa_impl);
        if (d[@"vsa_dense_layers"]) {
            NSArray *layers = d[@"vsa_dense_layers"];
            require([layers isKindOfClass:NSArray.class],
                    "vsa_dense_layers must be an array");
            for (id raw in layers) {
                require([raw isKindOfClass:NSNumber.class] &&
                            CFGetTypeID((__bridge CFTypeRef)raw) != CFBooleanGetTypeID() &&
                            [raw doubleValue] == [raw intValue] && [raw intValue] >= 0,
                        "vsa_dense_layers entries must be nonnegative integers");
                r.vsa_dense_layers.push_back([raw intValue]);
            }
        }
    } else {
        keys(d, @[
            @"schema_version", @"model", @"operation", @"inputs", @"outputs", @"sampling",
            @"execution", @"parameters", @"dump_tensors", @"model_variant", @"loras",
            @"lora_strategy"
        ]);
        r.model = string_value(d, @"model", r.model);
        r.model_variant = string_value(d, @"model_variant", r.model_variant);
        auto model_descriptor = module_for(r.model).describe();
        auto descriptor = to_dictionary(model_descriptor);
        r.operation = string_value(d, @"operation");
        require(!r.operation.empty(), "operation is required");
        NSArray *outputs = d[@"outputs"];
        require([outputs isKindOfClass:NSArray.class] && outputs.count == 1,
                "one primary output is required");
        auto output = dictionary(outputs[0], "output");
        keys(output, @[ @"kind", @"path", @"width", @"height", @"frames", @"fps", @"audio" ]);
        require([@(string_value(output, @"kind").c_str()) isEqual:descriptor[@"output"]],
                "output kind does not match model");
        r.output = string_value(output, @"path");
        r.width = number(output, @"width", [descriptor[@"default_width"] intValue]);
        r.height = number(output, @"height", [descriptor[@"default_height"] intValue]);
        r.frames = number(output, @"frames", [descriptor[@"default_frames"] intValue]);
        r.fps = number(output, @"fps", model_descriptor.fps ? model_descriptor.fps : 24);
        r.audio = boolean(output, @"audio", model_descriptor.default_audio);
        auto sampling = d[@"sampling"] ? dictionary(d[@"sampling"], "sampling") : @{};
        keys(sampling, @[ @"seed", @"steps" ]);
        r.seed = number(sampling, @"seed", 42);
        r.steps = number(sampling, @"steps", [descriptor[@"default_steps"] intValue]);
        auto execution = d[@"execution"] ? dictionary(d[@"execution"], "execution") : @{};
        keys_with_ltx_options(
            execution,
            @[ @"policy", @"profile", @"ane_manifest", @"encoder_ane_manifest", @"allow_approximation",
               @"residency", @"memory_budget_bytes", @"warmup_iterations",
               @"quantized_cache", @"memory_constrained", @"streaming" ]);
        r.execution = string_value(execution, @"policy", "gpu");
        r.profile = string_value(execution, @"profile");
        r.ane_manifest = string_value(execution, @"ane_manifest");
        r.encoder_ane_manifest = string_value(execution, @"encoder_ane_manifest");
        r.allow_approximation = boolean(execution, @"allow_approximation", false);
        r.residency = string_value(execution, @"residency", model_descriptor.default_residency);
        r.residency_specified = execution[@"residency"] != nil;
        r.memory_budget_specified = execution[@"memory_budget_bytes"] != nil;
        if (execution[@"streaming"])
            parse_streaming_config(execution[@"streaming"], r.streaming, "request");
        r.memory_budget_bytes = byte_count(execution, @"memory_budget_bytes", 0);
        if (execution[@"memory_constrained"])
            parse_memory_constrained(execution[@"memory_constrained"],
                                     r.memory_constrained);
        r.quantized_cache = string_value(execution, @"quantized_cache");
        parse_ltx_options(execution, r);
        r.warmup_iterations = number(execution, @"warmup_iterations", 0);
        require(r.warmup_iterations >= 0 && r.warmup_iterations <= 8,
                "execution.warmup_iterations must be 0...8");
        auto parameters = d[@"parameters"] ? dictionary(d[@"parameters"], "parameters") : @{};
        keys(parameters, @[ @"dynamic_text", @"compile_gpu", @"noise_path",
                            @"streaming_offload", @"vsa", @"vsa_sparsity",
                            @"vsa_tile_size", @"vsa_prefix_mode",
                            @"vsa_dense_first_n_steps", @"vsa_dense_layers",
                            @"vsa_impl" ]);
        r.compile_gpu = boolean(parameters, @"compile_gpu", false);
        r.dynamic_text = boolean(parameters, @"dynamic_text", true);
        r.noise_path = string_value(parameters, @"noise_path");
        r.streaming_offload = boolean(parameters, @"streaming_offload", false);
        r.streaming_offload_specified = parameters[@"streaming_offload"] != nil;
        r.vsa = boolean(parameters, @"vsa", false);
        r.vsa_sparsity = numeric(parameters, @"vsa_sparsity", r.vsa_sparsity);
        r.vsa_tile_size = number(parameters, @"vsa_tile_size", r.vsa_tile_size);
        r.vsa_prefix_mode = string_value(parameters, @"vsa_prefix_mode", r.vsa_prefix_mode);
        r.vsa_dense_first_n_steps = number(parameters, @"vsa_dense_first_n_steps", 0);
        r.vsa_impl = string_value(parameters, @"vsa_impl", r.vsa_impl);
        if (parameters[@"vsa_dense_layers"]) {
            NSArray *layers = parameters[@"vsa_dense_layers"];
            require([layers isKindOfClass:NSArray.class],
                    "vsa_dense_layers must be an array");
            for (id raw in layers) {
                require([raw isKindOfClass:NSNumber.class] &&
                            CFGetTypeID((__bridge CFTypeRef)raw) != CFBooleanGetTypeID() &&
                            [raw doubleValue] == [raw intValue] && [raw intValue] >= 0,
                        "vsa_dense_layers entries must be nonnegative integers");
                r.vsa_dense_layers.push_back([raw intValue]);
            }
        }
    }
    r.lora_strategy = string_value(d, @"lora_strategy", r.lora_strategy);
    r.dump = string_value(d, @"dump_tensors");
    if (d[@"loras"]) {
        NSArray *loras = d[@"loras"];
        require([loras isKindOfClass:NSArray.class] && loras.count <= 8,
                "loras must contain at most 8 adapters");
        for (id value in loras) {
            auto lora = dictionary(value, "lora");
            keys(lora, @[ @"path", @"strength", @"role" ]);
            LoRAAsset adapter;
            adapter.path = string_value(lora, @"path");
            adapter.strength = float(numeric(lora, @"strength", 1.0));
            adapter.role = string_value(lora, @"role", "transformer");
            require(!adapter.path.empty(), "lora path is required");
            require(adapter.strength >= -8.0f && adapter.strength <= 8.0f,
                    "lora strength must be -8...8");
            require(adapter.role == "transformer" || adapter.role == "text_encoder" ||
                        adapter.role == "refiner",
                    "unsupported lora role");
            r.loras.push_back(std::move(adapter));
        }
    }
    if (d[@"inputs"]) {
        NSArray *inputs = d[@"inputs"];
        require([inputs isKindOfClass:NSArray.class] && inputs.count <= 32,
                "inputs must be an array of at most 32 assets");
        bool text_seen = !r.prompt.empty();
        for (id value in inputs) {
            auto input = dictionary(value, "input");
            keys(input, @[
                @"kind", @"role", @"path", @"text", @"audio_path", @"include_embedded_audio",
                @"strength"
            ]);
            InputAsset a;
            a.kind = string_value(input, @"kind");
            a.role = string_value(input, @"role");
            if (a.kind == "text") {
                require(a.role == "prompt" && !text_seen, "one prompt input is required");
                r.prompt = string_value(input, @"text");
                text_seen = true;
                continue;
            }
            a.path = string_value(input, @"path");
            require(!a.path.empty() && !a.role.empty(), "media input requires path and role");
            a.audio_path = string_value(input, @"audio_path");
            a.include_audio = boolean(input, @"include_embedded_audio", true);
            a.strength = float(numeric(input, @"strength", a.role == "first_frame" ? 1. : .75));
            require(a.strength >= 0 && a.strength <= 1, "strength must be 0...1");
            require(a.kind == "image" || a.kind == "audio" || a.kind == "video",
                    "unsupported media kind");
            r.inputs.push_back(std::move(a));
        }
    }
    if (r.streaming.specified()) r.streaming_requested = r.streaming;
    resolve_profile(r);
    if (r.streaming.specified()) validate_streaming_config(r.streaming);
    validate_memory_constrained_request(
        r, [NSProcessInfo processInfo].physicalMemory);
    if (!r.encoder_ane_manifest.empty()) {
        require(r.allow_approximation,
                "encoder_ane_manifest requires allow_approximation=true");
        require(module_for(r.model).describe().supports_encoder_gpu_ane,
                "encoder_ane_manifest is currently supported for FLUX/Z-Image "
                "Qwen3, LTX Gemma4, and H3 Qwen3-VL encoders only");
    }
    if (r.execution == "gpu_ane") {
        require(!r.ane_manifest.empty(),
                "gpu_ane requires an explicit ANE manifest or partition directory");
        if (r.model == "ltx-2.5-distilled")
            validate_ltx_ane_profile(r.ane_manifest, r.width, r.height, r.frames, r.fps);
        else if (r.model == "wan2.1-1.3b-qad")
            validate_wan_ane_manifest(r.ane_manifest);
        else if (r.model == "flux2-klein-4b")
            require(r.allow_approximation,
                    "FLUX gpu_ane requires allow_approximation=true");
    }
    return r;
}
} // namespace tc
