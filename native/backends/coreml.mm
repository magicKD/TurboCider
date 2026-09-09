#include "coreml.hpp"
#include "coreml_partitions.hpp"
#include "../platform/apple/bridge.hpp"
#include "../platform/apple/platform.hpp"
#import <CoreML/CoreML.h>
#include <set>
#include <algorithm>
namespace tc {
class CoreMLBranch {
    MLModel *model_;
    Tensor output_storage_;
    MLMultiArray *output_;
    MLPredictionOptions *options_;
    int rows_, hidden_;
    bool flexible_;

  public:
    double model_load_seconds = 0, interface_setup_seconds = 0;
    double seconds = 0, warmup_seconds = 0;
    double first_runtime_seconds = 0, subsequent_runtime_seconds = 0;
    uint64_t calls = 0, copied_bytes = 0, warmup_calls = 0, runtime_calls = 0;
    uint64_t first_runtime_calls = 0, subsequent_runtime_calls = 0;
    CoreMLBranch(const std::filesystem::path &, int rows, int hidden,
                 const Tensor &output_storage, MLMultiArray *output_backing, bool flexible = false);
    void bind(int rows, const Tensor &storage, MLMultiArray *output);
    Tensor predict(const Tensor &packed_input, int actual_rows, bool warmup = false);
};
struct HybridSession::Impl {
    std::vector<std::unique_ptr<CoreMLBranch>> branches;
    std::vector<int> buckets;
    bool flexible = false;
};
HybridSession::~HybridSession() = default;

struct CoreMLPartitions::Impl {
    std::vector<std::unique_ptr<CoreMLBranch>> branches;
};

CoreMLPartitions::~CoreMLPartitions() = default;

CoreMLPartitions::CoreMLPartitions(const std::vector<std::filesystem::path> &paths,
                                   int rows, int hidden, const Event &event,
                                   std::atomic<bool> &cancelled)
    : impl_(std::make_unique<Impl>()), rows_(rows), hidden_(hidden) {
    require(rows > 0 && rows <= 65536 && hidden > 0 && hidden <= 16384 &&
                !paths.empty() && paths.size() <= 64, "invalid Core ML partition geometry");
    checkpoint(cancelled);
    auto storage = mx::contiguous(mx::zeros({1, rows, hidden}, mx::float16));
    mx::eval(storage);
    NSError *error = nil;
    auto backing = [[MLMultiArray alloc] initWithDataPointer:storage.data<mx::float16_t>()
        shape:@[@1, @(hidden), @1, @(rows)] dataType:MLMultiArrayDataTypeFloat16
        strides:@[@(size_t(rows) * hidden), @1, @(size_t(rows) * hidden), @(hidden)]
        deallocator:^(void *) {} error:&error];
    require(backing != nil, "Core ML partition output allocation failed");
    for (size_t index = 0; index < paths.size(); ++index) {
        checkpoint(cancelled);
        require(paths[index].is_absolute(), "Core ML partition path must be absolute");
        event("coreml_partition_load", int(index), int(paths.size()));
        impl_->branches.push_back(std::make_unique<CoreMLBranch>(paths[index], rows, hidden, storage, backing, false));
    }
    checkpoint(cancelled);
    event("coreml_partition_load", int(paths.size()), int(paths.size()));
}

Tensor CoreMLPartitions::predict(int block, const Tensor &input) {
    require(block >= 0 && size_t(block) < impl_->branches.size(), "invalid Core ML partition index");
    require(input.shape() == mx::Shape{1, rows_, hidden_} && input.dtype() == mx::float16 &&
                input.flags().row_contiguous, "Core ML partition requires contiguous FP16 [1,R,H]");
    mx::eval(input);
    return impl_->branches[block]->predict(input, rows_);
}

uint64_t CoreMLPartitions::calls() const {
    uint64_t total = 0;
    for (const auto &branch : impl_->branches) total += branch->calls;
    return total;
}

uint64_t CoreMLPartitions::copied_bytes() const {
    uint64_t total = 0;
    for (const auto &branch : impl_->branches) total += branch->copied_bytes;
    return total;
}

CoreMLBranch::CoreMLBranch(const std::filesystem::path &path, int rows, int hidden,
                           const Tensor &output_storage, MLMultiArray *output_backing, bool flexible)
    : output_storage_(output_storage), output_(output_backing), rows_(rows), hidden_(hidden), flexible_(flexible) {
    auto setup_begin = Clock::now();
    require(path.extension() == ".mlmodelc" && std::filesystem::is_directory(path),
            "expected compiled Core ML artifact: " + path.string());
    auto config = [MLModelConfiguration new];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    config.functionName = @"main";
    if (flexible) {
        auto hints = [MLOptimizationHints new];
        hints.reshapeFrequency = MLReshapeFrequencyHintInfrequent;
        config.optimizationHints = hints;
    }
    NSError *error = nil;
    auto model_load_begin = Clock::now();
    model_ = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(path.c_str())]
                               configuration:config
                                       error:&error];
    model_load_seconds =
        std::chrono::duration<double>(Clock::now() - model_load_begin).count();
    require(model_ != nil,
            "Core ML load failed: " +
                std::string(error ? error.localizedDescription.UTF8String : "unknown"));
    auto input = model_.modelDescription.inputDescriptionsByName[@"x"].multiArrayConstraint;
    auto output = model_.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint;
    require(input && output && input.dataType == MLMultiArrayDataTypeFloat16 &&
                output.dataType == MLMultiArrayDataTypeFloat16,
            "Core ML feature dtype mismatch");
    bind(rows, output_storage, output_backing);
    interface_setup_seconds =
        std::chrono::duration<double>(Clock::now() - setup_begin).count() - model_load_seconds;
}
void CoreMLBranch::bind(int rows, const Tensor &storage, MLMultiArray *output) {
    auto bind_begin = Clock::now();
    NSArray *shape = @[ @1, @(hidden_), @1, @(rows) ];
    auto constraint = model_.modelDescription.inputDescriptionsByName[@"x"].multiArrayConstraint;
    bool compatible = [constraint.shape isEqual:shape];
    auto flexible = constraint.shapeConstraint;
    if (flexible_ && flexible.type == MLMultiArrayShapeConstraintTypeEnumerated)
        compatible = [flexible.enumeratedShapes containsObject:shape];
    if (flexible_ && flexible.type == MLMultiArrayShapeConstraintTypeRange && flexible.sizeRangeForDimension.count == 4) {
        compatible = true;
        for (int i = 0; i < 4; ++i)
            compatible &= NSLocationInRange([shape[i] unsignedIntegerValue], [flexible.sizeRangeForDimension[i] rangeValue]);
    }
    auto outputConstraint = model_.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint;
    require(compatible && (flexible_ || [outputConstraint.shape isEqual:shape]),
            "Core ML feature shape ABI mismatch for " + std::to_string(rows) + " rows");
    require(output != nil, "Core ML shared output backing missing");
    rows_ = rows;
    output_storage_ = storage;
    output_ = output;
    options_ = [MLPredictionOptions new];
    options_.outputBackings = @{@"y" : output_};
    interface_setup_seconds += std::chrono::duration<double>(Clock::now() - bind_begin).count();
}
Tensor CoreMLBranch::predict(const Tensor &input, int actual, bool warmup) {
    // Input has been materialized before GPU attention submission. No writable alias
    // is exposed to callers; output storage is leased until the block completes.
    auto begin = Clock::now();
    NSError *error = nil;
    MLMultiArray *in =
        [[MLMultiArray alloc] initWithDataPointer:(void *)input.data<mx::float16_t>()
                                            shape:@[ @1, @(hidden_), @1, @(rows_) ]
                                         dataType:MLMultiArrayDataTypeFloat16
                                          strides:@[ @(rows_ * hidden_), @1, @(rows_ * hidden_), @(hidden_) ]
                                      deallocator:^(void *) {
                                      }
                                            error:&error];
    require(in != nil, "Core ML input binding failed");
    auto provider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{@"x" : [MLFeatureValue featureValueWithMultiArray:in]}
                     error:&error];
    auto result = [model_ predictionFromFeatures:provider options:options_ error:&error];
    require(result != nil,
            "Core ML prediction failed: " +
                std::string(error ? error.localizedDescription.UTF8String : "unknown"));
    auto actual_output = [result featureValueForName:@"y"].multiArrayValue;
    require(actual_output != nil &&
                [actual_output.shape isEqual:@[ @1, @(hidden_), @1, @(rows_) ]] &&
                actual_output.dataType == MLMultiArrayDataTypeFloat16,
            "Core ML returned an unexpected output shape or dtype");
    if (actual_output.dataPointer != output_.dataPointer) {
        copied_bytes += uint64_t(rows_) * uint64_t(hidden_) * 2;
        // Strides can differ when the framework declines the caller output backing.
        for (int row = 0; row < rows_; ++row)
            for (int c = 0; c < hidden_; ++c) {
                size_t offset = row * [actual_output.strides[3] unsignedLongLongValue] +
                                c * [actual_output.strides[1] unsignedLongLongValue];
                ((uint16_t *)output_storage_.data<mx::float16_t>())[size_t(row) * hidden_ + c] =
                    ((uint16_t *)actual_output.dataPointer)[offset];
            }
    }
    // The model coordinator and per-block eval guarantee that the prior
    // consumer has completed before this branch writes its next output. Core ML
    // writes directly into an MLX-owned shared buffer; no tensor escapes the block.
    const double elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
    ++calls;
    seconds += elapsed;
    if (warmup) {
        ++warmup_calls;
        warmup_seconds += elapsed;
    } else {
        ++runtime_calls;
        if (first_runtime_calls == 0) {
            ++first_runtime_calls;
            first_runtime_seconds += elapsed;
        } else {
            ++subsequent_runtime_calls;
            subsequent_runtime_seconds += elapsed;
        }
    }
    return slice_axis(output_storage_, 1, 0, actual);
}
HybridSession::HybridSession(const std::filesystem::path &file, const std::filesystem::path &model,
                             int tokens, const Event &event, std::atomic<bool> &cancelled,
                             int warmups, const std::filesystem::path &requested_checkpoint,
                             const std::vector<LoRAAsset> &requested_loras, int policy_rows)
    : impl_(std::make_unique<Impl>()), manifest(file.string()) {
    auto begin = Clock::now();
    auto d = read_json(file);
    require([d[@"schema_version"] isKindOfClass:NSNumber.class] &&
                [d[@"shape"] isKindOfClass:NSDictionary.class] &&
                [d[@"source"] isKindOfClass:NSDictionary.class] &&
                [d[@"artifacts"] isKindOfClass:NSDictionary.class],
            "invalid hybrid manifest containers");
    require([d[@"shape"][@"K"] isKindOfClass:NSNumber.class] &&
                [d[@"shape"][@"N"] isKindOfClass:NSNumber.class] &&
                [d[@"source"][@"checkpoint_bytes"] isKindOfClass:NSNumber.class],
            "invalid hybrid manifest numbers");
    require([d[@"schema_version"] intValue] == 2, "hybrid requires manifest schema 2");
    hidden = [d[@"shape"][@"K"] intValue];
    require(hidden > 0 && hidden <= 8192 && [d[@"shape"][@"N"] intValue] == hidden,
            "hybrid hidden dimension mismatch");
    NSArray *buckets = d[@"shape"][@"buckets"];
    const auto mode = string_value(d[@"shape"], @"input_mode", "fixed");
    impl_->flexible = mode == "enumerated" || mode == "range";
    require(mode == "fixed" || impl_->flexible, "unknown Core ML input shape mode");
    require([buckets isKindOfClass:NSArray.class] && buckets.count > 0 && buckets.count <= 128 &&
                (impl_->flexible || buckets.count == 1), "invalid Core ML input buckets");
    int previous = 0;
    for (id bucket in buckets) {
        require([bucket isKindOfClass:NSNumber.class] && [bucket doubleValue] == [bucket intValue] &&
                    [bucket intValue] > previous && [bucket intValue] <= 8192,
                "Core ML buckets must be increasing positive integers up to 8192");
        previous = [bucket intValue];
        impl_->buckets.push_back(previous);
    }
    set_tokens(tokens);
    require(!policy_rows || rows == policy_rows, "Core ML bucket has no matching measured policy");
    id manifest_mlp_width = d[@"shape"][@"mlp_width"];
    id manifest_mlp_start = d[@"shape"][@"ane_mlp_start"];
    id manifest_mlp_end = d[@"shape"][@"ane_mlp_end"];
    // Legacy FLUX manifests describe a full 9,216-channel MLP branch implicitly.
    // New prefix manifests make the split explicit; the C++ GPU branch must
    // compute every channel outside this ANE-owned prefix.
    mlp_width = [manifest_mlp_width isKindOfClass:NSNumber.class]
                    ? [manifest_mlp_width intValue]
                    : 9216;
    ane_mlp_start = [manifest_mlp_start isKindOfClass:NSNumber.class]
                        ? [manifest_mlp_start intValue]
                        : 0;
    ane_mlp_end = [manifest_mlp_end isKindOfClass:NSNumber.class]
                      ? [manifest_mlp_end intValue]
                      : mlp_width;
    id manifest_output_scale = d[@"shape"][@"output_scale"];
    output_scale = [manifest_output_scale isKindOfClass:NSNumber.class]
                       ? [manifest_output_scale floatValue]
                       : 1.f;
    require(mlp_width > 0 && mlp_width <= 65536 && ane_mlp_start == 0 &&
                ane_mlp_end > 0 && ane_mlp_end <= mlp_width && std::isfinite(output_scale) &&
                output_scale >= 1.f && output_scale <= 256.f,
            "unsupported Core ML MLP partition; expected a nonempty [0,N) prefix");
    auto checkpoint = requested_checkpoint.empty()
                          ? model / "transformer/diffusion_pytorch_model.safetensors"
                          : requested_checkpoint;
    std::filesystem::path source = string_value(d[@"source"], @"checkpoint");
    require(std::filesystem::equivalent(source, checkpoint) &&
                [d[@"source"][@"checkpoint_bytes"] unsignedLongLongValue] ==
                    std::filesystem::file_size(checkpoint),
            "artifact checkpoint provenance mismatch");
    id checkpoint_sha = d[@"source"][@"checkpoint_sha256"];
    if ([checkpoint_sha isKindOfClass:NSString.class]) {
        require(sha256_file(checkpoint) ==
                    std::string([(NSString *)checkpoint_sha UTF8String]),
                "artifact checkpoint SHA-256 mismatch");
        checkpoint_sha_verified = true;
    }
    const bool indexed_checkpoint = checkpoint.filename().string().ends_with(
        ".safetensors.index.json");
    NSDictionary *checkpoint_shards = d[@"source"][@"checkpoint_shards"];
    if (indexed_checkpoint) {
        auto index = read_json(checkpoint);
        NSDictionary *weight_map = index[@"weight_map"];
        require([weight_map isKindOfClass:NSDictionary.class] && weight_map.count > 0 &&
                    [checkpoint_shards isKindOfClass:NSDictionary.class],
                "indexed checkpoint requires a complete shard identity map");
        std::set<std::string> expected_shards;
        for (NSString *tensor in weight_map) {
            id value = weight_map[tensor];
            require([tensor isKindOfClass:NSString.class] && tensor.length > 0 &&
                        [value isKindOfClass:NSString.class],
                    "invalid checkpoint weight map");
            expected_shards.emplace([(NSString *)value UTF8String]);
        }
        require(checkpoint_shards.count == expected_shards.size(),
                "checkpoint shard identity map is incomplete");
        for (const auto &expected_name : expected_shards) {
            NSString *name = @(expected_name.c_str());
            require(name.length > 0 && [name rangeOfString:@"/"].location == NSNotFound &&
                        [name rangeOfString:@"\\"].location == NSNotFound &&
                        ![name isEqual:@"."] && ![name isEqual:@".."],
                    "invalid checkpoint shard name");
            auto shard = source.parent_path() / name.UTF8String;
            NSDictionary *identity = checkpoint_shards[name];
            require([identity isKindOfClass:NSDictionary.class] &&
                        [identity[@"bytes"] isKindOfClass:NSNumber.class] &&
                        std::filesystem::is_regular_file(shard) &&
                        std::filesystem::file_size(shard) ==
                            [identity[@"bytes"] unsignedLongLongValue],
                    "artifact checkpoint shard provenance mismatch");
            id shard_sha = identity[@"sha256"];
            require([shard_sha isKindOfClass:NSString.class] &&
                        sha256_file(shard) ==
                            std::string([(NSString *)shard_sha UTF8String]),
                    "artifact checkpoint shard SHA-256 mismatch");
        }
    } else
        require(checkpoint_shards == nil,
                "single-file checkpoint must not declare sharded provenance");
    id manifest_loras_value = d[@"source"][@"loras"];
    NSArray *manifest_loras = nil;
    if (manifest_loras_value != nil) {
        require([manifest_loras_value isKindOfClass:NSArray.class],
                "artifact LoRA provenance must be an array");
        manifest_loras = manifest_loras_value;
    }
    const NSUInteger manifest_lora_count = manifest_loras ? manifest_loras.count : 0;
    require(manifest_lora_count == requested_loras.size(),
            requested_loras.empty()
                ? "LoRA-bound Core ML artifact cannot serve a base request"
                : "Core ML artifact does not match the active LoRA set");
    for (NSUInteger index = 0; index < manifest_lora_count; ++index) {
        tc::checkpoint(cancelled);
        NSDictionary *identity = manifest_loras[index];
        require([identity isKindOfClass:NSDictionary.class] &&
                    [identity[@"path"] isKindOfClass:NSString.class] &&
                    [identity[@"bytes"] isKindOfClass:NSNumber.class] &&
                    [identity[@"sha256"] isKindOfClass:NSString.class] &&
                    [identity[@"role"] isKindOfClass:NSString.class] &&
                    [identity[@"strength"] isKindOfClass:NSNumber.class],
                "invalid Core ML LoRA provenance record");
        const auto &requested = requested_loras[index];
        std::error_code path_error;
        auto requested_path = std::filesystem::canonical(requested.path, path_error);
        require(!path_error && std::filesystem::is_regular_file(requested_path) &&
                    !std::filesystem::is_symlink(requested_path),
                "active LoRA file is missing or invalid: " + requested.path);
        std::filesystem::path manifest_path = string_value(identity, @"path");
        require(manifest_path.is_absolute() && std::filesystem::is_regular_file(manifest_path) &&
                    std::filesystem::equivalent(manifest_path, requested_path) &&
                    [identity[@"bytes"] unsignedLongLongValue] ==
                        std::filesystem::file_size(requested_path),
                "Core ML LoRA path or size mismatch");
        require(string_value(identity, @"role") == requested.role &&
                    std::abs([identity[@"strength"] doubleValue] -
                             double(requested.strength)) <= 1e-7,
                "Core ML LoRA role or strength mismatch");
        require(sha256_file(requested_path) == string_value(identity, @"sha256"),
                "Core ML LoRA SHA-256 mismatch");
    }
    lora_identity_verified = !requested_loras.empty();
    // Existing local manifests lack a full source SHA; explicitly research-only.
    block_count = int([d[@"artifacts"] count]);
    require(block_count > 0 && block_count <= 64,
            "hybrid manifest has an invalid block count");
    manifest_validation_seconds =
        std::chrono::duration<double>(Clock::now() - begin).count();
    // Blocks execute serially and z_block materializes the prior consumer
    // before the next prediction. One session-wide backing therefore avoids
    // retaining block_count identical rows*hidden FP16 buffers without
    // changing the prediction ABI or exposing a writable tensor to callers.
    auto output_setup_begin = Clock::now();
    auto output_storage = mx::contiguous(mx::zeros({1, rows, hidden}, mx::float16));
    mx::eval(output_storage);
    require(output_storage.data_size() == size_t(rows) * size_t(hidden) &&
                output_storage.flags().row_contiguous,
            "Core ML output backing must be fully materialized and contiguous");
    NSError *output_error = nil;
    auto output =
        [[MLMultiArray alloc] initWithDataPointer:output_storage.data<mx::float16_t>()
                                            shape:@[ @1, @(hidden), @1, @(rows) ]
                                         dataType:MLMultiArrayDataTypeFloat16
                                          strides:@[ @(rows * hidden), @1, @(rows * hidden), @(hidden) ]
                                      deallocator:^(void *) {
                                      }
                                            error:&output_error];
    require(output != nil, "Core ML shared backing allocation failed");
    output_backing_setup_seconds =
        std::chrono::duration<double>(Clock::now() - output_setup_begin).count();
    for (int i = 0; i < block_count; ++i) {
        tc::checkpoint(cancelled);
        event("coreml_load", i, block_count);
        NSString *k = [NSString stringWithFormat:@"%d", i];
        auto relative = string_value(d[@"artifacts"][k], @"int8_pc");
        require(!relative.empty(), "Core ML manifest missing block");
        auto path = file.parent_path() / relative;
        require(std::filesystem::weakly_canonical(path).string().starts_with(
                    std::filesystem::weakly_canonical(file.parent_path()).string() + "/"),
                "artifact path escapes manifest directory");
        impl_->branches.push_back(
            std::make_unique<CoreMLBranch>(path, rows, hidden, output_storage, output, impl_->flexible));
    }
    if (warmups) {
        auto warmup_begin = Clock::now();
        auto input = mx::zeros({1, rows, hidden}, mx::float16);
        mx::eval(input);
        for (int iteration = 0; iteration < warmups; ++iteration)
            for (int block = 0; block < block_count; ++block) {
                tc::checkpoint(cancelled);
                event("coreml_warmup", iteration * block_count + block,
                      warmups * block_count);
                auto result = impl_->branches[block]->predict(input, rows, true);
                mx::eval(result);
            }
        zero_input_warmup_seconds =
            std::chrono::duration<double>(Clock::now() - warmup_begin).count();
    }
    load_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
}
void HybridSession::set_tokens(int tokens) {
    require(tokens > 0, "Core ML input row count must be positive");
    auto chosen = std::lower_bound(impl_->buckets.begin(), impl_->buckets.end(), tokens);
    require(chosen != impl_->buckets.end(),
            "Core ML token capacity exceeded: request needs " + std::to_string(tokens) +
            " rows, artifact supports at most " + std::to_string(impl_->buckets.back()) +
            ". Select a larger/flexible artifact or use GPU.");
    if (rows == *chosen) return;
    const int selected = *chosen;
    if (!impl_->branches.empty()) {
        auto storage = mx::contiguous(mx::zeros({1, selected, hidden}, mx::float16));
        mx::eval(storage);
        NSError *error = nil;
        auto output = [[MLMultiArray alloc] initWithDataPointer:storage.data<mx::float16_t>()
            shape:@[ @1, @(hidden), @1, @(selected) ] dataType:MLMultiArrayDataTypeFloat16
            strides:@[ @(selected * hidden), @1, @(selected * hidden), @(hidden) ]
            deallocator:^(void *) {} error:&error];
        require(output != nil, "Core ML flexible backing allocation failed");
        for (auto &branch : impl_->branches) branch->bind(selected, storage, output);
    }
    rows = selected;
}
Tensor HybridSession::predict(int block, const Tensor &input) {
    try { return impl_->branches.at(block)->predict(input, input.shape(1)); }
    catch (const std::exception &error) {
        throw std::runtime_error("Core ML block " + std::to_string(block) +
            " (" + std::to_string(rows) + " rows): " + error.what());
    }
}
HybridMetrics HybridSession::metrics() const {
    HybridMetrics metrics;
    metrics.load_seconds = load_seconds;
    metrics.manifest_validation_seconds = manifest_validation_seconds;
    metrics.output_backing_setup_seconds = output_backing_setup_seconds;
    metrics.zero_input_warmup_seconds = zero_input_warmup_seconds;
    metrics.bucket = rows;
    metrics.hidden = hidden;
    metrics.block_count = block_count;
    metrics.mlp_width = mlp_width;
    metrics.ane_mlp_start = ane_mlp_start;
    metrics.ane_mlp_end = ane_mlp_end;
    metrics.output_scale = output_scale;
    metrics.checkpoint_sha_verified = checkpoint_sha_verified;
    metrics.lora_identity_verified = lora_identity_verified;
    for (auto &branch : impl_->branches) {
        metrics.model_load_seconds += branch->model_load_seconds;
        metrics.model_interface_setup_seconds += branch->interface_setup_seconds;
        metrics.calls += branch->calls;
        metrics.copied_bytes += branch->copied_bytes;
        metrics.prediction_seconds += branch->seconds;
        metrics.warmup_calls += branch->warmup_calls;
        metrics.runtime_calls += branch->runtime_calls;
        metrics.first_runtime_prediction_calls += branch->first_runtime_calls;
        metrics.subsequent_runtime_prediction_calls += branch->subsequent_runtime_calls;
        metrics.first_runtime_prediction_seconds += branch->first_runtime_seconds;
        metrics.subsequent_runtime_prediction_seconds += branch->subsequent_runtime_seconds;
    }
    return metrics;
}
} // namespace tc
