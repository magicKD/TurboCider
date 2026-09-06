#include "coreml.hpp"
#include "../platform/apple/bridge.hpp"
#include "../platform/apple/platform.hpp"
#import <CoreML/CoreML.h>
namespace tc {
class CoreMLBranch {
    MLModel *model_;
    MLMultiArray *output_;
    MLPredictionOptions *options_;
    Tensor output_storage_;
    int rows_, hidden_;

  public:
    double seconds = 0;
    uint64_t calls = 0, copied_bytes = 0;
    CoreMLBranch(const std::filesystem::path &, int rows, int hidden);
    Tensor predict(const Tensor &packed_input, int actual_rows);
};
struct HybridSession::Impl {
    std::vector<std::unique_ptr<CoreMLBranch>> branches;
};
HybridSession::~HybridSession() = default;

CoreMLBranch::CoreMLBranch(const std::filesystem::path &path, int rows, int hidden)
    : output_storage_(mx::contiguous(mx::zeros({1, rows, hidden}, mx::float16))), rows_(rows),
      hidden_(hidden) {
    require(path.extension() == ".mlmodelc" && std::filesystem::is_directory(path),
            "expected compiled Core ML artifact: " + path.string());
    auto config = [MLModelConfiguration new];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    config.functionName = @"main";
    NSError *error = nil;
    model_ = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(path.c_str())]
                               configuration:config
                                       error:&error];
    require(model_ != nil,
            "Core ML load failed: " +
                std::string(error ? error.localizedDescription.UTF8String : "unknown"));
    auto input = model_.modelDescription.inputDescriptionsByName[@"x"].multiArrayConstraint;
    auto output = model_.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint;
    NSArray *shape = @[ @1, @(hidden), @1, @(rows) ];
    require(input && output && [input.shape isEqual:shape] && [output.shape isEqual:shape] &&
                input.dataType == MLMultiArrayDataTypeFloat16 &&
                output.dataType == MLMultiArrayDataTypeFloat16,
            "Core ML feature ABI mismatch");
    mx::eval(output_storage_);
    require(output_storage_.data_size() == size_t(rows) * size_t(hidden) &&
                output_storage_.flags().row_contiguous,
            "Core ML output backing must be fully materialized and contiguous");
    output_ =
        [[MLMultiArray alloc] initWithDataPointer:output_storage_.data<mx::float16_t>()
                                            shape:shape
                                         dataType:MLMultiArrayDataTypeFloat16
                                          strides:@[ @(rows * hidden), @1, @(rows * hidden), @(hidden) ]
                                      deallocator:^(void *) {
                                      }
                                            error:&error];
    require(output_ != nil, "Core ML backing allocation failed");
    options_ = [MLPredictionOptions new];
    options_.outputBackings = @{@"y" : output_};
}
Tensor CoreMLBranch::predict(const Tensor &input, int actual) {
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
    require(actual_output != nil, "missing Core ML output");
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
    ++calls;
    seconds += std::chrono::duration<double>(Clock::now() - begin).count();
    return slice_axis(output_storage_, 1, 0, actual);
}
HybridSession::HybridSession(const std::filesystem::path &file, const std::filesystem::path &model,
                             int tokens, const Event &event, std::atomic<bool> &cancelled,
                             int warmups, const std::filesystem::path &requested_checkpoint)
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
    require([buckets isKindOfClass:NSArray.class] && buckets.count == 1 &&
                [buckets[0] isKindOfClass:NSNumber.class],
            "native hybrid requires a single fixed bucket");
    rows = [buckets[0] intValue];
    require(rows >= tokens && rows <= 8192, "Core ML token bucket cannot serve this request");
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
    require(mlp_width > 0 && mlp_width <= 65536 && ane_mlp_start == 0 &&
                ane_mlp_end > 0 && ane_mlp_end <= mlp_width,
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
        require(sha256_file(checkpoint) == std::string([(NSString *)checkpoint_sha UTF8String]),
                "artifact checkpoint SHA-256 mismatch");
        checkpoint_sha_verified = true;
    }
    // Existing local manifests lack a full source SHA; explicitly research-only.
    block_count = int([d[@"artifacts"] count]);
    require(block_count > 0 && block_count <= 64,
            "hybrid manifest has an invalid block count");
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
        impl_->branches.push_back(std::make_unique<CoreMLBranch>(path, rows, hidden));
    }
    if (warmups) {
        auto input = mx::zeros({1, rows, hidden}, mx::float16);
        mx::eval(input);
        for (int iteration = 0; iteration < warmups; ++iteration)
            for (int block = 0; block < block_count; ++block) {
                tc::checkpoint(cancelled);
                event("coreml_warmup", iteration * block_count + block,
                      warmups * block_count);
                auto result = impl_->branches[block]->predict(input, rows);
                mx::eval(result);
            }
    }
    load_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
}
Tensor HybridSession::predict(int block, const Tensor &input) {
    return impl_->branches.at(block)->predict(input, input.shape(1));
}
HybridMetrics HybridSession::metrics() const {
    HybridMetrics metrics;
    metrics.load_seconds = load_seconds;
    metrics.bucket = rows;
    metrics.hidden = hidden;
    metrics.block_count = block_count;
    metrics.mlp_width = mlp_width;
    metrics.ane_mlp_start = ane_mlp_start;
    metrics.ane_mlp_end = ane_mlp_end;
    metrics.checkpoint_sha_verified = checkpoint_sha_verified;
    for (auto &branch : impl_->branches) {
        metrics.calls += branch->calls;
        metrics.copied_bytes += branch->copied_bytes;
        metrics.prediction_seconds += branch->seconds;
    }
    return metrics;
}
} // namespace tc
