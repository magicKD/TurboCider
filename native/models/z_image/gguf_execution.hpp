#pragma once

#include "../../runtime/streaming/gguf_weight_pager.hpp"
#include "../../runtime/streaming/context.hpp"

namespace tc::z_image {

struct GgufExecutionPlan {
    streaming::Descriptor descriptor;
    streaming::Layout layout;
    StreamingConfig config;
    uint64_t packed_capacity_upper = 0, dense_capacity_upper = 0, read_capacity_upper = 0;
    uint64_t gpu_prepare_capacity_upper=0;
};

// Shared strict Z naming/geometry validation, also used by the experimental
// immutable packed-bank importer. No payload allocation or inference.
void validate_gguf_model_directory(const gguf::Directory &);
// Source-mixed profiles preserve original floating fields. packed_streamed
// includes interleaved refiners; packed_resident preserves its prior aliases.
GgufExecutionPlan describe_gguf_execution(
    std::shared_ptr<const streaming::SourceLease>, uint32_t prefetch_layers,
    uint32_t width, uint32_t height, uint32_t caption_rows, uint32_t steps,
    const std::string &precision_profile = "z-source-mixed-v1",
    const std::string &source_residency = "packed_resident");

} // namespace tc::z_image
