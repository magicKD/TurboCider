#pragma once

#include "../../runtime/streaming/gguf_weight_pager.hpp"
#include "../../runtime/streaming/context.hpp"

namespace tc::z_image {

struct GgufExecutionPlan {
    streaming::Descriptor descriptor;
    streaming::Layout layout;
    StreamingConfig config;
    uint64_t packed_capacity_upper = 0, dense_capacity_upper = 0;
};

// This first execution profile preserves original float fields and expands
// only the quantized main-block matrices. Already-floating fixed/refiner
// weights alias their packed-resident source buffers, not a second dense bank.
GgufExecutionPlan describe_gguf_execution(
    std::shared_ptr<const streaming::SourceLease>, uint32_t prefetch_layers,
    uint32_t width, uint32_t height, uint32_t caption_rows, uint32_t steps,
    const std::string &precision_profile = "z-source-mixed-v1");

} // namespace tc::z_image
