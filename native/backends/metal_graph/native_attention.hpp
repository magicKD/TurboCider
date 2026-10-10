#pragma once

#include "backend.hpp"

namespace tc::metal_graph {

// Head-major [H,L,D], contiguous D. A token range may retain a larger source
// head stride and a nonzero BufferView offset. Sources/readiness are retained
// by the graph; no MLX tensor or physical copy is part of this interface.
struct HeadView {
    BufferView storage;
    uint32_t heads = 0;
    uint32_t tokens = 0;
    uint32_t dimension = 128;
    size_t token_stride_bytes = 0;
    size_t head_stride_bytes = 0;
};
struct AttentionGeometry {
    uint32_t heads = 0; // 1..32; local production models use 30 or 32.
    uint32_t dimension = 128;
    uint32_t batch = 1;
    uint32_t kv_heads = 0; // 0 means equal heads; explicit divisor enables GQA.
    ScalarType type = ScalarType::BF16; // D128 BF16 or zero-padded D80 FP32 vision.
};
struct AttentionOptions {
    bool causal = false;
    // Key logical positions start at zero. This offset affects visibility,
    // never the physical Q/O pointer; suffix attention normally uses K-Q.
    uint32_t query_position_offset = 0;
    float scale = 0.08838834764831844f; // 1/sqrt(128), before exp2 conversion.
};
struct AttentionBindings {
    HeadView query, key, value, output;
    AttentionOptions options;
};

class NativeAttention {
public:
    static NativeAttention prepare(Backend&, AttentionGeometry);
    AttentionGeometry geometry() const noexcept { return geometry_; }
    static constexpr size_t temporary_bytes() noexcept { return 0; }
    static constexpr std::string_view precision_recipe() noexcept {
        return "steel-bf16-d128-online-softmax-fp32-accum-v1";
    }
    // Head-major BF16 output. No score matrix, concat, compilation or submit.
    // If K/V were concatenated, the caller budgets those buffers explicitly.
    // Explicit GQA divisor supported; batch>1, arbitrary masks/sinks unsupported.
    void encode(CommandGraph&, const AttentionBindings&) const;
private:
    NativeAttention() = default;
    AttentionGeometry geometry_;
    Pipeline noncausal_, causal_;
};

} // namespace tc::metal_graph
