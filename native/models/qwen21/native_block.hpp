#pragma once

#include "../../backends/metal_graph/native_attention.hpp"
#include "../../backends/metal_graph/native_linear.hpp"
#include "../../backends/metal_graph/native_lora.hpp"
#include "../../backends/metal_graph/native_qwen_ops.hpp"
#include "sequence.hpp"

#include <array>
#include <optional>

namespace tc::qwen21::native_graph {

using metal_graph::MatrixView;
using metal_graph::HeadView;

enum class BlockPhase : uint8_t { PrefixBuild, SteadyDecode };
struct BlockGeometry {
    uint32_t hidden = 4096;
    uint32_t heads = 32;
    uint32_t head_dim = 128;
    uint32_t ffn_width = 12288;
};
void validate_geometry(BlockGeometry);

struct Projection {
    MatrixView weight;
    std::optional<MatrixView> bias;
    // Original order, with a BF16 addition boundary after every correction.
    std::vector<metal_graph::OrderedLoRATerm> lora{};
};
struct BlockWeights {
    Projection query, key, value, attention_output;
    MatrixView query_norm, key_norm; // BF16 [1,D]
    MatrixView gate, up, down;       // Preserve original fused source strides.
    std::vector<metal_graph::OrderedLoRATerm> gate_lora{}, up_lora{}, down_lora{};
};
struct PrefixKv {
    HeadView key, value;
};
struct BlockBindings {
    MatrixView input, output; // BF16 [M,H], disjoint; no MLX in this interface.
    // Actual Qwen order: scale_attn, gate_attn, scale_ffn, gate_ffn.
    // Each is BF16 [1,H] or [M,H]; no AdaLN shift vectors.
    std::array<MatrixView,4> modulation;
    MatrixView cosine, sine; // FP32 [M,D/2], corresponding to current rows.
    BlockWeights weights;
    BlockPhase phase = BlockPhase::SteadyDecode;
    uint32_t prefix_tokens = 0;
    // PrefixBuild writes these after RoPE; SteadyDecode reads immutable cache.
    // The enclosing denoiser must validate source/layer/geometry/epoch identity
    // and GPU producer readiness before binding a prefix cache.
    std::optional<PrefixKv> prefix;
    // PrefixBuild only: contiguous partition of all M rows, with target last.
    // Each segment attends to keys [0,end); causal rows use logical start.
    std::vector<SequenceSegment> segments;
};

struct BlockWorkspace {
    uint32_t rows = 0, prefix_tokens = 0;
    BlockPhase phase = BlockPhase::SteadyDecode;
    size_t reserved_bytes = 0;
    MatrixView normalized_attention, raw_query, raw_key, raw_value;
    HeadView query, key, value, attended, joined_key, joined_value;
    MatrixView row_attention, projected_attention, attention_residual;
    MatrixView normalized_ffn, ffn_hidden, ffn_output;
    // Optional request-shared scratch, outside the unchanged base arena.
    // Each view has the current phase's rows. No simultaneous block execution.
    MatrixView lora_low, lora_gate, lora_up;
};

struct ParallelFfnWorkspace {
    MatrixView gpu_hidden;  // BF16 [M,F/2], may be a pitched view of ffn_hidden.
    MatrixView gpu_partial, ane_partial; // Distinct FP32 [M,H] destinations.
};
struct ParallelBlockPlan {
    metal_graph::CommandGraph before_ffn; // Attention through normalized_ffn.
    metal_graph::CommandGraph gpu_ffn;    // First half of source channels, FP32 down.
    metal_graph::CommandGraph after_join; // FP32 add→BF16, then gated residual.
    MatrixView ane_input; // Produced by before_ffn; NOT a readiness/read permit.
};

// Complete GPU block execution; preparation and workspace allocation are
// explicit. Native graph owns every dispatch, including attention. This is a
// building block, not yet the 32-layer DiT/session or a performance-qualified
// App backend. Hybrid LoRA remains a separate unsupported execution recipe.
class NativeBlock {
public:
    static NativeBlock prepare(metal_graph::Backend&, BlockGeometry = {});
    // Reuse the model's H->H projection and pointwise PSOs for all layers.
    static NativeBlock prepare_with_shared_ops(metal_graph::Backend&, BlockGeometry,
        metal_graph::NativeLinear, metal_graph::NativeQwenOps,
        std::optional<metal_graph::NativeLoRA> = {}, bool lora_ffn = false);
    BlockGeometry geometry() const noexcept { return geometry_; }
    static size_t workspace_bytes(BlockGeometry, uint32_t rows,
                                  uint32_t prefix_tokens, BlockPhase);
    BlockWorkspace allocate_workspace(metal_graph::Backend&, uint32_t rows,
        uint32_t prefix_tokens, BlockPhase,
        metal_graph::StorageMode = metal_graph::StorageMode::Private) const;
    // Alternate phase views over one request arena. The enclosing executor
    // must finish the preceding phase before reusing this backing.
    BlockWorkspace bind_workspace(const metal_graph::Buffer&, uint32_t rows,
                                  uint32_t prefix_tokens, BlockPhase) const;
    // All shape, cross-step alias and operator checks complete on a private
    // draft before the outer graph changes. No eval, submit, wait or GPU allocation.
    void encode(metal_graph::CommandGraph&, const BlockBindings&,
                const BlockWorkspace&) const;
    // The caller prepares the complement once with prepare_parallel(), never
    // per layer. This first plan supports steady decode and an equal channel
    // split; tiny shapes are operator tests, not production qualification.
    // All three graphs, extents and cross-graph aliases validate before return.
    // Restore is deliberately absent: the controller records it only after a
    // real ANE read permit, then appends after_join to that consumer draft.
    ParallelBlockPlan plan_parallel(const BlockBindings&, const BlockWorkspace&,
        const metal_graph::NativeFfn& prepared_gpu_complement,
        const ParallelFfnWorkspace&) const;
    static void validate_parallel_geometry(BlockGeometry, metal_graph::dense_ffn::Geometry,
                                          uint32_t rows, uint32_t prefix_tokens, BlockPhase);
private:
    NativeBlock(BlockGeometry, metal_graph::NativeLinear,
        metal_graph::NativeAttention, metal_graph::NativeQwenOps, metal_graph::NativeFfn,
        std::optional<metal_graph::NativeLoRA>, std::optional<metal_graph::NativeLinear>,
        std::optional<metal_graph::NativeLinear>);
    BlockGeometry geometry_;
    metal_graph::NativeLinear projection_;
    metal_graph::NativeAttention attention_;
    metal_graph::NativeQwenOps ops_;
    metal_graph::NativeFfn ffn_;
    std::optional<metal_graph::NativeLoRA> lora_;
    std::optional<metal_graph::NativeLinear> lora_gate_up_, lora_down_;
    void encode_before_ffn(metal_graph::CommandGraph&, const BlockBindings&,
                           const BlockWorkspace&) const;
};

// Retained token subview, preserving physical head stride. Does not copy K/V.
HeadView token_range(const HeadView&, uint32_t start, uint32_t count);

} // namespace tc::qwen21::native_graph
