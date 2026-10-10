#pragma once

#include "native_ffn.hpp"
#include "native_attention.hpp"
#include "kernels/qwen_ops.hpp"

#include <array>

namespace tc::metal_graph {

// Batch-one Qwen DiT pointwise/layout operators. Qwen's actual modulation is
// scale-only; the optional shift is explicit and must not be bound by that
// route. Every activation and modulation operand is BF16, except FP32 RoPE.
// These methods only append graph steps: callers establish cross-queue input
// readiness. Graph views retain storage owners until actual GPU completion.
class NativeQwenOps {
public:
    static NativeQwenOps prepare(Backend&, qwen_ops::Geometry = {});
    qwen_ops::Geometry geometry() const noexcept { return geometry_; }
    static constexpr std::string_view precision_recipe() noexcept {
        return qwen_ops::precision_recipe;
    }
    static constexpr std::string_view conditioning_recipe() noexcept {
        return qwen_ops::conditioning_recipe;
    }
    // No compilation, allocation, append or submission. A block builder must
    // validate all operations (and cross-operation scratch hazards) first.
    static void validate_layer_norm(qwen_ops::Geometry, const MatrixView& input,
        const MatrixView& output, float epsilon = 1e-6f);
    static void validate_norm_modulation(qwen_ops::Geometry, const MatrixView& input,
        const MatrixView& scale, const MatrixView& output, float epsilon = 1e-6f,
        const MatrixView* shift = nullptr);
    static void validate_modulation(qwen_ops::Geometry, const MatrixView& input,
        const MatrixView& scale, const MatrixView& output, const MatrixView* shift = nullptr);
    static void validate_gated_residual(qwen_ops::Geometry, const MatrixView& residual,
        const MatrixView& branch, const MatrixView& gate, const MatrixView& output,
        bool gate_is_tanh = false);
    static void validate_qk_norm_rope(qwen_ops::Geometry, const MatrixView& q,
        const MatrixView& k, const MatrixView& q_weight, const MatrixView& k_weight,
        const MatrixView& cosine, const MatrixView& sine,
        const HeadView& q_out, const HeadView& k_out, float epsilon = 1e-6f);
    static void validate_pack_heads(qwen_ops::Geometry, const MatrixView& row_major,
        const HeadView& head_major);
    static void validate_unpack_heads(qwen_ops::Geometry, const HeadView& head_major,
        const MatrixView& row_major);
    static void validate_copy_heads(qwen_ops::Geometry, const HeadView& source,
        const HeadView& destination);
    static void validate_silu(qwen_ops::Geometry, const MatrixView& input,
        const MatrixView& output);
    static void validate_gelu(qwen_ops::Geometry, const MatrixView& input,
        const MatrixView& output);
    static void validate_condition_norm(qwen_ops::Geometry, const MatrixView& text,
        const MatrixView& weight, const MatrixView& output, float epsilon = 1e-6f);
    static void validate_expand_modulation(qwen_ops::Geometry, const MatrixView& source,
        const std::array<MatrixView,4>& outputs, uint32_t prefix_rows);
    void encode_layer_norm(CommandGraph&, const MatrixView& input,
                           const MatrixView& output, float epsilon = 1e-6f) const;
    void encode_norm_modulation(CommandGraph&, const MatrixView& input,
        const MatrixView& scale, const MatrixView& output, float epsilon = 1e-6f,
        const MatrixView* shift = nullptr) const;
    void encode_modulation(CommandGraph&, const MatrixView& normalized,
        const MatrixView& scale, const MatrixView& output,
        const MatrixView* shift = nullptr) const;
    // gate_is_tanh=false consumes raw modulation; true consumes the already
    // BF16-rounded tanh gate. Both preserve product and residual BF16 rounding.
    void encode_gated_residual(CommandGraph&, const MatrixView& residual,
        const MatrixView& branch, const MatrixView& gate, const MatrixView& output,
        bool gate_is_tanh = false) const;
    // q/k: [M,H]; q/k_weight: [1,128]; cosine/sine: FP32 [M,64].
    // q_out/k_out use [heads,M,128] with explicit head/token strides.
    void encode_qk_norm_rope(CommandGraph&, const MatrixView& q,
        const MatrixView& k, const MatrixView& q_weight, const MatrixView& k_weight,
        const MatrixView& cosine, const MatrixView& sine,
        const HeadView& q_out, const HeadView& k_out, float epsilon = 1e-6f) const;
    void encode_pack_heads(CommandGraph&, const MatrixView& row_major,
                           const HeadView& head_major) const;
    void encode_unpack_heads(CommandGraph&, const HeadView& head_major,
                             const MatrixView& row_major) const;
    // Source/destination may be token subviews of larger head-major banks.
    // Copies the visible range; retains original head strides; no overlap.
    void encode_copy_heads(CommandGraph&, const HeadView& source,
                           const HeadView& destination) const;
    // BF16 [M,H]. SiLU rounds exp, denominator, reciprocal, positive-branch
    // subtraction and product, matching MLX Sigmoid<bfloat>. GELU matches Qwen DiT's
    // text_in: BF16 scalars and per-operation rounding, not vision's FP32 GELU.
    void encode_silu(CommandGraph&, const MatrixView& input, const MatrixView& output) const;
    void encode_gelu(CommandGraph&, const MatrixView& input, const MatrixView& output) const;
    // Text conditioning uses FP32 x*rsqrt(mean(x*x)+eps)*(float(weight)+1),
    // casting only the final result to BF16. This is not Q/K fast RMSNorm.
    void encode_condition_norm(CommandGraph&, const MatrixView& text,
        const MatrixView& weight, const MatrixView& output, float epsilon = 1e-6f) const;
    // source=[2,4H]; outputs=[M,H] each. Prefix rows consume source row 1,
    // remaining target rows consume row 0. No shifted or learned padding rows.
    void encode_expand_modulation(CommandGraph&, const MatrixView& source,
        const std::array<MatrixView,4>& outputs, uint32_t prefix_rows) const;
private:
    qwen_ops::Geometry geometry_;
    Pipeline norm_, modulation_, residual_, qk_rope_, pack_, unpack_, copy_heads_;
    Pipeline silu_, gelu_, condition_norm_, expand_modulation_;
};

} // namespace tc::metal_graph
