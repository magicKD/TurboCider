#pragma once

#include "native_ffn.hpp"
#include "kernels/native_lora.hpp"

#include <optional>
#include <span>

namespace tc::metal_graph {

// A/B retain their produced compact BF16 or FP32 source backings. b is the
// already resolved [output_count,rank] slice, not an expanded dense update.
// Vector order is the arithmetic order; the ordinal is provenance only.
struct OrderedLoRATerm {
    MatrixView a, b;
    float effective_scale = 0;
    uint32_t first_output = 0, output_count = 0;
    size_t source_adapter_ordinal = 0;
};

class NativeLoRA;

// Immutable, retained bindings for one split low/apply pair. Only a prepared
// NativeLoRA can construct this descriptor after validating the complete term,
// including B and the BF16 in-place output before any low work is recorded.
// It is not a completion/readiness certificate, an ordered-adapter scheduler,
// or permission to reuse low across a different term/variant/channel plan.
class ValidatedLoRAProjection {
public:
    native_lora::KernelVariant selected_variant() const noexcept {
        return checked_.selected_variant;
    }
    std::string_view selected_precision_recipe() const {
        return native_lora::recipe_for_variant(selected_variant());
    }
    const native_lora::CheckedTerm& parameters() const noexcept { return checked_.term; }
    const MatrixView& input() const noexcept { return input_; }
    const OrderedLoRATerm& term() const noexcept { return term_; }
    const MatrixView& output() const noexcept { return output_; }
    const MatrixView& low() const noexcept { return low_; }
private:
    ValidatedLoRAProjection(MatrixView input,OrderedLoRATerm term,MatrixView output,MatrixView low,
        native_lora::CheckedProjection checked,native_lora::Geometry geometry,
        native_lora::KernelVariant bundle_variant,void* device)
        :input_(std::move(input)),term_(std::move(term)),output_(std::move(output)),low_(std::move(low)),
         checked_(checked),geometry_(geometry),bundle_variant_(bundle_variant),device_(device) {}
    MatrixView input_;
    OrderedLoRATerm term_;
    MatrixView output_,low_;
    native_lora::CheckedProjection checked_;
    native_lora::Geometry geometry_;
    native_lora::KernelVariant bundle_variant_;
    void* device_ = nullptr;
    friend class NativeLoRA;
};

class NativeLoRA {
public:
    using Geometry = native_lora::Geometry;
    // Empty construction owns no PSO or scratch. A no-adapter projection must
    // retain its existing NativeLinear path, including its original bias.
    NativeLoRA() = default;
    // K/N are runtime upper bounds. One generic tiled FP32 bundle is prepared
    // outside the hot path and serves every bounded projection/rank geometry.
    static NativeLoRA prepare(Backend&, Geometry = {},
        native_lora::KernelVariant = native_lora::KernelVariant::Scalar);
    Geometry geometry() const noexcept { return geometry_; }
    native_lora::KernelVariant variant() const noexcept { return variant_; }
    // The selected bundle's recipe includes its scalar tail policy. SIMD
    // matrix reductions have separate qualification from the scalar default.
    std::string_view actual_precision_recipe() const {
        return native_lora::recipe_for_variant(variant_);
    }
    static constexpr std::string_view precision_recipe() noexcept {
        return native_lora::precision_recipe;
    }
    // Base is BF16 without bias. Each ordered adapter uses FP32 low and B
    // accumulation, then scale/add followed by exactly one BF16 cast. Bias is
    // BF16 and is added only after all terms. Exact logical base/output in-place
    // is allowed; every other overlapping read/write and scratch alias fails.
    // Empty terms permit only exact in-place/no-bias and require no low view.
    // All terms are validated before the caller graph is atomically appended.
    void encode_ordered_corrections(CommandGraph&, const MatrixView& input,
        std::span<const OrderedLoRATerm>, const MatrixView& base,
        const MatrixView& output, const MatrixView& low,
        std::optional<MatrixView> bias = {}) const;
    // Split recording uses the exact same Scalar/SG choice for both phases,
    // derived from the whole original term, never from A alone. No new PSOs,
    // bias or base-copy support. Output must already contain the previous BF16
    // value; one apply adds one scaled correction and narrows it once to BF16.
    ValidatedLoRAProjection validate_projection(const MatrixView& input,
        const OrderedLoRATerm&, const MatrixView& output_inplace,
        const MatrixView& fp32_low) const;
    void encode_low(CommandGraph&, const ValidatedLoRAProjection&) const;
    // Caller must prove low belongs to this same evaluated input/A/recipe and
    // is ready before this node executes, and serialize ordered adapter adds.
    // Recording an apply does not establish GPU/ANE or cross-graph readiness.
    void encode_apply_inplace(CommandGraph&, const ValidatedLoRAProjection&) const;
    // Separate pre-SiLU gate/up correction must precede this exact BF16 recipe.
    // No gate/up LoRA is applied to an already multiplied FFN hidden tensor.
    void encode_silu_up(CommandGraph&, const MatrixView& gate,
        const MatrixView& up, const MatrixView& hidden) const;
private:
    void validate_projection_descriptor(const ValidatedLoRAProjection&) const;
    Geometry geometry_;
    native_lora::KernelVariant variant_ = native_lora::KernelVariant::Scalar;
    void* device_ = nullptr;
    Pipeline low_, apply_, copy_, bias_, silu_up_;
    Pipeline low_sgmma_, apply_sgmma_;
};

} // namespace tc::metal_graph
