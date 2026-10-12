#pragma once

#include "native_ffn.hpp" // Reuse the frozen MatrixView without moving its API.
#include "kernels/dense_linear.hpp"

#include <optional>

namespace tc::metal_graph {

struct LinearBindings {
    MatrixView input;       // [M,K] BF16
    MatrixView weight;      // [N,K] BF16, out_in
    MatrixView output;      // [M,N] BF16 or explicitly FP32
    // BF16 [1,N]. Base matmul cast precedes addition; later LoRA support must
    // place this epilogue after all ordered adapter additions instead.
    std::optional<MatrixView> bias = {};
};

class NativeLinear {
public:
    static NativeLinear prepare(Backend&, dense_linear::Geometry);
    dense_linear::Geometry geometry() const noexcept { return geometry_; }
    static constexpr bool supports_bias() noexcept { return true; }
    static constexpr std::string_view precision_recipe() noexcept {
        return dense_linear::precision_recipe;
    }
    // No compilation, activation allocation, source copies or submit/waits.
    // The graph retains all source views and a copy of the parameter bytes.
    void encode(CommandGraph&, const LinearBindings&) const;
private:
    NativeLinear() = default; // Only prepare may create a capability-checked operator.
    dense_linear::Geometry geometry_;
    Pipeline bf16_, f32_, bias_bf16_, bias_f32_;
};

} // namespace tc::metal_graph
