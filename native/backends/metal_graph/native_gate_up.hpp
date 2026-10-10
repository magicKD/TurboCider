#pragma once

#include "native_ffn.hpp" // Retain the frozen MatrixView and geometry contract.
#include "kernels/dense_gate_up.hpp"

namespace tc::metal_graph {

// Optional paired base gate/up projections. This is not a complete FFN and
// neither applies LoRA nor invokes SiLU; both outputs remain independently
// rounded BF16 matrices for the original ordered adapter path.
class NativeGateUp {
public:
    using Geometry = dense_gate_up::Geometry;
    // One MPP PSO prepared outside the hot path. No tensor allocation or model
    // copies. Ordinary NativeFfn/NativeLoRA preparation does not create it.
    static NativeGateUp prepare(Backend&, Geometry);
    Geometry geometry() const noexcept { return geometry_; }
    static constexpr bool supports_bias() noexcept { return false; }
    static constexpr std::string_view precision_recipe() noexcept {
        return dense_gate_up::precision_recipe;
    }
    // Full [F,H] source views preserve physical row strides and byte offsets;
    // first_channel selects C weight rows and each [M,C] output starts at col0.
    // All layouts, aliases and devices are validated before atomic graph append.
    void encode(CommandGraph&, const MatrixView& input,
                const MatrixView& gate_weight, const MatrixView& up_weight,
                const MatrixView& gate_output, const MatrixView& up_output,
                uint32_t first_channel = 0) const;
private:
    NativeGateUp() = default;
    Geometry geometry_;
    void* device_ = nullptr;
    Pipeline paired_;
};

} // namespace tc::metal_graph
