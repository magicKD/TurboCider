#pragma once

#include "../../backends/mlx.hpp"

namespace tc::z_image {

// Keep the original numerical boundaries, including each projection's bias
// and LoRA handling. Only the identical gate/up ConvRot input dependency is
// shared. The down projection still rotates the post-SwiGLU activation.
// Explicit opt-in until full-request performance/quality qualification.
inline Tensor feed_forward(const Tensor &x, const Weights &weights,
                           const std::string &prefix, bool share_convrot) {
    if (share_convrot && weights.convrot(prefix + ".w1") &&
        weights.convrot(prefix + ".w3")) {
        auto gate_up = weights.project_many(x, {prefix + ".w1", prefix + ".w3"});
        return weights.project(silu(gate_up[0]) * gate_up[1], prefix + ".w2");
    }
    return weights.project(silu(weights.project(x, prefix + ".w1")) *
                               weights.project(x, prefix + ".w3"), prefix + ".w2");
}

} // namespace tc::z_image
