#pragma once
#include "../../backends/mlx.hpp"

namespace tc::qwen21 {
// Use independent output slices inside compiled FFNs. MLX multi-output split
// siblings can keep a traced graph's materialized captured weights alive after
// the callable is destroyed (the ordinary Transformer uses this boundary too).
inline Tensor runtime_ffn_activation(const Tensor &gate_up) {
    const int half = gate_up.shape(-1) / 2;
    require(half > 0 && half * 2 == gate_up.shape(-1), "invalid fused runtime FFN width");
    return silu(slice_axis(gate_up, -1, 0, half)) *
           slice_axis(gate_up, -1, half, gate_up.shape(-1));
}
} // namespace tc::qwen21
