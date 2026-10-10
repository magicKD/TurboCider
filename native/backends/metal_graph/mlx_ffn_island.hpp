#pragma once

#include "mlx_bridge.hpp"
#include "native_ffn.hpp"

namespace tc::metal_graph {

struct IslandResult {
    mlx::core::array output;
    mlx::core::array hidden;
    Completion completion;
};

// W0 implementation slice of the independent backend. Inputs are produced
// MLX leaves; both GPU dispatches are encoded/submitted by the native backend.
// This boundary waits for physical completion before returning MLX outputs.
// It is deliberately identified as an FFN island, not a full native DiT.
class MlxFfnIsland {
public:
    MlxFfnIsland(Backend&, dense_ffn::Geometry);
    IslandResult run(const mlx::core::array& input,
                     const mlx::core::array& gate,
                     const mlx::core::array& up,
                     const mlx::core::array& down,
                     uint32_t first_channel = 0,
                     bool fp32_partial = false) const;
private:
    Backend& backend_;
    NativeFfn ffn_;
};
} // namespace tc::metal_graph
