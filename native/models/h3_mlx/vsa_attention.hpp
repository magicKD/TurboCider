#pragma once

#include "vsa.hpp"
#include "../../backends/mlx.hpp"

namespace tc::h3_mlx {

struct VSAStats {
    double configured_sparsity = 0.0;
    double achieved_sparsity = 0.0;
    double video_keep = 0.0;
    int tile_size = 64;
    int num_prefix_tiles = 0;
    int num_video_tiles = 0;
    uint64_t attention_calls = 0;
    uint64_t sparse_calls = 0;
    std::string prefix_mode = "exempt";
    std::string implementation = "dense";
    std::string dense_fallback_reason;
    bool capture_debug = false;
    bool debug_captured = false;
    std::optional<Tensor> debug_scores;
    std::optional<Tensor> debug_block_indices;
    std::optional<Tensor> debug_q_pool;
    std::optional<Tensor> debug_k_pool;
};

Tensor vsa_attention(const Tensor &, const Tensor &, const Tensor &,
                     const VSAGeometry &, double sparsity,
                     VSAPrefixMode prefix_mode,
                     VSAImplementation implementation,
                     const Tensor *gate_compress,
                     VSAStats *stats);

} // namespace tc::h3_mlx
