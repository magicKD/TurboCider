#pragma once

#include "../../backends/mlx.hpp"
#include "../ltx_mlx/block.hpp"

namespace tc::ltx098 {

struct BlockConfig {
    int hidden = 2048;
    int heads = 32;
    float residual_epsilon = 1.0e-6f;
    float qk_epsilon = 1.0e-5f;
};

struct CrossAttentionKV {
    Tensor key, value;
};

// Video-only LTX 0.9.8 block. The checkpoint is not compatible with the
// 2.5 AV block: in particular its RoPE is interleaved and its cross-attention
// does not have prompt AdaLN. This component is not yet a complete pipeline.
class Block {
    Weights weights_;
    BlockConfig config_;
    Tensor attention(const Tensor &, const std::optional<Tensor> &,
                     const std::string &, const std::optional<ltx_mlx::RopeFrequencies> &,
                     const std::optional<Tensor> &, const CrossAttentionKV *) const;
    Tensor forward_impl(const Tensor &, const Tensor &, const Tensor &,
                        const std::optional<ltx_mlx::RopeFrequencies> &,
                        const std::optional<Tensor> &, const CrossAttentionKV *) const;

  public:
    void load(const std::filesystem::path &, int index, BlockConfig = {});
    size_t bytes() const { return weights_.bytes(); }
    CrossAttentionKV prepare_cross_attention(const Tensor &context) const;

    // Modulation is [1,1,6*hidden] or [1,tokens,6*hidden]; per-token
    // timesteps are required for image conditioning. RoPE is [1,heads,
    // tokens,head_dim/2], and the optional text mask is additive.
    Tensor forward(const Tensor &hidden, const Tensor &context,
                   const Tensor &modulation,
                   const std::optional<ltx_mlx::RopeFrequencies> &rope = {},
                   const std::optional<Tensor> &text_mask = {}) const;
    Tensor forward_cached(const Tensor &hidden, const Tensor &context, const Tensor &modulation,
                          const std::optional<ltx_mlx::RopeFrequencies> &rope,
                          const std::optional<Tensor> &text_mask, const CrossAttentionKV &) const;
};

} // namespace tc::ltx098
