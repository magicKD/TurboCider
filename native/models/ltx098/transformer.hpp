#pragma once

#include "block.hpp"

namespace tc::ltx098 {

// A resident, video-only 0.9.8 transformer. The surrounding pipeline owns
// image/text conditioning, Euler scheduling and VAE encode/decode.
class Transformer {
    Weights top_;
    std::vector<std::unique_ptr<Block>> blocks_;
    std::optional<ltx_mlx::RopeFrequencies> rope_;
    int rope_frames_ = 0, rope_height_ = 0, rope_width_ = 0;
    float rope_fps_ = 0;
    int evaluate_every_ = 1;

    const ltx_mlx::RopeFrequencies &positions(int frames, int height, int width, float fps);
    Tensor forward_impl(const Tensor &, const Tensor &, float, float, bool, float,
                        const std::optional<Tensor> &, const std::vector<CrossAttentionKV> *);

  public:
    // The block-count parameter is a probe seam. Production must use 28.
    void load(const std::filesystem::path &checkpoint, int block_count = 28, int evaluate_every = 1);
    Tensor prepare_text(const Tensor &text) const;
    // Cache is valid only for this transformer and this exact prepared text.
    std::vector<CrossAttentionKV> prepare_cross_attention(const Tensor &prepared_text) const;
    // Latents are [1,F,H,W,128]; prepared text [1,T,2048]. The first latent
    // frame receives sigma*(1-strength), the others receive sigma. The caller
    // owns Euler updates and preserves/merges the conditioning latents.
    Tensor forward(const Tensor &latents, const Tensor &prepared_text, float sigma,
                   float fps, bool image_conditioned, float strength,
                   const std::optional<Tensor> &text_mask = {});
    Tensor forward_cached(const Tensor &latents, const Tensor &prepared_text, float sigma,
                          float fps, bool image_conditioned, float strength,
                          const std::optional<Tensor> &text_mask, const std::vector<CrossAttentionKV> &);
};

} // namespace tc::ltx098
