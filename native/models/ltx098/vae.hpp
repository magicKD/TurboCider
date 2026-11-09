#pragma once

#include "../../backends/mlx.hpp"

namespace tc::ltx098 {

// Encoder for the LTX-Video 0.9.8 checkpoint. It is intentionally distinct
// from the LTX 2.5 VAE (different channels, residual counts and decoder).
class VAEEncoder {
    Weights weights_;
    Tensor convolution(const Tensor &, const std::string &) const;
    Tensor residual_stage(Tensor, int stage, int count) const;
    Tensor downsample(Tensor, int stage, int channels, int temporal, int spatial) const;

  public:
    void load(const std::filesystem::path &checkpoint);
    size_t bytes() const { return weights_.bytes(); }
    // BF16 pixels [1,frames,height,width,3] in [-1,1] -> normalized BF16
    // latents [1,(frames+7)/8,height/32,width/32,128].
    Tensor encode(const Tensor &pixels) const;
    Tensor normalize_latents(const Tensor &latents) const;
    Tensor denormalize_latents(const Tensor &latents) const;
};

class VAEDecoder {
    Weights weights_;
    Tensor convolution(const Tensor &, const std::string &) const;
    Tensor timestep_embedding(const Tensor &, const std::string &) const;
    Tensor residual_stage(Tensor, int stage, const Tensor &time) const;
    Tensor upsample(const Tensor &, int stage) const;

  public:
    void load(const std::filesystem::path &checkpoint);
    size_t bytes() const { return weights_.bytes(); }
    // The noise input is explicit so tests and callers can reproduce the
    // reference's 0.025 decode-noise injection. Result is unclamped BF16 BFHWC.
    Tensor decode(const Tensor &latents, const Tensor &noise,
                  float timestep = .05f, float noise_scale = .025f) const;
};

} // namespace tc::ltx098
