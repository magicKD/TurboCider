#pragma once
#include "vae.hpp"

namespace tc::seedvr2 {

struct Restoration {
    Tensor pixels, condition, velocity, latent;
    double encode_seconds = 0, denoise_seconds = 0, decode_seconds = 0;
};

class Pipeline {
    VAE vae_;
    DiT dit_;
    bool loaded_ = false;
  public:
    void load(const std::filesystem::path &, const Event &, std::atomic<bool> &);
    size_t bytes() const { return vae_.bytes() + dit_.bytes(); }
    const DiT &dit() const { return dit_; }
    const std::string &vae_conv_recipe() const { return vae_.conv_recipe(); }
    // Resized RGB BCTHW in [-1,1], padded to 4n+1 if necessary. A supplied
    // FP32 sampling-noise fixture makes backend comparisons RNG-independent.
    Restoration restore(const Tensor &pixels, const Tensor &noise,
                        const Event &, std::atomic<bool> &, const std::filesystem::path &dump = {},
                        const FFNRoute & = {});
    void clear() { dit_.clear(); vae_.clear(); loaded_ = false; }
};

} // namespace tc::seedvr2
