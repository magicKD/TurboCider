#pragma once

#include "vae_weights.hpp"

namespace tc::h3_mlx {

class AudioVAE {
    VAEWeights weights_;

    Tensor weight_norm(const std::string &) const;
    Tensor conv1d(const Tensor &, const Tensor &,
                  const std::optional<Tensor> & = {}, int stride = 1,
                  int padding = 0, int dilation = 1, int groups = 1) const;
    Tensor conv_transpose1d(const Tensor &, const Tensor &,
                            const std::optional<Tensor> &, int stride,
                            int padding) const;
    Tensor replicate_pad(const Tensor &, int left, int right) const;
    Tensor low_pass(const Tensor &, const Tensor &, int stride) const;
    Tensor up_sample(const Tensor &, const Tensor &, int ratio) const;
    Tensor activation(const Tensor &, const std::string &) const;
    Tensor amp_block(Tensor, const std::string &, int kernel,
                     const std::array<int, 3> &dilations) const;

  public:
    static constexpr int sampling_rate = 32000;
    static constexpr int latent_channels = 32;

    void load(const std::filesystem::path &, const Event &,
              std::atomic<bool> &);
    void unload();
    bool loaded() const { return weights_.bytes() != 0; }
    Tensor denormalize_latents(const Tensor &) const;
    Tensor decode(const Tensor &, const Event &, std::atomic<bool> &) const;
};

} // namespace tc::h3_mlx
