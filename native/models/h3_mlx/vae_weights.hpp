#pragma once

#include "../../backends/mlx.hpp"

#include <array>
#include <unordered_map>

namespace tc::h3_mlx {

class VAEWeights {
    std::unordered_map<std::string, Tensor> arrays_;

  public:
    void load(const std::filesystem::path &component_root,
              const std::vector<std::string> &prefixes,
              mx::Dtype storage_dtype, bool convert_conv3d_layout,
              const Event &, std::atomic<bool> &);
    void clear();
    bool has(const std::string &) const;
    const Tensor &at(const std::string &) const;
    size_t bytes() const;
};

struct VideoVAEConfig {
    int latent_channels = 24;
    int decoder_layers = 36;
    int num_heads = 32;
    int head_dim = 64;
    int register_tokens = 4;
    int spatial_ratio = 16;
    int temporal_ratio = 4;
    int clip_length = 17;
    int token_drop = 3;
    float rope_theta = 100.f;
    float rope_ratio = 0.75f;
    float norm_epsilon = 1e-5f;
    std::array<float, 24> latent_mean{};
    std::array<float, 24> latent_std{};
};

VideoVAEConfig load_video_vae_config(const std::filesystem::path &);

} // namespace tc::h3_mlx
