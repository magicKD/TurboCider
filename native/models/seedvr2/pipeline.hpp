#pragma once
#include "vae.hpp"

namespace tc::seedvr2 {

struct Restoration {
    Tensor pixels, condition, velocity, latent;
    double encode_seconds = 0, denoise_seconds = 0, decode_seconds = 0;
    double staging_seconds = 0, restore_seconds = 0;
};

class Pipeline {
    VAE vae_;
    DiT dit_;
    bool loaded_ = false;
    bool stage_dit_ = false, stable_profile_ = false;
    mx::Dtype dit_dtype_ = mx::float16;
    std::filesystem::path dit_path_;
    uintmax_t dit_source_bytes_ = 0;
    std::filesystem::file_time_type dit_source_modified_;
  public:
    // Session owner selects this policy before loading. No environment
    // override may silently change the request's declared residency.
    void set_stage_dit(bool enabled) {
        require(!loaded_, "select SeedVR2 DiT staging before loading");
        stage_dit_ = enabled;
    }
    bool stages_dit() const { return stage_dit_; }
    const char *weight_residency_recipe() const { return stage_dit_ ? "stage_dit" : "resident"; }
    void load(const std::filesystem::path &, const Event &, std::atomic<bool> &,
              mx::Dtype dit_dtype = mx::float16);
    size_t bytes() const { return vae_.bytes() + dit_.bytes(); }
    const DiT &dit() const { return dit_; }
    const std::string &vae_conv_recipe() const { return vae_.conv_recipe(); }
    const std::string &vae_norm_recipe() const { return vae_.norm_recipe(); }
    const std::string &vae_history_recipe() const { return vae_.history_recipe(); }
    const std::string &vae_pointwise_recipe() const { return vae_.pointwise_recipe(); }
    int vae_conv_tile_rows() const { return vae_.conv_tile_rows(); }
    // Resized RGB BCTHW in [-1,1], padded to 4n+1 if necessary. A supplied
    // FP32 sampling-noise fixture makes backend comparisons RNG-independent.
    Restoration restore(const Tensor &pixels, const Tensor &noise,
                        const Event &, std::atomic<bool> &, const std::filesystem::path &dump = {},
                        const FFNRoute & = {});
    void clear() { dit_.clear(); vae_.clear(); loaded_ = false; }
};

} // namespace tc::seedvr2
