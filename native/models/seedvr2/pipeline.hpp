#pragma once
#include "vae.hpp"

namespace tc::seedvr2 {

struct Restoration {
    Tensor pixels, condition, velocity, latent;
    double encode_seconds = 0, denoise_seconds = 0, decode_seconds = 0;
    double staging_seconds = 0, restore_seconds = 0;
    double input_release_seconds = 0;
    uint64_t released_input_logical_bytes = 0;
    bool pixels_streamed = false;
    std::shared_ptr<SpatialFileStats> spatial_stats;
    std::shared_ptr<SpatialFileStats> encoder_spatial_stats;
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
    Restoration restore_impl(const Tensor &, const Tensor &, const Event &,
                             std::atomic<bool> &, const std::filesystem::path &,
                             const FFNRoute &, Tensor *input_owner, const DecodedFrameSink & = {});
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
    const std::string &vae_upscale_recipe() const { return vae_.upscale_recipe(); }
    bool vae_shares_upscale_history() const {return vae_.shares_upscale_history();}
    uint64_t vae_shared_upscale_frames() const {return vae_.shared_upscale_frames();}
    uint64_t vae_shared_upscale_logical_bytes() const {return vae_.shared_upscale_logical_bytes();}
    const std::string &vae_spatial_recipe() const { return vae_.spatial_recipe(); }
    const std::string &vae_encoder_spatial_recipe() const { return vae_.encoder_spatial_recipe(); }
    int vae_conv_tile_rows() const { return vae_.conv_tile_rows(); }
    void configure_vae(const std::optional<SeedVR2VAEConfig> &config) { vae_.configure(config); }
#if defined(TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_EXPERIMENT) && TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_EXPERIMENT
    void use_experimental_causal_conv(ExperimentalCausalConv execute){vae_.use_experimental_causal_conv(std::move(execute));}
    void use_experimental_upscale_causal_conv(ExperimentalCausalConv execute){vae_.use_experimental_upscale_causal_conv(std::move(execute));}
#endif
    // Resized RGB BCTHW in [-1,1], padded to 4n+1 if necessary. A supplied
    // FP32 sampling-noise fixture makes backend comparisons RNG-independent.
    Restoration restore(const Tensor &pixels, const Tensor &noise,
                        const Event &, std::atomic<bool> &, const std::filesystem::path &dump = {},
                        const FFNRoute & = {});
    // Explicit ownership transfer for video/staged-image callers. Keeps the
    // non-owning interface's math and drops normalized/padded RGB handles only
    // after VAE encode has completed, before sampling/decode.
    Restoration restore_owned(Tensor pixels, const Tensor &noise,
                              const Event &, std::atomic<bool> &,
                              const std::filesystem::path &dump = {}, const FFNRoute & = {},
                              const DecodedFrameSink & = {});
    // Source EOF must be verified after encoding, before any sampling. This
    // bounds imported/resized RGB, not complete latent/DiT context or RAM.
    // Complete input tensor dumps require the buffered path.
    Restoration restore_source(int source_frames, int height, int width, const Tensor &noise,
                               const EncodedFrameSource &, const std::function<void()> &verify_complete_source,
                               const Event &, std::atomic<bool> &, const FFNRoute & = {},
                               const DecodedFrameSink & = {});
    void clear() { dit_.clear(); vae_.clear(); loaded_ = false; }
};

} // namespace tc::seedvr2
