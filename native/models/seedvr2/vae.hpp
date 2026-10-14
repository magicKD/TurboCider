#pragma once
#include "model.hpp"
#include "frame_history.hpp"

namespace tc::seedvr2 {
// A completed FP16 BCTHW frame. The consumer must not retain it if bounded
// decoded-output residency is required. Indices exclude temporal padding.
using DecodedFrameSink = std::function<void(int, const Tensor &)>;

class VAE {
    TensorBank weights_;
    std::string conv_recipe_ = "singleton2d";
    std::string norm_recipe_ = "metal";
    std::string history_recipe_ = "compact";
    std::string pointwise_recipe_ = "matmul";
    int conv_tile_rows_ = 16384;
    using CompactCaches = std::unordered_map<std::string, Tensor>;
    struct Caches { CompactCaches compact; FrameHistoryMap frames; };
    Tensor conv(const Tensor &, const std::string &, Caches *, bool initial,
                bool downsample = false, bool owned_normalized_singleton = false) const;
    Tensor conv_frames(const Tensor &, const std::string &, FrameHistoryMap *, bool initial,
                       bool downsample, bool owned_normalized_singleton) const;
    Tensor group_norm(const Tensor &, const std::string &, bool silu = true) const;
    Tensor resnet(const Tensor &, const std::string &, Caches *, bool) const;
    Tensor mid_block(const Tensor &, const std::string &, Caches *, bool) const;
    Tensor upsample(const Tensor &, const std::string &, Caches *, bool, bool temporal) const;
    Tensor encode_slice(const Tensor &, Caches *, bool, const Event &, std::atomic<bool> &) const;
    Tensor decode_slice(const Tensor &, Caches *, bool, const Event &, std::atomic<bool> &,
                        const DecodedFrameSink & = {}, int *next_frame = nullptr, int frame_limit = 0) const;
  public:
    void load(const std::filesystem::path &);
    void configure(const std::optional<SeedVR2VAEConfig> &);
    size_t bytes() const { return weights_.bytes(); }
    const std::string &conv_recipe() const { return conv_recipe_; }
    int conv_tile_rows() const { return conv_tile_rows_; }
    const std::string &norm_recipe() const { return norm_recipe_; }
    const std::string &history_recipe() const { return history_recipe_; }
    const std::string &pointwise_recipe() const { return pointwise_recipe_; }
    // External BCTHW, RGB pixels in [-1,1]. Returns scaled latent means only.
    Tensor encode(const Tensor &, const Event &, std::atomic<bool> &) const;
    // Scaled BCTHW latents -> RGB BCTHW in [-1,1]. Temporal slicing carries
    // original-precision causal histories, never independent image inference.
    Tensor decode(const Tensor &, const Event &, std::atomic<bool> &) const;
    void decode_stream(const Tensor &, int output_frames, const DecodedFrameSink &,
                       const Event &, std::atomic<bool> &) const;
    void clear() { weights_.clear(); }
};

} // namespace tc::seedvr2
