#pragma once

#include "vae_weights.hpp"

namespace tc::h3_mlx {

class VideoVAE {
    VAEWeights weights_;
    VideoVAEConfig config_;

    Tensor conv3d(const Tensor &, const std::string &, int temporal_padding,
                  int spatial_padding = 0, const std::array<int, 3> &stride = {1, 1, 1}) const;
    Tensor rms_affine(const Tensor &, const std::string &, float) const;
    Tensor layer_norm(const Tensor &, const std::string &, const std::string &, float) const;
    std::pair<Tensor, Tensor> rotary(const Tensor &) const;
    Tensor attention(const Tensor &, const std::string &, const Tensor &, const Tensor &) const;
    Tensor feed_forward(const Tensor &, const std::string &) const;
    Tensor decode_clip(const Tensor &, const Event &, std::atomic<bool> &) const;
    Tensor decode_clip_tiled(const Tensor &, int, int, int, int,
                             const Event &, std::atomic<bool> &) const;

    static std::vector<int> split_tiles(int length, int tile_size, int min_overlap,
                                        int compression, std::vector<int> &overlaps);
    static Tensor blend(const Tensor &, const Tensor &, int extent, int axis);

  public:
    void load(const std::filesystem::path &, const Event &, std::atomic<bool> &);
    void unload();
    bool loaded() const { return weights_.bytes() != 0; }
    const VideoVAEConfig &config() const { return config_; }
    Tensor denormalize_latents(const Tensor &) const;
    Tensor denormalize_pixels(const Tensor &) const;
    Tensor decode(const Tensor &, int target_frames, int target_height,
                  int target_width, bool tiled, const Event &,
                  std::atomic<bool> &) const;
};

} // namespace tc::h3_mlx
