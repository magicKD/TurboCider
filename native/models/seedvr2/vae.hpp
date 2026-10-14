#pragma once
#include "model.hpp"
#include "frame_history.hpp"
#include "spatial_files.hpp"

namespace tc::seedvr2 {
// A completed FP16 BCTHW frame. The consumer must not retain it if bounded
// decoded-output residency is required. Indices exclude temporal padding.
using DecodedFrameSink = std::function<void(int, const Tensor &)>;
// Each call supplies exactly one materialized, normalized FP32 BCTHW RGB
// source frame. Indices address only real frames; padding is internal.
using EncodedFrameSource = std::function<Tensor(int)>;

class VAE {
    TensorBank weights_;
    std::string conv_recipe_ = "singleton2d";
    std::string norm_recipe_ = "metal";
    std::string history_recipe_ = "compact";
    std::string pointwise_recipe_ = "matmul";
    std::string spatial_recipe_ = "full";
    SpatialTileOptions spatial_options_;
    int conv_tile_rows_ = 16384;
    using CompactCaches = std::unordered_map<std::string, Tensor>;
    struct Caches {
        CompactCaches compact; FrameHistoryMap frames;
        bool temporal = true;
        std::shared_ptr<SpatialTileCache> spatial;
        std::atomic<bool> *cancelled = nullptr;
    };
    Tensor conv(const Tensor &, const std::string &, Caches *, bool initial,
                bool downsample = false, bool owned_normalized_singleton = false) const;
    Tensor conv_frames(const Tensor &, const std::string &, FrameHistoryMap *, bool initial,
                       bool downsample, bool owned_normalized_singleton) const;
    Tensor group_norm(const Tensor &, const std::string &, bool silu = true) const;
    Tensor norm_conv(const Tensor &, const std::string &norm_key, const std::string &conv_key,
                     Caches *, bool initial) const;
    Tensor spatial_conv(const Tensor &shape_source, const std::function<Tensor(int,int)> &rows,
                        const std::string &, Caches *, bool initial) const;
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
    const std::string &spatial_recipe() const { return spatial_recipe_; }
    // External BCTHW, RGB pixels in [-1,1]. Returns scaled latent means only.
    Tensor encode(const Tensor &, const Event &, std::atomic<bool> &) const;
    // Preserve the original initial-five/subsequent-four temporal partition
    // and global spatial context, with bounded normalized input chunks. The
    // caller must additionally verify the complete source EOF before sampling.
    Tensor encode_stream(int source_frames, int height, int width, const EncodedFrameSource &,
                         const Event &, std::atomic<bool> &) const;
    // Scaled BCTHW latents -> RGB BCTHW in [-1,1]. Temporal slicing carries
    // original-precision causal histories, never independent image inference.
    Tensor decode(const Tensor &, const Event &, std::atomic<bool> &) const;
    void decode_stream(const Tensor &, int output_frames, const DecodedFrameSink &,
                       const Event &, std::atomic<bool> &) const;
    // Explicit scheduler; decode_configured selects it only on request.
    // Omitted request settings and environment defaults retain the full route.
    // Global statistics/all attention stay unchanged; only eligible high-res
    // norm/conv stages and lossless causal histories use row/file scheduling.
    void decode_spatial_stream(const Tensor &, int output_frames, const DecodedFrameSink &,
                               const SpatialTileOptions &, std::shared_ptr<SpatialFileStats> &,
                               const Event &, std::atomic<bool> &) const;
    // Request-selected decode boundary. Buffered output gathers only owned RGB
    // frames; stream sinks consume frames directly. Default remains unchanged.
    Tensor decode_configured(const Tensor &, int output_frames, const DecodedFrameSink &,
                             std::shared_ptr<SpatialFileStats> &, const Event &, std::atomic<bool> &) const;
    void clear() { weights_.clear(); }
};

} // namespace tc::seedvr2
