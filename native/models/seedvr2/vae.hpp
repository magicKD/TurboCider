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
// Offline observation only: never replaces an operator or changes its input.
// Callers must finish GPU/host access before returning, not retain borrowed views.
using UpscaleInputObserver = std::function<void(const std::string &, const Tensor &, bool, bool)>;
#if defined(TURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT) && TURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT
// Offline complete-VAE qualification only. The executor returns the completed
// FP16 spatial projection with separate bias/shuffle, never temporal phases.
using ExperimentalUpscaleProjection = std::function<Tensor(const Tensor &,const Tensor &,const Tensor &)>;
#endif

class VAE {
    TensorBank weights_;
    std::string conv_recipe_ = "singleton2d";
    std::string norm_recipe_ = "metal";
    std::string history_recipe_ = "compact";
    std::string pointwise_recipe_ = "matmul";
    std::string upscale_recipe_ = "native";
    std::string spatial_recipe_ = "full";
    std::string encoder_spatial_recipe_ = "full";
    SpatialTileOptions spatial_options_;
    UpscaleInputObserver upscale_input_observer_;
#if defined(TURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT) && TURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT
    ExperimentalUpscaleProjection experimental_upscale_projection_;
#endif
    int conv_tile_rows_ = 16384;
    using CompactCaches = std::unordered_map<std::string, Tensor>;
    struct Caches {
        CompactCaches compact; FrameHistoryMap frames;
        bool temporal = true;
        bool spatial_head_only = false;
        bool spatial_enabled = true;
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
    Tensor encode_stream_impl(int,int,int,const EncodedFrameSource &,const SpatialTileOptions *,
                              std::shared_ptr<SpatialFileStats> *,const Event &,std::atomic<bool> &) const;
  public:
    void load(const std::filesystem::path &);
    void configure(const std::optional<SeedVR2VAEConfig> &);
    size_t bytes() const { return weights_.bytes(); }
    const std::string &conv_recipe() const { return conv_recipe_; }
    int conv_tile_rows() const { return conv_tile_rows_; }
    const std::string &norm_recipe() const { return norm_recipe_; }
    const std::string &history_recipe() const { return history_recipe_; }
    const std::string &pointwise_recipe() const { return pointwise_recipe_; }
    const std::string &upscale_recipe() const { return upscale_recipe_; }
    const std::string &spatial_recipe() const { return spatial_recipe_; }
    const std::string &encoder_spatial_recipe() const { return encoder_spatial_recipe_; }
    void observe_upscale_inputs(UpscaleInputObserver observer) { upscale_input_observer_ = std::move(observer); }
#if defined(TURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT) && TURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT
    void use_experimental_upscale_projection(ExperimentalUpscaleProjection executor) {
        require(bool(executor) && upscale_recipe_=="native","experimental projection requires the native outer recipe");
        experimental_upscale_projection_=std::move(executor);
        upscale_recipe_="coreml_projection_probe";
    }
#endif
    // External BCTHW, RGB pixels in [-1,1]. Returns scaled latent means only.
    Tensor encode(const Tensor &, const Event &, std::atomic<bool> &) const;
    // Offline exact-context encoder-head scheduling: the high-resolution
    // singleton head uses the existing row/file scheduler; all temporal
    // downsampling, later stages and attention retain the original path.
    Tensor encode_spatial(const Tensor &, const SpatialTileOptions &,
                          std::shared_ptr<SpatialFileStats> &, const Event &, std::atomic<bool> &) const;
    Tensor encode_configured(const Tensor &,std::shared_ptr<SpatialFileStats> &,const Event &,std::atomic<bool> &) const;
    // Preserve the original initial-five/subsequent-four temporal partition
    // and global spatial context, with bounded normalized input chunks. The
    // caller must additionally verify the complete source EOF before sampling.
    Tensor encode_stream(int source_frames, int height, int width, const EncodedFrameSource &,
                         const Event &, std::atomic<bool> &) const;
    Tensor encode_stream_spatial(int source_frames,int height,int width,const EncodedFrameSource &,
                                 const SpatialTileOptions &,std::shared_ptr<SpatialFileStats> &,
                                 const Event &,std::atomic<bool> &) const;
    Tensor encode_stream_configured(int,int,int,const EncodedFrameSource &,std::shared_ptr<SpatialFileStats> &,
                                    const Event &,std::atomic<bool> &) const;
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
    void clear() {
        weights_.clear();
#if defined(TURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT) && TURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT
        experimental_upscale_projection_={};
#endif
    }
};

} // namespace tc::seedvr2
