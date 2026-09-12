#pragma once

#include "checkpoint.hpp"
#include "geometry.hpp"
#include "vsa_attention.hpp"

namespace tc::h3_mlx {

struct DiTOutput {
    Tensor video;
    Tensor audio;
};

struct DiTDebugCapture {
    std::optional<Tensor> video_embed;
    std::optional<Tensor> audio_embed;
    std::optional<Tensor> text_embed;
    std::optional<Tensor> packed_embed;
    std::optional<Tensor> first_norm1;
    std::optional<Tensor> first_modulated1;
    std::optional<Tensor> first_attention;
    std::optional<Tensor> first_query;
    std::optional<Tensor> first_key;
    std::optional<Tensor> first_value;
    std::optional<Tensor> first_gate;
    std::optional<Tensor> first_attended;
    std::optional<Tensor> first_after_attention;
    std::optional<Tensor> first_norm2;
    std::optional<Tensor> first_modulated2;
    std::optional<Tensor> first_feed_forward;
    std::optional<Tensor> first_block;
};

class DiT {
    const Checkpoint &weights_;

    Tensor cast_for(const Tensor &, const std::string &prefix) const;
    Tensor rms(const Tensor &, const std::string &weight, float epsilon) const;
    std::pair<Tensor, Tensor> rope(const PackedLayout &) const;
    Tensor apply_rotary(const Tensor &, const Tensor &, const Tensor &) const;
    Tensor attention(const Tensor &, const std::string &prefix,
                     const std::optional<std::pair<Tensor, Tensor>> &rope,
                     DiTDebugCapture * = nullptr,
                     const VSAGeometry * = nullptr,
                     double vsa_sparsity = 0.0,
                     VSAPrefixMode vsa_prefix_mode = VSAPrefixMode::exempt,
                     VSAImplementation vsa_implementation = VSAImplementation::automatic,
                     const Tensor *gate_compress = nullptr,
                     VSAStats *vsa_stats = nullptr) const;
    Tensor feed_forward(const Tensor &, const std::string &prefix) const;
    Tensor refine_text(const Tensor &) const;
    Tensor block(const Tensor &, int index, const Tensor &adaln_indices,
                 const Tensor &cosine, const Tensor &sine,
                 int step_index, const VSAConfig &, const VSAGeometry *, VSAStats *,
                 DiTDebugCapture * = nullptr) const;

  public:
    explicit DiT(const Checkpoint &weights) : weights_(weights) {}
    DiTOutput forward(const Tensor &video_rows, const Tensor &audio_rows,
                      const Tensor &text_rows, const PackedLayout &,
                      const RowTimesteps &, const Event &,
                      std::atomic<bool> &, int step_index = 0,
                      const VSAConfig & = {}, VSAStats * = nullptr,
                      DiTDebugCapture * = nullptr) const;
};

Tensor scheduler_step(const Tensor &sample, const Tensor &velocity,
                      const Scheduler &, int step_index);

} // namespace tc::h3_mlx
