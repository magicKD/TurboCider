#pragma once

#include "../../backends/mlx.hpp"

#include <optional>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tc::ltx_mlx {

struct RopeFrequencies {
    Tensor cosine;
    Tensor sine;
};

class CompiledBlockForward;

// A C++/MLX implementation of the LTX BasicAVTransformerBlock.  The class is
// intentionally independent of the legacy C/Metal runtime at first: it can be
// loaded and probed one block at a time, which makes numerical parity and
// memory ownership observable before the production session is switched over.
class BasicAVTransformerBlock {
    friend class CompiledBlockForward;

    Weights weights_;
    int block_index_ = 0;
    int video_dim_ = 4096;
    int audio_dim_ = 2048;
    int video_heads_ = 32;
    int audio_heads_ = 32;
    int video_head_dim_ = 128;
    int audio_head_dim_ = 64;
    int av_heads_ = 32;
    int av_head_dim_ = 64;
    float norm_eps_ = 1.0e-6f;
    bool bf16_fences_ = false;
    mutable uint64_t forward_count_ = 0;

    Tensor rms_norm(const Tensor &, float) const;
    std::vector<Tensor> unpack(const Tensor &, const std::string &, int,
                               int) const;
    Tensor apply_rope(const Tensor &, const RopeFrequencies &) const;
    std::pair<Tensor, Tensor> apply_rope_pair(
        const Tensor &, const Tensor &, const RopeFrequencies &) const;
    Tensor attention(const Tensor &, const std::optional<Tensor> &, const std::string &,
                     int, int, int, int, int, const std::optional<RopeFrequencies> &,
                     const std::optional<RopeFrequencies> &, const std::optional<Tensor> &,
                     bool) const;
    Tensor feed_forward(const Tensor &, const std::string &) const;

  public:
    BasicAVTransformerBlock() = default;

    void load(const std::filesystem::path &, int block_index = 0,
              int convrot_group_size = 64);
    void clear();
    void set_bf16_fences(bool enabled) { bf16_fences_ = enabled; }
    bool bf16_fences() const { return bf16_fences_; }
    size_t bytes() const { return weights_.bytes(); }
    const Weights &weights() const { return weights_; }

    std::pair<Tensor, Tensor> forward(
        const Tensor &video_hidden,
        const Tensor &audio_hidden,
        const Tensor &video_adaln_params,
        const Tensor &audio_adaln_params,
        const Tensor &video_prompt_adaln_params,
        const Tensor &audio_prompt_adaln_params,
        const Tensor &av_video_params,
        const Tensor &av_audio_params,
        const Tensor &a2v_gate_params,
        const Tensor &v2a_gate_params,
        const std::optional<Tensor> &video_text_embeds = {},
        const std::optional<Tensor> &audio_text_embeds = {},
        const std::optional<RopeFrequencies> &video_rope = {},
        const std::optional<RopeFrequencies> &audio_rope = {},
        const std::optional<RopeFrequencies> &video_cross_rope = {},
        const std::optional<RopeFrequencies> &audio_cross_rope = {},
        const std::optional<Tensor> &video_attention_mask = {},
        const std::optional<Tensor> &audio_attention_mask = {},
        const std::optional<Tensor> &video_cross_attention_mask = {}) const;
};

// A single compiled LTX block graph whose weights are explicit array inputs.
// This mirrors ltx-2-mlx's `mx.compile(shared, inputs=shared)` behavior: one
// graph can execute every resident block and can also survive streamed block
// rebinding without retaining checkpoint-sized constants in the graph cache.
class CompiledBlockForward {
    using Graph = std::function<std::vector<Tensor>(
        const std::vector<Tensor> &)>;

    std::vector<std::string> weight_keys_;
    Graph graph_;
    bool video_text_ = false;
    bool audio_text_ = false;
    bool video_rope_ = false;
    bool audio_rope_ = false;
    bool video_cross_rope_ = false;
    bool audio_cross_rope_ = false;
    bool video_attention_mask_ = false;
    bool audio_attention_mask_ = false;
    bool video_cross_attention_mask_ = false;
    bool metal_convrot_ = false;
    bool bf16_fences_ = false;

  public:
    CompiledBlockForward(
        const BasicAVTransformerBlock &prototype,
        const std::optional<Tensor> &video_text_embeds,
        const std::optional<Tensor> &audio_text_embeds,
        const std::optional<RopeFrequencies> &video_rope,
        const std::optional<RopeFrequencies> &audio_rope,
        const std::optional<RopeFrequencies> &video_cross_rope,
        const std::optional<RopeFrequencies> &audio_cross_rope,
        const std::optional<Tensor> &video_attention_mask,
        const std::optional<Tensor> &audio_attention_mask,
        const std::optional<Tensor> &video_cross_attention_mask);

    bool compatible(
        const BasicAVTransformerBlock &block,
        const std::optional<Tensor> &video_text_embeds,
        const std::optional<Tensor> &audio_text_embeds,
        const std::optional<RopeFrequencies> &video_rope,
        const std::optional<RopeFrequencies> &audio_rope,
        const std::optional<RopeFrequencies> &video_cross_rope,
        const std::optional<RopeFrequencies> &audio_cross_rope,
        const std::optional<Tensor> &video_attention_mask,
        const std::optional<Tensor> &audio_attention_mask,
        const std::optional<Tensor> &video_cross_attention_mask) const;

    std::pair<Tensor, Tensor> forward(
        const BasicAVTransformerBlock &block,
        const Tensor &video_hidden,
        const Tensor &audio_hidden,
        const Tensor &video_adaln_params,
        const Tensor &audio_adaln_params,
        const Tensor &video_prompt_adaln_params,
        const Tensor &audio_prompt_adaln_params,
        const Tensor &av_ca_video_params,
        const Tensor &av_ca_audio_params,
        const Tensor &av_ca_a2v_gate_params,
        const Tensor &av_ca_v2a_gate_params,
        const std::optional<Tensor> &video_text_embeds,
        const std::optional<Tensor> &audio_text_embeds,
        const std::optional<RopeFrequencies> &video_rope,
        const std::optional<RopeFrequencies> &audio_rope,
        const std::optional<RopeFrequencies> &video_cross_rope,
        const std::optional<RopeFrequencies> &audio_cross_rope,
        const std::optional<Tensor> &video_attention_mask,
        const std::optional<Tensor> &audio_attention_mask,
        const std::optional<Tensor> &video_cross_attention_mask) const;
};

RopeFrequencies precompute_rope(const Tensor &positions, int inner_dim,
                                int num_heads, const std::vector<int> &max_pos,
                                float theta = 10000.0f,
                                bool double_precision_grid = true);

} // namespace tc::ltx_mlx
