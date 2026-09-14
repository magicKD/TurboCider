#pragma once

#include "block.hpp"

#include <functional>
#include <chrono>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace tc::ltx_mlx {

struct PreparedTransformerInputs {
    Tensor video_hidden;
    Tensor audio_hidden;
    Tensor video_adaln;
    Tensor audio_adaln;
    Tensor video_prompt;
    Tensor audio_prompt;
    Tensor av_video;
    Tensor av_audio;
    Tensor a2v_gate;
    Tensor v2a_gate;
    Tensor video_embedded_timestep;
    Tensor audio_embedded_timestep;
    std::optional<RopeFrequencies> video_rope;
    std::optional<RopeFrequencies> audio_rope;
    std::optional<RopeFrequencies> video_cross_rope;
    std::optional<RopeFrequencies> audio_cross_rope;
};

struct BlockCacheMetrics {
    uint64_t loads = 0;
    uint64_t hits = 0;
    uint64_t evictions = 0;
    uint64_t loaded_bytes = 0;
    uint64_t slot_allocations = 0;
    uint64_t slot_refills = 0;
    double load_seconds = 0.0;
    size_t resident_bytes = 0;
    size_t peak_resident_bytes = 0;
};

// Capacity-limited block owner used by resident and low-memory paths. A block
// is evicted only after its output has been evaluated by the caller.
class BlockCache {
    struct Entry {
        std::unique_ptr<BasicAVTransformerBlock> block;
        uint64_t stamp = 0;
    };

    std::filesystem::path checkpoint_;
    size_t capacity_ = 1;
    size_t pinned_count_ = 0;
    size_t refill_slots_ = 1;
    int convrot_group_size_ = 64;
    uint64_t stamp_ = 0;
    std::unordered_map<int, Entry> entries_;
    BlockCacheMetrics metrics_;

    std::unique_ptr<BasicAVTransformerBlock> take_refill_slot();
    void update_resident_metrics();

  public:
    BlockCache() = default;
    void configure(std::filesystem::path checkpoint, size_t capacity,
                   int convrot_group_size = 64);
    BasicAVTransformerBlock &get(int index);
    void clear();
    const BlockCacheMetrics &metrics() const { return metrics_; }
    size_t capacity() const { return capacity_; }
    size_t pinned_count() const { return pinned_count_; }
    size_t refill_slots() const { return refill_slots_; }
};

struct ModelConfig {
    int num_layers = 48;
    int video_dim = 4096;
    int audio_dim = 2048;
    int video_patch_channels = 128;
    int audio_patch_channels = 128;
    int video_heads = 32;
    int audio_heads = 32;
    int video_head_dim = 128;
    int audio_head_dim = 64;
    int av_heads = 32;
    int av_head_dim = 64;
    int timestep_embedding_dim = 256;
    int timestep_scale_multiplier = 1000;
    float av_ca_timestep_scale_multiplier = 1.0f;
    float rope_theta = 10000.0f;
    float norm_eps = 1.0e-6f;
    std::vector<int> video_max_pos{20, 2048, 2048};
    std::vector<int> audio_max_pos{20};
    bool double_precision_rope = true;
};

struct TransformerRunOptions {
    size_t block_cache_capacity = 48;
    int convrot_group_size = 64;
    int block_count = 48;
    bool force_eval_each_block = true;
    float av_ca_timestep_scale_multiplier = 1.0f;
};

class Transformer {
    Weights top_weights_;
    BlockCache blocks_;
    ModelConfig config_;
    TransformerRunOptions options_;
    std::filesystem::path checkpoint_;
    std::unique_ptr<CompiledBlockForward> compiled_block_;
    // A stage reuses the exact same position arrays for every scheduler
    // evaluation.  Keep one identity-bound materialized RoPE bundle for the
    // opt-in cache experiment; changing either MLX array invalidates all four
    // frequency tensors, so probe callers with arbitrary positions cannot
    // receive a shape-only stale result.
    mutable std::uintptr_t rope_video_positions_id_ = 0;
    mutable std::uintptr_t rope_audio_positions_id_ = 0;
    mutable std::optional<RopeFrequencies> cached_video_rope_;
    mutable std::optional<RopeFrequencies> cached_audio_rope_;
    mutable std::optional<RopeFrequencies> cached_video_cross_rope_;
    mutable std::optional<RopeFrequencies> cached_audio_cross_rope_;
    bool loaded_ = false;

    Tensor timestep_embedding(const Tensor &, const std::string &) const;
    std::pair<Tensor, Tensor> adaln(const Tensor &, const std::string &, int,
                                    int) const;
    PreparedTransformerInputs prepare(const Tensor &, const Tensor &, float,
                                      const Tensor &, const Tensor &) const;
    Tensor output_block(const Tensor &, const Tensor &, const std::string &,
                        const std::string &) const;
    void load_top_weights(const std::filesystem::path &);

  public:
    Transformer() = default;
    void load(const std::filesystem::path &, const TransformerRunOptions & = {});
    void clear();
    const ModelConfig &config() const { return config_; }
    const BlockCacheMetrics &cache_metrics() const { return blocks_.metrics(); }
    size_t block_cache_capacity() const { return blocks_.capacity(); }
    size_t pinned_block_count() const { return blocks_.pinned_count(); }
    size_t refill_slot_count() const { return blocks_.refill_slots(); }
    size_t top_weight_bytes() const { return top_weights_.bytes(); }

    // Validation hook: expose the top-level patch/timestep preparation without
    // running the block stack. It is intentionally named separately from the
    // production forward API so callers cannot confuse it with a denoiser step.
    PreparedTransformerInputs prepare_for_probe(
        const Tensor &, const Tensor &, float, const Tensor &, const Tensor &) const;

    // Inputs are patchified token tensors: [1, rows, 128]. Text tensors are
    // already connector-projected: video [1,text,4096], audio [1,text,2048].
    // This API deliberately matches the boundary produced by the existing C
    // connector so the session can switch only the denoiser implementation.
    std::pair<Tensor, Tensor> forward(
        const Tensor &video_latent, const Tensor &audio_latent, float sigma,
        const Tensor &video_text, const Tensor &audio_text,
        const Tensor &video_positions, const Tensor &audio_positions,
        const std::optional<Tensor> &video_attention_mask = {},
        const std::optional<Tensor> &audio_attention_mask = {},
        const std::optional<Tensor> &video_cross_attention_mask = {});
};

} // namespace tc::ltx_mlx
