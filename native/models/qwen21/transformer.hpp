#pragma once
#include "../../backends/mlx.hpp"
#include <array>
#include "sequence.hpp"

namespace tc::qwen21 {
struct ReferenceLatents {
    Tensor latents; // [1,height*width,channels], normalized VAE latents
    ReferenceGeometry geometry;
};

struct TransformerConfig {
    int layers = 32;
    int heads = 32;
    int head_dim = 128;
    int channels = 64;
    int context_dim = 4096;
    std::array<int, 3> rope_axes{16, 56, 56};
    float epsilon = 1e-6f;
    int hidden() const { return heads * head_dim; }
};

// Normally one request owns a Transformer. The resident Session can explicitly
// retain one instance for a matched-conditioning prefix-KV experiment.
class Transformer {
  public:
    Transformer(const Weights &, TransformerConfig = {}, const std::vector<Tensor> *fused_qkv = nullptr);
    Transformer(const Transformer &) = delete;
    Transformer &operator=(const Transformer &) = delete;
    void reset();
    bool prefix_matches(const Tensor &text, int height, int width,
                        const std::vector<ReferenceLatents> &references) const;
    Tensor forward(const Tensor &latents, const Tensor &text, float timestep,
                   int latent_height, int latent_width, bool cache_prefix = true,
                   std::unordered_map<std::string, Tensor> *trace = nullptr,
                   const std::vector<ReferenceLatents> &references = {});
    size_t cached_layers() const { return prefix_.size(); }
    // The caller restores lazy checkpoint arrays before each forward. After
    // each GPU block completes, discard its compiled constants and weights.
    void set_layer_release(std::function<void(const std::string &)> fn) {
        release_layer_ = std::move(fn);
    }
    // Decode-only FFN split: prefill remains exact GPU so cached
    // conditioning is unchanged. Caller owns the callback's runtime.
    using DecodeMLP = std::function<Tensor(int, const Tensor &)>;
    void set_decode_mlp(DecodeMLP fn, int first_block = 0) {
        require(first_block >= 0 && first_block < config_.layers,
                "Qwen21 decode MLP first block is outside the transformer");
        decode_mlp_ = std::move(fn); decode_first_block_ = first_block; decode_blocks_.clear();
    }
    void set_prefill_mlp(DecodeMLP fn, int first_block = 0, bool capture_tile_tails = false) {
        require(first_block >= 0 && first_block < config_.layers,
                "Qwen21 tiled prefill first block is outside the transformer");
        prefill_mlp_ = std::move(fn); prefill_first_block_ = first_block;
        capture_tile_tails_ = capture_tile_tails; prefill_blocks_.clear();
    }
    const Tensor &prefill_tile_tail(int block) const {
        require(block >= 0 && block < config_.layers &&
                    prefill_tile_tails_.size() == size_t(config_.layers) &&
                    prefill_tile_tails_[block].has_value(),
                "Qwen21 repeated prefill missing the matched prefix FFN tile tail");
        return *prefill_tile_tails_[block];
    }
    // Explicit diagnostic: capture each decode FFN output for reuse on the
    // immediately following denoise step. Never enabled by default.
    enum class FFNCacheMode { Off, Capture, Reuse, ReuseLast16, ReuseEvenAndCapture };
    void set_ffn_cache_mode(FFNCacheMode mode) { ffn_cache_mode_ = mode; }
    void clear_step_cache() {
        cached_ffn_.clear(); ffn_cache_mode_ = FFNCacheMode::Off;
        db_prev_front_residual_.reset(); db_middle_residual_.reset();
        db_cached_steps_ = db_consecutive_steps_ = 0; db_step_ = -1;
    }
    // Request-local decode-only DBCache: front eight blocks always run; the
    // cached middle aggregate residual is reused only after a threshold test.
    void configure_db_cache(bool enabled, float threshold = 0.08f,
                            int total_steps = 0, int max_consecutive = 2) {
        db_cache_enabled_ = enabled; db_threshold_ = threshold; db_total_steps_ = total_steps;
        db_max_consecutive_ = max_consecutive;
        clear_step_cache();
    }
    void set_db_cache_step(int step) { db_step_ = step; }
    int db_cached_steps() const { return db_cached_steps_; }
    static constexpr int db_front_blocks = 8;
    static constexpr int db_back_blocks = 0;

  private:
    struct KV { Tensor key, value; };
    const Weights &weights_;
    std::function<void(const std::string &)> release_layer_;
    TransformerConfig config_;
    // Session-owned diagnostic bank; Q/K/V are concatenated once per model
    // load rather than rebuilding 3 GiB of matrices on every image request.
    const std::vector<Tensor> *fused_qkv_ = nullptr;
    bool metal_qk_rope_ = false;
    bool metal_qk_norm_rope_ = false;
    int reference_local_attention_ = 0; // 1: later references; 2: last reference only; 3: later references in final 16 blocks
    bool profile_gpu_blocks_ = false;
    bool profile_gpu_ops_ = false;
    bool profile_prefill_segments_ = false;
    bool prefill_last_target_only_ = false;
    std::vector<KV> prefix_;
    FFNCacheMode ffn_cache_mode_ = FFNCacheMode::Off;
    std::vector<Tensor> cached_ffn_;
    bool db_cache_enabled_ = false;
    float db_threshold_ = 0.08f;
    int db_max_consecutive_ = 2;
    int db_step_ = -1, db_total_steps_ = 0, db_cached_steps_ = 0, db_consecutive_steps_ = 0;
    std::optional<Tensor> db_prev_front_residual_, db_middle_residual_;
    std::optional<Tensor> cached_text_;
    std::vector<Tensor> cached_references_;
    // Only the unfinished 1024-row FFN tile's prefix input is needed to
    // reproduce the same W8A8 tile boundary on a repeated target prefill.
    std::vector<std::optional<Tensor>> prefill_tile_tails_;
    bool capture_tile_tails_ = false;
    std::vector<ReferenceGeometry> reference_geometry_;
    SequenceGeometry sequence_;
    std::optional<Tensor> cosine_, sine_;
    int text_length_ = 0, height_ = 0, width_ = 0;
    using BlockFunction = std::function<std::vector<Tensor>(const std::vector<Tensor> &)>;
    std::vector<BlockFunction> prefill_blocks_, decode_blocks_, capture_blocks_, reuse_blocks_,
                               reuse_last16_blocks_, half_reuse_blocks_;
    DecodeMLP decode_mlp_;
    DecodeMLP prefill_mlp_;
    int prefill_first_block_ = 0, decode_first_block_ = 0;
    Tensor embedding(float timestep, mx::Dtype) const;
    void geometry(int text_length, int height, int width, const std::vector<ReferenceGeometry> &);
};
} // namespace tc::qwen21
