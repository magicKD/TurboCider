#include "model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>

namespace tc::ltx_mlx {
namespace {

int configured_eval_every_blocks() {
    const char *value = std::getenv("TURBOCIDER_LTX_MLX_EVAL_EVERY");
    if (!value || !*value) return 0;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end || parsed < 1 || parsed > 48) return 0;
    return static_cast<int>(parsed);
}

Tensor timestep_grid(const Tensor &timestep) {
    constexpr int kDim = 256;
    constexpr int kHalf = kDim / 2;
    auto exponent = -std::log(10000.0f) *
                    mx::arange(0, kHalf, mx::float32) / float(kHalf);
    auto frequencies = mx::exp(exponent);
    auto args = mx::expand_dims(mx::astype(timestep, mx::float32), -1) * frequencies;
    return mx::concatenate({mx::cos(args), mx::sin(args)}, -1);
}

} // namespace

void BlockCache::configure(std::filesystem::path checkpoint, size_t capacity,
                           int convrot_group_size) {
    clear();
    checkpoint_ = std::move(checkpoint);
    capacity_ = std::max<size_t>(1, capacity);
    /* Keep a prefix permanently resident and use a small rotating refill
     * pool for the remaining blocks.  A sequential 0..47 traversal otherwise
     * turns a plain LRU into a full-model reload on every denoise step. */
    refill_slots_ = capacity_ < 48 ? 1u : 0u;
    pinned_count_ = capacity_ - refill_slots_;
    convrot_group_size_ = convrot_group_size;
    stamp_ = 0;
}

void BlockCache::update_resident_metrics() {
    metrics_.resident_bytes = 0;
    for (const auto &[_, entry] : entries_)
        metrics_.resident_bytes += entry.block->bytes();
    metrics_.peak_resident_bytes =
        std::max(metrics_.peak_resident_bytes, metrics_.resident_bytes);
}

std::unique_ptr<BasicAVTransformerBlock> BlockCache::take_refill_slot() {
    if (entries_.size() < capacity_) return {};
    auto victim = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (static_cast<size_t>(it->first) < pinned_count_)
            continue;
        if (victim == entries_.end() ||
            it->second.stamp < victim->second.stamp)
            victim = it;
    }
    require(victim != entries_.end(),
            "LTX block cache cannot evict its pinned prefix");
    auto block = std::move(victim->second.block);
    entries_.erase(victim);
    ++metrics_.evictions;
    ++metrics_.slot_refills;
    update_resident_metrics();
    return block;
}

BasicAVTransformerBlock &BlockCache::get(int index) {
    require(index >= 0 && index < 48, "invalid LTX transformer block index");
    auto found = entries_.find(index);
    if (found != entries_.end()) {
        found->second.stamp = ++stamp_;
        ++metrics_.hits;
        return *found->second.block;
    }
    auto block = take_refill_slot();
    if (!block) {
        block = std::make_unique<BasicAVTransformerBlock>();
        ++metrics_.slot_allocations;
    }
    const auto load_started = std::chrono::steady_clock::now();
    block->load(checkpoint_, index, convrot_group_size_);
    metrics_.loaded_bytes += block->bytes();
    metrics_.load_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - load_started).count();
    entries_.emplace(index, Entry{std::move(block), ++stamp_});
    ++metrics_.loads;
    update_resident_metrics();
    return *entries_.at(index).block;
}

void BlockCache::clear() {
    entries_.clear();
    metrics_ = {};
}

void Transformer::load_top_weights(const std::filesystem::path &checkpoint) {
    top_weights_.clear();
    top_weights_.set_metal_convrot(
        std::getenv("TURBOCIDER_LTX_MLX_METAL_CONVROT") != nullptr);
    // The MLX safetensors loader is mmap-backed. Loading the shared prefix and
    // dropping block/connector subtrees avoids materializing a second 20 GiB
    // copy while keeping one immutable source identity for all block loads.
    top_weights_.load_file(checkpoint, "model.diffusion_model.");
    top_weights_.erase_prefix("transformer_blocks.");
    top_weights_.erase_prefix("video_embeddings_connector.");
    top_weights_.erase_prefix("audio_embeddings_connector.");
    top_weights_.pack_convrot_q8(options_.convrot_group_size, mx::float32);
    top_weights_.cast_unquantized_float32(mx::bfloat16);
    top_weights_.materialize();
    require(top_weights_.has("patchify_proj.weight") &&
                top_weights_.has("audio_patchify_proj.weight") &&
                top_weights_.has("proj_out.weight") &&
                top_weights_.has("audio_proj_out.weight"),
            "LTX checkpoint is missing top-level projection weights");
}

void Transformer::load(const std::filesystem::path &checkpoint,
                       const TransformerRunOptions &options) {
    clear();
    require(std::filesystem::is_regular_file(checkpoint),
            "LTX MLX checkpoint is missing: " + checkpoint.string());
    options_ = options;
    require(options_.block_count >= 1 && options_.block_count <= config_.num_layers,
            "LTX MLX block count must be between 1 and 48");
    require(std::isfinite(options_.av_ca_timestep_scale_multiplier) &&
                options_.av_ca_timestep_scale_multiplier > 0.0f,
            "LTX MLX AV timestep scale multiplier must be positive");
    config_.av_ca_timestep_scale_multiplier =
        options_.av_ca_timestep_scale_multiplier;
    checkpoint_ = checkpoint;
    load_top_weights(checkpoint);
    blocks_.configure(checkpoint_, options_.block_cache_capacity,
                      options_.convrot_group_size);
    loaded_ = true;
}

void Transformer::clear() {
    compiled_block_.reset();
    cached_video_rope_.reset();
    cached_audio_rope_.reset();
    cached_video_cross_rope_.reset();
    cached_audio_cross_rope_.reset();
    rope_video_positions_id_ = 0;
    rope_audio_positions_id_ = 0;
    blocks_.clear();
    top_weights_.clear();
    checkpoint_.clear();
    loaded_ = false;
}

PreparedTransformerInputs Transformer::prepare_for_probe(
    const Tensor &video_latent, const Tensor &audio_latent, float sigma,
    const Tensor &video_positions, const Tensor &audio_positions) const {
    require(loaded_, "LTX MLX transformer is not loaded");
    return prepare(video_latent, audio_latent, sigma,
                   video_positions, audio_positions);
}

Tensor Transformer::timestep_embedding(const Tensor &base,
                                       const std::string &prefix) const {
    auto first = top_weights_.project(base,
                                      prefix + ".emb.timestep_embedder.linear_1");
    auto second = tc::silu(first);
    return top_weights_.project(second,
                                prefix + ".emb.timestep_embedder.linear_2");
}

std::pair<Tensor, Tensor> Transformer::adaln(const Tensor &base,
                                             const std::string &prefix,
                                             int params, int dim) const {
    auto embedded = timestep_embedding(base, prefix);
    auto values = top_weights_.project(tc::silu(embedded), prefix + ".linear");
    require(values.shape() == mx::Shape{1, params * dim},
            "LTX top-level AdaLN output geometry mismatch");
    return {std::move(values), std::move(embedded)};
}

PreparedTransformerInputs Transformer::prepare(
    const Tensor &video_latent, const Tensor &audio_latent, float sigma,
    const Tensor &video_positions, const Tensor &audio_positions) const {
    require(video_latent.ndim() == 3 && video_latent.shape(0) == 1 &&
                video_latent.shape(-1) == config_.video_patch_channels,
            "LTX video latent token geometry mismatch");
    require(audio_latent.ndim() == 3 && audio_latent.shape(0) == 1 &&
                audio_latent.shape(-1) == config_.audio_patch_channels,
            "LTX audio latent token geometry mismatch");
    auto video = top_weights_.project(mx::astype(video_latent, mx::bfloat16),
                                      "patchify_proj");
    auto audio = top_weights_.project(mx::astype(audio_latent, mx::bfloat16),
                                      "audio_patchify_proj");
    const std::array<float, 1> base_timestep_values{
        sigma * config_.timestep_scale_multiplier};
    auto base_timestep = Tensor(base_timestep_values.data(), {1}, mx::float32);
    auto base = timestep_grid(base_timestep);
    auto video_a = adaln(base, "adaln_single", 9, config_.video_dim);
    auto audio_a = adaln(base, "audio_adaln_single", 9, config_.audio_dim);
    auto video_prompt = adaln(base, "prompt_adaln_single", 2, config_.video_dim);
    auto audio_prompt = adaln(base, "audio_prompt_adaln_single", 2, config_.audio_dim);
    const std::array<float, 1> gate_timestep_values{
        sigma * config_.av_ca_timestep_scale_multiplier};
    auto gate_base = timestep_grid(
        Tensor(gate_timestep_values.data(), {1}, mx::float32));
    // TurboCider's current C/Metal runtime and the exported parity fixtures
    // use one unscaled AV-cross timestep for scale/shift and residual gates.
    auto av_video = adaln(gate_base,
                          "av_ca_video_scale_shift_adaln_single", 4,
                          config_.video_dim);
    auto av_audio = adaln(gate_base,
                          "av_ca_audio_scale_shift_adaln_single", 4,
                          config_.audio_dim);
    auto a2v_gate = adaln(gate_base, "av_ca_a2v_gate_adaln_single", 1,
                          config_.video_dim);
    auto v2a_gate = adaln(gate_base, "av_ca_v2a_gate_adaln_single", 1,
                          config_.audio_dim);

    PreparedTransformerInputs result{
        std::move(video), std::move(audio),
        std::move(video_a.first), std::move(audio_a.first),
        std::move(video_prompt.first), std::move(audio_prompt.first),
        std::move(av_video.first), std::move(av_audio.first),
        std::move(a2v_gate.first), std::move(v2a_gate.first),
        std::move(video_a.second), std::move(audio_a.second),
        std::nullopt, std::nullopt, std::nullopt, std::nullopt};
    const bool cache_rope =
        std::getenv("TURBOCIDER_LTX_MLX_CACHE_ROPE") != nullptr;
    const bool rope_cache_hit = cache_rope && video_positions.size() &&
        audio_positions.size() &&
        rope_video_positions_id_ == video_positions.id() &&
        rope_audio_positions_id_ == audio_positions.id() &&
        cached_video_rope_.has_value() && cached_audio_rope_.has_value() &&
        cached_video_cross_rope_.has_value() &&
        cached_audio_cross_rope_.has_value();
    if (rope_cache_hit) {
        result.video_rope = cached_video_rope_;
        result.audio_rope = cached_audio_rope_;
        result.video_cross_rope = cached_video_cross_rope_;
        result.audio_cross_rope = cached_audio_cross_rope_;
        return result;
    }
    if (video_positions.size()) {
        result.video_rope = precompute_rope(
            video_positions, config_.video_heads * config_.video_head_dim,
            config_.video_heads, config_.video_max_pos, config_.rope_theta,
            config_.double_precision_rope);
        result.video_cross_rope = precompute_rope(
            mx::slice(video_positions, {0, 0, 0},
                      {1, video_positions.shape(1), 1}),
            config_.av_heads * config_.av_head_dim, config_.av_heads,
            {config_.video_max_pos[0]}, config_.rope_theta,
            config_.double_precision_rope);
    }
    if (audio_positions.size()) {
        result.audio_rope = precompute_rope(
            audio_positions, config_.audio_heads * config_.audio_head_dim,
            config_.audio_heads, config_.audio_max_pos, config_.rope_theta,
            config_.double_precision_rope);
        result.audio_cross_rope = precompute_rope(
            mx::slice(audio_positions, {0, 0, 0},
                      {1, audio_positions.shape(1), 1}),
            config_.av_heads * config_.av_head_dim, config_.av_heads,
            {config_.audio_max_pos[0]}, config_.rope_theta,
            config_.double_precision_rope);
    }
    if (cache_rope && result.video_rope && result.audio_rope &&
        result.video_cross_rope && result.audio_cross_rope) {
        // Retaining a lazy graph would rebuild its trigonometric work when a
        // later scheduler evaluation consumes it. Materialize once, then keep
        // only the finished frequency arrays for the lifetime of this stage's
        // position-array identity.
        mx::eval({result.video_rope->cosine, result.video_rope->sine,
                  result.audio_rope->cosine, result.audio_rope->sine,
                  result.video_cross_rope->cosine,
                  result.video_cross_rope->sine,
                  result.audio_cross_rope->cosine,
                  result.audio_cross_rope->sine});
        rope_video_positions_id_ = video_positions.id();
        rope_audio_positions_id_ = audio_positions.id();
        cached_video_rope_ = result.video_rope;
        cached_audio_rope_ = result.audio_rope;
        cached_video_cross_rope_ = result.video_cross_rope;
        cached_audio_cross_rope_ = result.audio_cross_rope;
    }
    return result;
}

Tensor Transformer::output_block(const Tensor &hidden, const Tensor &embedded,
                                 const std::string &table_key,
                                 const std::string &projection_key) const {
    auto normalized = mx::fast::layer_norm(hidden, {}, {}, config_.norm_eps);
    auto table = top_weights_.at(table_key);
    auto values = table + mx::expand_dims(embedded, 1);
    auto shift = mx::slice(values, {0, 0, 0}, {1, 1, hidden.shape(-1)});
    auto scale = mx::slice(values, {0, 1, 0}, {1, 2, hidden.shape(-1)});
    auto modulated = normalized * (1.0f + scale) + shift;
    return top_weights_.project(modulated, projection_key);
}

std::pair<Tensor, Tensor> Transformer::forward(
    const Tensor &video_latent, const Tensor &audio_latent, float sigma,
    const Tensor &video_text, const Tensor &audio_text,
    const Tensor &video_positions, const Tensor &audio_positions,
    const std::optional<Tensor> &video_attention_mask,
    const std::optional<Tensor> &audio_attention_mask,
    const std::optional<Tensor> &video_cross_attention_mask) {
    require(loaded_, "LTX MLX transformer is not loaded");
    auto state = prepare(video_latent, audio_latent, sigma,
                         video_positions, audio_positions);
    auto active_video_text = mx::astype(video_text, mx::bfloat16);
    auto active_audio_text = mx::astype(audio_text, mx::bfloat16);
    const int eval_every = configured_eval_every_blocks();
    for (int index = 0; index < options_.block_count; ++index) {
        auto &block = blocks_.get(index);
        auto output = [&]() -> std::pair<Tensor, Tensor> {
          if (std::getenv("TURBOCIDER_LTX_MLX_COMPILE_BLOCK")) {
            if (!compiled_block_ || !compiled_block_->compatible(
                    block, active_video_text, active_audio_text,
                    state.video_rope, state.audio_rope,
                    state.video_cross_rope, state.audio_cross_rope,
                    video_attention_mask, audio_attention_mask,
                    video_cross_attention_mask)) {
                compiled_block_ = std::make_unique<CompiledBlockForward>(
                    block, active_video_text, active_audio_text,
                    state.video_rope, state.audio_rope,
                    state.video_cross_rope, state.audio_cross_rope,
                    video_attention_mask, audio_attention_mask,
                    video_cross_attention_mask);
            }
            return compiled_block_->forward(
                block, state.video_hidden, state.audio_hidden,
                state.video_adaln, state.audio_adaln,
                state.video_prompt, state.audio_prompt,
                state.av_video, state.av_audio, state.a2v_gate, state.v2a_gate,
                active_video_text, active_audio_text, state.video_rope,
                state.audio_rope, state.video_cross_rope,
                state.audio_cross_rope, video_attention_mask,
                audio_attention_mask, video_cross_attention_mask);
          }
            return block.forward(
                state.video_hidden, state.audio_hidden,
                state.video_adaln, state.audio_adaln,
                state.video_prompt, state.audio_prompt,
                state.av_video, state.av_audio, state.a2v_gate, state.v2a_gate,
                active_video_text, active_audio_text, state.video_rope,
                state.audio_rope, state.video_cross_rope,
                state.audio_cross_rope, video_attention_mask,
                audio_attention_mask, video_cross_attention_mask);
        }();
        state.video_hidden = std::move(output.first);
        state.audio_hidden = std::move(output.second);
        const bool periodic_flush = eval_every > 0 &&
            ((index + 1) % eval_every == 0 || index + 1 == options_.block_count);
        if (options_.force_eval_each_block || periodic_flush)
            mx::eval(state.video_hidden, state.audio_hidden);
    }
    auto video = output_block(state.video_hidden, state.video_embedded_timestep,
                              "scale_shift_table", "proj_out");
    auto audio = output_block(state.audio_hidden, state.audio_embedded_timestep,
                              "audio_scale_shift_table", "audio_proj_out");
    mx::eval(video, audio);
    return {std::move(video), std::move(audio)};
}

} // namespace tc::ltx_mlx
