#include "transformer.hpp"
#include "activation.hpp"

#include <cmath>

namespace tc::ltx098 {
namespace {

Tensor time_grid(const Tensor &timestep) {
    constexpr int half = 128;
    auto frequencies = mx::exp(-std::log(10000.0f) * mx::arange(0, half, mx::float32) / float(half));
    auto angles = mx::expand_dims(mx::astype(timestep, mx::float32), -1) * frequencies;
    return mx::concatenate({mx::cos(angles), mx::sin(angles)}, -1);
}

} // namespace

void Transformer::load(const std::filesystem::path &checkpoint, int block_count, int evaluate_every) {
    require(block_count >= 1 && block_count <= 28 && evaluate_every >= 0 && evaluate_every <= 28,
            "invalid LTX 0.9.8 block count/evaluation interval");
    require(std::filesystem::is_regular_file(checkpoint), "LTX 0.9.8 checkpoint is missing");
    top_.clear();
    blocks_.clear();
    rope_.reset();
    top_.load_file(checkpoint, "model.diffusion_model.");
    top_.erase_prefix("transformer_blocks.");
    for (const auto *name : {"patchify_proj.weight", "caption_projection.linear_1.weight",
                              "caption_projection.linear_2.weight", "adaln_single.emb.timestep_embedder.linear_1.weight",
                              "adaln_single.emb.timestep_embedder.linear_2.weight", "adaln_single.linear.weight",
                              "scale_shift_table", "proj_out.weight"})
        require(top_.has(name), "LTX 0.9.8 top-level weight missing: " + std::string(name));
    require(top_.at("patchify_proj.weight").shape() == mx::Shape{2048, 128} &&
                top_.at("caption_projection.linear_1.weight").shape() == mx::Shape{2048, 4096} &&
                top_.at("scale_shift_table").shape() == mx::Shape{2, 2048},
            "checkpoint is not the LTX 0.9.8 2B transformer geometry");
    top_.materialize();
    blocks_.reserve(block_count);
    for (int index = 0; index < block_count; ++index) {
        auto block = std::make_unique<Block>();
        block->load(checkpoint, index);
        blocks_.push_back(std::move(block));
    }
    evaluate_every_ = evaluate_every;
}

Tensor Transformer::prepare_text(const Tensor &text) const {
    require(text.ndim() == 3 && text.shape(0) == 1 && text.shape(1) > 0 && text.shape(2) == 4096 &&
                text.dtype() == mx::bfloat16, "LTX 0.9.8 requires BF16 [1,text,4096] T5 conditioning");
    return top_.project(gelu_tanh(top_.project(text, "caption_projection.linear_1")), "caption_projection.linear_2");
}

std::vector<CrossAttentionKV> Transformer::prepare_cross_attention(const Tensor &text) const {
    require(!blocks_.empty() && text.ndim() == 3 && text.shape(0) == 1 && text.shape(1) > 0 &&
                text.shape(2) == 2048 && text.dtype() == mx::bfloat16,
            "LTX 0.9.8 cross-attention cache requires prepared BF16 text");
    std::vector<CrossAttentionKV> result;
    result.reserve(blocks_.size());
    for (const auto &block : blocks_) {
        auto kv = block->prepare_cross_attention(text);
        mx::eval(kv.key, kv.value);
        result.push_back(std::move(kv));
    }
    return result;
}

const ltx_mlx::RopeFrequencies &Transformer::positions(int frames, int height, int width, float fps) {
    if (rope_ && rope_frames_ == frames && rope_height_ == height && rope_width_ == width && rope_fps_ == fps) return *rope_;
    std::vector<float> coordinates;
    coordinates.reserve(size_t(frames) * height * width * 3);
    for (int frame = 0; frame < frames; ++frame)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                // 0.9.8 uses the original non-causal corner coordinates.
                coordinates.push_back(float(frame * 8) / fps);
                coordinates.push_back(float(y * 32));
                coordinates.push_back(float(x * 32));
            }
    auto positions = Tensor(coordinates.data(), {1, frames * height * width, 3}, mx::float32);
    auto value = ltx_mlx::precompute_rope(positions, 2048, 32, {20, 2048, 2048}, 10000.f, false);
    // The reference casts its rotation matrix to the hidden-state dtype.
    value.cosine = mx::astype(value.cosine, mx::bfloat16);
    value.sine = mx::astype(value.sine, mx::bfloat16);
    mx::eval(value.cosine, value.sine);
    rope_ = std::move(value);
    rope_frames_ = frames; rope_height_ = height; rope_width_ = width; rope_fps_ = fps;
    return *rope_;
}

Tensor Transformer::forward(const Tensor &latents, const Tensor &text, float sigma, float fps,
                            bool image_conditioned, float strength, const std::optional<Tensor> &mask) {
    return forward_impl(latents, text, sigma, fps, image_conditioned, strength, mask, nullptr);
}

Tensor Transformer::forward_cached(const Tensor &latents, const Tensor &text, float sigma, float fps,
                                   bool image_conditioned, float strength, const std::optional<Tensor> &mask,
                                   const std::vector<CrossAttentionKV> &cached) {
    return forward_impl(latents, text, sigma, fps, image_conditioned, strength, mask, &cached);
}

Tensor Transformer::forward_impl(const Tensor &latents, const Tensor &text, float sigma, float fps,
                                 bool image_conditioned, float strength, const std::optional<Tensor> &mask,
                                 const std::vector<CrossAttentionKV> *cached) {
    require(!blocks_.empty(), "LTX 0.9.8 transformer is not loaded");
    require(latents.ndim() == 5 && latents.shape(0) == 1 && latents.shape(1) > 0 && latents.shape(2) > 0 &&
                latents.shape(3) > 0 && latents.shape(4) == 128 && latents.dtype() == mx::bfloat16,
            "LTX 0.9.8 requires BF16 [1,F,H,W,128] latents");
    require(text.ndim() == 3 && text.shape(0) == 1 && text.shape(1) > 0 && text.shape(2) == 2048 &&
                text.dtype() == mx::bfloat16, "LTX 0.9.8 prepared text geometry mismatch");
    require(std::isfinite(sigma) && sigma >= 0 && sigma <= 1 && std::isfinite(fps) && fps > 0 &&
                std::isfinite(strength) && strength >= 0 && strength <= 1, "invalid LTX 0.9.8 sigma/fps/strength");
    require(!cached || cached->size() == blocks_.size(), "LTX 0.9.8 cross-attention cache block count mismatch");
    const int frames = latents.shape(1), height = latents.shape(2), width = latents.shape(3);
    const int frame_tokens = height * width, token_count = frames * frame_tokens;
    const bool two_times = image_conditioned && frames > 1 && strength != 0;
    std::vector<float> times = two_times ? std::vector<float>{sigma * (1.f - strength) * 1000.f, sigma * 1000.f}
                                         : std::vector<float>{sigma * (image_conditioned ? (1.f - strength) : 1.f) * 1000.f};
    auto input_time = Tensor(times.data(), {int(times.size())}, mx::float32);
    auto embedded = top_.project(tc::silu(top_.project(mx::astype(time_grid(input_time), mx::bfloat16),
                                     "adaln_single.emb.timestep_embedder.linear_1")),
                                 "adaln_single.emb.timestep_embedder.linear_2");
    auto modulation = top_.project(tc::silu(embedded), "adaln_single.linear");
    auto expand_times = [&](const Tensor &value) {
        const int dim = value.shape(1);
        if (!two_times) return mx::reshape(value, {1, 1, dim});
        auto first = mx::reshape(mx::slice(value, {0, 0}, {1, dim}), {1, 1, dim});
        auto rest = mx::reshape(mx::slice(value, {1, 0}, {2, dim}), {1, 1, dim});
        return mx::concatenate({mx::broadcast_to(first, {1, frame_tokens, dim}),
                                mx::broadcast_to(rest, {1, token_count - frame_tokens, dim})}, 1);
    };
    auto per_token_modulation = expand_times(modulation);
    auto per_token_embedded = expand_times(embedded);
    auto hidden = top_.project(mx::reshape(latents, {1, token_count, 128}), "patchify_proj");
    const auto &rope = positions(frames, height, width, fps);
    for (size_t index = 0; index < blocks_.size(); ++index) {
        hidden = cached ? blocks_[index]->forward_cached(hidden, text, per_token_modulation, rope, mask, (*cached)[index])
                        : blocks_[index]->forward(hidden, text, per_token_modulation, rope, mask);
        if (evaluate_every_ && (index + 1) % size_t(evaluate_every_) == 0) mx::eval(hidden);
    }
    auto table = mx::reshape(top_.at("scale_shift_table"), {1, 1, 2, 2048});
    auto affine = table + mx::expand_dims(per_token_embedded, 2);
    auto shift = mx::squeeze(mx::slice(affine, {0, 0, 0, 0}, {1, per_token_embedded.shape(1), 1, 2048}), 2);
    auto scale = mx::squeeze(mx::slice(affine, {0, 0, 1, 0}, {1, per_token_embedded.shape(1), 2, 2048}), 2);
    auto normalized = mx::fast::layer_norm(hidden, {}, {}, 1.0e-6f);
    return mx::reshape(top_.project(normalized * (Tensor(1.f, hidden.dtype()) + scale) + shift, "proj_out"),
                       {1, frames, height, width, 128});
}

} // namespace tc::ltx098
