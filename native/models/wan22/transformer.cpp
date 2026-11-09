#include "transformer.hpp"
#include "linear.hpp"
#include "../../components/diffusion/wan.hpp"

#include <array>
#include <cmath>

namespace tc::wan22 {
namespace {

Tensor silu(const Tensor &value) {
    static auto graph = mx::compile([](const std::vector<Tensor> &args) {
        auto input = mx::astype(args[0], mx::float32);
        return std::vector<Tensor>{mx::astype(input * mx::sigmoid(input), args[0].dtype())};
    }, true);
    return graph({value})[0];
}

Tensor gelu_tanh(const Tensor &value) {
    static auto graph = mx::compile([](const std::vector<Tensor> &args) {
        auto x = mx::astype(args[0], mx::float32);
        auto result = .5f * x * (1.f + mx::tanh(.7978845608028654f * (x + .044715f * x * x * x)));
        return std::vector<Tensor>{mx::astype(result, args[0].dtype())};
    }, true);
    return graph({value})[0];
}

std::pair<Tensor, Tensor> frequencies(int frames, int height, int width, int head_dim) {
    const int spatial = 2 * (head_dim / 6);
    const std::array<int, 3> dimensions{head_dim - 2 * spatial, spatial, spatial};
    require(dimensions[0] > 0 && dimensions[1] > 0 && dimensions[2] > 0 &&
                dimensions[0] % 2 == 0 && dimensions[1] % 2 == 0,
            "unsupported Wan2.2 rotary head dimension");
    auto positions = mx::arange(frames * height * width, mx::int32);
    std::vector<Tensor> cosine, sine;
    for (int axis = 0; axis < 3; ++axis) {
        const int inner = axis == 0 ? height * width : (axis == 1 ? width : 1);
        const int extent = axis == 0 ? frames : (axis == 1 ? height : width);
        auto coordinate = mx::remainder(mx::floor_divide(positions, Tensor(inner)), Tensor(extent));
        auto inverse = 1.f / mx::power(Tensor(10000.f),
                         mx::arange(0, dimensions[axis], 2, mx::float32) / float(dimensions[axis]));
        auto angles = mx::expand_dims(mx::astype(coordinate, mx::float32), 1) * inverse;
        cosine.push_back(mx::repeat(mx::cos(angles), 2, 1));
        sine.push_back(mx::repeat(mx::sin(angles), 2, 1));
    }
    return {mx::concatenate(cosine, 1), mx::concatenate(sine, 1)};
}

} // namespace

void Transformer::load(const std::filesystem::path &path, TransformerConfig config) {
    require(std::filesystem::is_regular_file(path) && config.channels > 0 && config.hidden > 0 &&
                config.heads > 0 && config.hidden % config.heads == 0 && config.intermediate > 0 &&
                config.layers > 0 && config.layers <= 30 && config.text_dim > 0 &&
                config.frequency_dim > 0 && config.frequency_dim % 2 == 0 && config.epsilon > 0,
            "invalid Wan2.2 checkpoint or transformer config");
    config_ = config;
    top_.clear();
    blocks_.clear();
    top_.load_file(path);
    top_.erase_prefix("blocks.");
    if (config.layout == CheckpointLayout::comfy) prepare_comfy_weights(top_, false);
    for (const auto *name : {"patch_embedding.weight", "condition_embedder.time_embedder.linear_1.weight",
                              "condition_embedder.time_embedder.linear_2.weight", "condition_embedder.time_proj.weight",
                              "condition_embedder.text_embedder.linear_1.weight", "condition_embedder.text_embedder.linear_2.weight",
                              "scale_shift_table", "proj_out.weight"})
        require(top_.has(name), "Wan2.2 top-level weight is missing: " + std::string(name));
    require(top_.at("patch_embedding.weight").shape() == mx::Shape{config.hidden, config.channels, 1, 2, 2} &&
                top_.at("proj_out.weight").shape() == mx::Shape{config.channels * 4, config.hidden} &&
                top_.at("scale_shift_table").shape() == mx::Shape{1, 2, config.hidden} &&
                top_.at("condition_embedder.text_embedder.linear_1.weight").shape() == mx::Shape{config.hidden, config.text_dim},
            "Wan2.2 checkpoint top-level geometry mismatch");
    top_.materialize();
    blocks_.reserve(config.layers);
    for (int index = 0; index < config.layers; ++index) {
        auto block = std::make_unique<Block>();
        block->load(path, index, {config.hidden, config.heads, config.intermediate, config.epsilon, config.layout});
        blocks_.push_back(std::move(block));
    }
}

Tensor Transformer::prepare_text(const Tensor &text, const Trace &trace) const {
    require(text.ndim() == 3 && text.shape(0) == 1 && text.shape(1) > 0 && text.shape(2) == config_.text_dim &&
                text.dtype() == mx::bfloat16, "Wan2.2 text must be BF16 [1,tokens,text_dim]");
    auto context = linear(top_, gelu_tanh(linear(top_, text, "condition_embedder.text_embedder.linear_1")),
                          "condition_embedder.text_embedder.linear_2");
    if (trace) trace("context", context);
    return context;
}

std::vector<CrossAttentionKV> Transformer::prepare_cross_attention(const Tensor &context,
                                                                   std::atomic<bool> &cancelled) const {
    require(!blocks_.empty(), "Wan2.2 transformer is not loaded");
    std::vector<CrossAttentionKV> result;
    result.reserve(blocks_.size());
    for (const auto &block : blocks_) {
        checkpoint(cancelled);
        result.push_back(block->prepare_cross_attention(context));
    }
    return result;
}

Tensor Transformer::forward(const Tensor &latents, const Tensor &context, float timestep,
                            const Event &event, std::atomic<bool> &cancelled, const Trace &trace,
                            const std::vector<CrossAttentionKV> *cached_cross) const {
    const auto &c = config_;
    require(!blocks_.empty() && latents.ndim() == 5 && latents.shape(0) == 1 && latents.shape(1) == c.channels &&
                latents.shape(2) >= 2 && latents.shape(3) > 0 && latents.shape(3) % 2 == 0 &&
                latents.shape(4) > 0 && latents.shape(4) % 2 == 0 && latents.dtype() == mx::bfloat16 &&
                context.ndim() == 3 && context.shape(0) == 1 && context.shape(2) == c.hidden &&
                context.dtype() == mx::bfloat16 && std::isfinite(timestep) && timestep >= 0 && timestep <= 1000,
            "invalid Wan2.2 latent, context or timestep");
    require(!cached_cross || cached_cross->size() == blocks_.size(), "invalid Wan2.2 cross-attention cache size");
    const int f = latents.shape(2), h = latents.shape(3), w = latents.shape(4);
    const int first_frame_tokens = (h / 2) * (w / 2), token_count = f * first_frame_tokens;
    auto patch = mx::reshape(latents, {1, c.channels, f, h / 2, 2, w / 2, 2});
    patch = mx::reshape(mx::transpose(patch, {0, 2, 3, 5, 1, 4, 6}), {1, token_count, c.channels * 4});
    auto hidden = linear(top_, patch, "patch_embedding");
    if (trace) trace("hidden_input", hidden);
    const float times[] = {0.f, timestep};
    auto time = components::wan_timestep_embedding(Tensor(times, {2}, mx::float32), c.frequency_dim);
    time = linear(top_, silu(linear(top_, mx::astype(time, mx::bfloat16),
                    "condition_embedder.time_embedder.linear_1")), "condition_embedder.time_embedder.linear_2");
    auto modulation = linear(top_, silu(time), "condition_embedder.time_proj");
    modulation = mx::reshape(modulation, {1, 2, 6, c.hidden});
    auto [cosine, sine] = frequencies(f, h / 2, w / 2, c.hidden / c.heads);
    if (trace) {
        trace("modulation", modulation);
        trace("cosine", cosine);
        trace("sine", sine);
    }
    for (int index = 0; index < c.layers; ++index) {
        checkpoint(cancelled);
        event("wan22_dit_block", index, c.layers);
        hidden = blocks_[index]->forward(hidden, context, modulation, cosine, sine, first_frame_tokens, {},
                                         cached_cross ? &cached_cross->at(index) : nullptr);
        mx::eval(hidden);
        if (trace) trace("block_" + std::to_string(index), hidden);
    }
    auto packed = mx::reshape(mx::astype(top_.at("scale_shift_table"), hidden.dtype()), {1, 1, 2, c.hidden}) +
                  mx::reshape(time, {1, 2, 1, c.hidden});
    auto expand = [&](int which) {
        auto entry = mx::squeeze(mx::slice(packed, {0, 0, which, 0}, {1, 2, which + 1, c.hidden}), 2);
        auto first = mx::slice(entry, {0, 0, 0}, {1, 1, c.hidden});
        auto rest = mx::slice(entry, {0, 1, 0}, {1, 2, c.hidden});
        return mx::concatenate({mx::broadcast_to(first, {1, first_frame_tokens, c.hidden}),
                                mx::broadcast_to(rest, {1, token_count - first_frame_tokens, c.hidden})}, 1);
    };
    auto shift = expand(0), scale = expand(1);
    auto normalized = mx::astype(mx::fast::layer_norm(mx::astype(hidden, mx::float32),
                                                       std::nullopt, std::nullopt, c.epsilon), hidden.dtype());
    auto multiplier = Tensor(1.f, hidden.dtype()) + scale;
    normalized = mx::astype(mx::astype(shift, mx::float32) + mx::astype(normalized, mx::float32) *
                            mx::astype(multiplier, mx::float32), hidden.dtype());
    auto output = linear(top_, normalized, "proj_out");
    output = mx::reshape(output, {1, f, h / 2, w / 2, 1, 2, 2, c.channels});
    output = mx::reshape(mx::transpose(output, {0, 7, 1, 4, 2, 5, 3, 6}), {1, c.channels, f, h, w});
    mx::eval(output);
    event("wan22_dit_block", c.layers, c.layers);
    checkpoint(cancelled);
    return output;
}

} // namespace tc::wan22
