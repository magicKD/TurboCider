#include "dit.hpp"
#include "../../components/diffusion/wan.hpp"

#include <cmath>

namespace tc::wan {
namespace {
Tensor gelu(const Tensor &x) {
    // The Python reference's gelu_approx is itself compiled. Keep this exact
    // formula (not sigmoid GELU) with the same FP16 intermediate semantics.
    static auto graph = mx::compile([](const std::vector<Tensor> &a) {
        auto x = a[0];
        auto scalar = [&](float value) { return Tensor(value, x.dtype()); };
        return std::vector<Tensor>{scalar(.5f) * x * (scalar(1.f) + mx::tanh(
            scalar(float(std::sqrt(2. / M_PI))) *
                (x + scalar(.044715f) * mx::power(x, scalar(3.f)))))};
    }, true);
    return graph({x})[0];
}
} // namespace

Tensor DiT::normalize(const Tensor &x) const {
    auto f = mx::astype(x, mx::float32);
    auto centered = f - mx::mean(f, -1, true);
    return centered * mx::rsqrt(mx::mean(mx::square(centered), -1, true) +
                                weights_.config().epsilon);
}

Tensor DiT::rms_normalize(const Tensor &x, const std::string &key) const {
    auto f = mx::astype(x, mx::float32);
    auto y = f * mx::rsqrt(mx::mean(mx::square(f), -1, true) + weights_.config().epsilon);
    // Wan casts normalized activations BEFORE multiplying the learned scale.
    return mx::astype(y, x.dtype()) * weights_.at(key);
}

Tensor DiT::attention(const Tensor &q, const Tensor &k, const Tensor &v) const {
    const auto &c = weights_.config();
    auto to_heads = [&](const Tensor &x) {
        return mx::transpose(mx::reshape(x, {x.shape(0), -1, c.heads, c.head_dim}), {0, 2, 1, 3});
    };
    auto y = mx::fast::scaled_dot_product_attention(to_heads(q), to_heads(k), to_heads(v),
                                                   1.f / std::sqrt(float(c.head_dim)));
    return mx::reshape(mx::transpose(y, {0, 2, 1, 3}), {q.shape(0), q.shape(1), c.hidden()});
}

Tensor DiT::rotary(const Tensor &x, const Tensor &cosine, const Tensor &sine) const {
    const auto &c = weights_.config();
    auto shape = mx::Shape{x.shape(0), x.shape(1), c.heads, c.head_dim};
    auto f = mx::reshape(mx::astype(x, mx::float32), shape);
    auto pairs = mx::reshape(f, {shape[0], shape[1], shape[2], c.head_dim / 2, 2});
    auto halves = mx::split(pairs, 2, -1);
    auto rotated = mx::reshape(mx::concatenate({-halves[1], halves[0]}, -1), shape);
    auto cs = mx::reshape(cosine, {1, shape[1], 1, c.head_dim});
    auto sn = mx::reshape(sine, {1, shape[1], 1, c.head_dim});
    return mx::reshape(mx::astype(f * cs + rotated * sn, x.dtype()), x.shape());
}

Tensor DiT::patch_embed(const Tensor &latent) const {
    const auto &c = weights_.config();
    require(latent.ndim() == 5 && latent.shape(0) == 1 && latent.shape(1) == c.channels &&
                latent.dtype() == mx::float16, "Wan DiT expects FP16 BCTHW latents, batch one");
    auto g = components::wan_patch_grid({latent.shape(2), latent.shape(3), latent.shape(4)},
                                        {c.patch[0], c.patch[1], c.patch[2]});
    auto x = mx::reshape(latent, {1, c.channels, g.frames, c.patch[0], g.height,
                                  c.patch[1], g.width, c.patch[2]});
    x = mx::reshape(mx::transpose(x, {0, 2, 4, 6, 1, 3, 5, 7}),
                     {1, g.tokens(), c.channels * c.patch[0] * c.patch[1] * c.patch[2]});
    return weights_.linear(x, "patch_embedding");
}

std::vector<Tensor> DiT::condition(const Tensor &timestep, const Tensor &text) const {
    const auto &c = weights_.config();
    require(timestep.shape() == mx::Shape{1} && text.ndim() == 3 && text.shape(0) == 1 &&
                text.shape(2) == c.text_dim && text.dtype() == mx::float16,
            "invalid Wan timestep or text conditioning shape/dtype");
    auto freq = mx::astype(components::wan_timestep_embedding(timestep, c.frequency_dim), mx::float16);
    auto time = weights_.linear(silu(weights_.linear(freq,
        "condition_embedder.time_embedder.linear_1")), "condition_embedder.time_embedder.linear_2");
    auto modulation = mx::reshape(weights_.linear(silu(time), "condition_embedder.time_proj"),
                                  {1, 6, c.hidden()});
    auto context = weights_.linear(gelu(weights_.linear(text,
        "condition_embedder.text_embedder.linear_1")), "condition_embedder.text_embedder.linear_2");
    return {time, modulation, context};
}

std::vector<Tensor> DiT::pre_ffn(const Tensor &input, const Tensor &context, const Tensor &modulation,
                                 const Tensor &cosine, const Tensor &sine, int index) const {
    const auto &c = weights_.config();
    require(index >= 0 && index < c.layers, "invalid Wan DiT block index");
    const auto p = "blocks." + std::to_string(index) + ".";
    auto project = [&](const Tensor &x, const std::string &name) { return weights_.linear(x, p + name); };
    auto shifts = mx::split(weights_.at(p + "scale_shift_table") + mx::astype(modulation, mx::float32), 6, 1);
    auto normalized = mx::astype(normalize(input) * (1.f + shifts[1]) + shifts[0], input.dtype());
    auto q = rotary(rms_normalize(project(normalized, "to_q"), p + "norm_q.weight"), cosine, sine);
    auto k = rotary(rms_normalize(project(normalized, "to_k"), p + "norm_k.weight"), cosine, sine);
    auto a = project(attention(q, k, project(normalized, "to_v")), "to_out");
    auto residual = input + a * shifts[2];
    normalized = normalize(residual) * weights_.at(p + "self_attn_residual_norm.norm.weight") +
                 weights_.at(p + "self_attn_residual_norm.norm.bias");
    normalized = mx::astype(normalized, input.dtype());
    residual = mx::astype(residual, input.dtype());
    q = rms_normalize(project(normalized, "attn2.to_q"), p + "attn2.norm_q.weight");
    if (context.shape(1) == 0) {
        a = mx::zeros_like(q);
    } else {
        k = rms_normalize(project(context, "attn2.to_k"), p + "attn2.norm_k.weight");
        a = attention(q, k, project(context, "attn2.to_v"));
    }
    residual = residual + project(a, "attn2.to_out");
    normalized = mx::astype(normalize(residual) * (1.f + shifts[4]) + shifts[3], input.dtype());
    residual = mx::astype(residual, input.dtype());
    return {normalized, residual, shifts[5]};
}

Tensor DiT::block(const Tensor &input, const Tensor &context, const Tensor &modulation,
                   const Tensor &cosine, const Tensor &sine, int index) const {
    auto state = pre_ffn(input, context, modulation, cosine, sine, index);
    const auto p = "blocks." + std::to_string(index) + ".ffn.";
    auto ff = weights_.linear(gelu(weights_.linear(state[0], p + "fc_in")), p + "fc_out");
    return mx::astype(state[1] + ff * state[2], input.dtype());
}

Tensor DiT::output(const Tensor &x, const Tensor &time, const mx::Shape &latent_shape) const {
    const auto &c = weights_.config();
    require(latent_shape.size() == 5 && latent_shape[0] == 1 && latent_shape[1] == c.channels,
            "invalid Wan output latent shape");
    auto g = components::wan_patch_grid({latent_shape[2], latent_shape[3], latent_shape[4]},
                                        {c.patch[0], c.patch[1], c.patch[2]});
    auto shifts = mx::split(weights_.at("scale_shift_table") + mx::expand_dims(time, 1), 2, 1);
    auto y = mx::astype(normalize(x) * (Tensor(1.f, shifts[1].dtype()) + shifts[1]) + shifts[0], mx::float16);
    y = weights_.linear(y, "proj_out");
    y = mx::reshape(y, {1, g.frames, g.height, g.width, c.patch[0], c.patch[1], c.patch[2], c.channels});
    return mx::reshape(mx::transpose(y, {0, 7, 1, 4, 2, 5, 3, 6}), latent_shape);
}

Tensor DiT::forward(const Tensor &latent, const Tensor &text, const Tensor &timestep,
                    const Tensor &cosine, const Tensor &sine,
                    const Event &event, std::atomic<bool> &cancelled) const {
    checkpoint(cancelled);
    auto x = patch_embed(latent);
    require(cosine.shape() == mx::Shape{x.shape(1), weights_.config().head_dim} &&
                sine.shape() == cosine.shape() && cosine.dtype() == mx::float32 && sine.dtype() == mx::float32,
            "invalid Wan rotary embedding shape/dtype");
    auto conditioning = condition(timestep, text);
    for (int index = 0; index < weights_.config().layers; ++index) {
        checkpoint(cancelled);
        event("wan_dit_block", index, weights_.config().layers);
        x = block(x, conditioning[2], conditioning[1], cosine, sine, index);
        mx::eval(x);
    }
    auto y = output(x, conditioning[0], latent.shape());
    mx::eval(y);
    checkpoint(cancelled);
    event("wan_dit_block", weights_.config().layers, weights_.config().layers);
    return y;
}

Tensor DiT::forward_hybrid(const Tensor &latent, const Tensor &text, const Tensor &timestep,
                           const Tensor &cosine, const Tensor &sine, const SplitFFN &ffn,
                           const Event &event, std::atomic<bool> &cancelled) const {
    require(bool(ffn), "Wan hybrid FFN callback missing");
    checkpoint(cancelled);
    auto x = patch_embed(latent);
    require(cosine.shape() == mx::Shape{x.shape(1), weights_.config().head_dim} && sine.shape() == cosine.shape(),
            "invalid Wan hybrid rotary shape");
    auto c = condition(timestep, text);
    for (int index = 0; index < weights_.config().layers; ++index) {
        checkpoint(cancelled);
        event("wan_dit_hybrid", index, weights_.config().layers);
        auto state = pre_ffn(x, c[2], c[1], cosine, sine, index);
        x = ffn(index, state[0], state[1], state[2]);
        mx::eval(x); // Consume the borrowed Core ML output before reusing it.
    }
    x = output(x, c[0], latent.shape());
    mx::eval(x);
    checkpoint(cancelled);
    event("wan_dit_hybrid", weights_.config().layers, weights_.config().layers);
    return x;
}

Tensor DiT::forward_compiled(const Tensor &latent, const Tensor &text, const Tensor &timestep,
                             const Tensor &cosine, const Tensor &sine,
                             const Event &event, std::atomic<bool> &cancelled) {
    checkpoint(cancelled);
    std::vector<Tensor> inputs{latent, text, timestep, cosine, sine};
    std::vector<mx::Shape> signature;
    for (const auto &input : inputs) signature.push_back(input.shape());
    require(latent.dtype() == mx::float16 && text.dtype() == mx::float16 &&
                timestep.dtype() == mx::float32 && cosine.dtype() == mx::float32 && sine.dtype() == mx::float32,
            "invalid compiled Wan input dtypes");
    if (!compiled_ || signature != signature_) {
        // Retire the previous graph before tracing another shape. Captured
        // weight materializations must not accumulate across video sizes.
        compiled_ = {};
        signature_.clear();
        auto graph = [this](const std::vector<Tensor> &a) {
            auto x = patch_embed(a[0]);
            require(a[3].shape() == mx::Shape{x.shape(1), weights_.config().head_dim} &&
                        a[4].shape() == a[3].shape(), "invalid compiled Wan rotary shape");
            auto c = condition(a[2], a[1]);
            for (int index = 0; index < weights_.config().layers; ++index)
                x = block(x, c[2], c[1], a[3], a[4], index);
            return std::vector<Tensor>{output(x, c[0], a[0].shape())};
        };
        compiled_ = mx::compile(std::function<std::vector<Tensor>(const std::vector<Tensor> &)>(graph));
        signature_ = std::move(signature);
    }
    event("wan_dit_compiled", 0, 1);
    auto output = compiled_(inputs).at(0);
    mx::eval(output);
    checkpoint(cancelled);
    event("wan_dit_compiled", 1, 1);
    return output;
}

} // namespace tc::wan
