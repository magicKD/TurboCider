#include "transformer.hpp"
#include <cmath>

namespace tc::qwen21 {
namespace {
Tensor rotate(const Tensor &x, const Tensor &cosine, const Tensor &sine) {
    auto shape = x.shape();
    auto paired_shape = shape;
    paired_shape.back() /= 2;
    paired_shape.push_back(2);
    auto pairs = mx::split(mx::reshape(mx::astype(x, mx::float32), paired_shape), 2, -1);
    auto a = mx::squeeze(pairs[0], -1), b = mx::squeeze(pairs[1], -1);
    auto c = mx::reshape(cosine, {1, 1, shape[2], shape[3] / 2});
    auto s = mx::reshape(sine, {1, 1, shape[2], shape[3] / 2});
    return mx::astype(mx::reshape(mx::stack({a * c - b * s, a * s + b * c}, -1), shape), x.dtype());
}
Tensor layer_norm(const Tensor &x, float eps) {
    return mx::fast::layer_norm(x, {}, {}, eps);
}
Tensor gelu(const Tensor &x) {
    static auto compiled = mx::compile([](const std::vector<Tensor> &args) {
        const auto &a = args[0];
        auto scalar = [&](float v) { return Tensor(v, a.dtype()); };
        return std::vector<Tensor>{scalar(0.5f) * a *
            (scalar(1.f) + mx::tanh(scalar(0.7978845608028654f) *
                                  (a + scalar(0.044715f) * mx::power(a, scalar(3.f)))))};
    }, true);
    return compiled({x})[0];
}
}

Transformer::Transformer(const Weights &weights, TransformerConfig config)
    : weights_(weights), config_(config) {
    require(config.layers > 0 && config.heads > 0 && config.head_dim > 0 &&
            config.channels > 0 && config.context_dim > 0, "invalid Qwen21 transformer dimensions");
    int sum = 0;
    for (int axis : config.rope_axes) {
        require(axis > 0 && axis % 2 == 0, "Qwen21 RoPE axes must be positive and even");
        sum += axis;
    }
    require(sum == config.head_dim, "Qwen21 RoPE axes must sum to head dimension");
}

void Transformer::reset() {
    prefix_.clear();
    cached_text_.reset();
    cached_references_.clear();
}

Tensor Transformer::embedding(float timestep, mx::Dtype dtype) const {
    // Reference constructs these constants in NumPy double then casts to MLX
    // float32. Small frequency errors can cross a BF16 rounding boundary.
    static const Tensor frequencies = [] {
        std::vector<float> values(128);
        for (int i = 0; i < 128; ++i)
            values[i] = float(std::exp(-std::log(10000.0) * double(i) / 128.0));
        return Tensor(values.data(), {128}, mx::float32);
    }();
    auto times = Tensor(std::vector<float>{timestep, 0.f}.data(), {2, 1}, mx::float32);
    auto angles = (times * 1000.f) * frequencies;
    auto input = mx::astype(mx::concatenate({mx::cos(angles), mx::sin(angles)}, -1), dtype);
    return linear(silu(linear(input, weights_, "time_text_embed.timestep_embedder.linear_1")),
                  weights_, "time_text_embed.timestep_embedder.linear_2");
}

void Transformer::geometry(int text_length, int height, int width, const std::vector<ReferenceGeometry> &references) {
    if (cosine_ && text_length == text_length_ && height == height_ && width == width_ && references == reference_geometry_) return;
    auto sequence = make_sequence_geometry(text_length, height, width, references);
    reset();
    prefill_blocks_.clear();
    decode_blocks_.clear();
    text_length_ = text_length;
    height_ = height;
    width_ = width;
    reference_geometry_ = references;
    sequence_ = std::move(sequence);
    int tokens = int(sequence_.positions.size());
    std::vector<float> cos_values, sin_values;
    cos_values.reserve(size_t(tokens) * config_.head_dim / 2);
    sin_values.reserve(cos_values.capacity());
    for (int token = 0; token < tokens; ++token) {
        const auto &position = sequence_.positions[token];
        for (int axis = 0; axis < 3; ++axis) {
            int dim = config_.rope_axes[axis];
            for (int i = 0; i < dim; i += 2) {
                float omega = 1.f / std::pow(10000.f, float(i) / dim);
                float angle = float(position[axis]) * omega;
                cos_values.push_back(std::cos(angle));
                sin_values.push_back(std::sin(angle));
            }
        }
    }
    cosine_ = Tensor(cos_values.data(), {tokens, config_.head_dim / 2}, mx::float32);
    sine_ = Tensor(sin_values.data(), {tokens, config_.head_dim / 2}, mx::float32);
}

Tensor Transformer::forward(const Tensor &latents, const Tensor &text, float timestep,
                            int height, int width, bool cache_prefix,
                            std::unordered_map<std::string, Tensor> *trace,
                            const std::vector<ReferenceLatents> &references) {
    require(height > 0 && width > 0 && latents.ndim() == 3 && latents.shape(0) == 1 &&
            latents.shape(1) == height * width && latents.shape(2) == config_.channels,
            "Qwen21 latent shape must be [1,H*W,channels]");
    require(text.ndim() == 3 && text.shape(0) == 1 && text.shape(1) > 0 &&
            text.shape(2) == config_.context_dim && text.dtype() == latents.dtype(),
            "Qwen21 text shape/dtype mismatch");
    require(std::isfinite(timestep) && timestep >= 0.f && timestep <= 1.f,
            "Qwen21 timestep must be a normalized sigma in [0,1]");
    std::vector<ReferenceGeometry> reference_geometry;
    bool same_references = references.size() == cached_references_.size();
    for (size_t i = 0; i < references.size(); ++i) {
        const auto &r = references[i];
        require(r.latents.ndim() == 3 && r.latents.shape(0) == 1 && r.latents.dtype() == latents.dtype() &&
                r.latents.shape(2) == config_.channels && r.latents.shape(1) == int64_t(r.geometry.height) * r.geometry.width,
                "Qwen21 reference latent shape/dtype mismatch");
        reference_geometry.push_back(r.geometry);
        if (i >= cached_references_.size() || r.latents.id() != cached_references_[i].id()) same_references = false;
    }
    geometry(text.shape(1), height, width, reference_geometry);
    // MLX arrays are immutable values: identity binds the cache without a GPU
    // synchronization/hash of a many-megabyte conditioning tensor each step.
    if (!cache_prefix || !cached_text_ || cached_text_->id() != text.id() || !same_references) reset();
    bool reuse = cache_prefix && prefix_.size() == size_t(config_.layers);
    const bool split_mlp = reuse && bool(decode_mlp_);
    require(!split_mlp || !trace, "Qwen21 MLP split trace is not supported");
    auto temb = embedding(timestep, latents.dtype());
    auto modulation = linear(silu(temb), weights_, "modulation.1");
    if (trace) { trace->emplace("temb", slice_axis(temb, 0, 0, 1)); trace->emplace("modulation", slice_axis(modulation, 0, 0, 1)); }
    auto mod_target = mx::split(mx::reshape(slice_axis(modulation, 0, 0, 1), {1, 1, 4 * config_.hidden()}), 4, -1);
    auto mods = mod_target;
    Tensor hidden = linear(latents, weights_, "img_in");
    int prefix_length = sequence_.prefix_length;
    if (!reuse) {
        auto f = mx::astype(text, mx::float32);
        auto scale = mx::astype(weights_.at("txt_in.text_norm.weight"), mx::float32) + 1.f;
        auto normalized = mx::astype(f * mx::rsqrt(mx::mean(f * f, -1, true) + config_.epsilon) * scale, text.dtype());
        auto context = linear(gelu(linear(normalized, weights_, "txt_in.in_layer")), weights_, "txt_in.out_layer");
        std::vector<Tensor> parts;
        for (const auto &segment : sequence_.segments) {
            if (segment.causal)
                parts.push_back(slice_axis(context, 1, segment.text_start, segment.text_start + segment.end - segment.start));
            else if (segment.image_index < int(references.size()))
                parts.push_back(linear(references[segment.image_index].latents, weights_, "img_in"));
            else parts.push_back(hidden);
        }
        hidden = mx::concatenate(parts, 1);
        auto mod_prefix = mx::split(mx::reshape(slice_axis(modulation, 0, 1, 2), {1, 1, 4 * config_.hidden()}), 4, -1);
        for (int i = 0; i < 4; ++i)
            mods[i] = mx::concatenate({mx::broadcast_to(mod_prefix[i], {1, prefix_length, config_.hidden()}),
                                      mx::broadcast_to(mod_target[i], {1, height * width, config_.hidden()})}, 1);
    }
    auto cosine = reuse ? slice_axis(*cosine_, 0, prefix_length, cosine_->shape(0)) : *cosine_;
    auto sine = reuse ? slice_axis(*sine_, 0, prefix_length, sine_->shape(0)) : *sine_;
    if (trace) {
        trace->emplace("input", hidden);
        trace->emplace("cosine", cosine);
        trace->emplace("sine", sine);
        for (int i = 0; i < 4; ++i) trace->emplace("mod" + std::to_string(i), mods[i]);
    }
    std::vector<KV> new_prefix;
    if (trace) { prefill_blocks_.clear(); decode_blocks_.clear(); }
    // Decode modulation is identical across blocks. Materialize its gate once
    // per forward, and fuse the split-path residual's elementwise operations.
    // The full-GPU block retains its existing compiled arithmetic.
    std::optional<Tensor> split_gate;
    if (split_mlp) split_gate = mx::tanh(mods[3]);
    static auto split_residual = mx::compile([](const std::vector<Tensor> &a) {
        return std::vector<Tensor>{a[0] + a[1] * a[2]};
    });
    for (int i = 0; i < config_.layers; ++i) {
        auto &functions = reuse ? decode_blocks_ : prefill_blocks_;
        if (functions.size() <= size_t(i)) {
            functions.push_back(mx::compile([this, i, reuse, prefix_length, split_mlp, tracing = trace != nullptr](const std::vector<Tensor> &args) {
                auto hidden = args[0];
                std::vector<Tensor> mods(args.begin() + 1, args.begin() + 5);
                const auto &cosine = args[5], &sine = args[6];
                std::string p = "transformer_blocks." + std::to_string(i);
                auto input = layer_norm(hidden, config_.epsilon) * (Tensor(1.f, hidden.dtype()) + mods[0]);
                auto attention_input = input;
                auto q = heads(linear(input, weights_, p + ".attn.to_q"), config_.heads, config_.head_dim);
                auto k = heads(linear(input, weights_, p + ".attn.to_k"), config_.heads, config_.head_dim);
                auto v = heads(linear(input, weights_, p + ".attn.to_v"), config_.heads, config_.head_dim);
                q = rotate(mx::fast::rms_norm(q, weights_.at(p + ".attn.norm_q.weight"), config_.epsilon), cosine, sine);
                k = rotate(mx::fast::rms_norm(k, weights_.at(p + ".attn.norm_k.weight"), config_.epsilon), cosine, sine);
                Tensor output = hidden;
                auto pk = k, pv = v;
                if (reuse) {
                    k = mx::concatenate({args[7], k}, 2);
                    v = mx::concatenate({args[8], v}, 2);
                    output = attend(q, k, v);
                } else {
                    pk = slice_axis(k, 2, 0, prefix_length);
                    pv = slice_axis(v, 2, 0, prefix_length);
                    std::vector<Tensor> segments;
                    for (const auto &segment : sequence_.segments)
                        segments.push_back(attend(slice_axis(q, 2, segment.start, segment.end),
                            slice_axis(k, 2, 0, segment.end), slice_axis(v, 2, 0, segment.end),
                            false, {}, false, segment.causal ? "causal" : ""));
                    output = mx::concatenate(segments, 1);
                }
                auto projected = linear(output, weights_, p + ".attn.to_out.0");
                hidden = hidden + mx::tanh(mods[1]) * projected;
                auto after_attention = hidden;
                input = layer_norm(hidden, config_.epsilon) * (Tensor(1.f, hidden.dtype()) + mods[2]);
                if (split_mlp) return std::vector<Tensor>{hidden, input};
                auto ff = input;
                if (weights_.has(p + ".img_mlp.gate_up.weight")) {
                    auto gate_up = mx::split(linear(input, weights_, p + ".img_mlp.gate_up"), 2, -1);
                    ff = silu(gate_up[0]) * gate_up[1];
                } else {
                    ff = silu(linear(input, weights_, p + ".img_mlp.gate_layer")) * linear(input, weights_, p + ".img_mlp.proj");
                }
                hidden = hidden + mx::tanh(mods[3]) * linear(ff, weights_, p + ".img_mlp.out");
                if (tracing) return std::vector<Tensor>{hidden, pk, pv, attention_input, q, k, v, output, projected, after_attention, input, ff};
                return reuse ? std::vector<Tensor>{hidden} : std::vector<Tensor>{hidden, pk, pv};
            }));
        }
        std::vector<Tensor> args{hidden, mods[0], mods[1], mods[2], mods[3], cosine, sine};
        if (reuse) {
            args.push_back(prefix_[i].key);
            args.push_back(prefix_[i].value);
        }
        auto outputs = functions[i](args);
        hidden = outputs[0];
        if (split_mlp) {
            auto feed = decode_mlp_(i, outputs[1]);
            hidden = split_residual({hidden, *split_gate, feed})[0];
            mx::eval(hidden); // consume shared Core ML output before next block
        }
        if (trace) trace->emplace("block" + std::to_string(i), hidden);
        if (trace && i == 0) {
            const char *names[] = {"attention_input", "q", "k", "v", "attention", "projected", "after_attention", "mlp_input", "ff"};
            for (int j = 0; j < 9; ++j) trace->emplace(names[j], outputs[3 + j]);
        }
        if (!reuse && cache_prefix) new_prefix.push_back({outputs[1], outputs[2]});
    }
    if (!reuse) hidden = slice_axis(hidden, 1, prefix_length, hidden.shape(1));
    auto out_scale = mx::reshape(slice_axis(linear(silu(temb), weights_, "norm_out.linear"), 0, 0, 1), {1, 1, config_.hidden()});
    auto result = linear(layer_norm(hidden, config_.epsilon) * (Tensor(1.f, hidden.dtype()) + out_scale), weights_, "proj_out");
    if (cache_prefix && !reuse) {
        prefix_ = std::move(new_prefix);
        cached_text_ = text;
        for (const auto &r : references) cached_references_.push_back(r.latents);
    }
    if (trace) { prefill_blocks_.clear(); decode_blocks_.clear(); }
    return result;
}
} // namespace tc::qwen21
