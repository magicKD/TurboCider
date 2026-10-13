// Native MLX adaptation of ByteDance-Seed/SeedVR's 3B one-step transformer.
// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tc::seedvr2 {
namespace {
constexpr int kWidth = 2560, kHeads = 20, kHeadDim = 128, kLayers = 32;
constexpr float kEpsilon = 1e-5f;

Tensor take_part(const Tensor &x, int axis, int start, int end) {
    mx::Shape lo(x.ndim(), 0), hi = x.shape();
    lo.at(axis) = start;
    hi.at(axis) = end;
    return mx::slice(x, lo, hi);
}
Tensor swish(const Tensor &x) {
    auto f = mx::astype(x, mx::float32);
    return mx::astype(f * mx::sigmoid(f), x.dtype());
}
Tensor rms_norm(const Tensor &x, const std::optional<Tensor> &weight = {}) {
    return mx::fast::rms_norm(x, weight, kEpsilon);
}
Tensor rotary(const Tensor &x, const Tensor &cosine, const Tensor &sine) {
    // The three axes rotate 3 * 42 leading head features. The final two
    // features are deliberately untouched, as in the source 3B checkpoint.
    auto rotated = take_part(x, x.ndim() - 1, 0, 126);
    auto shape = rotated.shape();
    shape.back() = 63;
    shape.push_back(2);
    auto pairs = mx::reshape(mx::astype(rotated, mx::float32), shape);
    auto even = take_part(pairs, pairs.ndim() - 1, 0, 1);
    auto odd = take_part(pairs, pairs.ndim() - 1, 1, 2);
    auto cs = mx::expand_dims(mx::expand_dims(cosine, -2), -1);
    auto sn = mx::expand_dims(mx::expand_dims(sine, -2), -1);
    auto y = mx::concatenate({even * cs - odd * sn, odd * cs + even * sn}, -1);
    return mx::concatenate({mx::astype(mx::reshape(y, rotated.shape()), x.dtype()),
                            take_part(x, x.ndim() - 1, 126, 128)}, -1);
}
std::pair<Tensor, Tensor> rope_angles(const std::array<int, 3> &shape,
                                    int text_offset, const Tensor &frequencies) {
    std::vector<float> positions;
    positions.reserve(size_t(shape[0]) * shape[1] * shape[2] * 3);
    for (int it = 0; it < shape[0]; ++it)
        for (int ih = 0; ih < shape[1]; ++ih)
            for (int iw = 0; iw < shape[2]; ++iw) {
                positions.push_back(float(it + text_offset));
                positions.push_back(float(ih));
                positions.push_back(float(iw));
            }
    auto pos = Tensor(positions.data(), {int(positions.size() / 3), 3, 1}, mx::float32);
    auto angles = mx::reshape(pos * mx::reshape(mx::astype(frequencies, mx::float32), {1, 1, 21}), {-1, 63});
    return {mx::cos(angles), mx::sin(angles)};
}
void dump_tensors(const std::filesystem::path &root, const std::string &name,
                  const std::unordered_map<std::string, Tensor> &tensors) {
    if (root.empty()) return;
    std::filesystem::create_directories(root);
    mx::save_safetensors((root / (name + ".safetensors")).string(), tensors);
}
} // namespace

void TensorBank::load(const std::filesystem::path &path, mx::Dtype dtype, bool convolution_filters) {
    require(tensors_.empty(), "SeedVR2 bank must be cleared before loading");
    auto loaded = mx::load_safetensors(path.string()).first;
    for (auto &[name, value] : loaded) {
        // Keep the checkpoint's original RoPE frequency precision. Projection
        // operands follow the requested compute dtype; weights are not quantized.
        if (name.ends_with(".rope.freqs")) value = mx::astype(value, mx::float32);
        else value = mx::astype(value, dtype);
        if (convolution_filters && value.ndim() == 5)
            value = mx::contiguous(mx::transpose(value, {0, 2, 3, 4, 1}));
        mx::eval(value);
        tensors_.emplace(name, std::move(value));
    }
}
const Tensor &TensorBank::at(const std::string &key) const {
    auto found = tensors_.find(key);
    require(found != tensors_.end(), "missing SeedVR2 tensor: " + key);
    return found->second;
}
bool TensorBank::has(const std::string &key) const { return tensors_.count(key); }
Tensor TensorBank::linear(const Tensor &x, const std::string &prefix) const {
    auto weight = mx::transpose(at(prefix + ".weight"));
    // Match a linear projection's fused bias epilogue: a separate BF16
    // matmul/add rounds twice and drifts noticeably after 32 residual blocks.
    if (has(prefix + ".bias")) return mx::addmm(at(prefix + ".bias"), x, weight);
    return mx::matmul(x, weight);
}
size_t TensorBank::bytes() const {
    size_t result = 0;
    for (const auto &[name, tensor] : tensors_) result += tensor.nbytes();
    return result;
}

void DiT::load(const std::filesystem::path &path, mx::Dtype dtype) {
    require(dtype == mx::bfloat16 || dtype == mx::float16 || dtype == mx::float32,
            "SeedVR2 DiT compute dtype must be BF16, FP16 or FP32");
    weights_.load(path, dtype);
    dtype_ = dtype;
    require(weights_.at("vid_in.proj.weight").shape() == mx::Shape{kWidth, 132} &&
            weights_.at("positive_conditioning").shape() == mx::Shape{58, 5120} &&
            weights_.at("blocks.31.mlp.all.proj_in_gate.weight").shape() == mx::Shape{6912, kWidth},
            "SeedVR2 requires the original Comfy-Org 3B checkpoint layout");
}
DiT::Layout DiT::make_layout(int frames, int height, int width, int tokens, bool shifted) const {
    auto windows = make_windows(frames, height, width, shifted);
    std::map<std::array<int, 3>, std::vector<Window>> groups;
    for (const auto &window : windows) groups[window.shape()].push_back(window);
    const auto &freq = weights_.at("blocks.0.attn.rope.rope.freqs");
    std::vector<int32_t> reverse(size_t(frames) * height * width, -1);
    Layout result{{}, Tensor(0), Tensor(0), Tensor(0), int(windows.size())};
    int offset = 0;
    for (const auto &[shape, same] : groups) {
        std::vector<int32_t> indices;
        for (const auto &window : same) {
            auto part = window_indices(window, height, width);
            indices.insert(indices.end(), part.begin(), part.end());
        }
        for (auto index : indices) {
            require(reverse.at(index) == -1, "overlapping SeedVR2 attention windows");
            reverse[index] = offset++;
        }
        auto [cs, sn] = rope_angles(shape, tokens, freq);
        result.groups.push_back({shape, int(same.size()),
            Tensor(indices.data(), {int(indices.size())}, mx::int32), cs, sn});
    }
    require(std::find(reverse.begin(), reverse.end(), -1) == reverse.end(), "incomplete SeedVR2 attention windows");
    result.reverse = Tensor(reverse.data(), {int(reverse.size())}, mx::int32);
    auto positions = mx::reshape(mx::arange(tokens, mx::float32), {tokens, 1});
    auto angles = positions * mx::reshape(freq, {1, 21});
    angles = mx::concatenate({angles, angles, angles}, -1);
    result.text_cosine = mx::cos(angles);
    result.text_sine = mx::sin(angles);
    std::vector<Tensor> arrays{result.reverse, result.text_cosine, result.text_sine};
    for (const auto &group : result.groups)
        arrays.insert(arrays.end(), {group.indices, group.cosine, group.sine});
    mx::eval(arrays);
    return result;
}
const DiT::Layout &DiT::layout(int frames, int height, int width, int tokens, bool shifted) {
    const std::array<int, 5> key{frames, height, width, tokens, shifted};
    auto found = layouts_.find(key);
    if (found == layouts_.end()) found = layouts_.emplace(key, make_layout(frames, height, width, tokens, shifted)).first;
    return found->second;
}
Tensor DiT::time_embedding(float timestep) const {
    auto frequencies = mx::exp(mx::arange(128, mx::float32) * (-std::log(10000.f) / 128.f));
    auto angles = frequencies * timestep;
    auto x = mx::reshape(mx::astype(mx::concatenate({mx::sin(angles), mx::cos(angles)}), dtype_), {1, 256});
    x = swish(weights_.linear(x, "emb_in.proj_in"));
    x = swish(weights_.linear(x, "emb_in.proj_hid"));
    return weights_.linear(x, "emb_in.proj_out");
}
Tensor DiT::modulate(const Tensor &input, const Tensor &time, int layer,
                     const std::string &prefix, bool output, bool learned) const {
    auto modulation = mx::reshape(time, {kWidth, 2, 3});
    auto at = [&](int column) {
        return mx::reshape(mx::slice(modulation, {0, layer, column}, {kWidth, layer + 1, column + 1}), {kWidth});
    };
    const auto tag = prefix.ends_with("vid_out_ada") ? "out" : (layer ? "mlp" : "attn");
    if (output) {
        auto gate = at(2);
        if (learned && weights_.has(prefix + "." + tag + "_gate")) gate = gate + weights_.at(prefix + "." + tag + "_gate");
        return input * gate;
    }
    // AdaSingle has no implicit '+1': checkpoint scale already contains it.
    auto shift = at(0) + weights_.at(prefix + "." + tag + "_shift");
    auto scale = at(1) + weights_.at(prefix + "." + tag + "_scale");
    // Stock ComfyUI MPS routes AdaSingle through its generic RMS-AdaLN
    // interface. Preserve that interface's BF16 scale round-trip so matched
    // comparisons do not silently change the low-precision numerical recipe.
    // This remains scale*x+shift: there is no implicit model '+1'.
    scale = Tensor(1.f, scale.dtype()) + (scale - Tensor(1.f, scale.dtype()));
    return rms_norm(input) * scale + shift;
}
std::vector<Tensor> DiT::embed(const Tensor &latent, const Tensor &condition) const {
    require(latent.ndim() == 5 && latent.shape(0) == 1 && latent.shape(1) == 16 &&
            condition.shape() == latent.shape() && latent.shape(3) % 2 == 0 && latent.shape(4) % 2 == 0,
            "SeedVR2 expects matching batch-one BCTHW latents, 16 channels and even spatial dimensions");
    const int t = latent.shape(2), h = latent.shape(3), w = latent.shape(4);
    auto x = mx::transpose(mx::astype(latent, dtype_), {0, 2, 3, 4, 1});
    auto c = mx::transpose(mx::astype(condition, dtype_), {0, 2, 3, 4, 1});
    auto mask = mx::ones({1, t, h, w, 1}, dtype_);
    auto video = mx::concatenate({x, c, mask}, -1);
    video = mx::reshape(video, {t, h / 2, 2, w / 2, 2, 33});
    video = mx::reshape(mx::transpose(video, {0, 1, 3, 2, 4, 5}), {-1, 132});
    return {weights_.linear(video, "vid_in.proj"), weights_.linear(weights_.at("positive_conditioning"), "txt_in")};
}
std::vector<Tensor> DiT::attention(const Tensor &video, const Tensor &text, int index, const Layout &layout) const {
    const auto base = "blocks." + std::to_string(index) + ".attn.";
    const auto vid_branch = index < 10 ? "vid" : "all";
    const auto txt_branch = index < 10 ? "txt" : "all";
    auto vqkv = weights_.linear(video, base + "proj_qkv." + vid_branch);
    auto tqkv = mx::reshape(weights_.linear(text, base + "proj_qkv." + txt_branch), {text.shape(0), 3, kHeads, kHeadDim});
    auto split_qkv = [&](const Tensor &qkv, int axis) {
        std::vector<Tensor> result;
        for (int i = 0; i < 3; ++i) result.push_back(mx::squeeze(take_part(qkv, axis, i, i + 1), axis));
        return result;
    };
    auto t = split_qkv(tqkv, 1);
    t[0] = rotary(rms_norm(t[0], weights_.at(base + "norm_q." + txt_branch + ".weight")), layout.text_cosine, layout.text_sine);
    t[1] = rotary(rms_norm(t[1], weights_.at(base + "norm_k." + txt_branch + ".weight")), layout.text_cosine, layout.text_sine);
    std::vector<Tensor> videos, texts;
    for (const auto &group : layout.groups) {
        const int n = group.shape[0] * group.shape[1] * group.shape[2];
        auto v = split_qkv(mx::reshape(mx::take(vqkv, group.indices, 0), {group.count, n, 3, kHeads, kHeadDim}), 2);
        v[0] = rotary(rms_norm(v[0], weights_.at(base + "norm_q." + vid_branch + ".weight")), group.cosine, group.sine);
        v[1] = rotary(rms_norm(v[1], weights_.at(base + "norm_k." + vid_branch + ".weight")), group.cosine, group.sine);
        auto combine = [&](int which) {
            auto repeated = mx::broadcast_to(mx::expand_dims(t[which], 0), {group.count, text.shape(0), kHeads, kHeadDim});
            return mx::transpose(mx::concatenate({v[which], repeated}, 1), {0, 2, 1, 3});
        };
        auto out = mx::fast::scaled_dot_product_attention(combine(0), combine(1), combine(2), 1.f / std::sqrt(float(kHeadDim)));
        out = mx::transpose(out, {0, 2, 1, 3});
        videos.push_back(mx::reshape(take_part(out, 1, 0, n), {-1, kWidth}));
        texts.push_back(mx::sum(mx::astype(mx::reshape(take_part(out, 1, n, n + text.shape(0)),
                                {group.count, text.shape(0), kWidth}), mx::float32), 0));
    }
    auto vid = mx::take(mx::concatenate(videos, 0), layout.reverse, 0);
    auto txt = texts[0];
    for (size_t i = 1; i < texts.size(); ++i) txt = txt + texts[i];
    txt = mx::astype(txt / float(layout.windows), text.dtype());
    return {weights_.linear(vid, base + "proj_out." + vid_branch), weights_.linear(txt, base + "proj_out." + txt_branch)};
}
Tensor DiT::feed_forward(const Tensor &x, const std::string &prefix) const {
    return weights_.linear(swish(weights_.linear(x, prefix + ".proj_in_gate")) *
                            weights_.linear(x, prefix + ".proj_in"), prefix + ".proj_out");
}
std::vector<Tensor> DiT::block(const Tensor &video, const Tensor &text, const Tensor &time,
                              int index, int frames, int height, int width, const FFNRoute &ffn) {
    require(index >= 0 && index < kLayers, "invalid SeedVR2 DiT block");
    const auto base = "blocks." + std::to_string(index) + ".";
    const auto vb = index < 10 ? "vid" : "all", tb = index < 10 ? "txt" : "all";
    auto vn = modulate(video, time, 0, base + "ada." + vb, false);
    // The final block has video-only modulation/MLP but still attends to text.
    auto tn = index == 31 ? rms_norm(text) : modulate(text, time, 0, base + "ada." + tb, false);
    auto attn = attention(vn, tn, index, layout(frames, height, width, text.shape(0), index % 2));
    auto v = video + modulate(attn[0], time, 0, base + "ada." + vb, true);
    auto t = text + (index == 31 ? attn[1] : modulate(attn[1], time, 0, base + "ada." + tb, true));
    vn = modulate(v, time, 1, base + "ada." + vb, false);
    auto feed = ffn ? ffn(index, vn) : feed_forward(vn, base + "mlp." + vb);
    require(feed.shape() == vn.shape() && feed.dtype() == vn.dtype(), "SeedVR2 FFN route changed its output contract");
    v = v + modulate(feed, time, 1, base + "ada." + vb, true);
    if (index != 31) {
        tn = modulate(t, time, 1, base + "ada." + tb, false);
        t = t + modulate(feed_forward(tn, base + "mlp." + tb), time, 1, base + "ada." + tb, true);
    } else {
        // vid_only MMModule leaves text unchanged in the MLP and modulation;
        // the source block still adds that unchanged value to its residual.
        t = t + t;
    }
    return {v, t};
}
Tensor DiT::ffn_full(const Tensor &input, int index) const {
    require(index >= 0 && index < kLayers, "invalid SeedVR2 FFN block");
    const auto prefix = "blocks." + std::to_string(index) + ".mlp." + (index < 10 ? "vid" : "all");
    return feed_forward(input, prefix);
}
Tensor DiT::ffn_partial(const Tensor &input, int index, int first, int last) const {
    require(index >= 0 && index < kLayers && first >= 0 && last > first && last <= 6912,
            "invalid SeedVR2 FFN channel slice");
    const auto p = "blocks." + std::to_string(index) + ".mlp." + (index < 10 ? "vid" : "all") + ".";
    auto gate = mx::matmul(input, mx::transpose(take_part(weights_.at(p + "proj_in_gate.weight"), 0, first, last)));
    auto up = mx::matmul(input, mx::transpose(take_part(weights_.at(p + "proj_in.weight"), 0, first, last)));
    auto hidden = swish(gate) * up;
    return mx::matmul(hidden, mx::transpose(take_part(weights_.at(p + "proj_out.weight"), 1, first, last)));
}
Tensor DiT::output(const Tensor &video, const Tensor &time, const mx::Shape &shape) const {
    // Source cache aliases the output's layer-0 modulation to the preceding
    // attention modulation (interleaved D x 2 x 3), not D x 6 contiguous chunks.
    auto normalized = rms_norm(video, weights_.at("vid_out_norm.weight"));
    auto mod = mx::reshape(time, {kWidth, 2, 3});
    auto component = [&](int col) { return mx::reshape(mx::slice(mod, {0, 0, col}, {kWidth, 1, col + 1}), {kWidth}); };
    auto y = normalized * (component(1) + weights_.at("vid_out_ada.out_scale")) +
             component(0) + weights_.at("vid_out_ada.out_shift");
    y = weights_.linear(y, "vid_out.proj");
    const int t = shape.at(2), h = shape.at(3), w = shape.at(4);
    y = mx::reshape(y, {1, t, h / 2, w / 2, 2, 2, 16});
    return mx::reshape(mx::transpose(y, {0, 6, 1, 2, 4, 3, 5}), shape);
}
Tensor DiT::forward(const Tensor &latent, const Tensor &condition, float timestep,
                    const Event &event, std::atomic<bool> &cancelled, const std::filesystem::path &dump,
                    const FFNRoute &ffn) {
    checkpoint(cancelled);
    auto state = embed(latent, condition);
    auto time = time_embedding(timestep);
    mx::eval({state[0], state[1], time});
    dump_tensors(dump, "embedding", {{"video", state[0]}, {"text", state[1]}, {"time", time}});
    for (int index = 0; index < kLayers; ++index) {
        checkpoint(cancelled);
        if (event) event("seedvr2_dit", index, kLayers);
        state = block(state[0], state[1], time, index, latent.shape(2), latent.shape(3) / 2, latent.shape(4) / 2, ffn);
        mx::eval(state);
        if (!dump.empty() && (index == 0 || index == 1 || index == 9 || index == 10 || index == 31))
            dump_tensors(dump, "block-" + std::to_string(index), {{"video", state[0]}, {"text", state[1]}});
    }
    auto result = output(state[0], time, latent.shape());
    mx::eval(result);
    dump_tensors(dump, "velocity", {{"velocity", result}});
    checkpoint(cancelled);
    if (event) event("seedvr2_dit", kLayers, kLayers);
    return result;
}
} // namespace tc::seedvr2
