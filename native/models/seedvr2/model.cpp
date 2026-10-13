// Native MLX adaptation of ByteDance-Seed/SeedVR's 3B one-step transformer.
// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>

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
                                    int text_offset, const Tensor &frequencies, bool fp16_angles) {
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
    if (fp16_angles) angles = mx::astype(mx::astype(angles,mx::float16),mx::float32);
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
    linear_recipe_ = "mlx";
    compute_dtype_ = dtype;
    if (!convolution_filters) {
        if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_DIT_LINEAR_RECIPE")) linear_recipe_ = value;
        require(linear_recipe_ == "mlx" || linear_recipe_ == "fp32",
                "SeedVR2 DiT linear recipe must be mlx or fp32");
    }
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
    auto input = mx::astype(x,compute_dtype_);
    auto weight = mx::transpose(at(prefix + ".weight"));
    if (linear_recipe_ == "fp32" && compute_dtype_ != mx::float32) {
        auto value = mx::astype(input,mx::float32), matrix = mx::astype(weight,mx::float32);
        auto projected = has(prefix+".bias") ? mx::addmm(mx::astype(at(prefix+".bias"),mx::float32),value,matrix)
            : mx::matmul(value,matrix);
        return mx::astype(projected,compute_dtype_);
    }
    // Match a linear projection's fused bias epilogue: a separate BF16
    // matmul/add rounds twice and drifts noticeably after 32 residual blocks.
    if (has(prefix + ".bias")) return mx::addmm(at(prefix + ".bias"), input, weight);
    return mx::matmul(input, weight);
}
size_t TensorBank::bytes() const {
    size_t result = 0;
    for (const auto &[name, tensor] : tensors_) result += tensor.nbytes();
    return result;
}

void DiT::load(const std::filesystem::path &path, mx::Dtype dtype) {
    require(dtype == mx::bfloat16 || dtype == mx::float16 || dtype == mx::float32,
            "SeedVR2 DiT compute dtype must be BF16, FP16 or FP32");
    rms_recipe_ = "mlx";
    if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_DIT_RMS_RECIPE")) rms_recipe_ = value;
    require(rms_recipe_ == "mlx" || rms_recipe_ == "fp32_affine",
            "SeedVR2 DiT RMS recipe must be mlx or fp32_affine");
    time_recipe_ = dtype == mx::float16 ? "host_fp32" : "mlx";
    if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_DIT_TIME_RECIPE")) time_recipe_ = value;
    require(time_recipe_ == "mlx" || time_recipe_ == "fp32_gemv" || time_recipe_ == "host_fp32",
            "SeedVR2 DiT time recipe must be mlx, fp32_gemv or host_fp32");
    attention_recipe_ = dtype == mx::float16 ? "fp16" : "mlx";
    if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_DIT_ATTENTION_RECIPE")) attention_recipe_ = value;
    require(attention_recipe_ == "mlx" || attention_recipe_ == "fp32" || attention_recipe_ == "fp16",
            "SeedVR2 DiT attention recipe must be mlx, fp16 or fp32");
    rope_recipe_ = dtype == mx::float16 ? "fp16_angles" : "fp32_angles";
    if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_DIT_ROPE_RECIPE")) rope_recipe_ = value;
    require(rope_recipe_ == "fp32_angles" || rope_recipe_ == "fp16_angles",
            "SeedVR2 DiT RoPE recipe must be fp32_angles or fp16_angles");
    residual_recipe_ = dtype == mx::float16 ? "fp32" : "native";
    if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_DIT_RESIDUAL_RECIPE")) residual_recipe_ = value;
    require(residual_recipe_ == "native" || residual_recipe_ == "fp32",
            "SeedVR2 DiT residual recipe must be native or fp32");
    weights_.load(path, dtype);
    dtype_ = dtype;
    require(weights_.at("vid_in.proj.weight").shape() == mx::Shape{kWidth, 132} &&
            weights_.at("positive_conditioning").shape() == mx::Shape{58, 5120} &&
            weights_.at("blocks.31.mlp.all.proj_in_gate.weight").shape() == mx::Shape{6912, kWidth},
            "SeedVR2 requires the original Comfy-Org 3B checkpoint layout");
}
Tensor DiT::rms_norm(const Tensor &x, const std::optional<Tensor> &weight) const {
    if (residual_recipe_ == "fp32")
        return mx::fast::rms_norm(mx::astype(x,mx::float32),
            weight ? std::optional<Tensor>(mx::astype(*weight,mx::float32)) : std::nullopt,kEpsilon);
    if (rms_recipe_ == "fp32_affine" && weight && x.dtype() != mx::float32) {
        // MLX 0.32's half/BF16 affine kernel rounds normalized values before
        // multiplying weight. The pinned Torch oracle keeps that product in
        // FP32 and casts once. This diagnostic keeps the same source operands,
        // epsilon and FP32 statistics, but removes the extra affine rounding.
        return mx::astype(mx::fast::rms_norm(mx::astype(x, mx::float32),
            mx::astype(*weight, mx::float32), kEpsilon), x.dtype());
    }
    return mx::fast::rms_norm(x, weight, kEpsilon);
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
        auto [cs, sn] = rope_angles(shape, tokens, freq,rope_recipe_=="fp16_angles");
        result.groups.push_back({shape, int(same.size()),
            Tensor(indices.data(), {int(indices.size())}, mx::int32), cs, sn});
    }
    require(std::find(reverse.begin(), reverse.end(), -1) == reverse.end(), "incomplete SeedVR2 attention windows");
    result.reverse = Tensor(reverse.data(), {int(reverse.size())}, mx::int32);
    auto positions = mx::reshape(mx::arange(tokens, mx::float32), {tokens, 1});
    auto angles = positions * mx::reshape(freq, {1, 21});
    if (rope_recipe_ == "fp16_angles") angles = mx::astype(mx::astype(angles,mx::float16),mx::float32);
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
Tensor DiT::time_embedding(float timestep, const std::filesystem::path &dump) const {
    const auto time_dtype = residual_recipe_ == "fp32" ? mx::float32 : dtype_;
    auto frequencies = mx::exp(mx::arange(128, mx::float32) * (-std::log(10000.f) / 128.f));
    auto angles = frequencies * timestep;
    auto x = mx::reshape(mx::astype(mx::concatenate({mx::sin(angles), mx::cos(angles)}), time_dtype), {1, 256});
    if (time_recipe_ == "host_fp32") {
        std::array<float,256> raw;
        const float logarithm = std::log(10000.f);
        for (int index = 0; index < 128; ++index) {
            const float exponent = -logarithm*float(index)/128.f;
            const float phase = std::exp(exponent)*timestep;
            raw[index] = std::sin(phase);
            raw[index+128] = std::cos(phase);
        }
        x = mx::astype(Tensor(raw.data(),{1,256},mx::float32),time_dtype);
    }
    auto project = [&](const Tensor &value, const std::string &prefix) {
        if (time_recipe_ == "mlx") return weights_.linear(value,prefix);
        // Preserve BF16-rounded loaded coefficients and per-projection output
        // casts. Only accumulation/bias arithmetic avoids the GEMV fallback's
        // intermediate BF16 rounding; no parameters are substituted.
        return mx::astype(mx::addmm(mx::astype(weights_.at(prefix+".bias"),mx::float32),
            mx::astype(value,mx::float32),mx::transpose(mx::astype(weights_.at(prefix+".weight"),mx::float32))),time_dtype);
    };
    std::unordered_map<std::string, Tensor> trace;
    if (!dump.empty()) trace.emplace("raw", x);
    x = project(x, "emb_in.proj_in");
    if (!dump.empty()) trace.emplace("proj_in", x);
    x = swish(x);
    if (!dump.empty()) trace.emplace("act_in", x);
    x = project(x, "emb_in.proj_hid");
    if (!dump.empty()) trace.emplace("proj_hid", x);
    x = swish(x);
    if (!dump.empty()) trace.emplace("act_hid", x);
    x = project(x, "emb_in.proj_out");
    if (!dump.empty()) {
        trace.emplace("proj_out", x);
        dump_tensors(dump, "time-detail", trace);
    }
    return x;
}
Tensor DiT::modulate(const Tensor &input, const Tensor &time, int layer,
                     const std::string &prefix, bool output, bool learned, bool rms_interface_roundtrip) const {
    auto modulation = mx::reshape(time, {kWidth, 2, 3});
    auto at = [&](int column) {
        return mx::reshape(mx::slice(modulation, {0, layer, column}, {kWidth, layer + 1, column + 1}), {kWidth});
    };
    const auto tag = prefix.ends_with("vid_out_ada") ? "out" : (layer ? "mlp" : "attn");
    if (output) {
        auto gate = at(2);
        if (learned && weights_.has(prefix + "." + tag + "_gate")) gate = gate + weights_.at(prefix + "." + tag + "_gate");
        return residual_recipe_ == "fp32" ? mx::astype(input,mx::float32)*mx::astype(gate,mx::float32) : input*gate;
    }
    // AdaSingle has no implicit '+1': checkpoint scale already contains it.
    auto shift = at(0) + weights_.at(prefix + "." + tag + "_shift");
    auto scale = at(1) + weights_.at(prefix + "." + tag + "_scale");
    // Stock ComfyUI MPS routes AdaSingle through its generic RMS-AdaLN
    // interface. Preserve that interface's BF16 scale round-trip so matched
    // comparisons do not silently change the low-precision numerical recipe.
    // This remains scale*x+shift: there is no implicit model '+1'.
    if (rms_interface_roundtrip)
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
    auto embedded_video = weights_.linear(video,"vid_in.proj");
    auto embedded_text = weights_.linear(weights_.at("positive_conditioning"),"txt_in");
    if (residual_recipe_ == "fp32") {
        embedded_video = mx::astype(embedded_video,mx::float32);
        embedded_text = mx::astype(embedded_text,mx::float32);
    }
    return {embedded_video,embedded_text};
}
std::vector<Tensor> DiT::attention(const Tensor &video, const Tensor &text, int index, const Layout &layout,
                                 std::unordered_map<std::string,Tensor> *trace) const {
    const auto base = "blocks." + std::to_string(index) + ".attn.";
    const auto vid_branch = index < 10 ? "vid" : "all";
    const auto txt_branch = index < 10 ? "txt" : "all";
    auto vqkv = weights_.linear(video, base + "proj_qkv." + vid_branch);
    auto tqkv = mx::reshape(weights_.linear(text, base + "proj_qkv." + txt_branch), {text.shape(0), 3, kHeads, kHeadDim});
    if (trace) {
        trace->emplace("attn_vid_qkv",vqkv);
        trace->emplace("attn_txt_qkv",mx::reshape(tqkv,{text.shape(0),3*kWidth}));
    }
    auto split_qkv = [&](const Tensor &qkv, int axis) {
        std::vector<Tensor> result;
        for (int i = 0; i < 3; ++i) result.push_back(mx::squeeze(take_part(qkv, axis, i, i + 1), axis));
        return result;
    };
    auto t = split_qkv(tqkv, 1);
    t[0] = rms_norm(t[0], weights_.at(base + "norm_q." + txt_branch + ".weight"));
    t[1] = rms_norm(t[1], weights_.at(base + "norm_k." + txt_branch + ".weight"));
    if (trace) {
        trace->emplace("attn_txt_normq",t[0]); trace->emplace("attn_txt_normk",t[1]);
    }
    t[0] = rotary(t[0],layout.text_cosine,layout.text_sine);
    t[1] = rotary(t[1],layout.text_cosine,layout.text_sine);
    if (trace) {
        trace->emplace("attn_txt_ropeq",t[0]); trace->emplace("attn_txt_ropek",t[1]);
    }
    std::vector<Tensor> videos, texts;
    for (const auto &group : layout.groups) {
        const int n = group.shape[0] * group.shape[1] * group.shape[2];
        auto v = split_qkv(mx::reshape(mx::take(vqkv, group.indices, 0), {group.count, n, 3, kHeads, kHeadDim}), 2);
        v[0] = rms_norm(v[0],weights_.at(base+"norm_q."+vid_branch+".weight"));
        v[1] = rms_norm(v[1],weights_.at(base+"norm_k."+vid_branch+".weight"));
        if (trace) {
            trace->emplace("attn_vid_normq",mx::reshape(v[0],{-1,kHeads,kHeadDim}));
            trace->emplace("attn_vid_normk",mx::reshape(v[1],{-1,kHeads,kHeadDim}));
        }
        v[0] = rotary(v[0],group.cosine,group.sine);
        v[1] = rotary(v[1],group.cosine,group.sine);
        if (trace) {
            trace->emplace("attn_vid_ropeq",mx::reshape(v[0],{-1,kHeads,kHeadDim}));
            trace->emplace("attn_vid_ropek",mx::reshape(v[1],{-1,kHeads,kHeadDim}));
        }
        auto combine = [&](int which) {
            auto repeated = mx::broadcast_to(mx::expand_dims(t[which], 0), {group.count, text.shape(0), kHeads, kHeadDim});
            return mx::transpose(mx::concatenate({v[which], repeated}, 1), {0, 2, 1, 3});
        };
        auto queries = combine(0), keys = combine(1), values = combine(2);
        if (attention_recipe_ == "fp32") {
            queries = mx::astype(queries,mx::float32);
            keys = mx::astype(keys,mx::float32);
            values = mx::astype(values,mx::float32);
        }
        if (residual_recipe_ == "fp32") values = mx::astype(values,queries.dtype());
        if (attention_recipe_ == "fp16") {
            queries = mx::astype(queries,mx::float16);
            keys = mx::astype(keys,mx::float16);
            values = mx::astype(values,mx::float16);
        }
        auto out = mx::fast::scaled_dot_product_attention(queries, keys, values, 1.f / std::sqrt(float(kHeadDim)));
        out = mx::astype(out,video.dtype());
        out = mx::transpose(out, {0, 2, 1, 3});
        videos.push_back(mx::reshape(take_part(out, 1, 0, n), {-1, kWidth}));
        texts.push_back(mx::sum(mx::astype(mx::reshape(take_part(out, 1, n, n + text.shape(0)),
                                {group.count, text.shape(0), kWidth}), mx::float32), 0));
    }
    auto vid = mx::take(mx::concatenate(videos, 0), layout.reverse, 0);
    auto txt = texts[0];
    for (size_t i = 1; i < texts.size(); ++i) txt = txt + texts[i];
    txt = mx::astype(txt / float(layout.windows), text.dtype());
    if (trace) { trace->emplace("attn_vid_raw",vid); trace->emplace("attn_txt_raw",txt); }
    auto projected_video = weights_.linear(vid,base+"proj_out."+vid_branch);
    auto projected_text = weights_.linear(txt,base+"proj_out."+txt_branch);
    if (trace) { trace->emplace("attn_vid_out",projected_video); trace->emplace("attn_txt_out",projected_text); }
    return {projected_video,projected_text};
}
Tensor DiT::feed_forward(const Tensor &x, const std::string &prefix) const {
    if (residual_recipe_ == "fp32") {
        auto gate = mx::astype(weights_.linear(x,prefix+".proj_in_gate"),mx::float32);
        auto up = mx::astype(weights_.linear(x,prefix+".proj_in"),mx::float32);
        return weights_.linear(gate*mx::sigmoid(gate)*up,prefix+".proj_out");
    }
    return weights_.linear(swish(weights_.linear(x, prefix + ".proj_in_gate")) *
                            weights_.linear(x, prefix + ".proj_in"), prefix + ".proj_out");
}
std::vector<Tensor> DiT::block(const Tensor &video, const Tensor &text, const Tensor &time,
                              int index, int frames, int height, int width, const FFNRoute &ffn,
                              const std::filesystem::path &dump) {
    require(index >= 0 && index < kLayers, "invalid SeedVR2 DiT block");
    const auto base = "blocks." + std::to_string(index) + ".";
    const auto vb = index < 10 ? "vid" : "all", tb = index < 10 ? "txt" : "all";
    // Last attention norm covers both branches while AdaSingle covers video
    // only. The source therefore normalizes separately and applies plain
    // scale*x+shift, without the generic RMS-AdaLN scale-1/+1 round-trip.
    auto vn = modulate(video, time, 0, base + "ada." + vb, false,true,index!=31);
    // The final block has video-only modulation/MLP but still attends to text.
    auto tn = index == 31 ? rms_norm(text) : modulate(text, time, 0, base + "ada." + tb, false);
    const auto &windows = layout(frames,height,width,text.shape(0),index%2);
    const bool detailed = index == 0 && !dump.empty() && windows.groups.size()==1 && windows.groups[0].count==1;
    std::unordered_map<std::string,Tensor> trace;
    if (detailed) { trace.emplace("attn_vid_in",vn); trace.emplace("attn_txt_in",tn); }
    auto attn = attention(vn, tn, index, windows,detailed ? &trace : nullptr);
    auto v = video + modulate(attn[0], time, 0, base + "ada." + vb, true);
    auto t = text + (index == 31 ? attn[1] : modulate(attn[1], time, 0, base + "ada." + tb, true));
    vn = modulate(v, time, 1, base + "ada." + vb, false);
    if (detailed) trace.emplace("mlp_vid_in",vn);
    auto feed_input = mx::astype(vn,dtype_);
    auto feed = ffn ? ffn(index,feed_input) : feed_forward(feed_input,base+"mlp."+vb);
    require(feed.shape() == vn.shape() && feed.dtype() == dtype_, "SeedVR2 FFN route changed its output contract");
    if (detailed) trace.emplace("mlp_vid_out",feed);
    v = v + modulate(feed, time, 1, base + "ada." + vb, true);
    if (index != 31) {
        tn = modulate(t, time, 1, base + "ada." + tb, false);
        auto text_feed = feed_forward(tn,base+"mlp."+tb);
        if (detailed) { trace.emplace("mlp_txt_in",tn); trace.emplace("mlp_txt_out",text_feed); }
        t = t + modulate(text_feed, time, 1, base + "ada." + tb, true);
    } else {
        // vid_only MMModule leaves text unchanged in the MLP and modulation;
        // the source block still adds that unchanged value to its residual.
        t = t + t;
    }
    if (detailed) { trace.emplace("block_vid",v); trace.emplace("block_txt",t); dump_tensors(dump,"block0-detail",trace); }
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
    auto hidden = residual_recipe_ == "fp32"
        ? mx::astype(swish(mx::astype(gate,mx::float32))*mx::astype(up,mx::float32),input.dtype())
        : swish(gate)*up;
    return mx::matmul(hidden, mx::transpose(take_part(weights_.at(p + "proj_out.weight"), 1, first, last)));
}
Tensor DiT::output(const Tensor &video, const Tensor &time, const mx::Shape &shape) const {
    // Source cache aliases the output's layer-0 modulation to the preceding
    // attention modulation (interleaved D x 2 x 3), not D x 6 contiguous chunks.
    auto normalized = rms_norm(video, weights_.at("vid_out_norm.weight"));
    auto mod = mx::reshape(time, {kWidth, 2, 3});
    auto component = [&](int col) { return mx::reshape(mx::slice(mod, {0, 0, col}, {kWidth, 1, col + 1}), {kWidth}); };
    auto y = normalized * (component(1) + weights_.at("vid_out_ada.out_scale")) +
             (component(0) + weights_.at("vid_out_ada.out_shift"));
    if (residual_recipe_ == "fp32") {
        const auto prefix = "vid_out.proj";
        y = mx::addmm(mx::astype(weights_.at(std::string(prefix)+".bias"),mx::float32),mx::astype(y,mx::float32),
            mx::transpose(mx::astype(weights_.at(std::string(prefix)+".weight"),mx::float32)));
    } else y = weights_.linear(y, "vid_out.proj");
    const int t = shape.at(2), h = shape.at(3), w = shape.at(4);
    y = mx::reshape(y, {1, t, h / 2, w / 2, 2, 2, 16});
    return mx::reshape(mx::transpose(y, {0, 6, 1, 2, 4, 3, 5}), shape);
}
Tensor DiT::forward(const Tensor &latent, const Tensor &condition, float timestep,
                    const Event &event, std::atomic<bool> &cancelled, const std::filesystem::path &dump,
                    const FFNRoute &ffn) {
    checkpoint(cancelled);
    auto state = embed(latent, condition);
    auto time = time_embedding(timestep,dump);
    mx::eval({state[0], state[1], time});
    dump_tensors(dump, "embedding", {{"video", state[0]}, {"text", state[1]}, {"time", time}});
    for (int index = 0; index < kLayers; ++index) {
        checkpoint(cancelled);
        if (event) event("seedvr2_dit", index, kLayers);
        state = block(state[0], state[1], time, index, latent.shape(2), latent.shape(3) / 2, latent.shape(4) / 2, ffn,dump);
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
