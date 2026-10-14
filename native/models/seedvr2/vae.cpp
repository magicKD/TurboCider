// Native MLX adaptation of ByteDance-Seed/SeedVR's causal s8/c16/t4 VAE.
// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "vae.hpp"
#include "metal/conv_tiled.hpp"
#include "metal/group_norm.hpp"
#include "metal/causal_history.hpp"
#include <charconv>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cstdio>

namespace tc::seedvr2 {
namespace {
Tensor part(const Tensor &x, int axis, int first, int last) {
    mx::Shape start(x.ndim(), 0), stop = x.shape();
    start.at(axis) = first;
    stop.at(axis) = last;
    return mx::slice(x, start, stop);
}
Tensor swish(const Tensor &x) { return x * mx::sigmoid(x); }
void fence(const Tensor &x, std::atomic<bool> &cancelled) {
    mx::eval(x);
    checkpoint(cancelled);
}
} // namespace

void VAE::load(const std::filesystem::path &path) {
    configure({});
    weights_.load(path, mx::float16, true);
    require(weights_.at("encoder.conv_in.weight").shape() == mx::Shape{128, 3, 3, 3, 3} &&
            weights_.at("decoder.conv_in.weight").shape() == mx::Shape{512, 3, 3, 3, 16},
            "SeedVR2 requires the original s8/c16/t4 VAE checkpoint");
}
void VAE::configure(const std::optional<SeedVR2VAEConfig> &config) {
    conv_recipe_ = "singleton2d";
    norm_recipe_ = "metal";
    history_recipe_ = "compact";
    pointwise_recipe_ = "matmul";
    conv_tile_rows_ = 16384;
    if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_VAE_CONV_RECIPE")) conv_recipe_ = value;
    require(conv_recipe_ == "conv3d" || conv_recipe_ == "taps2d" || conv_recipe_ == "singleton2d" ||
            conv_recipe_ == "tiled3d" || conv_recipe_ == "mixed3d",
            "SeedVR2 VAE conv recipe must be conv3d, taps2d, singleton2d, tiled3d or mixed3d");
    if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_VAE_NORM_RECIPE")) norm_recipe_ = value;
    require(norm_recipe_ == "mlx" || norm_recipe_ == "metal", "SeedVR2 VAE norm recipe must be mlx or metal");
    if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_VAE_HISTORY_RECIPE")) history_recipe_ = value;
    require(history_recipe_ == "concat" || history_recipe_ == "compact" || history_recipe_ == "frames",
            "SeedVR2 VAE history recipe must be concat, compact or frames");
    if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_VAE_POINTWISE_RECIPE")) pointwise_recipe_ = value;
    require(pointwise_recipe_ == "conv" || pointwise_recipe_ == "matmul",
            "SeedVR2 VAE pointwise recipe must be conv or matmul");
    if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_VAE_CONV_TILE_ROWS")) {
        const char *end = value + std::strlen(value);
        const auto parsed = std::from_chars(value, end, conv_tile_rows_);
        require(parsed.ec == std::errc{} && parsed.ptr == end && conv_tile_rows_ >= 128 &&
                conv_tile_rows_ <= 65536 && conv_tile_rows_ % 128 == 0,
                "SeedVR2 convolution tile rows must be a 128-aligned integer in 128...65536");
    }
    if (config) {
        require(config->encoder == "buffered" || config->encoder == "streaming", "invalid request-local SeedVR2 VAE encoder recipe");
        require(config->convolution == "singleton2d" || config->convolution == "conv3d" ||
                config->convolution == "mixed3d" || config->convolution == "tiled3d",
                "invalid request-local SeedVR2 VAE convolution recipe");
        require(config->tile_rows >= 128 && config->tile_rows <= 65536 && config->tile_rows % 128 == 0,
                "invalid request-local SeedVR2 VAE tile rows");
        conv_recipe_ = config->convolution;
        require(config->history == "compact" || config->history == "frames", "invalid request-local SeedVR2 VAE history recipe");
        history_recipe_ = config->history;
        conv_tile_rows_ = config->tile_rows;
    }
    require(history_recipe_ != "frames" || conv_recipe_ != "taps2d", "frame history does not support experimental taps2d");
}
Tensor VAE::conv(const Tensor &input, const std::string &key, Caches *state,
                 bool initial, bool downsample, bool owned_normalized_singleton) const {
    auto cache = state ? &state->compact : nullptr;
    // Channel-last throughout the VAE, including owned history tensors.
    const auto &weight = weights_.at(key + ".weight");
    const int kernel_t = weight.shape(1);
    const int stride_t = downsample && kernel_t > 1 ? 2 : 1;
    const int stride_s = downsample ? 2 : 1;
    if (pointwise_recipe_ == "matmul" && !downsample &&
            kernel_t == 1 && weight.shape(2) == 1 && weight.shape(3) == 1) {
        // A 1x1x1 shortcut is a channel projection, with no stencil or causal
        // history. Keep FP16 operands/FP32 GEMM accumulation instead of
        // materializing full-resolution FP32 conv2d inputs and workspaces.
        auto shape = input.shape();
        shape.back() = weight.shape(0);
        auto projected = mx::matmul(mx::reshape(input, {-1, weight.shape(4)}),
            mx::transpose(mx::reshape(weight, {weight.shape(0), weight.shape(4)})));
        return mx::reshape(projected, shape) + weights_.at(key + ".bias");
    }
    if (history_recipe_ == "frames")
        return conv_frames(input,key,state ? &state->frames : nullptr,initial,downsample,
                           owned_normalized_singleton && norm_recipe_ == "metal");
    // Restrict bounded direct stencils to the final decoder's largest
    // 256-channel temporal convolution; smaller operators retain the faster
    // MLX route. No spatial crop, independent tile norm or halo approximation.
    const bool bounded_stencil = conv_recipe_ == "tiled3d" || (conv_recipe_ == "mixed3d" &&
        !initial && kernel_t > 1 && int64_t(input.shape(2))*input.shape(3) >= 1024*1024 && input.shape(4) >= 256);
    if (bounded_stencil) {
        const int halo = kernel_t - 1;
        const auto found = cache ? cache->find(key) : CompactCaches::iterator{};
        const bool stored = halo && cache && !initial && found != cache->end();
        const auto head = stored ? found->second : (halo ? part(input, 1, 0, 1) : input);
        const int head_count = stored ? head.shape(1) : halo;
        if (cache && kernel_t > stride_t) {
            auto history = causal_history(input, head, kernel_t - stride_t, stored);
            mx::eval(history);
            cache->insert_or_assign(key, history);
        }
        if (initial && input.shape(1) == 1 && kernel_t > 1) {
            // Replicated first-frame operands allow an exact FP32 temporal
            // coefficient sum. The bounded direct GEMM avoids FP16 Winograd
            // overflow and the much larger FP32 spatial workspaces.
            auto folded = mx::expand_dims(mx::sum(mx::astype(weight, mx::float32), 1), 1);
            return metal::conv_tiled(input, input, 0, false, folded, stride_t, stride_s,
                                     downsample ? 0 : weight.shape(2) / 2, downsample, conv_tile_rows_) +
                   weights_.at(key + ".bias");
        }
        return metal::conv_tiled(input, head, head_count, !stored, weight, stride_t, stride_s,
                                 downsample ? 0 : weight.shape(2) / 2, downsample, conv_tile_rows_) +
               weights_.at(key + ".bias");
    }
    const int halo = kernel_t - 1;
    const auto found = cache ? cache->find(key) : CompactCaches::iterator{};
    const bool stored = halo && cache && !initial && found != cache->end();
    auto head = stored ? found->second : (halo ? part(input, 1, 0, 1) : input);
    const bool fold_first = conv_recipe_ != "conv3d" && initial && input.shape(1) == 1 && kernel_t > 1;
    auto x = input;
    // Folded singleton convolutions do not consume a physical temporal halo.
    // The reference recipe retains its original concatenation for diagnostics.
    if (halo && (!fold_first || history_recipe_ == "concat"))
        x = mx::concatenate({stored ? head : mx::repeat(head, halo, 1), input}, 1);
    if (cache && kernel_t > stride_t) {
        const int count = kernel_t - stride_t;
        // The legacy tail copy incidentally materialized x before replacing
        // the old history. Preserve that lifetime boundary for normal
        // convolutions: otherwise a lazy x pins both old and new histories
        // through the convolution workspace. Folded singletons need no halo.
        if (history_recipe_ == "compact" && !fold_first) mx::eval(x);
        // Force an owned contiguous copy: a view would pin the full activation
        // until the next slice, defeating bounded temporal residency.
        auto history = history_recipe_ == "compact" ? causal_history(input, head, count, stored)
            : mx::copy(mx::contiguous(part(x, 1, x.shape(1) - count, x.shape(1))));
        mx::eval(history);
        cache->insert_or_assign(key, history);
    }
    head = Tensor(0);
    if (downsample) {
        x = mx::pad(x, std::vector<std::pair<int, int>>{{0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 0}});
    }
    const int spatial_pad = downsample ? 0 : weight.shape(2) / 2;
    const int outputs_t = (input.shape(1) + halo - kernel_t) / stride_t + 1;
    Tensor y(0);
    if (fold_first) {
        // All temporal operands are the replicated first frame. Folding the
        // temporal filter avoids repeating their convolution products.
        // Keep the effective filter and convolution accumulator in FP32.
        // The FP16 2D backend can overflow its internal transformed operands
        // at real-media resolutions even when the final convolution is finite.
        auto filter = mx::sum(mx::astype(weight, mx::float32), 1);
        auto frame = mx::reshape(input, {input.shape(0), input.shape(2), input.shape(3), input.shape(4)});
        if (downsample) frame = mx::pad(frame, std::vector<std::pair<int, int>>{{0,0},{0,1},{0,1},{0,0}});
        y = mx::conv2d(mx::astype(frame, mx::float32), filter, {stride_s, stride_s}, {spatial_pad, spatial_pad});
        y = mx::astype(y, input.dtype());
        y = mx::expand_dims(y, 1);
    } else if (conv_recipe_ == "taps2d" ||
               ((conv_recipe_ == "singleton2d" || conv_recipe_ == "mixed3d") && kernel_t == 1)) {
        Tensor accumulated(0);
        for (int tap = 0; tap < kernel_t; ++tap) {
            std::vector<int32_t> indices(outputs_t);
            for (int i = 0; i < outputs_t; ++i) indices[i] = i * stride_t + tap;
            auto frames = mx::take(x, Tensor(indices.data(), {outputs_t}, mx::int32), 1);
            frames = mx::reshape(frames, {input.shape(0) * outputs_t, x.shape(2), x.shape(3), x.shape(4)});
            auto filter = mx::squeeze(part(weight, 1, tap, tap + 1), 1);
            auto partial = mx::conv2d(mx::astype(frames, mx::float32), mx::astype(filter, mx::float32),
                                      {stride_s, stride_s}, {spatial_pad, spatial_pad});
            // Bound each temporal tap workspace, instead of retaining three
            // convolution workspaces in a lazy sum. FP32 joins avoid adding
            // further FP16 rounding between the temporal contributions.
            mx::eval(partial);
            accumulated = tap ? accumulated + partial : partial;
            mx::eval(accumulated);
        }
        y = mx::astype(accumulated, input.dtype());
        y = mx::reshape(y, {input.shape(0), outputs_t, y.shape(1), y.shape(2), y.shape(3)});
    } else {
        y = mx::conv3d(x, weight, {stride_t, stride_s, stride_s}, {0, spatial_pad, spatial_pad});
    }
    return y + weights_.at(key + ".bias");
}
Tensor VAE::conv_frames(const Tensor &input, const std::string &key, FrameHistoryMap *cache,
                        bool initial, bool downsample, bool retain_owned_singleton) const {
    const auto &weight = weights_.at(key+".weight");
    const int kernel_t = weight.shape(1), halo = kernel_t-1;
    const int stride_t = downsample && kernel_t > 1 ? 2 : 1, stride_s = downsample ? 2 : 1;
    const auto found = cache ? cache->find(key) : FrameHistoryMap::iterator{};
    const bool stored = halo && cache && !initial && found != cache->end();
    auto previous = stored ? found->second : FrameHistory{};
    auto first = halo ? part(input,1,0,1) : input;
    const bool bounded = conv_recipe_ == "tiled3d" || (conv_recipe_ == "mixed3d" &&
        !initial && kernel_t > 1 && int64_t(input.shape(2))*input.shape(3) >= 1024*1024 && input.shape(4) >= 256);
    const bool fold_first = conv_recipe_ != "conv3d" && initial && input.shape(1) == 1 && halo;
    auto x = input;
    if (!bounded && halo && !fold_first) {
        auto operands = stored ? previous : FrameHistory(halo,first);
        operands.push_back(input);
        x = mx::concatenate(operands,1);
        // Materialize the only required concatenation before retiring old
        // history owners. There is no intermediate two-frame history copy.
        mx::eval(x);
    }
    if (cache && kernel_t > stride_t) {
        auto next = advance_frame_history(input,previous,kernel_t-stride_t,stored,retain_owned_singleton);
        cache->insert_or_assign(key,std::move(next));
    }
    const int pad = downsample ? 0 : weight.shape(2)/2;
    if (bounded) {
        if (fold_first) {
            auto folded = mx::expand_dims(mx::sum(mx::astype(weight,mx::float32),1),1);
            return metal::conv_tiled(input,input,0,false,folded,stride_t,stride_s,pad,downsample,conv_tile_rows_) +
                   weights_.at(key+".bias");
        }
        const auto &head = stored ? previous.front() : first;
        const auto *second = stored ? &previous.back() : nullptr;
        return metal::conv_tiled(input,head,stored ? int(previous.size()) : halo,!stored,weight,
                                 stride_t,stride_s,pad,downsample,conv_tile_rows_,second) + weights_.at(key+".bias");
    }
    previous.clear(); first = Tensor(0);
    if (downsample) x = mx::pad(x,std::vector<std::pair<int,int>>{{0,0},{0,0},{0,1},{0,1},{0,0}});
    Tensor y(0);
    if (fold_first) {
        auto filter = mx::sum(mx::astype(weight,mx::float32),1);
        auto frame = mx::reshape(input,{input.shape(0),input.shape(2),input.shape(3),input.shape(4)});
        if (downsample) frame = mx::pad(frame,std::vector<std::pair<int,int>>{{0,0},{0,1},{0,1},{0,0}});
        y = mx::expand_dims(mx::astype(mx::conv2d(mx::astype(frame,mx::float32),filter,
            {stride_s,stride_s},{pad,pad}),input.dtype()),1);
    } else if (conv_recipe_ != "conv3d" && kernel_t == 1) {
        const int t = input.shape(1);
        std::vector<int32_t> indices(t);
        for (int i = 0; i < t; ++i) indices[i] = i;
        auto frames = mx::take(x,Tensor(indices.data(),{t},mx::int32),1);
        frames = mx::reshape(frames,{input.shape(0)*t,x.shape(2),x.shape(3),x.shape(4)});
        auto filter = mx::squeeze(weight,1);
        y = mx::astype(mx::conv2d(mx::astype(frames,mx::float32),mx::astype(filter,mx::float32),
            {stride_s,stride_s},{pad,pad}),input.dtype());
        y = mx::reshape(y,{input.shape(0),t,y.shape(1),y.shape(2),y.shape(3)});
    } else y = mx::conv3d(x,weight,{stride_t,stride_s,stride_s},{0,pad,pad});
    return y + weights_.at(key+".bias");
}
Tensor VAE::group_norm(const Tensor &x, const std::string &key, bool activate) const {
    if (norm_recipe_ == "metal")
        return metal::group_norm(x, weights_.at(key + ".weight"), weights_.at(key + ".bias"), activate);
    const int b = x.shape(0), t = x.shape(1), h = x.shape(2), w = x.shape(3), c = x.shape(4);
    require(c % 32 == 0, "invalid SeedVR2 VAE GroupNorm channels");
    auto grouped = mx::reshape(x, {b * t, h * w, 32, c / 32});
    grouped = mx::reshape(mx::transpose(grouped, {0, 2, 1, 3}), {b * t, 32, h * w * (c / 32)});
    // All spatial positions and channels within a group, per frame. FP32
    // reductions and affine arithmetic precede the single FP16 output cast.
    grouped = mx::fast::layer_norm(mx::astype(grouped, mx::float32), {}, {}, 1e-6f);
    auto result = mx::reshape(mx::transpose(mx::reshape(grouped, {b * t, 32, h * w, c / 32}), {0, 2, 1, 3}), x.shape());
    result = result * mx::astype(weights_.at(key + ".weight"), mx::float32) +
             mx::astype(weights_.at(key + ".bias"), mx::float32);
    result = mx::astype(result, x.dtype());
    return activate ? swish(result) : result;
}
Tensor VAE::resnet(const Tensor &x, const std::string &key, Caches *cache, bool initial) const {
    auto y = conv(group_norm(x, key + ".norm1"), key + ".conv1", cache, initial,false,true);
    mx::eval(y);
    if (std::getenv("TURBOCIDER_SEEDVR2_VAE_PROFILE"))
        require(mx::all(mx::isfinite(y)).item<bool>(), "nonfinite SeedVR2 VAE convolution: " + key + ".conv1");
    y = conv(group_norm(y, key + ".norm2"), key + ".conv2", cache, initial,false,true);
    auto residual = weights_.has(key + ".conv_shortcut.weight") ? conv(x, key + ".conv_shortcut", cache, initial) : x;
    y = y + residual;
    mx::eval(y);
    if (std::getenv("TURBOCIDER_SEEDVR2_VAE_PROFILE")) {
        require(mx::all(mx::isfinite(y)).item<bool>(), "nonfinite SeedVR2 VAE residual: " + key);
        std::fprintf(stderr, "vae_resnet %s T=%d H=%d C=%d active=%zu peak=%zu cache=%zu\n",
                     key.c_str(), y.shape(1), y.shape(2), y.shape(4),
                     mx::get_active_memory(), mx::get_peak_memory(), mx::get_cache_memory());
    }
    return y;
}
Tensor VAE::mid_block(const Tensor &input, const std::string &key, Caches *cache, bool initial) const {
    auto x = resnet(input, key + ".resnets.0", cache, initial);
    const auto p = key + ".attentions.0";
    auto y = group_norm(x, p + ".group_norm", false);
    const int bt = x.shape(0) * x.shape(1), hw = x.shape(2) * x.shape(3), c = x.shape(4);
    y = mx::reshape(y, {bt, hw, c});
    auto heads = [&](const std::string &name) { return mx::expand_dims(weights_.linear(y, p + "." + name), 1); };
    auto q = mx::astype(heads("to_q"), mx::float32);
    auto k = mx::astype(heads("to_k"), mx::float32);
    auto v = mx::astype(heads("to_v"), mx::float32);
    mx::eval({q, k, v});
    std::vector<Tensor> attended;
    // This VAE has a single 512-wide head, outside MLX's fused full-SDPA
    // geometry. Bound the otherwise quadratic score matrix by query rows;
    // every query still attends to every spatial key in its own frame.
    // FP32 scores prevent overflow in the half-precision composite fallback.
    for (int frame = 0; frame < bt; ++frame) {
        auto keys = part(k, 0, frame, frame + 1);
        auto values = part(v, 0, frame, frame + 1);
        std::vector<Tensor> chunks;
        for (int first = 0; first < hw; first += 512) {
            auto queries = part(part(q, 0, frame, frame + 1), 2, first, std::min(first + 512, hw));
            auto chunk = mx::fast::scaled_dot_product_attention(queries, keys, values, 1.f / std::sqrt(float(c)));
            chunk = mx::astype(chunk, x.dtype());
            mx::eval(chunk);
            chunks.push_back(chunk);
        }
        attended.push_back(mx::concatenate(chunks, 2));
    }
    y = mx::concatenate(attended, 0);
    y = mx::reshape(y, {bt, hw, c});
    x = x + mx::reshape(weights_.linear(y, p + ".to_out.0"), x.shape());
    return resnet(x, key + ".resnets.1", cache, initial);
}
Tensor VAE::upsample(const Tensor &input, const std::string &key, Caches *cache,
                     bool initial, bool temporal) const {
    const int b = input.shape(0), t = input.shape(1), h = input.shape(2), w = input.shape(3), c = input.shape(4);
    const int tr = temporal ? 2 : 1;
    const auto &weight = weights_.at(key + ".upscale_conv.weight");
    auto y = mx::matmul(input, mx::transpose(mx::reshape(weight, {4 * tr * c, c}))) +
             weights_.at(key + ".upscale_conv.bias");
    // Expansion channels are [spatial_row, spatial_col, temporal_phase, c].
    y = mx::reshape(y, {b, t, h, w, 2, 2, tr, c});
    y = mx::reshape(mx::transpose(y, {0, 1, 6, 2, 4, 3, 5, 7}), {b, t * tr, h * 2, w * 2, c});
    if (temporal && initial) {
        // The causal first frame has one phase, not two. Keep phase zero and
        // skip phase one only for the initial slice, never subsequent slices.
        y = mx::concatenate({part(y, 1, 0, 1), part(y, 1, 2, y.shape(1))}, 1);
    }
    return conv(y, key + ".conv", cache, initial);
}
Tensor VAE::encode_slice(const Tensor &input, Caches *cache, bool initial,
                         const Event &event, std::atomic<bool> &cancelled) const {
    // The full-resolution spatial-only head is causal per frame. Running it
    // one frame at a time bounds convolution workspaces without changing the
    // temporal downsamplers' phase or processing frames independently.
    if (event) event("seedvr2_vae_encode", 0, 5);
    std::vector<Tensor> head;
    for (int frame = 0; frame < input.shape(1); ++frame) {
        const bool first = initial && frame == 0;
        auto y = conv(part(input, 1, frame, frame + 1), "encoder.conv_in", cache, first);
        for (int j = 0; j < 2; ++j) {
            checkpoint(cancelled);
            y = resnet(y, "encoder.down_blocks.0.resnets." + std::to_string(j), cache, first);
        }
        y = conv(y, "encoder.down_blocks.0.downsamplers.0.conv", cache, first, true);
        fence(y, cancelled);
        head.push_back(y);
    }
    auto x = mx::concatenate(head, 1);
    for (int i = 1; i < 4; ++i) {
        if (event) event("seedvr2_vae_encode", i, 5);
        const auto p = "encoder.down_blocks." + std::to_string(i);
        for (int j = 0; j < 2; ++j) {
            checkpoint(cancelled);
            x = resnet(x, p + ".resnets." + std::to_string(j), cache, initial);
        }
        if (i < 3) x = conv(x, p + ".downsamplers.0.conv", cache, initial, true);
        fence(x, cancelled);
    }
    x = mid_block(x, "encoder.mid_block", cache, initial);
    x = conv(group_norm(x, "encoder.conv_norm_out"), "encoder.conv_out", cache, initial,false,true);
    x = part(x, 4, 0, 16) * Tensor(.9152f, x.dtype());
    fence(x, cancelled);
    if (event) event("seedvr2_vae_encode", 5, 5);
    return x;
}
Tensor VAE::decode_slice(const Tensor &input, Caches *cache, bool initial,
                         const Event &event, std::atomic<bool> &cancelled,
                         const DecodedFrameSink &sink, int *next_frame, int frame_limit) const {
    auto x = conv(input / Tensor(.9152f, input.dtype()), "decoder.conv_in", cache, initial);
    x = mid_block(x, "decoder.mid_block", cache, initial);
    fence(x, cancelled);
    for (int i = 0; i < 2; ++i) {
        if (event) event("seedvr2_vae_decode", i, 5);
        const auto p = "decoder.up_blocks." + std::to_string(i);
        for (int j = 0; j < 3; ++j) {
            checkpoint(cancelled);
            x = resnet(x, p + ".resnets." + std::to_string(j), cache, initial);
        }
        if (i == 0) x = upsample(x, p + ".upsamplers.0", cache, initial, true);
        fence(x, cancelled);
    }
    // From the final temporal upsampler onward, carry causal histories through
    // each output frame. No full-resolution five-frame activation/workspace
    // needs to coexist with the resident 3B transformer bank.
    std::vector<Tensor> frames;
    for (int frame = 0; frame < x.shape(1); ++frame) {
        const bool first = initial && frame == 0;
        auto expanded = upsample(part(x, 1, frame, frame + 1),
                                 "decoder.up_blocks.1.upsamplers.0", cache, first, true);
        fence(expanded, cancelled);
        for (int phase = 0; phase < expanded.shape(1); ++phase) {
            const bool first_output = first && phase == 0;
            auto y = part(expanded, 1, phase, phase + 1);
            for (int i = 2; i < 4; ++i) {
                if (event) event("seedvr2_vae_decode", i, 5);
                const auto p = "decoder.up_blocks." + std::to_string(i);
                for (int j = 0; j < 3; ++j) {
                    checkpoint(cancelled);
                    y = resnet(y, p + ".resnets." + std::to_string(j), cache, first_output);
                }
                if (i == 2) y = upsample(y, p + ".upsamplers.0", cache, first_output, false);
                fence(y, cancelled);
            }
            y = conv(group_norm(y, "decoder.conv_norm_out"), "decoder.conv_out", cache, first_output,false,true);
            fence(y, cancelled);
            if (sink) {
                require(next_frame != nullptr, "missing streaming decoder frame counter");
                if (*next_frame < frame_limit) {
                    sink(*next_frame, mx::transpose(y, {0,4,1,2,3}));
                    checkpoint(cancelled);
                }
                ++*next_frame;
            } else frames.push_back(y);
        }
    }
    if (sink) {
        if (event) event("seedvr2_vae_decode", 5, 5);
        return Tensor(0);
    }
    x = mx::concatenate(frames, 1);
    fence(x, cancelled);
    if (event) event("seedvr2_vae_decode", 5, 5);
    return x;
}
Tensor VAE::encode(const Tensor &pixels, const Event &event, std::atomic<bool> &cancelled) const {
    require(pixels.ndim() == 5 && pixels.shape(0) == 1 && pixels.shape(1) == 3 &&
            pixels.shape(3) % 8 == 0 && pixels.shape(4) % 8 == 0 &&
            (pixels.shape(2) == 1 || pixels.shape(2) % 4 == 1),
            "SeedVR2 VAE encode requires batch-one RGB BCTHW, T=4n+1 and spatial multiples of eight");
    auto input = mx::transpose(mx::astype(pixels, mx::float16), {0, 2, 3, 4, 1});
    std::vector<Tensor> slices;
    Caches cache;
    // A one-frame image has no future slice that could consume a history.
    // Avoid storing every full-resolution residual input twice in this case.
    auto histories = input.shape(1) == 1 ? nullptr : &cache;
    for (int start = 0; start < input.shape(1);) {
        const int stop = std::min(input.shape(1), start + (start == 0 ? 5 : 4));
        auto result = encode_slice(part(input, 1, start, stop), histories, start == 0, event, cancelled);
        slices.push_back(result);
        start = stop;
    }
    auto result = mx::transpose(mx::concatenate(slices, 1), {0, 4, 1, 2, 3});
    fence(result, cancelled);
    return result;
}
Tensor VAE::encode_stream(int source_frames, int height, int width, const EncodedFrameSource &source,
                         const Event &event, std::atomic<bool> &cancelled) const {
    require(source_frames >= 1 && source_frames <= 121 && height > 0 && width > 0 &&
            height % 8 == 0 && width % 8 == 0 && bool(source),
            "SeedVR2 streaming encode requires a complete <=121-frame source and spatial multiples of eight");
    const int total = padded_frames(source_frames);
    Caches cache;
    auto histories = total == 1 ? nullptr : &cache;
    std::vector<Tensor> slices;
    Tensor last(0);
    int consumed = 0;
    for (int start = 0; start < total;) {
        checkpoint(cancelled);
        const int stop = std::min(total,start+(start == 0 ? 5 : 4));
        std::vector<Tensor> frames;
        frames.reserve(stop-start);
        for (int index = start; index < stop; ++index) {
            checkpoint(cancelled);
            if (index < source_frames) {
                require(index == consumed,"streaming encode source index is not sequential");
                auto frame = source(index);
                checkpoint(cancelled);
                require(frame.shape() == mx::Shape{1,3,1,height,width} && frame.dtype() == mx::float32,
                        "streaming encoder source must return normalized FP32 RGB BCTHW singleton frames");
                last = mx::transpose(mx::astype(frame,mx::float16),{0,2,3,4,1});
                // A completed FP16 allocation cannot pin the caller's FP32
                // resize/source graph through a lazy conversion. The caller
                // can reuse its host RGB24 buffer before the next source call.
                fence(last,cancelled);
                ++consumed;
            } else require(consumed == source_frames && last.ndim() == 5,"missing final source frame for padding");
            frames.push_back(last);
        }
        auto input = mx::concatenate(frames,1);
        fence(input,cancelled);
        frames.clear();
        auto encoded = encode_slice(input,histories,start == 0,event,cancelled);
        slices.push_back(encoded);
        start = stop;
        if (start < source_frames) last = Tensor(0);
    }
    require(consumed == source_frames,"streaming encoder did not consume the complete declared source");
    auto result = mx::transpose(mx::concatenate(slices,1),{0,4,1,2,3});
    fence(result,cancelled);
    return result;
}
Tensor VAE::decode(const Tensor &latent, const Event &event, std::atomic<bool> &cancelled) const {
    require(latent.ndim() == 5 && latent.shape(0) == 1 && latent.shape(1) == 16 && latent.shape(2) >= 1,
            "SeedVR2 VAE decode requires batch-one 16-channel BCTHW");
    auto input = mx::transpose(mx::astype(latent, mx::float16), {0, 2, 3, 4, 1});
    std::vector<Tensor> slices;
    Caches cache;
    auto histories = input.shape(1) == 1 ? nullptr : &cache;
    for (int start = 0; start < input.shape(1);) {
        const int stop = std::min(input.shape(1), start + (start == 0 ? 2 : 1));
        auto result = decode_slice(part(input, 1, start, stop), histories, start == 0, event, cancelled);
        slices.push_back(result);
        start = stop;
    }
    auto result = mx::transpose(mx::concatenate(slices, 1), {0, 4, 1, 2, 3});
    fence(result, cancelled);
    return result;
}
void VAE::decode_stream(const Tensor &latent, int output_frames, const DecodedFrameSink &sink,
                        const Event &event, std::atomic<bool> &cancelled) const {
    require(latent.ndim() == 5 && latent.shape(0) == 1 && latent.shape(1) == 16 && latent.shape(2) >= 1,
            "SeedVR2 VAE streaming decode requires batch-one 16-channel BCTHW");
    const int total = 1 + 4 * (latent.shape(2) - 1);
    require(bool(sink) && output_frames > 0 && output_frames <= total && total - output_frames <= 3,
            "invalid SeedVR2 streaming decoder sink or padded frame count");
    auto input = mx::transpose(mx::astype(latent, mx::float16), {0,2,3,4,1});
    Caches cache;
    auto histories = input.shape(1) == 1 ? nullptr : &cache;
    int next = 0;
    for (int start = 0; start < input.shape(1);) {
        checkpoint(cancelled);
        const int stop = std::min(input.shape(1), start + (start == 0 ? 2 : 1));
        decode_slice(part(input,1,start,stop),histories,start == 0,event,cancelled,sink,&next,output_frames);
        start = stop;
    }
    require(next == total, "SeedVR2 streaming decoder did not produce the complete padded sequence");
}
} // namespace tc::seedvr2
