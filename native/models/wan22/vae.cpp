#include "vae.hpp"

#include <cmath>

namespace tc::wan22 {
namespace {

Tensor patchify(const Tensor &image) {
    const int height = image.shape(2), width = image.shape(3);
    auto patches = mx::reshape(image, {1, 3, height / 2, 2, width / 2, 2});
    // Wan's channel order is c, width-within-patch, height-within-patch.
    patches = mx::transpose(patches, {0, 1, 5, 3, 2, 4});
    return mx::reshape(patches, {1, 12, height / 2, width / 2});
}

Tensor down_shortcut(const Tensor &x, int channels, int temporal, int spatial) {
    const int c = x.shape(1), h = x.shape(2), w = x.shape(3);
    const int factor = temporal * spatial * spatial;
    require(c * factor % channels == 0 && h % spatial == 0 && w % spatial == 0,
            "invalid Wan2.2 VAE downsample shortcut geometry");
    auto expanded = mx::expand_dims(x, 2);
    if (temporal == 2) expanded = mx::concatenate({mx::zeros_like(expanded), expanded}, 2);
    expanded = mx::reshape(expanded, {1, c, 1, temporal, h / spatial, spatial, w / spatial, spatial});
    expanded = mx::transpose(expanded, {0, 1, 3, 5, 7, 2, 4, 6});
    expanded = mx::reshape(expanded, {1, channels, c * factor / channels, h / spatial, w / spatial});
    return mx::mean(expanded, 2);
}

} // namespace

void VAEEncoder::load(const std::filesystem::path &checkpoint) {
    require(std::filesystem::is_regular_file(checkpoint), "Wan2.2 VAE checkpoint is missing");
    weights_.clear();
    weights_.load_file(checkpoint);
    weights_.erase_prefix("decoder.");
    weights_.erase_prefix("conv2.");
    for (const auto &key : weights_.sorted_keys())
        if (key.find(".time_conv.") != std::string::npos) weights_.erase(key);
    require(weights_.has("encoder.conv1.weight") &&
                weights_.at("encoder.conv1.weight").shape() == mx::Shape{160, 12, 3, 3, 3} &&
                weights_.has("conv1.weight") && weights_.at("conv1.weight").shape() == mx::Shape{96, 96, 1, 1, 1},
            "checkpoint is not the 48-channel Wan2.2 VAE encoder");
    auto names = weights_.sorted_keys();
    std::vector<Tensor> arrays;
    arrays.reserve(names.size());
    for (const auto &name : names) {
        auto value = weights_.at(name);
        if (name.ends_with(".weight") && value.ndim() == 5) {
            // At the clean first frame, earlier causal frames are zero. Only
            // the final temporal kernel slice can contribute.
            value = mx::squeeze(slice_axis(value, 2, value.shape(2) - 1, value.shape(2)), 2);
        }
        if (name.ends_with(".weight") && value.ndim() == 4)
            value = mx::transpose(value, {0, 2, 3, 1});
        if (value.dtype() != mx::bfloat16) value = mx::astype(value, mx::bfloat16);
        arrays.push_back(value);
    }
    weights_.bind_arrays(names, arrays);
    weights_.materialize();
}

Tensor VAEEncoder::conv(const Tensor &x, const std::string &prefix, int padding, int stride) const {
    auto input = mx::transpose(x, {0, 2, 3, 1});
    auto value = mx::conv2d(input, weights_.at(prefix + ".weight"), {stride, stride}, {padding, padding});
    value = value + weights_.at(prefix + ".bias");
    return mx::transpose(value, {0, 3, 1, 2});
}

Tensor VAEEncoder::normalize(const Tensor &x, const std::string &prefix) const {
    auto value = mx::astype(x, mx::float32);
    auto length = mx::sqrt(mx::sum(value * value, 1, true));
    auto gamma = mx::reshape(mx::astype(weights_.at(prefix + ".gamma"), mx::float32), {1, x.shape(1), 1, 1});
    return mx::astype((value / mx::maximum(length, Tensor(1e-12f))) * std::sqrt(float(x.shape(1))) * gamma,
                      x.dtype());
}

Tensor VAEEncoder::residual(const Tensor &x, const std::string &prefix) const {
    auto shortcut = weights_.has(prefix + ".shortcut.weight") ? conv(x, prefix + ".shortcut", 0) : x;
    auto hidden = conv(silu(normalize(x, prefix + ".residual.0")), prefix + ".residual.2");
    hidden = conv(silu(normalize(hidden, prefix + ".residual.3")), prefix + ".residual.6");
    return hidden + shortcut;
}

Tensor VAEEncoder::middle(const Tensor &x) const {
    auto hidden = residual(x, "encoder.middle.0");
    constexpr auto prefix = "encoder.middle.1";
    const int c = hidden.shape(1), h = hidden.shape(2), w = hidden.shape(3);
    auto qkv = conv(normalize(hidden, std::string(prefix) + ".norm"), std::string(prefix) + ".to_qkv", 0);
    qkv = mx::reshape(mx::transpose(qkv, {0, 2, 3, 1}), {1, h * w, 3 * c});
    auto parts = mx::split(qkv, 3, -1);
    auto scores = mx::matmul(parts[0], mx::transpose(parts[1], {0, 2, 1})) * (1.f / std::sqrt(float(c)));
    auto attended = mx::matmul(mx::softmax(scores, -1), parts[2]);
    attended = mx::transpose(mx::reshape(attended, {1, h, w, c}), {0, 3, 1, 2});
    hidden = hidden + conv(attended, std::string(prefix) + ".proj", 0);
    return residual(hidden, "encoder.middle.2");
}

Tensor VAEEncoder::encode_first_frame(const Tensor &pixels) const {
    require(pixels.ndim() == 5 && pixels.shape(0) == 1 && pixels.shape(1) == 3 && pixels.shape(2) == 1 &&
                pixels.shape(3) >= 32 && pixels.shape(3) % 16 == 0 &&
                pixels.shape(4) >= 32 && pixels.shape(4) % 16 == 0 && pixels.dtype() == mx::bfloat16,
            "Wan2.2 VAE needs BF16 [1,3,1,16H,16W] first-frame pixels");
    auto hidden = conv(patchify(mx::squeeze(pixels, 2)), "encoder.conv1");
    for (int stage = 0; stage < 4; ++stage) {
        const auto prefix = "encoder.downsamples." + std::to_string(stage) + ".downsamples.";
        auto shortcut = hidden;
        hidden = residual(hidden, prefix + "0");
        hidden = residual(hidden, prefix + "1");
        if (stage < 3) {
            hidden = mx::pad(hidden, {{0, 0}, {0, 0}, {0, 1}, {0, 1}});
            hidden = conv(hidden, prefix + "2.resample.1", 0, 2);
        }
        shortcut = down_shortcut(shortcut, hidden.shape(1), stage > 0 && stage < 3 ? 2 : 1,
                                 stage < 3 ? 2 : 1);
        hidden = hidden + shortcut;
        mx::eval(hidden);
    }
    hidden = middle(hidden);
    hidden = conv(silu(normalize(hidden, "encoder.head.0")), "encoder.head.2");
    hidden = slice_axis(conv(hidden, "conv1", 0), 1, 0, 48);
    return mx::expand_dims(hidden, 2);
}

void VAEDecoder::load(const std::filesystem::path &checkpoint) {
    require(std::filesystem::is_regular_file(checkpoint), "Wan2.2 VAE checkpoint is missing");
    weights_.clear();
    weights_.load_file(checkpoint);
    weights_.erase_prefix("encoder.");
    weights_.erase_prefix("conv1.");
    require(weights_.has("decoder.conv1.weight") &&
                weights_.at("decoder.conv1.weight").shape() == mx::Shape{1024, 48, 3, 3, 3} &&
                weights_.has("conv2.weight") && weights_.at("conv2.weight").shape() == mx::Shape{48, 48, 1, 1, 1},
            "checkpoint is not the 48-channel Wan2.2 VAE decoder");
    auto names = weights_.sorted_keys();
    std::vector<Tensor> arrays;
    arrays.reserve(names.size());
    for (const auto &name : names) {
        auto value = weights_.at(name);
        if (name.ends_with(".weight") && value.ndim() == 5) value = mx::transpose(value, {0, 2, 3, 4, 1});
        else if (name.ends_with(".weight") && value.ndim() == 4) value = mx::transpose(value, {0, 2, 3, 1});
        if (value.dtype() != mx::bfloat16) value = mx::astype(value, mx::bfloat16);
        arrays.push_back(value);
    }
    weights_.bind_arrays(names, arrays);
    weights_.materialize();
}

Tensor VAEDecoder::causal_conv(const Tensor &x, const std::string &prefix, Cache &cache,
                               bool store_history) const {
    const auto &weight = weights_.at(prefix + ".weight");
    require(x.ndim() == 5 && weight.ndim() == 5 && weight.shape(4) == x.shape(4),
            "invalid Wan2.2 VAE causal convolution geometry");
    const int temporal = weight.shape(1), pad_h = weight.shape(2) / 2, pad_w = weight.shape(3) / 2;
    require((temporal == 1 || temporal == 3) && weight.shape(2) % 2 == 1 && weight.shape(3) % 2 == 1,
            "unsupported Wan2.2 VAE convolution kernel");
    auto source = x;
    int previous = 0;
    if (store_history && temporal > 1) {
        auto found = cache.find(prefix);
        if (found != cache.end()) {
            previous = found->second.shape(1);
            source = mx::concatenate({found->second, x}, 1);
        }
        cache.insert_or_assign(prefix, slice_axis(source, 1, std::max(0, source.shape(1) - temporal + 1), source.shape(1)));
    }
    const int left = std::max(0, temporal - 1 - previous);
    auto padded = mx::pad(source, {{0, 0}, {left, 0}, {pad_h, pad_h}, {pad_w, pad_w}, {0, 0}});
    auto result = mx::conv3d(padded, weight, {1, 1, 1}, {0, 0, 0});
    return result + mx::reshape(weights_.at(prefix + ".bias"), {1, 1, 1, 1, result.shape(4)});
}

Tensor VAEDecoder::spatial_conv(const Tensor &x, const std::string &prefix, int padding) const {
    const int frames = x.shape(1), height = x.shape(2), width = x.shape(3), channels = x.shape(4);
    auto input = mx::reshape(x, {frames, height, width, channels});
    auto value = mx::conv2d(input, weights_.at(prefix + ".weight"), {1, 1}, {padding, padding});
    value = value + weights_.at(prefix + ".bias");
    return mx::reshape(value, {1, frames, value.shape(1), value.shape(2), value.shape(3)});
}

Tensor VAEDecoder::normalize(const Tensor &x, const std::string &prefix) const {
    auto value = mx::astype(x, mx::float32);
    auto length = mx::sqrt(mx::sum(value * value, -1, true));
    auto gamma = mx::reshape(mx::astype(weights_.at(prefix + ".gamma"), mx::float32), {1, 1, 1, 1, x.shape(4)});
    return mx::astype((value / mx::maximum(length, Tensor(1e-12f))) * std::sqrt(float(x.shape(4))) * gamma,
                      x.dtype());
}

Tensor VAEDecoder::residual(const Tensor &x, const std::string &prefix, Cache &cache) const {
    auto shortcut = weights_.has(prefix + ".shortcut.weight")
        ? causal_conv(x, prefix + ".shortcut", cache, false) : x;
    auto hidden = causal_conv(silu(normalize(x, prefix + ".residual.0")), prefix + ".residual.2", cache);
    hidden = causal_conv(silu(normalize(hidden, prefix + ".residual.3")), prefix + ".residual.6", cache);
    return hidden + shortcut;
}

Tensor VAEDecoder::middle(const Tensor &x, Cache &cache) const {
    auto hidden = residual(x, "decoder.middle.0", cache);
    const int frames = hidden.shape(1), height = hidden.shape(2), width = hidden.shape(3), channels = hidden.shape(4);
    auto qkv = spatial_conv(normalize(hidden, "decoder.middle.1.norm"), "decoder.middle.1.to_qkv", 0);
    qkv = mx::reshape(qkv, {frames, height * width, 3 * channels});
    auto parts = mx::split(qkv, 3, -1);
    auto scores = mx::matmul(parts[0], mx::transpose(parts[1], {0, 2, 1})) * (1.f / std::sqrt(float(channels)));
    auto attended = mx::matmul(mx::softmax(scores, -1), parts[2]);
    attended = mx::reshape(attended, {1, frames, height, width, channels});
    hidden = hidden + spatial_conv(attended, "decoder.middle.1.proj", 0);
    return residual(hidden, "decoder.middle.2", cache);
}

Tensor VAEDecoder::upsample(const Tensor &x, int stage, Cache &cache,
                            std::unordered_set<std::string> &seen) const {
    auto prefix = "decoder.upsamples." + std::to_string(stage) + ".upsamples.3";
    auto hidden = x;
    if (stage < 2) {
        auto temporal = prefix + ".time_conv";
        // The official chunked VAE emits the first latent frame directly;
        // its first temporal-convolution cache entry is only a sentinel.
        if (!seen.insert(temporal).second) {
            hidden = causal_conv(hidden, temporal, cache);
            const int frames = hidden.shape(1), height = hidden.shape(2), width = hidden.shape(3), channels = x.shape(4);
            hidden = mx::reshape(hidden, {1, frames, height, width, 2, channels});
            hidden = mx::transpose(hidden, {0, 1, 4, 2, 3, 5});
            hidden = mx::reshape(hidden, {1, frames * 2, height, width, channels});
        }
    }
    hidden = mx::repeat(mx::repeat(hidden, 2, 2), 2, 3);
    return spatial_conv(hidden, prefix + ".resample.1");
}

Tensor VAEDecoder::decode(const Tensor &latents, const Event &event, std::atomic<bool> *cancelled) const {
    require(latents.ndim() == 5 && latents.shape(0) == 1 && latents.shape(1) == 48 &&
                latents.shape(2) > 0 && latents.shape(3) > 0 && latents.shape(4) > 0 &&
                latents.dtype() == mx::bfloat16,
            "Wan2.2 VAE needs BF16 [1,48,F,H,W] latents");
    Cache cache;
    std::unordered_set<std::string> seen;
    std::vector<Tensor> chunks;
    const int frames = latents.shape(2);
    for (int frame = 0; frame < frames; ++frame) {
        if (cancelled) checkpoint(*cancelled);
        if (event) event("wan22_video_vae", frame, frames);
        auto latent = slice_axis(latents, 2, frame, frame + 1);
        auto hidden = mx::transpose(latent, {0, 2, 3, 4, 1});
        hidden = causal_conv(hidden, "conv2", cache, false);
        hidden = middle(causal_conv(hidden, "decoder.conv1", cache), cache);
        for (int stage = 0; stage < 4; ++stage) {
            auto shortcut = hidden;
            auto prefix = "decoder.upsamples." + std::to_string(stage) + ".upsamples.";
            for (int block = 0; block < 3; ++block)
                hidden = residual(hidden, prefix + std::to_string(block), cache);
            if (stage < 3) {
                hidden = upsample(hidden, stage, cache, seen);
                const int channels = hidden.shape(4), temporal = stage < 2 ? 2 : 1;
                const int t = shortcut.shape(1), h = shortcut.shape(2), w = shortcut.shape(3), c = shortcut.shape(4);
                require(channels * temporal * 4 % c == 0, "invalid Wan2.2 VAE upsample shortcut geometry");
                auto expanded = mx::transpose(shortcut, {0, 4, 1, 2, 3});
                expanded = mx::repeat(expanded, channels * temporal * 4 / c, 1);
                expanded = mx::reshape(expanded, {1, channels, temporal, 2, 2, t, h, w});
                expanded = mx::transpose(expanded, {0, 1, 5, 2, 6, 3, 7, 4});
                expanded = mx::reshape(expanded, {1, channels, t * temporal, h * 2, w * 2});
                if (frame == 0 && temporal > 1) expanded = slice_axis(expanded, 2, temporal - 1, t * temporal);
                expanded = mx::transpose(expanded, {0, 2, 3, 4, 1});
                hidden = hidden + expanded;
            }
            mx::eval(hidden);
        }
        hidden = causal_conv(silu(normalize(hidden, "decoder.head.0")), "decoder.head.2", cache);
        const int t = hidden.shape(1), h = hidden.shape(2), w = hidden.shape(3);
        require(hidden.shape(4) == 12, "invalid Wan2.2 VAE output patch");
        hidden = mx::reshape(hidden, {1, t, h, w, 3, 2, 2});
        hidden = mx::transpose(hidden, {0, 1, 2, 6, 3, 5, 4});
        hidden = mx::reshape(hidden, {1, t, h * 2, w * 2, 3});
        mx::eval(hidden);
        chunks.push_back(hidden);
    }
    auto pixels = mx::concatenate(chunks, 1);
    require(pixels.shape(1) == 4 * frames - 3, "Wan2.2 VAE produced an invalid frame count");
    if (cancelled) checkpoint(*cancelled);
    if (event) event("wan22_video_vae", frames, frames);
    return mx::transpose(pixels, {0, 4, 1, 2, 3});
}

} // namespace tc::wan22
