#include "vae.hpp"
#include <cmath>

namespace tc::qwen21 {
namespace {
const float means[] = {
    .5126f,.7721f,-.0631f,1.3506f,-.7855f,-2.1025f,-.3458f,1.3722f,
    1.8873f,-1.7177f,-.651f,.2732f,.7562f,-.6163f,-1.0277f,3.8363f,
    2.021f,.0472f,.932f,2.0087f,2.4954f,-.1391f,-1.4249f,1.8464f,
    -.5236f,1.2826f,3.7046f,-1.3035f,2.7286f,-1.4518f,-1.9036f,-1.9955f,
    -.0342f,-1.0265f,-.7636f,3.0555f,.0746f,-3.0751f,-.1076f,1.7376f,
    -1.0914f,-1.9435f,-.2784f,-1.368f,.4809f,-.4433f,.3764f,.5729f,
    -2.0595f,1.096f,-1.326f,-2.0211f,-5.0179f,.5275f,4.0162f,1.8505f,
    .3026f,1.9373f,1.4937f,.2632f,.5547f,-1.7121f,-.1562f,.0304f};
const float deviations[] = {
    3.2001f,3.2936f,3.4321f,3.0091f,3.1061f,4.0379f,4.0705f,3.791f,
    3.0785f,3.65f,3.9308f,3.0904f,2.8778f,3.7675f,3.732f,5.0756f,
    3.2864f,4.0397f,3.1317f,4.0443f,2.9249f,3.9454f,3.0988f,4.2489f,
    3.4896f,3.8513f,3.9323f,3.4719f,3.7498f,4.283f,3.5694f,4.2467f,
    3.9037f,3.2947f,5.077f,3.5075f,3.27f,3.4767f,2.8063f,5.1125f,
    3.5327f,4.7833f,3.1286f,4.1819f,3.8527f,3.8312f,3.5605f,4.3875f,
    3.9624f,4.0168f,3.5643f,4.055f,5.5614f,4.2963f,4.408f,3.4959f,
    3.8747f,3.7608f,3.5735f,3.149f,3.7662f,3.6746f,3.4563f,3.8161f};

Tensor up_shortcut(const Tensor &x, int channels, int temporal) {
    int h = x.shape(2), w = x.shape(3);
    int factor = temporal * 4;
    require(channels * factor % x.shape(1) == 0, "invalid Qwen21 VAE up shortcut");
    auto repeated = mx::repeat(x, channels * factor / x.shape(1), 1);
    auto expanded = mx::reshape(repeated, {1, channels, temporal, 2, 2, 1, h, w});
    expanded = mx::transpose(expanded, {0, 1, 5, 2, 6, 3, 7, 4});
    expanded = mx::reshape(expanded, {1, channels, temporal, h * 2, w * 2});
    return mx::squeeze(slice_axis(expanded, 2, temporal - 1, temporal), 2);
}
Tensor down_shortcut(const Tensor &x, int channels, int temporal, int spatial) {
    int c = x.shape(1), h = x.shape(2), w = x.shape(3);
    int factor = temporal * spatial * spatial;
    require(c * factor % channels == 0 && h % spatial == 0 && w % spatial == 0,
            "invalid Qwen21 VAE down shortcut");
    auto expanded = mx::expand_dims(x, 2);
    if (temporal == 2)
        expanded = mx::concatenate({mx::zeros_like(expanded), expanded}, 2);
    expanded = mx::reshape(expanded, {1, c, 1, temporal, h / spatial, spatial, w / spatial, spatial});
    expanded = mx::transpose(expanded, {0, 1, 3, 5, 7, 2, 4, 6});
    expanded = mx::reshape(expanded, {1, channels, c * factor / channels, h / spatial, w / spatial});
    return mx::mean(expanded, 2);
}
}

std::string VAE::canonical_key(const std::string &key) {
    std::string result = key;
    if (result.starts_with("conv1.")) result.replace(0, 5, "quant_conv");
    else if (result.starts_with("conv2.")) result.replace(0, 5, "post_quant_conv");
    result = std::regex_replace(result, std::regex("^(encoder|decoder)\\.conv1\\."), "$1.conv_in.");
    result = std::regex_replace(result, std::regex("^(encoder|decoder)\\.head\\.0\\."), "$1.norm_out.");
    result = std::regex_replace(result, std::regex("^(encoder|decoder)\\.head\\.2\\."), "$1.conv_out.");
    result = std::regex_replace(result, std::regex("\\.middle\\.0\\."), ".mid_block.resnets.0.");
    result = std::regex_replace(result, std::regex("\\.middle\\.1\\."), ".mid_block.attentions.0.");
    result = std::regex_replace(result, std::regex("\\.middle\\.2\\."), ".mid_block.resnets.1.");
    result = std::regex_replace(result, std::regex("decoder\\.upsamples\\.([0-4])\\.upsamples\\.3\\."), "decoder.up_blocks.$1.upsampler.");
    result = std::regex_replace(result, std::regex("decoder\\.upsamples\\.([0-4])\\.upsamples\\.([0-2])\\."), "decoder.up_blocks.$1.resnets.$2.");
    result = std::regex_replace(result, std::regex("encoder\\.downsamples\\.([0-4])\\.downsamples\\.2\\."), "encoder.down_blocks.$1.downsampler.");
    result = std::regex_replace(result, std::regex("encoder\\.downsamples\\.([0-4])\\.downsamples\\.([0-1])\\."), "encoder.down_blocks.$1.resnets.$2.");
    result = std::regex_replace(result, std::regex("\\.residual\\.0\\."), ".norm1.");
    result = std::regex_replace(result, std::regex("\\.residual\\.2\\."), ".conv1.");
    result = std::regex_replace(result, std::regex("\\.residual\\.3\\."), ".norm2.");
    result = std::regex_replace(result, std::regex("\\.residual\\.6\\."), ".conv2.");
    result = std::regex_replace(result, std::regex("\\.shortcut\\."), ".conv_shortcut.");
    return result;
}

VAE::VAE(const Weights &weights)
    : mean_(means, {1, 64, 1, 1}, mx::float32), std_(deviations, {1, 64, 1, 1}, mx::float32) {
    for (const auto &key : weights.sorted_keys()) {
        if (key.find("time_conv") != std::string::npos) continue;
        auto name = canonical_key(key);
        auto value = weights.at(key);
        if (name.ends_with(".weight")) {
            if (value.ndim() == 5) {
                require(value.shape(2) == 1, "Qwen21 image VAE requires single-frame spatial kernels");
                value = mx::squeeze(value, 2);
            }
            require(value.ndim() == 4, "invalid Qwen21 VAE convolution: " + name);
            value = mx::transpose(value, {0, 2, 3, 1});
        }
        require(values_.emplace(name, value).second, "duplicate Qwen21 VAE tensor: " + name);
    }
}

Tensor VAE::conv(const Tensor &x, const std::string &p, int padding, int stride) const {
    auto y = mx::conv2d(mx::transpose(x, {0, 2, 3, 1}), values_.at(p + ".weight"),
                        {stride, stride}, {padding, padding});
    y = y + values_.at(p + ".bias");
    return mx::transpose(y, {0, 3, 1, 2});
}
Tensor VAE::normalize(const Tensor &x, const std::string &p) const {
    auto f = mx::astype(x, mx::float32);
    auto l2 = mx::sqrt(mx::sum(f * f, 1, true));
    auto gamma = mx::reshape(values_.at(p + ".gamma"), {1, x.shape(1), 1, 1});
    return mx::astype((f / mx::maximum(l2, Tensor(1e-12f))) * std::sqrt(float(x.shape(1))) * gamma, x.dtype());
}
Tensor VAE::residual(const Tensor &x, const std::string &p) const {
    auto skip = values_.count(p + ".conv_shortcut.weight") ? conv(x, p + ".conv_shortcut", 0) : x;
    auto hidden = conv(silu(normalize(x, p + ".norm1")), p + ".conv1");
    return conv(silu(normalize(hidden, p + ".norm2")), p + ".conv2") + skip;
}
Tensor VAE::middle(const Tensor &x, const std::string &p) const {
    auto hidden = residual(x, p + ".resnets.0");
    auto a = p + ".attentions.0";
    int c = hidden.shape(1), h = hidden.shape(2), w = hidden.shape(3);
    auto qkv = conv(normalize(hidden, a + ".norm"), a + ".to_qkv", 0);
    qkv = mx::reshape(mx::transpose(qkv, {0, 2, 3, 1}), {1, h * w, 3 * c});
    auto parts = mx::split(qkv, 3, -1);
    auto scores = mx::matmul(parts[0], mx::transpose(parts[1], {0, 2, 1})) * (1.f / std::sqrt(float(c)));
    auto output = mx::matmul(mx::softmax(scores, -1), parts[2]);
    output = mx::transpose(mx::reshape(output, {1, h, w, c}), {0, 3, 1, 2});
    hidden = hidden + conv(output, a + ".proj", 0);
    return residual(hidden, p + ".resnets.1");
}

Tensor VAE::decode(const Tensor &latents, const Event &event, std::atomic<bool> &cancelled) const {
    require(latents.ndim() == 4 && latents.shape(0) == 1 && latents.shape(1) == 64 &&
            latents.shape(2) > 0 && latents.shape(3) > 0, "Qwen21 VAE latent shape must be [1,64,H,W]");
    checkpoint(cancelled);
    auto hidden = conv(latents * std_ + mean_, "post_quant_conv", 0);
    hidden = middle(conv(hidden, "decoder.conv_in"), "decoder.mid_block");
    for (int block = 0; block < 5; ++block) {
        checkpoint(cancelled);
        auto p = "decoder.up_blocks." + std::to_string(block);
        auto skip = hidden;
        for (int i = 0; i < 3; ++i) hidden = residual(hidden, p + ".resnets." + std::to_string(i));
        if (block < 4) {
            hidden = conv(mx::repeat(mx::repeat(hidden, 2, 2), 2, 3), p + ".upsampler.resample.1");
            hidden = hidden + up_shortcut(skip, hidden.shape(1), block < 3 ? 2 : 1);
        }
        mx::eval(hidden);
        if (event) event("qwen21_vae_decode", block + 1, 5);
    }
    return conv(silu(normalize(hidden, "decoder.norm_out")), "decoder.conv_out");
}

Tensor VAE::encode(const Tensor &image, const Event &event, std::atomic<bool> &cancelled,
                   std::unordered_map<std::string, Tensor> *trace) const {
    require(image.ndim() == 4 && image.shape(0) == 1 && (image.shape(1) == 3 || image.shape(1) == 4) &&
            image.shape(2) > 0 && image.shape(3) > 0 && image.shape(2) % 16 == 0 && image.shape(3) % 16 == 0,
            "Qwen21 VAE image must be RGB/RGBA NCHW with dimensions divisible by 16");
    checkpoint(cancelled);
    auto hidden = image;
    if (image.shape(1) == 3)
        hidden = mx::concatenate({image, mx::ones({1, 1, image.shape(2), image.shape(3)}, image.dtype())}, 1);
    hidden = conv(hidden, "encoder.conv_in");
    if (trace) trace->emplace("conv_in", hidden);
    for (int block = 0; block < 5; ++block) {
        checkpoint(cancelled);
        auto p = "encoder.down_blocks." + std::to_string(block);
        auto skip = hidden;
        for (int i = 0; i < 2; ++i) {
            hidden = residual(hidden, p + ".resnets." + std::to_string(i));
            if (trace) trace->emplace("down_" + std::to_string(block) + "_res_" + std::to_string(i), hidden);
        }
        if (block < 4) {
            hidden = mx::pad(hidden, {{0, 0}, {0, 0}, {0, 1}, {0, 1}});
            hidden = conv(hidden, p + ".downsampler.resample.1", 0, 2);
        }
        auto shortcut = down_shortcut(skip, hidden.shape(1), block > 0 && block < 4 ? 2 : 1, block < 4 ? 2 : 1);
        if (trace) {
            trace->emplace("down_" + std::to_string(block) + "_shortcut", shortcut);
            trace->emplace("down_" + std::to_string(block) + "_main", hidden);
        }
        hidden = hidden + shortcut;
        mx::eval(hidden);
        if (trace) trace->emplace("down_" + std::to_string(block), hidden);
        if (event) event("qwen21_vae_encode", block + 1, 5);
    }
    hidden = middle(hidden, "encoder.mid_block");
    if (trace) trace->emplace("middle", hidden);
    hidden = conv(silu(normalize(hidden, "encoder.norm_out")), "encoder.conv_out");
    hidden = slice_axis(conv(hidden, "quant_conv", 0), 1, 0, 64);
    if (trace) trace->emplace("mean", hidden);
    return (hidden - mean_) / std_;
}
} // namespace tc::qwen21
