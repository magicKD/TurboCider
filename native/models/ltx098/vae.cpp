#include "vae.hpp"
#include "vae_padding.hpp"

#include <cmath>

namespace tc::ltx098 {
namespace {

// The patch and residual downsampling layouts match the existing LTX 2.5
// native VAE; the 0.9.8 stage geometry and weights are different.
Tensor patchify(const Tensor &input) {
    const int b = input.shape(0), f = input.shape(1), h = input.shape(2), w = input.shape(3), c = input.shape(4);
    auto value = mx::reshape(input, {b, f, h / 4, 4, w / 4, 4, c});
    value = mx::transpose(value, {0, 1, 2, 4, 6, 5, 3});
    return mx::reshape(value, {b, f, h / 4, w / 4, c * 16});
}

Tensor space_to_depth(const Tensor &input, int temporal, int spatial) {
    const int b = input.shape(0), f = input.shape(1), h = input.shape(2), w = input.shape(3), c = input.shape(4);
    require(f % temporal == 0 && h % spatial == 0 && w % spatial == 0,
            "LTX 0.9.8 VAE downsample geometry mismatch");
    auto value = mx::reshape(input, {b, f / temporal, temporal, h / spatial, spatial,
                                     w / spatial, spatial, c});
    value = mx::transpose(value, {0, 1, 3, 5, 7, 2, 4, 6});
    return mx::reshape(value, {b, f / temporal, h / spatial, w / spatial, c * temporal * spatial * spatial});
}

Tensor pixel_norm(const Tensor &input) {
    return mx::fast::rms_norm(input, std::nullopt, 1.0e-8f);
}

void prepare_convolutions(Weights &weights) {
    auto names = weights.sorted_keys();
    std::vector<Tensor> arrays;
    arrays.reserve(names.size());
    for (const auto &name : names) {
        auto value = weights.at(name);
        if (value.ndim() == 5 && name.ends_with(".weight")) value = mx::transpose(value, {0, 2, 3, 4, 1});
        if (value.dtype() != mx::bfloat16) value = mx::astype(value, mx::bfloat16);
        arrays.push_back(value);
    }
    weights.bind_arrays(names, arrays);
    weights.materialize();
}

Tensor depth_to_space(const Tensor &input, int channels) {
    const int b = input.shape(0), f = input.shape(1), h = input.shape(2), w = input.shape(3);
    require(input.shape(4) == channels * 8, "LTX 0.9.8 VAE upsample geometry mismatch");
    auto value = mx::reshape(input, {b, f, h, w, channels, 2, 2, 2});
    value = mx::transpose(value, {0, 1, 5, 2, 6, 3, 7, 4});
    value = mx::reshape(value, {b, f * 2, h * 2, w * 2, channels});
    return mx::slice(value, {0, 1, 0, 0, 0}, {b, f * 2, h * 2, w * 2, channels});
}

Tensor unpatchify(const Tensor &input) {
    const int b = input.shape(0), f = input.shape(1), h = input.shape(2), w = input.shape(3);
    require(input.shape(4) == 48, "LTX 0.9.8 VAE output patch geometry mismatch");
    auto value = mx::reshape(input, {b, f, h, w, 3, 4, 4});
    value = mx::transpose(value, {0, 1, 2, 6, 3, 5, 4});
    return mx::reshape(value, {b, f, h * 4, w * 4, 3});
}

Tensor timestep_grid(const Tensor &time) {
    constexpr int half = 128;
    auto frequencies = mx::exp(-std::log(10000.0f) * mx::arange(0, half, mx::float32) / float(half));
    auto angles = mx::expand_dims(mx::astype(time, mx::float32), -1) * frequencies;
    return mx::astype(mx::concatenate({mx::cos(angles), mx::sin(angles)}, -1), mx::bfloat16);
}

} // namespace

void VAEEncoder::load(const std::filesystem::path &checkpoint) {
    require(std::filesystem::is_regular_file(checkpoint), "LTX 0.9.8 VAE checkpoint is missing");
    weights_.clear();
    weights_.load_file(checkpoint, "vae.");
    weights_.erase_prefix("decoder.");
    for (const auto *name : {"encoder.conv_in.conv.weight", "encoder.conv_out.conv.weight",
                              "per_channel_statistics.mean-of-means", "per_channel_statistics.std-of-means"})
        require(weights_.has(name), "LTX 0.9.8 encoder weight is missing: " + std::string(name));
    require(weights_.at("encoder.conv_out.conv.weight").shape() == mx::Shape{129, 2048, 3, 3, 3},
            "checkpoint is not the LTX 0.9.8 VAE encoder geometry");
    prepare_convolutions(weights_);
}

Tensor VAEEncoder::convolution(const Tensor &input, const std::string &prefix) const {
    const int b = input.shape(0), h = input.shape(2), w = input.shape(3), c = input.shape(4);
    auto first = mx::slice(input, {0, 0, 0, 0, 0}, {b, 1, h, w, c});
    auto padded = mx::concatenate({first, first, input}, 1);
    padded = mx::pad(padded, {{0, 0}, {0, 0}, {1, 1}, {1, 1}, {0, 0}}, Tensor(0.f, input.dtype()));
    auto value = mx::conv3d(padded, weights_.at(prefix + ".weight"),
                            std::tuple<int, int, int>{1, 1, 1}, std::tuple<int, int, int>{0, 0, 0});
    return value + mx::reshape(weights_.at(prefix + ".bias"), {1, 1, 1, 1, value.shape(4)});
}

Tensor VAEEncoder::residual_stage(Tensor value, int stage, int count) const {
    for (int block = 0; block < count; ++block) {
        const auto prefix = "encoder.down_blocks." + std::to_string(stage) + ".res_blocks." + std::to_string(block);
        auto branch = convolution(tc::silu(pixel_norm(value)), prefix + ".conv1.conv");
        branch = convolution(tc::silu(pixel_norm(branch)), prefix + ".conv2.conv");
        value = value + branch;
    }
    return value;
}

Tensor VAEEncoder::downsample(Tensor input, int stage, int channels, int temporal, int spatial) const {
    if (temporal == 2) {
        auto first = mx::slice(input, {0, 0, 0, 0, 0}, {input.shape(0), 1, input.shape(2), input.shape(3), input.shape(4)});
        input = mx::concatenate({first, input}, 1);
    }
    auto skip = space_to_depth(input, temporal, spatial);
    const int group = input.shape(4) * temporal * spatial * spatial / channels;
    require(group > 0 && input.shape(4) * temporal * spatial * spatial == channels * group,
            "LTX 0.9.8 VAE residual group mismatch");
    if (group > 1)
        skip = mx::mean(mx::reshape(skip, {skip.shape(0), skip.shape(1), skip.shape(2), skip.shape(3), channels, group}), -1);
    const auto name = "encoder.down_blocks." + std::to_string(stage) + ".conv.conv";
    return space_to_depth(convolution(input, name), temporal, spatial) + skip;
}

Tensor VAEEncoder::encode(const Tensor &pixels) const {
    require(pixels.ndim() == 5 && pixels.shape(0) == 1 && pixels.shape(1) > 0 &&
                (pixels.shape(1) - 1) % 8 == 0 && pixels.shape(2) >= 32 && pixels.shape(2) % 32 == 0 &&
                pixels.shape(3) >= 32 && pixels.shape(3) % 32 == 0 && pixels.shape(4) == 3 && pixels.dtype() == mx::bfloat16,
            "LTX 0.9.8 VAE expects BF16 [1,8n+1,32h,32w,3] pixels");
    auto value = convolution(patchify(pixels), "encoder.conv_in.conv");
    value = downsample(residual_stage(value, 0, 4), 1, 256, 1, 2);
    mx::eval(value);
    value = downsample(residual_stage(value, 2, 6), 3, 512, 2, 1);
    mx::eval(value);
    value = downsample(residual_stage(value, 4, 6), 5, 1024, 2, 2);
    mx::eval(value);
    value = downsample(residual_stage(value, 6, 2), 7, 2048, 2, 2);
    mx::eval(value);
    value = convolution(tc::silu(pixel_norm(residual_stage(value, 8, 2))), "encoder.conv_out.conv");
    value = mx::slice(value, {0, 0, 0, 0, 0}, {value.shape(0), value.shape(1), value.shape(2), value.shape(3), 128});
    return normalize_latents(value);
}

Tensor VAEEncoder::normalize_latents(const Tensor &value) const {
    require(value.ndim() == 5 && value.shape(4) == 128 && value.dtype() == mx::bfloat16,
            "LTX 0.9.8 latent normalization requires BF16 BFHWC");
    auto mean = mx::reshape(weights_.at("per_channel_statistics.mean-of-means"), {1, 1, 1, 1, 128});
    auto stddev = mx::reshape(weights_.at("per_channel_statistics.std-of-means"), {1, 1, 1, 1, 128});
    return (value - mean) / stddev;
}

Tensor VAEEncoder::denormalize_latents(const Tensor &value) const {
    require(value.ndim() == 5 && value.shape(4) == 128 && value.dtype() == mx::bfloat16,
            "LTX 0.9.8 latent denormalization requires BF16 BFHWC");
    auto mean = mx::reshape(weights_.at("per_channel_statistics.mean-of-means"), {1, 1, 1, 1, 128});
    auto stddev = mx::reshape(weights_.at("per_channel_statistics.std-of-means"), {1, 1, 1, 1, 128});
    return value * stddev + mean;
}

void VAEDecoder::load(const std::filesystem::path &checkpoint) {
    require(std::filesystem::is_regular_file(checkpoint), "LTX 0.9.8 VAE checkpoint is missing");
    weights_.clear();
    weights_.load_file(checkpoint, "vae.");
    weights_.erase_prefix("encoder.");
    for (const auto *name : {"decoder.conv_in.conv.weight", "decoder.conv_out.conv.weight",
                              "decoder.timestep_scale_multiplier", "decoder.last_scale_shift_table",
                              "per_channel_statistics.mean-of-means", "per_channel_statistics.std-of-means"})
        require(weights_.has(name), "LTX 0.9.8 decoder weight is missing: " + std::string(name));
    require(weights_.at("decoder.conv_in.conv.weight").shape() == mx::Shape{1024, 128, 3, 3, 3} &&
                weights_.at("decoder.last_scale_shift_table").shape() == mx::Shape{2, 128},
            "checkpoint is not the timestep-conditioned LTX 0.9.8 decoder geometry");
    prepare_convolutions(weights_);
}

Tensor VAEDecoder::convolution(const Tensor &input, const std::string &prefix) const {
    const int h = input.shape(2), w = input.shape(3);
    require(h >= 2 && w >= 2, "LTX 0.9.8 decoder reflect padding needs at least 2x2 latents");
    auto padded = vae_padding::apply(input);
    auto value = mx::conv3d(padded, weights_.at(prefix + ".weight"),
                            std::tuple<int, int, int>{1, 1, 1}, std::tuple<int, int, int>{0, 0, 0});
    return value + mx::reshape(weights_.at(prefix + ".bias"), {1, 1, 1, 1, value.shape(4)});
}

Tensor VAEDecoder::timestep_embedding(const Tensor &time, const std::string &prefix) const {
    auto input = timestep_grid(time);
    auto hidden = weights_.project(input, prefix + ".timestep_embedder.linear_1");
    return weights_.project(tc::silu(hidden), prefix + ".timestep_embedder.linear_2");
}

Tensor VAEDecoder::residual_stage(Tensor value, int stage, const Tensor &time) const {
    const int channels = value.shape(4);
    const auto prefix = "decoder.up_blocks." + std::to_string(stage);
    auto embedded = mx::reshape(timestep_embedding(time, prefix + ".time_embedder"), {4, channels});
    const Tensor one(1.f, mx::bfloat16);
    for (int block = 0; block < 5; ++block) {
        const auto name = prefix + ".res_blocks." + std::to_string(block);
        auto packed = weights_.at(name + ".scale_shift_table") + embedded;
        std::vector<Tensor> params;
        for (int index = 0; index < 4; ++index)
            params.push_back(mx::reshape(mx::slice(packed, {index, 0}, {index + 1, channels}), {1, 1, 1, 1, channels}));
        auto branch = convolution(tc::silu(pixel_norm(value) * (one + params[1]) + params[0]), name + ".conv1.conv");
        branch = convolution(tc::silu(pixel_norm(branch) * (one + params[3]) + params[2]), name + ".conv2.conv");
        value = value + branch;
        // Keep decoder graphs bounded at long video lengths.
        mx::eval(value);
    }
    return value;
}

Tensor VAEDecoder::upsample(const Tensor &input, int stage) const {
    const int channels = input.shape(4);
    require(channels % 8 == 0, "LTX 0.9.8 residual upsample requires channels divisible by 8");
    auto branch = depth_to_space(convolution(input, "decoder.up_blocks." + std::to_string(stage) + ".conv.conv"), channels / 2);
    auto skip = mx::tile(depth_to_space(input, channels / 8), {1, 1, 1, 1, 4});
    return branch + skip;
}

Tensor VAEDecoder::decode(const Tensor &latents, const Tensor &noise, float timestep, float noise_scale) const {
    require(latents.ndim() == 5 && latents.shape(0) == 1 && latents.shape(1) > 0 &&
                latents.shape(2) >= 2 && latents.shape(3) >= 2 && latents.shape(4) == 128 &&
                latents.dtype() == mx::bfloat16 && noise.shape() == latents.shape() && noise.dtype() == latents.dtype() &&
                std::isfinite(timestep) && timestep >= 0 && timestep <= 1 &&
                std::isfinite(noise_scale) && noise_scale >= 0 && noise_scale <= 1,
            "LTX 0.9.8 decoder expects matching BF16 [1,F,H>=2,W>=2,128] latents/noise and valid time/scale");
    const Tensor noise_factor(noise_scale, mx::bfloat16), signal_factor(1.f - noise_scale, mx::bfloat16);
    auto value = noise * noise_factor + latents * signal_factor;
    auto mean = mx::reshape(weights_.at("per_channel_statistics.mean-of-means"), {1, 1, 1, 1, 128});
    auto stddev = mx::reshape(weights_.at("per_channel_statistics.std-of-means"), {1, 1, 1, 1, 128});
    value = convolution(value * stddev + mean, "decoder.conv_in.conv");
    auto scaled_time = mx::reshape(Tensor(timestep, mx::bfloat16) * weights_.at("decoder.timestep_scale_multiplier"), {1});
    value = upsample(residual_stage(value, 0, scaled_time), 1);
    value = upsample(residual_stage(value, 2, scaled_time), 3);
    value = upsample(residual_stage(value, 4, scaled_time), 5);
    value = residual_stage(value, 6, scaled_time);
    auto embedded = mx::reshape(timestep_embedding(scaled_time, "decoder.last_time_embedder"), {2, 128});
    auto params = embedded + weights_.at("decoder.last_scale_shift_table");
    auto shift = mx::reshape(mx::slice(params, {0, 0}, {1, 128}), {1, 1, 1, 1, 128});
    auto scale = mx::reshape(mx::slice(params, {1, 0}, {2, 128}), {1, 1, 1, 1, 128});
    value = convolution(tc::silu(pixel_norm(value) * (Tensor(1.f, mx::bfloat16) + scale) + shift), "decoder.conv_out.conv");
    return unpatchify(value);
}

} // namespace tc::ltx098
