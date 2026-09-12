#include "audio_vae.hpp"

#include <array>
#include <cmath>

namespace tc::h3_mlx {
namespace {
constexpr std::array<int, 7> decoder_rates{5, 5, 2, 2, 2, 2, 2};
constexpr std::array<int, 7> decoder_kernels{9, 9, 4, 4, 4, 4, 4};
constexpr std::array<int, 3> resblock_kernels{3, 7, 11};
constexpr std::array<int, 3> resblock_dilations{1, 3, 5};

constexpr std::array<float, 32> latent_mean{
    -0.0202116875f, 0.3876466480f, -0.0439827980f, -0.2859151494f,
    0.0817968622f, -0.3578264117f, 0.0406238101f, -0.0155253448f,
    -0.2233624756f, 0.1821006835f, 0.2941778898f, -0.0790116787f,
    -0.0568150729f, -0.3699028194f, -0.3161631525f, 0.5905951262f,
    -0.0521395691f, 0.0136731602f, -0.0369164795f, 0.0973266065f,
    -0.3394662440f, -0.3068567812f, -0.2450459898f, -0.0346985236f,
    0.0286803227f, -0.2121777982f, -0.1678263098f, 0.3221288025f,
    -0.1223055869f, 0.4356604815f, -0.0502599180f, 0.3979258239f,
};
constexpr std::array<float, 32> latent_std{
    1.6895524263f, 2.7626373768f, 1.7945344448f, 1.6801681519f,
    1.6390227079f, 2.7788298130f, 1.7659089565f, 1.6199758053f,
    2.6336526871f, 1.8539357185f, 2.5056498051f, 1.8110191822f,
    1.9579657316f, 1.6685497761f, 1.4922469854f, 3.2986702919f,
    1.9491804838f, 1.8720003366f, 1.8334083557f, 1.6488070488f,
    1.6176958084f, 1.9131449461f, 1.5695245266f, 1.6943659782f,
    1.8318420649f, 1.5540637970f, 1.9344930649f, 1.5991982222f,
    1.7180459499f, 1.6307219267f, 1.8661226034f, 1.5613768101f,
};

Tensor contiguous_transpose(const Tensor &value,
                            const std::vector<int> &axes) {
    return mx::contiguous(mx::transpose(value, axes));
}
} // namespace

void AudioVAE::load(const std::filesystem::path &root, const Event &event,
                    std::atomic<bool> &cancelled) {
    weights_.load(root, {"dec_in_proj.", "decoder."}, mx::float32, false,
                  event, cancelled);
    for (const auto &key : {
             "dec_in_proj.weight", "dec_in_proj.bias",
             "decoder.conv_pre.weight_v", "decoder.conv_pre.weight_g",
             "decoder.conv_pre.bias", "decoder.conv_post.weight_v",
             "decoder.conv_post.weight_g",
             "decoder.activation_post.act.alpha",
             "decoder.activation_post.act.beta",
             "decoder.activation_post.upsample.filter",
             "decoder.activation_post.downsample.lowpass.filter"})
        require(weights_.has(key), "H3 audio VAE is missing tensor: " +
                                    std::string(key));
}

void AudioVAE::unload() { weights_.clear(); }

Tensor AudioVAE::weight_norm(const std::string &prefix) const {
    const auto &value = weights_.at(prefix + ".weight_v");
    const auto &gain = weights_.at(prefix + ".weight_g");
    auto norm = mx::sqrt(mx::sum(value * value, std::vector<int>{1, 2}, true));
    return value * (gain / norm);
}

Tensor AudioVAE::conv1d(const Tensor &input, const Tensor &weight,
                        const std::optional<Tensor> &bias, int stride,
                        int padding, int dilation, int groups) const {
    require(input.ndim() == 3 && weight.ndim() == 3,
            "invalid H3 audio VAE conv1d geometry");
    auto output = mx::conv1d(
        contiguous_transpose(input, {0, 2, 1}),
        contiguous_transpose(weight, {0, 2, 1}), stride, padding, dilation,
        groups);
    if (bias) output = output + *bias;
    return contiguous_transpose(output, {0, 2, 1});
}

Tensor AudioVAE::conv_transpose1d(
    const Tensor &input, const Tensor &weight,
    const std::optional<Tensor> &bias, int stride, int padding) const {
    auto output = mx::conv_transpose1d(
        contiguous_transpose(input, {0, 2, 1}),
        contiguous_transpose(weight, {1, 2, 0}), stride, padding);
    if (bias) output = output + *bias;
    return contiguous_transpose(output, {0, 2, 1});
}

Tensor AudioVAE::replicate_pad(const Tensor &input, int left,
                               int right) const {
    if (!left && !right) return input;
    std::vector<Tensor> pieces;
    if (left)
        pieces.push_back(mx::repeat(slice_axis(input, -1, 0, 1), left, -1));
    pieces.push_back(input);
    if (right)
        pieces.push_back(mx::repeat(
            slice_axis(input, -1, input.shape(-1) - 1, input.shape(-1)),
            right, -1));
    return mx::concatenate(pieces, -1);
}

Tensor AudioVAE::low_pass(const Tensor &input, const Tensor &filter,
                          int stride) const {
    const int channels = input.shape(1);
    const int kernel = filter.size();
    const bool even = kernel % 2 == 0;
    auto padded = replicate_pad(input, kernel / 2 - int(even), kernel / 2);
    auto weight = mx::repeat(mx::reshape(filter, {1, kernel, 1}), channels, 0);
    auto output = mx::conv1d(contiguous_transpose(padded, {0, 2, 1}),
                             weight, stride, 0, 1, channels);
    return contiguous_transpose(output, {0, 2, 1});
}

Tensor AudioVAE::up_sample(const Tensor &input, const Tensor &filter,
                           int ratio) const {
    const int kernel = filter.size();
    const int pad = kernel / ratio - 1;
    const int pad_left = pad * ratio + (kernel - ratio) / 2;
    const int pad_right = pad * ratio + (kernel - ratio + 1) / 2;
    const int batch = input.shape(0), channels = input.shape(1);
    auto padded = replicate_pad(input, pad, pad);
    std::vector<Tensor> parts{mx::expand_dims(padded, -1)};
    for (int index = 1; index < ratio; ++index)
        parts.push_back(mx::zeros(
            {batch, channels, padded.shape(-1), 1}, input.dtype()));
    auto stuffed = mx::reshape(mx::concatenate(parts, -1),
                               {batch, channels, padded.shape(-1) * ratio});
    stuffed = mx::concatenate({
        mx::zeros({batch, channels, kernel - 1}, input.dtype()), stuffed,
        mx::zeros({batch, channels, kernel - 1}, input.dtype())}, -1);
    std::vector<int32_t> reverse_indices(kernel);
    for (int index = 0; index < kernel; ++index)
        reverse_indices[index] = kernel - index - 1;
    auto reversed = mx::take(
        filter, Tensor(reverse_indices.data(), {kernel}, mx::int32), 0);
    auto weight = mx::repeat(mx::reshape(reversed, {1, kernel, 1}),
                             channels, 0);
    auto output = mx::conv1d(contiguous_transpose(stuffed, {0, 2, 1}),
                             weight, 1, 0, 1, channels);
    output = contiguous_transpose(output, {0, 2, 1});
    const int out_length = (padded.shape(-1) - 1) * ratio + kernel;
    output = float(ratio) * slice_axis(output, -1, 0, out_length);
    return slice_axis(output, -1, pad_left, output.shape(-1) - pad_right);
}

Tensor AudioVAE::activation(const Tensor &input,
                            const std::string &prefix) const {
    auto up_filter = mx::reshape(weights_.at(prefix + ".upsample.filter"),
                                 {-1});
    auto down_filter = mx::reshape(
        weights_.at(prefix + ".downsample.lowpass.filter"), {-1});
    auto value = up_sample(input, up_filter, 2);
    auto alpha = mx::reshape(mx::exp(weights_.at(prefix + ".act.alpha")),
                             {1, -1, 1});
    auto beta = mx::reshape(mx::exp(weights_.at(prefix + ".act.beta")),
                            {1, -1, 1});
    auto sine = mx::sin(alpha * value);
    value = value + sine * sine / (beta + 1e-9f);
    return low_pass(value, down_filter, 2);
}

Tensor AudioVAE::amp_block(Tensor value, const std::string &prefix,
                           int kernel,
                           const std::array<int, 3> &dilations) const {
    for (int index = 0; index < int(dilations.size()); ++index) {
        auto residual = activation(
            value, prefix + ".activations." + std::to_string(2 * index));
        const auto conv1 = prefix + ".convs1." + std::to_string(index);
        residual = conv1d(residual, weight_norm(conv1),
                          weights_.at(conv1 + ".bias"), 1,
                          (kernel * dilations[index] - dilations[index]) / 2,
                          dilations[index]);
        residual = activation(
            residual, prefix + ".activations." +
                          std::to_string(2 * index + 1));
        const auto conv2 = prefix + ".convs2." + std::to_string(index);
        residual = conv1d(residual, weight_norm(conv2),
                          weights_.at(conv2 + ".bias"), 1,
                          (kernel - 1) / 2);
        value = value + residual;
    }
    return value;
}

Tensor AudioVAE::denormalize_latents(const Tensor &latents) const {
    require(latents.ndim() == 3 && latents.shape(1) == latent_channels,
            "invalid H3 audio latent geometry");
    auto mean = Tensor(latent_mean.data(), {1, latent_channels, 1},
                       mx::float32);
    auto std = Tensor(latent_std.data(), {1, latent_channels, 1}, mx::float32);
    return latents * std + mean;
}

Tensor AudioVAE::decode(const Tensor &latents, const Event &event,
                        std::atomic<bool> &cancelled) const {
    require(loaded(), "H3 audio VAE is not loaded");
    require(latents.ndim() == 3 && latents.shape(1) == latent_channels,
            "invalid H3 audio VAE decode input");
    auto hidden = conv1d(latents, weights_.at("dec_in_proj.weight"),
                         weights_.at("dec_in_proj.bias"));
    hidden = conv1d(hidden, weight_norm("decoder.conv_pre"),
                    weights_.at("decoder.conv_pre.bias"), 1, 3);
    for (int stage = 0; stage < int(decoder_rates.size()); ++stage) {
        checkpoint(cancelled);
        event("h3_mlx_audio_decode", stage,
              static_cast<int>(decoder_rates.size()));
        const auto up = "decoder.ups." + std::to_string(stage) + ".0";
        hidden = conv_transpose1d(
            hidden, weight_norm(up), weights_.at(up + ".bias"),
            decoder_rates[stage],
            (decoder_kernels[stage] - decoder_rates[stage]) / 2);
        std::optional<Tensor> residual_sum;
        for (int block = 0; block < int(resblock_kernels.size()); ++block) {
            auto output = amp_block(
                hidden,
                "decoder.resblocks." +
                    std::to_string(stage * int(resblock_kernels.size()) + block),
                resblock_kernels[block], resblock_dilations);
            residual_sum = residual_sum ? *residual_sum + output : output;
        }
        require(residual_sum.has_value(),
                "H3 audio VAE decoder has no residual blocks");
        hidden = *residual_sum / float(resblock_kernels.size());
    }
    hidden = activation(hidden, "decoder.activation_post");
    std::optional<Tensor> post_bias;
    if (weights_.has("decoder.conv_post.bias"))
        post_bias = weights_.at("decoder.conv_post.bias");
    hidden = conv1d(hidden, weight_norm("decoder.conv_post"), post_bias, 1,
                    3);
    hidden = mx::clip(hidden, Tensor(-1.f, hidden.dtype()),
                      Tensor(1.f, hidden.dtype()));
    mx::eval(hidden);
    checkpoint(cancelled);
    event("h3_mlx_audio_decode", static_cast<int>(decoder_rates.size()),
          static_cast<int>(decoder_rates.size()));
    return hidden;
}

} // namespace tc::h3_mlx
