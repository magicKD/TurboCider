#include "ltx_mlx_bwe.h"
#include "ltx_mlx_vocoder_layers.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace {

using ltx::mlx_vocoder::VocoderLayers;
using ltx::mlx_vocoder::checked_product;
using ltx::mlx_vocoder::dump_stage;
using ltx::mlx_vocoder::load_checkpoint;
using ltx::mlx_vocoder::set_error;

constexpr const char *kBwe = "vocoder.bwe_generator";
constexpr std::array<int, 5> kRates = {6, 5, 2, 2, 2};
constexpr std::array<int, 5> kUpsampleKernels = {12, 11, 4, 4, 4};

class BandwidthExtension {
public:
    explicit BandwidthExtension(const char *checkpoint_path) {
        const auto loaded = load_checkpoint(checkpoint_path, "missing BWE checkpoint path");
        auto materialized = layers_.prepare_weights(loaded.first, kRates.size(), {
            "vocoder.mel_stft.mel_basis",
            "vocoder.mel_stft.stft_fn.forward_basis",
            "vocoder.mel_stft.stft_fn.inverse_basis",
        }, /*additional_arrays=*/1);
        hann_filter_ = make_hann_filter();
        materialized.push_back(hann_filter_);
        mx::eval(materialized);
        layers_.detach_weights();
        hann_filter_.detach();
        if (layers_.weight_tensors() != 560u) {
            throw std::runtime_error("unexpected BWE tensor count");
        }
    }

    const VocoderLayers &layers() const { return layers_; }

    void extend(float *output, size_t output_elements,
                const float *input, size_t input_elements,
                uint32_t batch, uint32_t samples,
                const char *dump_directory = nullptr) const {
        if (!output || !input) {
            throw std::invalid_argument("missing BWE input/output buffer");
        }
        const size_t expected_input = checked_product(
            {batch, samples, 2u}, "BWE tensor element count overflow");
        const size_t expected_output = checked_product(
            {batch, samples * 3u, 2u}, "BWE tensor element count overflow");
        if (input_elements != expected_input) {
            throw std::invalid_argument("BWE input element count mismatch");
        }
        if (output_elements != expected_output) {
            throw std::invalid_argument("BWE output element count mismatch");
        }
        mx::array waveform(input,
                           {static_cast<int32_t>(batch),
                            static_cast<int32_t>(samples), 2});
        waveform = mx::astype(waveform, mx::float32);
        dump_stage(dump_directory, "00_waveform_16k_btc", waveform, "BWE");
        mx::array channels_first = mx::transpose(waveform, {0, 2, 1});
        const uint32_t padded_samples = (samples + 79u) / 80u * 80u;
        if (padded_samples != samples) {
            channels_first = mx::pad(
                channels_first,
                {{0, 0}, {0, 0},
                 {0, static_cast<int>(padded_samples - samples)}});
        }
        mx::array flat = mx::reshape(
            channels_first,
            {static_cast<int32_t>(batch * 2u),
             static_cast<int32_t>(padded_samples)});
        mx::array bwe_mel = mel_stft(flat);
        const int32_t frames = bwe_mel.shape(1);
        dump_stage(dump_directory, "01_bwe_mel_bc", bwe_mel, "BWE");
        bwe_mel = mx::reshape(
            bwe_mel,
            {static_cast<int32_t>(batch), 2, frames, 64});
        bwe_mel = mx::reshape(
            mx::transpose(bwe_mel, {0, 1, 3, 2}),
            {static_cast<int32_t>(batch), 128, frames});
        bwe_mel = mx::transpose(bwe_mel, {0, 2, 1});
        dump_stage(dump_directory, "02_bwe_mel_btm", bwe_mel, "BWE");
        mx::array residual = generator(bwe_mel, dump_directory);
        residual = mx::transpose(residual, {0, 2, 1});
        dump_stage(dump_directory, "08_residual_bct", residual, "BWE");

        auto channel_values = mx::unstack(channels_first, 1);
        std::vector<mx::array> resampled;
        resampled.reserve(channel_values.size());
        for (const auto &channel : channel_values) {
            resampled.push_back(hann_resample(channel, padded_samples));
        }
        mx::array skip = mx::stack(resampled, 1);
        dump_stage(dump_directory, "09_skip_bct", skip, "BWE");
        const int32_t shared = std::min(skip.shape(2), residual.shape(2));
        mx::array value = mx::slice(skip, {0, 0, 0},
                                   {skip.shape(0), skip.shape(1), shared}) +
            mx::slice(residual, {0, 0, 0},
                      {residual.shape(0), residual.shape(1), shared});
        value = mx::clip(value, mx::array(-1.0f), mx::array(1.0f));
        value = mx::slice(value, {0, 0, 0},
                          {value.shape(0), value.shape(1),
                           static_cast<int32_t>(samples * 3u)});
        value = mx::contiguous(mx::transpose(value, {0, 2, 1}));
        dump_stage(dump_directory, "10_waveform_48k_btc", value, "BWE");
        mx::eval(value);
        if (value.dtype() != mx::float32 || value.size() != output_elements) {
            throw std::runtime_error("BWE output shape/dtype mismatch");
        }
        std::memcpy(output, value.data<float>(), output_elements * sizeof(float));
    }

private:
    mx::array generator(mx::array value, const char *dump_directory) const {
        value = layers_.convolution(value, std::string(kBwe) + ".conv_pre", 1, 3, 1, true);
        dump_stage(dump_directory, "03_bwe_conv_pre", value, "BWE");
        for (int stage = 0; stage < 5; ++stage) {
            value = layers_.upsample_stage(
                value, stage, kRates[stage], kUpsampleKernels[stage]);
            dump_stage(dump_directory,
                       ("04_bwe_stage_" + std::to_string(stage)).c_str(),
                       value, "BWE");
        }
        value = layers_.activation(value, std::string(kBwe) + ".act_post");
        return layers_.convolution(
            value, std::string(kBwe) + ".conv_post", 1, 3, 1, false);
    }

    mx::array mel_stft(const mx::array &waveform) const {
        mx::array value = mx::reshape(
            waveform, {waveform.shape(0), waveform.shape(1), 1});
        value = mx::pad(value, {{0, 0}, {432, 0}, {0, 0}});
        value = mx::conv1d(
            value, layers_.weight("vocoder.mel_stft.stft_fn.forward_basis"), 80);
        mx::array real = mx::slice(value, {0, 0, 0},
                                  {value.shape(0), value.shape(1), 257});
        mx::array imaginary = mx::slice(
            value, {0, 0, 257},
            {value.shape(0), value.shape(1), 514});
        mx::array magnitude = mx::sqrt(
            real * real + imaginary * imaginary + mx::array(1.0e-9f));
        mx::array mel = mx::matmul(
            magnitude, mx::transpose(layers_.weight("vocoder.mel_stft.mel_basis")));
        return mx::log(mx::maximum(mel, mx::array(1.0e-5f)));
    }

    static mx::array make_hann_filter() {
        constexpr int ratio = 3;
        constexpr double rolloff = 0.99;
        constexpr double lowpass_width = 6.0;
        constexpr int width = 7;
        constexpr int kernel_size = 43;
        std::vector<float> values(kernel_size);
        for (int index = 0; index < kernel_size; ++index) {
            const double time = (static_cast<double>(index) / ratio - width) *
                rolloff;
            const double clamped = std::max(-lowpass_width,
                                            std::min(lowpass_width, time));
            const double window = std::pow(
                std::cos(clamped * M_PI / lowpass_width / 2.0), 2.0);
            const double sinc = time == 0.0 ? 1.0 :
                std::sin(M_PI * time) / (M_PI * time);
            values[index] = static_cast<float>(
                sinc * window * rolloff / ratio);
        }
        return mx::reshape(mx::array(values.data(), {kernel_size}),
                           {1, kernel_size, 1});
    }

    mx::array hann_resample(const mx::array &input,
                            uint32_t samples) const {
        constexpr int ratio = 3;
        constexpr int pad = 7;
        constexpr int kernel_size = 43;
        mx::array first = mx::repeat(
            mx::slice(input, {0, 0}, {input.shape(0), 1}), pad, 1);
        mx::array last = mx::repeat(
            mx::slice(input, {0, input.shape(1) - 1},
                      {input.shape(0), input.shape(1)}), pad, 1);
        mx::array padded = mx::concatenate({first, input, last}, 1);
        mx::array zero = mx::zeros_like(padded);
        mx::array value = mx::reshape(
            mx::stack({padded, zero, zero}, 2),
            {padded.shape(0), padded.shape(1) * ratio});
        const int32_t inserted = (padded.shape(1) - 1) * ratio + 1;
        value = mx::slice(value, {0, 0}, {value.shape(0), inserted});
        value = mx::reshape(value, {value.shape(0), value.shape(1), 1});
        value = mx::pad(value,
                        {{0, 0}, {kernel_size - 1, kernel_size - 1}, {0, 0}});
        value = mx::squeeze(mx::conv1d(value, hann_filter_), -1) *
            mx::array(static_cast<float>(ratio));
        value = mx::slice(value, {0, 42},
                          {value.shape(0), value.shape(1) - 40});
        return mx::slice(value, {0, 0},
                         {value.shape(0), static_cast<int32_t>(samples * ratio)});
    }

    VocoderLayers layers_{kBwe, "BWE", /*allow_matrix_weights=*/true};
    mx::array hann_filter_{0.0f};
};

}  // namespace

struct ltx_mlx_bwe {
    std::unique_ptr<BandwidthExtension> implementation;
};

extern "C" ltx_mlx_bwe *ltx_mlx_bwe_create(
        const char *checkpoint_path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    try {
        auto result = std::make_unique<ltx_mlx_bwe>();
        result->implementation =
            std::make_unique<BandwidthExtension>(checkpoint_path);
        return result.release();
    } catch (const std::exception &exception) {
        set_error(error, error_size, "create MLX BWE: %s", exception.what());
        return nullptr;
    }
}

extern "C" void ltx_mlx_bwe_free(ltx_mlx_bwe *bwe) {
    delete bwe;
}

extern "C" int ltx_mlx_bwe_get_info(
        const ltx_mlx_bwe *bwe, ltx_mlx_bwe_info *info) {
    if (!bwe || !bwe->implementation || !info) return 0;
    const auto &layers = bwe->implementation->layers();
    info->weight_tensors = layers.weight_tensors();
    info->source_weight_bytes = layers.source_weight_bytes();
    info->resident_weight_bytes = layers.resident_weight_bytes();
    return 1;
}

static int extend(ltx_mlx_bwe *bwe,
                  float *output, size_t output_elements,
                  const float *input, size_t input_elements,
                  uint32_t batch, uint32_t samples,
                  const char *dump_directory,
                  char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!bwe || !bwe->implementation) {
        set_error(error, error_size, "missing MLX BWE instance");
        return 0;
    }
    try {
        bwe->implementation->extend(output, output_elements, input,
                                    input_elements, batch, samples,
                                    dump_directory);
        return 1;
    } catch (const std::exception &exception) {
        set_error(error, error_size, "extend MLX BWE: %s", exception.what());
        return 0;
    }
}

extern "C" int ltx_mlx_bwe_extend_f32(
        ltx_mlx_bwe *bwe, float *output, size_t output_elements,
        const float *input, size_t input_elements, uint32_t batch,
        uint32_t samples, char *error, size_t error_size) {
    return extend(bwe, output, output_elements, input, input_elements,
                  batch, samples, nullptr, error, error_size);
}

extern "C" int ltx_mlx_bwe_extend_f32_debug(
        ltx_mlx_bwe *bwe, float *output, size_t output_elements,
        const float *input, size_t input_elements, uint32_t batch,
        uint32_t samples, const char *dump_directory,
        char *error, size_t error_size) {
    return extend(bwe, output, output_elements, input, input_elements,
                  batch, samples, dump_directory, error, error_size);
}

extern "C" void ltx_mlx_bwe_clear_cache(void) {
    mx::clear_cache();
}
