#include "ltx_mlx_bwe.h"

#include <mlx/mlx.h>

#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mx = mlx::core;

namespace {

constexpr const char *kBwe = "vocoder.bwe_generator";
constexpr std::array<int, 5> kRates = {6, 5, 2, 2, 2};
constexpr std::array<int, 5> kUpsampleKernels = {12, 11, 4, 4, 4};
constexpr std::array<int, 3> kResampleKernels = {3, 7, 11};
constexpr std::array<int, 3> kDilations = {1, 3, 5};

void set_error(char *error, size_t error_size, const char *format, ...) {
    if (!error || error_size == 0) return;
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

size_t checked_product(std::initializer_list<uint32_t> dimensions) {
    size_t result = 1;
    for (uint32_t dimension : dimensions) {
        if (dimension == 0 || result > SIZE_MAX / dimension) {
            throw std::overflow_error("BWE tensor element count overflow");
        }
        result *= dimension;
    }
    return result;
}

class BandwidthExtension {
public:
    explicit BandwidthExtension(const char *checkpoint_path) {
        if (!checkpoint_path || !checkpoint_path[0]) {
            throw std::invalid_argument("missing BWE checkpoint path");
        }
        const mx::Device gpu(mx::Device::gpu);
        if (!mx::is_available(gpu)) {
            throw std::runtime_error("MLX Metal GPU is not available");
        }
        mx::set_default_device(gpu);
        auto loaded = mx::load_safetensors(checkpoint_path);
        auto &source = loaded.first;
        const auto names = weight_names();
        std::vector<mx::array> materialized;
        materialized.reserve(names.size() + 1);
        for (const auto &name : names) {
            auto found = source.find(name);
            if (found == source.end()) {
                throw std::runtime_error("missing BWE tensor: " + name);
            }
            mx::array value = found->second;
            source_weight_bytes_ += value.nbytes();
            if (value.ndim() == 3) {
                if (is_transposed_convolution(name)) {
                    value = mx::transpose(value, {1, 2, 0});
                } else {
                    value = mx::transpose(value, {0, 2, 1});
                }
            } else if (value.ndim() != 1 && value.ndim() != 2) {
                throw std::runtime_error("unsupported BWE tensor rank: " + name);
            }
            value = mx::astype(value, mx::float32);
            resident_weight_bytes_ += value.nbytes();
            materialized.push_back(value);
            weights_.emplace(name, std::move(value));
        }
        hann_filter_ = make_hann_filter();
        materialized.push_back(hann_filter_);
        mx::eval(materialized);
        for (auto &[name, value] : weights_) {
            (void)name;
            value.detach();
        }
        hann_filter_.detach();
        weight_tensors_ = static_cast<uint32_t>(weights_.size());
        if (weight_tensors_ != 560u) {
            throw std::runtime_error("unexpected BWE tensor count");
        }
    }

    uint32_t weight_tensors() const { return weight_tensors_; }
    uint64_t source_weight_bytes() const { return source_weight_bytes_; }
    uint64_t resident_weight_bytes() const { return resident_weight_bytes_; }

    void extend(float *output, size_t output_elements,
                const float *input, size_t input_elements,
                uint32_t batch, uint32_t samples,
                const char *dump_directory = nullptr) const {
        if (!output || !input) {
            throw std::invalid_argument("missing BWE input/output buffer");
        }
        const size_t expected_input = checked_product({batch, samples, 2u});
        const size_t expected_output = checked_product({batch, samples * 3u, 2u});
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
        dump_stage(dump_directory, "00_waveform_16k_btc", waveform);
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
        dump_stage(dump_directory, "01_bwe_mel_bc", bwe_mel);
        bwe_mel = mx::reshape(
            bwe_mel,
            {static_cast<int32_t>(batch), 2, frames, 64});
        bwe_mel = mx::reshape(
            mx::transpose(bwe_mel, {0, 1, 3, 2}),
            {static_cast<int32_t>(batch), 128, frames});
        bwe_mel = mx::transpose(bwe_mel, {0, 2, 1});
        dump_stage(dump_directory, "02_bwe_mel_btm", bwe_mel);
        mx::array residual = generator(bwe_mel, dump_directory);
        residual = mx::transpose(residual, {0, 2, 1});
        dump_stage(dump_directory, "08_residual_bct", residual);

        auto channel_values = mx::unstack(channels_first, 1);
        std::vector<mx::array> resampled;
        resampled.reserve(channel_values.size());
        for (const auto &channel : channel_values) {
            resampled.push_back(hann_resample(channel, padded_samples));
        }
        mx::array skip = mx::stack(resampled, 1);
        dump_stage(dump_directory, "09_skip_bct", skip);
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
        dump_stage(dump_directory, "10_waveform_48k_btc", value);
        mx::eval(value);
        if (value.dtype() != mx::float32 || value.size() != output_elements) {
            throw std::runtime_error("BWE output shape/dtype mismatch");
        }
        std::memcpy(output, value.data<float>(), output_elements * sizeof(float));
    }

private:
    static bool is_transposed_convolution(const std::string &name) {
        return name.starts_with(std::string(kBwe) + ".ups.") &&
            name.ends_with(".weight");
    }

    static std::vector<std::string> weight_names() {
        std::vector<std::string> names = {
            std::string(kBwe) + ".conv_pre.weight",
            std::string(kBwe) + ".conv_pre.bias",
            std::string(kBwe) + ".conv_post.weight",
            "vocoder.mel_stft.mel_basis",
            "vocoder.mel_stft.stft_fn.forward_basis",
            "vocoder.mel_stft.stft_fn.inverse_basis",
        };
        for (int stage = 0; stage < 5; ++stage) {
            const std::string up = std::string(kBwe) + ".ups." +
                std::to_string(stage);
            names.push_back(up + ".weight");
            names.push_back(up + ".bias");
        }
        for (int block = 0; block < 15; ++block) {
            const std::string base = std::string(kBwe) + ".resblocks." +
                std::to_string(block);
            for (const char *set : {"1", "2"}) {
                for (int layer = 0; layer < 3; ++layer) {
                    const std::string convolution = base + ".convs" + set +
                        "." + std::to_string(layer);
                    names.push_back(convolution + ".weight");
                    names.push_back(convolution + ".bias");
                    const std::string activation = base + ".acts" + set +
                        "." + std::to_string(layer);
                    names.push_back(activation + ".act.alpha");
                    names.push_back(activation + ".act.beta");
                    names.push_back(activation + ".upsample.filter");
                    names.push_back(activation + ".downsample.lowpass.filter");
                }
            }
        }
        names.push_back(std::string(kBwe) + ".act_post.act.alpha");
        names.push_back(std::string(kBwe) + ".act_post.act.beta");
        names.push_back(std::string(kBwe) + ".act_post.upsample.filter");
        names.push_back(std::string(kBwe) + ".act_post.downsample.lowpass.filter");
        return names;
    }

    const mx::array &weight(const std::string &name) const {
        auto found = weights_.find(name);
        if (found == weights_.end()) {
            throw std::runtime_error("BWE weight lookup failed: " + name);
        }
        return found->second;
    }

    mx::array convolution(const mx::array &input, const std::string &prefix,
                          int stride, int padding, int dilation,
                          bool bias) const {
        mx::array value = mx::conv1d(input, weight(prefix + ".weight"),
                                     stride, padding, dilation);
        return bias ? value + mx::reshape(weight(prefix + ".bias"),
                                           {1, 1, value.shape(2)}) : value;
    }

    mx::array transposed_convolution(const mx::array &input,
                                     const std::string &prefix,
                                     int stride, int padding) const {
        mx::array value = mx::conv_transpose1d(
            input, weight(prefix + ".weight"), stride, padding);
        return value + mx::reshape(weight(prefix + ".bias"),
                                   {1, 1, value.shape(2)});
    }

    mx::array upsample_activation(const mx::array &input,
                                  const std::string &prefix) const {
        const int32_t batch = input.shape(0);
        const int32_t time = input.shape(1);
        const int32_t channels = input.shape(2);
        mx::array value = mx::reshape(
            mx::stack({input, mx::zeros_like(input)}, 2),
            {batch, time * 2, channels});
        value = mx::reshape(mx::transpose(value, {0, 2, 1}),
                            {batch * channels, time * 2, 1});
        const auto &filter = weight(prefix + ".filter");
        const int32_t kernel = filter.shape(1);
        const int32_t pad = kernel / 2;
        mx::array left = mx::repeat(
            mx::slice(value, {0, 0, 0}, {batch * channels, 1, 1}),
            pad, 1);
        mx::array right = mx::repeat(
            mx::slice(value, {0, time * 2 - 1, 0},
                      {batch * channels, time * 2, 1}),
            pad - 1, 1);
        value = mx::conv1d(mx::concatenate({left, value, right}, 1), filter);
        return mx::transpose(
            mx::reshape(value, {batch, channels, value.shape(1)}),
            {0, 2, 1}) * mx::array(2.0f);
    }

    mx::array downsample_activation(const mx::array &input,
                                    const std::string &prefix) const {
        const int32_t batch = input.shape(0);
        const int32_t time = input.shape(1);
        const int32_t channels = input.shape(2);
        mx::array value = mx::reshape(mx::transpose(input, {0, 2, 1}),
                                      {batch * channels, time, 1});
        const auto &filter = weight(prefix + ".lowpass.filter");
        const int32_t kernel = filter.shape(1);
        const int32_t even = kernel % 2 == 0 ? 1 : 0;
        const int32_t pad_left = kernel / 2 - even;
        const int32_t pad_right = kernel / 2;
        mx::array left = mx::repeat(
            mx::slice(value, {0, 0, 0}, {batch * channels, 1, 1}),
            pad_left, 1);
        mx::array right = mx::repeat(
            mx::slice(value, {0, time - 1, 0},
                      {batch * channels, time, 1}),
            pad_right, 1);
        value = mx::conv1d(mx::concatenate({left, value, right}, 1),
                           filter, 2);
        return mx::transpose(
            mx::reshape(value, {batch, channels, value.shape(1)}),
            {0, 2, 1});
    }

    mx::array activation(mx::array value, const std::string &prefix) const {
        value = upsample_activation(value, prefix + ".upsample");
        const mx::array alpha = mx::reshape(
            mx::exp(weight(prefix + ".act.alpha")),
            {1, 1, value.shape(2)});
        const mx::array beta = mx::reshape(
            mx::exp(weight(prefix + ".act.beta")),
            {1, 1, value.shape(2)});
        const mx::array sine = mx::sin(alpha * value);
        value = value + (mx::array(1.0f) / (beta + mx::array(1.0e-9f))) *
            sine * sine;
        return downsample_activation(value, prefix + ".downsample");
    }

    mx::array amp_block(mx::array value, int block) const {
        const std::string base = std::string(kBwe) + ".resblocks." +
            std::to_string(block);
        const int kernel = kResampleKernels[block % 3];
        for (int layer = 0; layer < 3; ++layer) {
            const mx::array residual = value;
            value = activation(value, base + ".acts1." + std::to_string(layer));
            const int dilation = kDilations[layer];
            value = convolution(
                value, base + ".convs1." + std::to_string(layer), 1,
                (kernel * dilation - dilation) / 2, dilation, true);
            value = activation(value, base + ".acts2." + std::to_string(layer));
            value = convolution(
                value, base + ".convs2." + std::to_string(layer), 1,
                kernel / 2, 1, true);
            value = value + residual;
        }
        return value;
    }

    mx::array generator(mx::array value, const char *dump_directory) const {
        value = convolution(value, std::string(kBwe) + ".conv_pre", 1, 3, 1, true);
        dump_stage(dump_directory, "03_bwe_conv_pre", value);
        for (int stage = 0; stage < 5; ++stage) {
            const std::string up = std::string(kBwe) + ".ups." +
                std::to_string(stage);
            value = transposed_convolution(
                value, up, kRates[stage],
                (kUpsampleKernels[stage] - kRates[stage]) / 2);
            mx::array sum = amp_block(value, stage * 3);
            sum = sum + amp_block(value, stage * 3 + 1);
            sum = sum + amp_block(value, stage * 3 + 2);
            value = sum / mx::array(3.0f);
            dump_stage(dump_directory,
                       ("04_bwe_stage_" + std::to_string(stage)).c_str(), value);
        }
        value = activation(value, std::string(kBwe) + ".act_post");
        return convolution(value, std::string(kBwe) + ".conv_post",
                           1, 3, 1, false);
    }

    mx::array mel_stft(const mx::array &waveform) const {
        mx::array value = mx::reshape(
            waveform, {waveform.shape(0), waveform.shape(1), 1});
        value = mx::pad(value, {{0, 0}, {432, 0}, {0, 0}});
        value = mx::conv1d(
            value, weight("vocoder.mel_stft.stft_fn.forward_basis"), 80);
        mx::array real = mx::slice(value, {0, 0, 0},
                                  {value.shape(0), value.shape(1), 257});
        mx::array imaginary = mx::slice(
            value, {0, 0, 257},
            {value.shape(0), value.shape(1), 514});
        mx::array magnitude = mx::sqrt(
            real * real + imaginary * imaginary + mx::array(1.0e-9f));
        mx::array mel = mx::matmul(
            magnitude, mx::transpose(weight("vocoder.mel_stft.mel_basis")));
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

    static void dump_stage(const char *directory, const char *name,
                           const mx::array &value) {
        if (!directory || !directory[0]) return;
        std::filesystem::create_directories(directory);
        mx::array output = mx::contiguous(mx::astype(value, mx::float32));
        mx::eval(output);
        const auto path = std::filesystem::path(directory) /
            (std::string(name) + ".f32");
        std::ofstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("cannot create BWE stage dump: " +
                                     path.string());
        }
        stream.write(reinterpret_cast<const char *>(output.data<float>()),
                     static_cast<std::streamsize>(output.nbytes()));
        if (!stream) {
            throw std::runtime_error("cannot write BWE stage dump: " +
                                     path.string());
        }
    }

    std::unordered_map<std::string, mx::array> weights_;
    mx::array hann_filter_{0.0f};
    uint32_t weight_tensors_ = 0;
    uint64_t source_weight_bytes_ = 0;
    uint64_t resident_weight_bytes_ = 0;
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
    info->weight_tensors = bwe->implementation->weight_tensors();
    info->source_weight_bytes = bwe->implementation->source_weight_bytes();
    info->resident_weight_bytes = bwe->implementation->resident_weight_bytes();
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
