#include "ltx_mlx_vocoder.h"

#include <mlx/mlx.h>

#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mx = mlx::core;

namespace {

constexpr const char *kBase = "vocoder.vocoder";
constexpr std::array<int, 6> kRates = {5, 2, 2, 2, 2, 2};
constexpr std::array<int, 6> kUpsampleKernels = {11, 4, 4, 4, 4, 4};
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
            throw std::overflow_error("vocoder tensor element count overflow");
        }
        result *= dimension;
    }
    return result;
}

class BaseVocoder {
public:
    explicit BaseVocoder(const char *checkpoint_path) {
        if (!checkpoint_path || !checkpoint_path[0]) {
            throw std::invalid_argument("missing vocoder checkpoint path");
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
        materialized.reserve(names.size());
        for (const auto &name : names) {
            auto found = source.find(name);
            if (found == source.end()) {
                throw std::runtime_error("missing base vocoder tensor: " + name);
            }
            mx::array value = found->second;
            source_weight_bytes_ += value.nbytes();
            if (value.ndim() == 3) {
                if (is_transposed_convolution(name)) {
                    /* PyTorch ConvTranspose1d stores IOK; MLX consumes OKI. */
                    value = mx::transpose(value, {1, 2, 0});
                } else {
                    /* PyTorch Conv1d/filter stores OIK; MLX consumes OKI. */
                    value = mx::transpose(value, {0, 2, 1});
                }
            } else if (value.ndim() != 1) {
                throw std::runtime_error(
                    "unsupported base vocoder tensor rank: " + name);
            }
            value = mx::astype(value, mx::float32);
            resident_weight_bytes_ += value.nbytes();
            materialized.push_back(value);
            weights_.emplace(name, std::move(value));
        }
        mx::eval(materialized);
        for (auto &[name, value] : weights_) {
            (void)name;
            value.detach();
        }
        weight_tensors_ = static_cast<uint32_t>(weights_.size());
        if (weight_tensors_ != 667u) {
            throw std::runtime_error("unexpected base vocoder tensor count");
        }
    }

    uint32_t weight_tensors() const { return weight_tensors_; }
    uint64_t source_weight_bytes() const { return source_weight_bytes_; }
    uint64_t resident_weight_bytes() const { return resident_weight_bytes_; }

    void decode(float *output, size_t output_elements,
                const uint16_t *input, size_t input_elements,
                uint32_t batch, uint32_t mel_time,
                const char *dump_directory = nullptr) const {
        if (!output || !input) {
            throw std::invalid_argument("missing base vocoder input/output buffer");
        }
        const uint32_t samples = mel_time * 160u;
        const size_t expected_input = checked_product({batch, 2u, mel_time, 64u});
        const size_t expected_output = checked_product({batch, samples, 2u});
        if (input_elements != expected_input) {
            throw std::invalid_argument("base vocoder input element count mismatch");
        }
        if (output_elements != expected_output) {
            throw std::invalid_argument("base vocoder output element count mismatch");
        }

        const mx::bfloat16_t *typed_input =
            reinterpret_cast<const mx::bfloat16_t *>(input);
        mx::array mel(typed_input,
                      {static_cast<int32_t>(batch), 2,
                       static_cast<int32_t>(mel_time), 64});
        mx::array value = mx::astype(
            mx::reshape(mx::transpose(mel, {0, 2, 1, 3}),
                        {static_cast<int32_t>(batch),
                         static_cast<int32_t>(mel_time), 128}),
            mx::float32);
        dump_stage(dump_directory, "00_mel_btm", value);
        value = convolution(value, std::string(kBase) + ".conv_pre", 1, 3, 1, true);
        dump_stage(dump_directory, "01_conv_pre", value);
        for (int stage = 0; stage < 6; ++stage) {
            const std::string up = std::string(kBase) + ".ups." +
                std::to_string(stage);
            value = transposed_convolution(
                value, up, kRates[stage],
                (kUpsampleKernels[stage] - kRates[stage]) / 2);
            mx::array sum = amp_block(value, stage * 3);
            sum = sum + amp_block(value, stage * 3 + 1);
            sum = sum + amp_block(value, stage * 3 + 2);
            value = sum / mx::array(3.0f);
            dump_stage(dump_directory,
                       ("02_stage_" + std::to_string(stage)).c_str(), value);
        }
        value = activation(value, std::string(kBase) + ".act_post");
        dump_stage(dump_directory, "08_act_post", value);
        value = convolution(value, std::string(kBase) + ".conv_post",
                            1, 3, 1, false);
        value = mx::tanh(value);
        dump_stage(dump_directory, "09_waveform_16k", value);
        value = mx::contiguous(value);
        mx::eval(value);
        if (value.dtype() != mx::float32 || value.size() != output_elements) {
            throw std::runtime_error("base vocoder output shape/dtype mismatch");
        }
        std::memcpy(output, value.data<float>(), output_elements * sizeof(float));
    }

private:
    static bool is_transposed_convolution(const std::string &name) {
        const auto marker = std::string(kBase) + ".ups.";
        return name.starts_with(marker) && name.ends_with(".weight");
    }

    static std::vector<std::string> weight_names() {
        std::vector<std::string> names = {
            std::string(kBase) + ".conv_pre.weight",
            std::string(kBase) + ".conv_pre.bias",
            std::string(kBase) + ".conv_post.weight",
        };
        for (int stage = 0; stage < 6; ++stage) {
            const std::string up = std::string(kBase) + ".ups." +
                std::to_string(stage);
            names.push_back(up + ".weight");
            names.push_back(up + ".bias");
        }
        for (int block = 0; block < 18; ++block) {
            const std::string base = std::string(kBase) + ".resblocks." +
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
        names.push_back(std::string(kBase) + ".act_post.act.alpha");
        names.push_back(std::string(kBase) + ".act_post.act.beta");
        names.push_back(std::string(kBase) + ".act_post.upsample.filter");
        names.push_back(std::string(kBase) + ".act_post.downsample.lowpass.filter");
        return names;
    }

    const mx::array &weight(const std::string &name) const {
        auto found = weights_.find(name);
        if (found == weights_.end()) {
            throw std::runtime_error("base vocoder weight lookup failed: " + name);
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
        mx::array zeros = mx::zeros_like(input);
        mx::array value = mx::reshape(mx::stack({input, zeros}, 2),
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
        value = mx::concatenate({left, value, right}, 1);
        value = mx::conv1d(value, filter);
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
        value = mx::concatenate({left, value, right}, 1);
        value = mx::conv1d(value, filter, 2);
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
        const std::string base = std::string(kBase) + ".resblocks." +
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
            throw std::runtime_error("cannot create vocoder stage dump: " +
                                     path.string());
        }
        stream.write(reinterpret_cast<const char *>(output.data<float>()),
                     static_cast<std::streamsize>(output.nbytes()));
        if (!stream) {
            throw std::runtime_error("cannot write vocoder stage dump: " +
                                     path.string());
        }
    }

    std::unordered_map<std::string, mx::array> weights_;
    uint32_t weight_tensors_ = 0;
    uint64_t source_weight_bytes_ = 0;
    uint64_t resident_weight_bytes_ = 0;
};

}  // namespace

struct ltx_mlx_vocoder {
    std::unique_ptr<BaseVocoder> implementation;
};

extern "C" ltx_mlx_vocoder *ltx_mlx_vocoder_create_base(
        const char *checkpoint_path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    try {
        auto result = std::make_unique<ltx_mlx_vocoder>();
        result->implementation = std::make_unique<BaseVocoder>(checkpoint_path);
        return result.release();
    } catch (const std::exception &exception) {
        set_error(error, error_size, "create MLX base vocoder: %s",
                  exception.what());
        return nullptr;
    }
}

extern "C" void ltx_mlx_vocoder_free(ltx_mlx_vocoder *vocoder) {
    delete vocoder;
}

extern "C" int ltx_mlx_vocoder_get_info(
        const ltx_mlx_vocoder *vocoder, ltx_mlx_vocoder_info *info) {
    if (!vocoder || !vocoder->implementation || !info) return 0;
    info->weight_tensors = vocoder->implementation->weight_tensors();
    info->source_weight_bytes = vocoder->implementation->source_weight_bytes();
    info->resident_weight_bytes = vocoder->implementation->resident_weight_bytes();
    return 1;
}

static int decode_base(ltx_mlx_vocoder *vocoder,
                       float *output, size_t output_elements,
                       const uint16_t *input, size_t input_elements,
                       uint32_t batch, uint32_t mel_time,
                       const char *dump_directory,
                       char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!vocoder || !vocoder->implementation) {
        set_error(error, error_size, "missing MLX base vocoder instance");
        return 0;
    }
    try {
        vocoder->implementation->decode(output, output_elements, input,
                                        input_elements, batch, mel_time,
                                        dump_directory);
        return 1;
    } catch (const std::exception &exception) {
        set_error(error, error_size, "decode MLX base vocoder: %s",
                  exception.what());
        return 0;
    }
}

extern "C" int ltx_mlx_vocoder_decode_base_bf16(
        ltx_mlx_vocoder *vocoder, float *output, size_t output_elements,
        const uint16_t *input, size_t input_elements, uint32_t batch,
        uint32_t mel_time, char *error, size_t error_size) {
    return decode_base(vocoder, output, output_elements, input, input_elements,
                       batch, mel_time, nullptr, error, error_size);
}

extern "C" int ltx_mlx_vocoder_decode_base_bf16_debug(
        ltx_mlx_vocoder *vocoder, float *output, size_t output_elements,
        const uint16_t *input, size_t input_elements, uint32_t batch,
        uint32_t mel_time, const char *dump_directory,
        char *error, size_t error_size) {
    return decode_base(vocoder, output, output_elements, input, input_elements,
                       batch, mel_time, dump_directory, error, error_size);
}

extern "C" void ltx_mlx_vocoder_clear_cache(void) {
    mx::clear_cache();
}
