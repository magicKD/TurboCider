#pragma once

#include <mlx/mlx.h>

#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ltx::mlx_vocoder {

namespace mx = mlx::core;

inline void set_error(char *error, size_t error_size, const char *format, ...) {
    if (!error || error_size == 0) return;
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

inline size_t checked_product(std::initializer_list<uint32_t> dimensions,
                              const char *error) {
    size_t result = 1;
    for (uint32_t dimension : dimensions) {
        if (dimension == 0 || result > SIZE_MAX / dimension) {
            throw std::overflow_error(error);
        }
        result *= dimension;
    }
    return result;
}

inline auto load_checkpoint(const char *path, const char *missing_path_error) {
    if (!path || !path[0]) {
        throw std::invalid_argument(missing_path_error);
    }
    const mx::Device gpu(mx::Device::gpu);
    if (!mx::is_available(gpu)) {
        throw std::runtime_error("MLX Metal GPU is not available");
    }
    mx::set_default_device(gpu);
    return mx::load_safetensors(path);
}

inline void dump_stage(const char *directory, const char *name,
                       const mx::array &value, const char *description) {
    if (!directory || !directory[0]) return;
    std::filesystem::create_directories(directory);
    mx::array output = mx::contiguous(mx::astype(value, mx::float32));
    mx::eval(output);
    const auto path = std::filesystem::path(directory) /
        (std::string(name) + ".f32");
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(std::string("cannot create ") + description +
                                 " stage dump: " + path.string());
    }
    stream.write(reinterpret_cast<const char *>(output.data<float>()),
                 static_cast<std::streamsize>(output.nbytes()));
    if (!stream) {
        throw std::runtime_error(std::string("cannot write ") + description +
                                 " stage dump: " + path.string());
    }
}

class VocoderLayers {
public:
    VocoderLayers(const char *prefix, const char *description,
                  bool allow_matrix_weights = false)
        : prefix_(prefix), description_(description),
          allow_matrix_weights_(allow_matrix_weights) {}

    std::vector<mx::array> prepare_weights(
            const std::unordered_map<std::string, mx::array> &source,
            int stages, std::initializer_list<const char *> extra_weights = {},
            size_t additional_arrays = 0) {
        const auto names = weight_names(stages, extra_weights);
        const auto upsample_prefix = prefix_ + ".ups.";
        std::vector<mx::array> materialized;
        materialized.reserve(names.size() + additional_arrays);
        for (const auto &name : names) {
            auto found = source.find(name);
            if (found == source.end()) {
                throw std::runtime_error("missing " + description_ + " tensor: " + name);
            }
            mx::array value = found->second;
            source_weight_bytes_ += value.nbytes();
            if (value.ndim() == 3) {
                if (name.starts_with(upsample_prefix) && name.ends_with(".weight")) {
                    /* PyTorch ConvTranspose1d stores IOK; MLX consumes OKI. */
                    value = mx::transpose(value, {1, 2, 0});
                } else {
                    /* PyTorch Conv1d/filter stores OIK; MLX consumes OKI. */
                    value = mx::transpose(value, {0, 2, 1});
                }
            } else if (value.ndim() != 1 && !(allow_matrix_weights_ && value.ndim() == 2)) {
                throw std::runtime_error("unsupported " + description_ + " tensor rank: " + name);
            }
            value = mx::astype(value, mx::float32);
            resident_weight_bytes_ += value.nbytes();
            materialized.push_back(value);
            weights_.emplace(name, std::move(value));
        }
        return materialized;
    }

    void detach_weights() {
        for (auto &[name, value] : weights_) {
            (void)name;
            value.detach();
        }
    }

    uint32_t weight_tensors() const { return static_cast<uint32_t>(weights_.size()); }
    uint64_t source_weight_bytes() const { return source_weight_bytes_; }
    uint64_t resident_weight_bytes() const { return resident_weight_bytes_; }

    const mx::array &weight(const std::string &name) const {
        auto found = weights_.find(name);
        if (found == weights_.end()) {
            throw std::runtime_error(description_ + " weight lookup failed: " + name);
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

    mx::array upsample_stage(const mx::array &input, int stage,
                             int rate, int kernel) const {
        const std::string prefix = prefix_ + ".ups." + std::to_string(stage);
        const auto value = transposed_convolution(
            input, prefix, rate, (kernel - rate) / 2);
        mx::array sum = amp_block(value, stage * 3);
        sum = sum + amp_block(value, stage * 3 + 1);
        sum = sum + amp_block(value, stage * 3 + 2);
        return sum / mx::array(3.0f);
    }

private:
    static constexpr std::array<int, 3> kResampleKernels = {3, 7, 11};
    static constexpr std::array<int, 3> kDilations = {1, 3, 5};

    std::vector<std::string> weight_names(
            int stages, std::initializer_list<const char *> extra_weights) const {
        std::vector<std::string> names = {
            prefix_ + ".conv_pre.weight",
            prefix_ + ".conv_pre.bias",
            prefix_ + ".conv_post.weight",
        };
        names.insert(names.end(), extra_weights.begin(), extra_weights.end());
        for (int stage = 0; stage < stages; ++stage) {
            const std::string up = prefix_ + ".ups." +
                std::to_string(stage);
            names.push_back(up + ".weight");
            names.push_back(up + ".bias");
        }
        for (int block = 0; block < stages * 3; ++block) {
            const std::string base = prefix_ + ".resblocks." +
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
        names.push_back(prefix_ + ".act_post.act.alpha");
        names.push_back(prefix_ + ".act_post.act.beta");
        names.push_back(prefix_ + ".act_post.upsample.filter");
        names.push_back(prefix_ + ".act_post.downsample.lowpass.filter");
        return names;
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

    mx::array amp_block(mx::array value, int block) const {
        const std::string base = prefix_ + ".resblocks." +
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

    std::string prefix_;
    std::string description_;
    bool allow_matrix_weights_;
    std::unordered_map<std::string, mx::array> weights_;
    uint64_t source_weight_bytes_ = 0;
    uint64_t resident_weight_bytes_ = 0;
};

}  // namespace ltx::mlx_vocoder
