#include "ltx_mlx_upsampler.h"

#include <mlx/mlx.h>
#include <mlx/memory.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mx = mlx::core;

namespace {

void set_error(char *error, size_t error_size, const char *format, ...) {
    if (!error || error_size == 0) return;
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

size_t checked_product(const std::vector<uint32_t> &dimensions) {
    size_t result = 1;
    for (uint32_t dimension : dimensions) {
        if (dimension == 0 || result > SIZE_MAX / dimension) {
            throw std::overflow_error("tensor element count overflow");
        }
        result *= dimension;
    }
    return result;
}

std::vector<std::string> expected_weight_names(void) {
    std::vector<std::string> names = {
        "initial_conv.weight",
        "initial_conv.bias",
        "initial_norm.weight",
        "initial_norm.bias",
    };
    const char *stages[] = {
        "res_blocks", "post_upsample_res_blocks"};
    const char *suffixes[] = {
        "conv1.weight", "conv1.bias",
        "norm1.weight", "norm1.bias",
        "conv2.weight", "conv2.bias",
        "norm2.weight", "norm2.bias",
    };
    for (const char *stage : stages) {
        for (uint32_t block = 0; block < 4; ++block) {
            for (const char *suffix : suffixes) {
                names.push_back(
                    std::string(stage) + "." + std::to_string(block) +
                    "." + suffix);
            }
        }
    }
    names.push_back("upsampler.0.weight");
    names.push_back("upsampler.0.bias");
    names.push_back("final_conv.weight");
    names.push_back("final_conv.bias");
    return names;
}

class Upsampler {
 public:
    explicit Upsampler(const char *checkpoint_path) {
        if (!checkpoint_path || !checkpoint_path[0]) {
            throw std::invalid_argument("missing upsampler checkpoint path");
        }
        const mx::Device gpu(mx::Device::gpu);
        if (!mx::is_available(gpu)) {
            throw std::runtime_error("MLX Metal GPU is not available");
        }
        mx::set_default_device(gpu);

        auto loaded = mx::load_safetensors(checkpoint_path);
        auto &source_weights = loaded.first;
        std::vector<mx::array> materialize;
        const std::vector<std::string> names = expected_weight_names();
        materialize.reserve(names.size());
        for (const std::string &name : names) {
            const auto found = source_weights.find(name);
            if (found == source_weights.end()) {
                throw std::runtime_error("missing upsampler tensor: " + name);
            }
            mx::array value = found->second;
            weight_bytes_ += value.nbytes();
            if (value.ndim() == 5) {
                value = mx::transpose(value, {0, 2, 3, 4, 1});
            } else if (value.ndim() == 4) {
                value = mx::transpose(value, {0, 2, 3, 1});
            } else if (value.ndim() != 1) {
                throw std::runtime_error(
                    "unsupported upsampler tensor rank: " + name);
            }
            if (value.dtype() != mx::bfloat16) {
                value = mx::astype(value, mx::bfloat16);
            }
            materialize.push_back(value);
            weights_.emplace(name, std::move(value));
        }
        mx::eval(materialize);
        for (auto &[name, value] : weights_) {
            (void)name;
            value.detach();
        }

        const mx::array &initial = weight("initial_conv.weight");
        hidden_channels_ = static_cast<uint32_t>(initial.shape(0));
        input_channels_ = static_cast<uint32_t>(initial.shape(4));
        if (input_channels_ != 128 || hidden_channels_ == 0 ||
            hidden_channels_ % 32 != 0) {
            throw std::runtime_error("unsupported upsampler channel geometry");
        }
        weight_tensors_ = static_cast<uint32_t>(weights_.size());
    }

    uint32_t input_channels(void) const { return input_channels_; }
    uint32_t hidden_channels(void) const { return hidden_channels_; }
    uint32_t weight_tensors(void) const { return weight_tensors_; }
    uint64_t weight_bytes(void) const { return weight_bytes_; }

    void run(uint16_t *output,
             size_t output_elements,
             const uint16_t *input,
             size_t input_elements,
             uint32_t batch,
             uint32_t frames,
             uint32_t height,
             uint32_t width) const {
        if (!output || !input) {
            throw std::invalid_argument("missing upsampler input/output buffer");
        }
        const size_t expected_input = checked_product(
            {batch, frames, height, width, input_channels_});
        const size_t expected_output = checked_product(
            {batch, frames, height * 2u, width * 2u, input_channels_});
        if (input_elements != expected_input) {
            throw std::invalid_argument("upsampler input element count mismatch");
        }
        if (output_elements != expected_output) {
            throw std::invalid_argument("upsampler output element count mismatch");
        }

        const mx::bfloat16_t *typed_input =
            reinterpret_cast<const mx::bfloat16_t *>(input);
        mx::array value(
            typed_input,
            {static_cast<int32_t>(batch), static_cast<int32_t>(frames),
             static_cast<int32_t>(height), static_cast<int32_t>(width),
             static_cast<int32_t>(input_channels_)});
        value = silu(group_norm(
            convolution3d(value, "initial_conv"), "initial_norm"));
        for (uint32_t block = 0; block < 4; ++block) {
            value = residual_block(value, "res_blocks", block);
        }
        value = spatial_upsample(value);
        for (uint32_t block = 0; block < 4; ++block) {
            value = residual_block(
                value, "post_upsample_res_blocks", block);
        }
        value = mx::contiguous(convolution3d(value, "final_conv"));
        if (value.dtype() != mx::bfloat16) {
            value = mx::astype(value, mx::bfloat16);
        }
        mx::eval(value);
        if (value.size() != output_elements) {
            throw std::runtime_error("upsampler output shape mismatch");
        }
        std::memcpy(output, value.data<uint16_t>(),
                    output_elements * sizeof(uint16_t));
    }

 private:
    const mx::array &weight(const std::string &name) const {
        const auto found = weights_.find(name);
        if (found == weights_.end()) {
            throw std::runtime_error("upsampler weight lookup failed: " + name);
        }
        return found->second;
    }

    static mx::array silu(const mx::array &value) {
        return value * mx::sigmoid(value);
    }

    mx::array convolution3d(const mx::array &input,
                            const std::string &prefix) const {
        mx::array result = mx::conv3d(
            input, weight(prefix + ".weight"),
            std::tuple<int, int, int>{1, 1, 1},
            std::tuple<int, int, int>{1, 1, 1});
        return result + mx::reshape(
            weight(prefix + ".bias"),
            {1, 1, 1, 1, result.shape(4)});
    }

    mx::array convolution2d(const mx::array &input,
                            const std::string &prefix) const {
        mx::array result = mx::conv2d(
            input, weight(prefix + ".weight"), {1, 1}, {1, 1});
        return result + mx::reshape(
            weight(prefix + ".bias"),
            {1, 1, 1, result.shape(3)});
    }

    mx::array group_norm(const mx::array &input,
                         const std::string &prefix) const {
        const int32_t batch = input.shape(0);
        const int32_t depth = input.shape(1);
        const int32_t height = input.shape(2);
        const int32_t width = input.shape(3);
        const int32_t channels = input.shape(4);
        constexpr int32_t groups = 32;
        const int32_t group_size = channels / groups;
        const int32_t spatial = depth * height * width;
        mx::array value = mx::reshape(
            input, {batch, spatial, groups, group_size});
        value = mx::transpose(value, {0, 2, 1, 3});
        value = mx::reshape(value, {batch, groups, spatial * group_size});
        value = mx::fast::layer_norm(
            value, std::nullopt, std::nullopt, 1.0e-5f);
        value = mx::reshape(
            value, {batch, groups, spatial, group_size});
        value = mx::transpose(value, {0, 2, 1, 3});
        value = mx::reshape(
            value, {batch, depth, height, width, channels});
        return value * mx::reshape(
                           weight(prefix + ".weight"),
                           {1, 1, 1, 1, channels}) +
               mx::reshape(
                   weight(prefix + ".bias"),
                   {1, 1, 1, 1, channels});
    }

    mx::array residual_block(const mx::array &input,
                             const char *stage,
                             uint32_t block) const {
        const std::string base =
            std::string(stage) + "." + std::to_string(block);
        mx::array value = convolution3d(input, base + ".conv1");
        value = silu(group_norm(value, base + ".norm1"));
        value = convolution3d(value, base + ".conv2");
        value = group_norm(value, base + ".norm2");
        return silu(value + input);
    }

    mx::array spatial_upsample(const mx::array &input) const {
        const int32_t batch = input.shape(0);
        const int32_t frames = input.shape(1);
        const int32_t height = input.shape(2);
        const int32_t width = input.shape(3);
        const int32_t channels = input.shape(4);
        mx::array value = mx::reshape(
            input, {batch * frames, height, width, channels});
        value = convolution2d(value, "upsampler.0");
        value = mx::reshape(
            value, {batch, frames, height, width, channels, 2, 2});
        value = mx::transpose(value, {0, 1, 2, 5, 3, 6, 4});
        return mx::reshape(
            value, {batch, frames, height * 2, width * 2, channels});
    }

    std::unordered_map<std::string, mx::array> weights_;
    uint32_t input_channels_ = 0;
    uint32_t hidden_channels_ = 0;
    uint32_t weight_tensors_ = 0;
    uint64_t weight_bytes_ = 0;
};

}  // namespace

struct ltx_mlx_upsampler {
    std::unique_ptr<Upsampler> implementation;
};

extern "C" ltx_mlx_upsampler *ltx_mlx_upsampler_create(
    const char *checkpoint_path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    try {
        auto result = std::make_unique<ltx_mlx_upsampler>();
        result->implementation = std::make_unique<Upsampler>(checkpoint_path);
        return result.release();
    } catch (const std::exception &exception) {
        set_error(error, error_size, "create MLX upsampler: %s",
                  exception.what());
        return nullptr;
    }
}

extern "C" void ltx_mlx_upsampler_free(ltx_mlx_upsampler *upsampler) {
    delete upsampler;
}

extern "C" int ltx_mlx_upsampler_get_info(
    const ltx_mlx_upsampler *upsampler, ltx_mlx_upsampler_info *info) {
    if (!upsampler || !upsampler->implementation || !info) return 0;
    info->input_channels = upsampler->implementation->input_channels();
    info->hidden_channels = upsampler->implementation->hidden_channels();
    info->weight_tensors = upsampler->implementation->weight_tensors();
    info->weight_bytes = upsampler->implementation->weight_bytes();
    return 1;
}

extern "C" int ltx_mlx_upsampler_run_tokens_bf16(
    ltx_mlx_upsampler *upsampler,
    uint16_t *output,
    size_t output_elements,
    const uint16_t *input,
    size_t input_elements,
    uint32_t batch,
    uint32_t frames,
    uint32_t height,
    uint32_t width,
    char *error,
    size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!upsampler || !upsampler->implementation) {
        set_error(error, error_size, "missing MLX upsampler instance");
        return 0;
    }
    try {
        upsampler->implementation->run(
            output, output_elements, input, input_elements,
            batch, frames, height, width);
        return 1;
    } catch (const std::exception &exception) {
        set_error(error, error_size, "run MLX upsampler: %s",
                  exception.what());
        return 0;
    }
}

extern "C" uint64_t ltx_mlx_active_memory_bytes(void) {
    return static_cast<uint64_t>(mx::get_active_memory());
}

extern "C" uint64_t ltx_mlx_cache_memory_bytes(void) {
    return static_cast<uint64_t>(mx::get_cache_memory());
}

extern "C" uint64_t ltx_mlx_peak_memory_bytes(void) {
    return static_cast<uint64_t>(mx::get_peak_memory());
}

extern "C" void ltx_mlx_clear_cache(void) {
    mx::clear_cache();
}
