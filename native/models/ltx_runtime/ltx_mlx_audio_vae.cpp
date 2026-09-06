#include "ltx_mlx_audio_vae.h"

#include <mlx/mlx.h>

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
            throw std::overflow_error("audio VAE tensor element count overflow");
        }
        result *= dimension;
    }
    return result;
}

class AudioVAE {
public:
    explicit AudioVAE(const char *checkpoint_path) {
        if (!checkpoint_path || !checkpoint_path[0]) {
            throw std::invalid_argument("missing audio VAE checkpoint path");
        }
        const mx::Device gpu(mx::Device::gpu);
        if (!mx::is_available(gpu)) {
            throw std::runtime_error("MLX Metal GPU is not available");
        }
        mx::set_default_device(gpu);
        auto loaded = mx::load_safetensors(checkpoint_path);
        auto &source = loaded.first;

        const std::vector<std::string> names = weight_names();
        std::vector<mx::array> materialized;
        materialized.reserve(names.size());
        for (const auto &name : names) {
            auto found = source.find(name);
            if (found == source.end()) {
                throw std::runtime_error("missing audio VAE tensor: " + name);
            }
            mx::array value = found->second;
            weight_bytes_ += value.nbytes();
            const bool convolution = name.ends_with(".weight") &&
                name.find("per_channel_statistics") == std::string::npos;
            if (convolution) {
                if (value.ndim() != 4) {
                    throw std::runtime_error(
                        "expected rank-4 audio VAE convolution weight: " + name);
                }
                /* Safetensors stores OIHW; MLX conv2d consumes OHWI. */
                value = mx::transpose(value, {0, 2, 3, 1});
            } else if (value.ndim() != 1) {
                throw std::runtime_error(
                    "expected rank-1 audio VAE vector: " + name);
            }
            if (value.dtype() != mx::bfloat16) {
                value = mx::astype(value, mx::bfloat16);
            }
            materialized.push_back(value);
            weights_.emplace(name, std::move(value));
        }
        mx::eval(materialized);
        for (auto &[name, value] : weights_) {
            (void)name;
            value.detach();
        }
        weight_tensors_ = static_cast<uint32_t>(weights_.size());
    }

    uint32_t weight_tensors() const { return weight_tensors_; }
    uint64_t weight_bytes() const { return weight_bytes_; }

    void decode(uint16_t *output, size_t output_elements,
                const uint16_t *input, size_t input_elements,
                uint32_t batch, uint32_t tokens,
                const char *dump_directory = nullptr) const {
        if (!output || !input) {
            throw std::invalid_argument("missing audio VAE input/output buffer");
        }
        const uint32_t mel_time = tokens * 4u - 3u;
        const size_t expected_input = checked_product({batch, tokens, 128u});
        const size_t expected_output = checked_product(
            {batch, 2u, mel_time, 64u});
        if (input_elements != expected_input) {
            throw std::invalid_argument("audio VAE input element count mismatch");
        }
        if (output_elements != expected_output) {
            throw std::invalid_argument("audio VAE output element count mismatch");
        }

        const mx::bfloat16_t *typed_input =
            reinterpret_cast<const mx::bfloat16_t *>(input);
        mx::array latent(typed_input,
                         {static_cast<int32_t>(batch),
                         static_cast<int32_t>(tokens), 128});
        dump_stage(dump_directory, "00_input_blc", latent);
        mx::array value = mx::reshape(latent,
            {static_cast<int32_t>(batch), static_cast<int32_t>(tokens), 8, 16});

        const mx::array mean = mx::reshape(
            weight("audio_vae.per_channel_statistics.mean-of-means"),
            {1, 1, 128});
        const mx::array standard_deviation = mx::reshape(
            weight("audio_vae.per_channel_statistics.std-of-means"),
            {1, 1, 128});
        value = mx::reshape(value, {
            static_cast<int32_t>(batch), static_cast<int32_t>(tokens), 128});
        value = value * standard_deviation + mean;
        dump_stage(dump_directory, "01_denormalized_blc", value);
        value = mx::reshape(value, {
            static_cast<int32_t>(batch), static_cast<int32_t>(tokens), 8, 16});
        value = mx::transpose(value, {0, 1, 3, 2});
        dump_stage(dump_directory, "02_nhwc", value);

        value = convolution(value, "audio_vae.decoder.conv_in.conv", true);
        dump_stage(dump_directory, "03_conv_in", value);
        value = residual_block(value, "audio_vae.decoder.mid.block_1", 512, 512);
        dump_stage(dump_directory, "04_mid_block_1", value);
        value = residual_block(value, "audio_vae.decoder.mid.block_2", 512, 512);
        dump_stage(dump_directory, "05_mid_block_2", value);
        value = up_block(value, 2, 512, 512, 3, true);
        dump_stage(dump_directory, "06_up_2", value);
        value = up_block(value, 1, 512, 256, 3, true);
        dump_stage(dump_directory, "07_up_1", value);
        value = up_block(value, 0, 256, 128, 3, false);
        dump_stage(dump_directory, "08_up_0", value);
        value = silu(pixel_norm(value));
        dump_stage(dump_directory, "09_pre_out", value);
        value = convolution(value,
                            "audio_vae.decoder.conv_out.conv", true);
        dump_stage(dump_directory, "10_conv_out_nhwc", value);
        value = mx::transpose(value, {0, 3, 1, 2});
        dump_stage(dump_directory, "11_output_bctf", value);
        value = mx::contiguous(value);
        if (value.dtype() != mx::bfloat16) value = mx::astype(value, mx::bfloat16);
        mx::eval(value);
        if (value.size() != output_elements) {
            throw std::runtime_error("audio VAE output shape mismatch");
        }
        std::memcpy(output, value.data<uint16_t>(),
                    output_elements * sizeof(uint16_t));
    }

private:
    static std::vector<std::string> weight_names() {
        std::vector<std::string> names = {
            "audio_vae.per_channel_statistics.mean-of-means",
            "audio_vae.per_channel_statistics.std-of-means",
            "audio_vae.decoder.conv_in.conv.weight",
            "audio_vae.decoder.conv_in.conv.bias",
            "audio_vae.decoder.conv_out.conv.weight",
            "audio_vae.decoder.conv_out.conv.bias",
        };
        names.push_back("audio_vae.decoder.mid.block_1.conv1.conv.weight");
        names.push_back("audio_vae.decoder.mid.block_1.conv1.conv.bias");
        names.push_back("audio_vae.decoder.mid.block_1.conv2.conv.weight");
        names.push_back("audio_vae.decoder.mid.block_1.conv2.conv.bias");
        names.push_back("audio_vae.decoder.mid.block_2.conv1.conv.weight");
        names.push_back("audio_vae.decoder.mid.block_2.conv1.conv.bias");
        names.push_back("audio_vae.decoder.mid.block_2.conv2.conv.weight");
        names.push_back("audio_vae.decoder.mid.block_2.conv2.conv.bias");
        const int channels[] = {512, 256, 128};
        const int inputs[] = {512, 512, 256};
        const int indices[] = {2, 1, 0};
        for (int stage = 0; stage < 3; ++stage) {
            for (int block = 0; block < 3; ++block) {
                const std::string prefix = "audio_vae.decoder.up." +
                    std::to_string(indices[stage]) + ".block." +
                    std::to_string(block);
                const int input_channels = block == 0 ? inputs[stage] : channels[stage];
                names.push_back(prefix + ".conv1.conv.weight");
                names.push_back(prefix + ".conv1.conv.bias");
                names.push_back(prefix + ".conv2.conv.weight");
                names.push_back(prefix + ".conv2.conv.bias");
                if (input_channels != channels[stage]) {
                    names.push_back(prefix + ".nin_shortcut.conv.weight");
                    names.push_back(prefix + ".nin_shortcut.conv.bias");
                }
            }
            if (stage < 2) {
                const std::string prefix = "audio_vae.decoder.up." +
                    std::to_string(indices[stage]) + ".upsample.conv.conv";
                names.push_back(prefix + ".weight");
                names.push_back(prefix + ".bias");
            }
        }
        return names;
    }

    const mx::array &weight(const std::string &name) const {
        auto found = weights_.find(name);
        if (found == weights_.end()) {
            throw std::runtime_error("audio VAE weight lookup failed: " + name);
        }
        return found->second;
    }

    static mx::array silu(const mx::array &value) {
        /* Python mlx.nn.silu is a shapeless compiled activation.  Preserve
         * that graph boundary so BF16 fusion and rounding match the oracle. */
        static auto compiled = mx::compile(
            [](const std::vector<mx::array> &inputs) {
                return std::vector<mx::array>{
                    inputs[0] * mx::sigmoid(inputs[0])};
            }, true);
        return compiled({value})[0];
    }

    static mx::array pixel_norm(const mx::array &value) {
        return mx::fast::rms_norm(value, std::nullopt, 1.0e-6f);
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
            throw std::runtime_error("cannot create audio VAE stage dump: " +
                                     path.string());
        }
        stream.write(reinterpret_cast<const char *>(output.data<float>()),
                     static_cast<std::streamsize>(output.nbytes()));
        if (!stream) {
            throw std::runtime_error("cannot write audio VAE stage dump: " +
                                     path.string());
        }
    }

    mx::array convolution(const mx::array &input, const std::string &prefix,
                          bool causal) const {
        const auto &kernel = weight(prefix + ".weight");
        const int32_t kernel_height = kernel.shape(1);
        const int32_t kernel_width = kernel.shape(2);
        const int32_t pad_height = kernel_height / 2;
        const int32_t pad_width = kernel_width / 2;
        mx::array padded = input;
        if (causal) {
            padded = mx::pad(padded,
                {{0, 0}, {kernel_height - 1, 0},
                 {pad_width, pad_width}, {0, 0}},
                mx::array(0.0f, input.dtype()));
        }
        mx::array result = mx::conv2d(
            padded, kernel, {1, 1}, causal ? std::pair<int, int>{0, 0} :
                std::pair<int, int>{pad_height, pad_width});
        return result + mx::reshape(weight(prefix + ".bias"),
                                    {1, 1, 1, result.shape(3)});
    }

    mx::array residual_block(mx::array value, const std::string &prefix,
                             int input_channels, int output_channels) const {
        mx::array residual = value;
        value = convolution(silu(pixel_norm(value)), prefix + ".conv1.conv", true);
        value = convolution(silu(pixel_norm(value)), prefix + ".conv2.conv", true);
        if (input_channels != output_channels) {
            residual = convolution(residual, prefix + ".nin_shortcut.conv", false);
        }
        return value + residual;
    }

    mx::array up_block(mx::array value, int stage, int input_channels,
                       int output_channels, int blocks, bool upsample) const {
        const std::string prefix = "audio_vae.decoder.up." +
            std::to_string(stage);
        for (int block = 0; block < blocks; ++block) {
            value = residual_block(value, prefix + ".block." +
                                   std::to_string(block),
                                   block == 0 ? input_channels : output_channels,
                                   output_channels);
        }
        if (upsample) {
            value = mx::repeat(value, 2, 1);
            value = mx::repeat(value, 2, 2);
            value = convolution(value, prefix + ".upsample.conv.conv", true);
            value = mx::slice(value, {0, 1, 0, 0},
                              {value.shape(0), value.shape(1),
                               value.shape(2), value.shape(3)});
        }
        return value;
    }

    std::unordered_map<std::string, mx::array> weights_;
    uint32_t weight_tensors_ = 0;
    uint64_t weight_bytes_ = 0;
};

}  // namespace

struct ltx_mlx_audio_vae {
    std::unique_ptr<AudioVAE> implementation;
};

extern "C" ltx_mlx_audio_vae *ltx_mlx_audio_vae_create(
    const char *checkpoint_path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    try {
        auto result = std::make_unique<ltx_mlx_audio_vae>();
        result->implementation = std::make_unique<AudioVAE>(checkpoint_path);
        return result.release();
    } catch (const std::exception &exception) {
        set_error(error, error_size, "create MLX audio VAE: %s", exception.what());
        return nullptr;
    }
}

extern "C" void ltx_mlx_audio_vae_free(ltx_mlx_audio_vae *vae) {
    delete vae;
}

extern "C" int ltx_mlx_audio_vae_get_info(
        const ltx_mlx_audio_vae *vae, ltx_mlx_audio_vae_info *info) {
    if (!vae || !vae->implementation || !info) return 0;
    info->weight_tensors = vae->implementation->weight_tensors();
    info->weight_bytes = vae->implementation->weight_bytes();
    return 1;
}

extern "C" int ltx_mlx_audio_vae_decode_bf16(
        ltx_mlx_audio_vae *vae, uint16_t *output, size_t output_elements,
        const uint16_t *input, size_t input_elements, uint32_t batch,
        uint32_t tokens, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!vae || !vae->implementation) {
        set_error(error, error_size, "missing MLX audio VAE instance");
        return 0;
    }
    try {
        vae->implementation->decode(output, output_elements, input,
                                    input_elements, batch, tokens, nullptr);
        return 1;
    } catch (const std::exception &exception) {
        set_error(error, error_size, "decode MLX audio VAE: %s", exception.what());
        return 0;
    }
}

extern "C" int ltx_mlx_audio_vae_decode_bf16_debug(
        ltx_mlx_audio_vae *vae, uint16_t *output, size_t output_elements,
        const uint16_t *input, size_t input_elements, uint32_t batch,
        uint32_t tokens, const char *dump_directory,
        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!vae || !vae->implementation) {
        set_error(error, error_size, "missing MLX audio VAE instance");
        return 0;
    }
    try {
        vae->implementation->decode(output, output_elements, input,
                                    input_elements, batch, tokens,
                                    dump_directory);
        return 1;
    } catch (const std::exception &exception) {
        set_error(error, error_size, "decode MLX audio VAE: %s", exception.what());
        return 0;
    }
}

extern "C" void ltx_mlx_audio_vae_clear_cache(void) {
    mx::clear_cache();
}
