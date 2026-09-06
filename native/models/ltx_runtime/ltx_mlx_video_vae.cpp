#include "ltx_runtime_config.h"
#include "ltx_mlx_video_vae.h"

#include <mlx/mlx.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
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

std::vector<std::string> decoder_weight_names(void) {
    std::vector<std::string> names = {
        "per_channel_statistics.mean-of-means",
        "per_channel_statistics.std-of-means",
        "decoder.conv_in.conv.weight",
        "decoder.conv_in.conv.bias",
        "decoder.up_blocks.1.conv.conv.weight",
        "decoder.up_blocks.1.conv.conv.bias",
        "decoder.up_blocks.3.conv.conv.weight",
        "decoder.up_blocks.3.conv.conv.bias",
        "decoder.up_blocks.5.conv.conv.weight",
        "decoder.up_blocks.5.conv.conv.bias",
        "decoder.up_blocks.7.conv.conv.weight",
        "decoder.up_blocks.7.conv.conv.bias",
        "decoder.conv_out.conv.weight",
        "decoder.conv_out.conv.bias",
    };
    const uint32_t stages[] = {0, 2, 4, 6, 8};
    const uint32_t blocks[] = {2, 2, 4, 6, 4};
    for (size_t stage_index = 0;
         stage_index < sizeof(stages) / sizeof(stages[0]);
         ++stage_index) {
        for (uint32_t block = 0; block < blocks[stage_index]; ++block) {
            for (uint32_t convolution = 1; convolution <= 2; ++convolution) {
                const std::string prefix =
                    "decoder.up_blocks." + std::to_string(stages[stage_index]) +
                    ".res_blocks." + std::to_string(block) +
                    ".conv" + std::to_string(convolution) + ".conv";
                names.push_back(prefix + ".weight");
                names.push_back(prefix + ".bias");
            }
        }
    }
    return names;
}

std::vector<std::string> encoder_weight_names(void) {
    std::vector<std::string> names = {
        "per_channel_statistics.mean-of-means",
        "per_channel_statistics.std-of-means",
        "encoder.conv_in.conv.weight",
        "encoder.conv_in.conv.bias",
        "encoder.down_blocks.1.conv.conv.weight",
        "encoder.down_blocks.1.conv.conv.bias",
        "encoder.down_blocks.3.conv.conv.weight",
        "encoder.down_blocks.3.conv.conv.bias",
        "encoder.down_blocks.5.conv.conv.weight",
        "encoder.down_blocks.5.conv.conv.bias",
        "encoder.down_blocks.7.conv.conv.weight",
        "encoder.down_blocks.7.conv.conv.bias",
        "encoder.conv_out.conv.weight",
        "encoder.conv_out.conv.bias",
    };
    const uint32_t stages[] = {0, 2, 4, 6, 8};
    const uint32_t blocks[] = {4, 6, 4, 2, 2};
    for (size_t stage_index = 0;
         stage_index < sizeof(stages) / sizeof(stages[0]);
         ++stage_index) {
        for (uint32_t block = 0; block < blocks[stage_index]; ++block) {
            for (uint32_t convolution = 1; convolution <= 2; ++convolution) {
                const std::string prefix =
                    "encoder.down_blocks." +
                    std::to_string(stages[stage_index]) +
                    ".res_blocks." + std::to_string(block) +
                    ".conv" + std::to_string(convolution) + ".conv";
                names.push_back(prefix + ".weight");
                names.push_back(prefix + ".bias");
            }
        }
    }
    return names;
}

class VideoVAE {
 public:
    explicit VideoVAE(const char *checkpoint_path, bool encoder) :
        encoder_(encoder) {
        if (!checkpoint_path || !checkpoint_path[0]) {
            throw std::invalid_argument("missing video VAE checkpoint path");
        }
        const mx::Device gpu(mx::Device::gpu);
        if (!mx::is_available(gpu)) {
            throw std::runtime_error("MLX Metal GPU is not available");
        }
        mx::set_default_device(gpu);

        auto loaded = mx::load_safetensors(checkpoint_path);
        auto &source_weights = loaded.first;
        std::vector<mx::array> materialize;
        const std::vector<std::string> names = encoder ?
            encoder_weight_names() : decoder_weight_names();
        materialize.reserve(names.size());
        for (const std::string &name : names) {
            const auto found = source_weights.find(name);
            if (found == source_weights.end()) {
                throw std::runtime_error("missing video VAE tensor: " + name);
            }
            mx::array value = found->second;
            weight_bytes_ += value.nbytes();
            if (name.ends_with(".weight")) {
                if (value.ndim() != 5) {
                    throw std::runtime_error(
                        "expected rank-5 convolution weight: " + name);
                }
                value = mx::transpose(value, {0, 2, 3, 4, 1});
            } else if (value.ndim() != 1) {
                throw std::runtime_error(
                    "expected rank-1 video VAE tensor: " + name);
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
        weight_tensors_ = static_cast<uint32_t>(weights_.size());
    }

    uint32_t weight_tensors(void) const { return weight_tensors_; }
    uint64_t weight_bytes(void) const { return weight_bytes_; }

    void encode(uint16_t *output,
                size_t output_elements,
                const uint16_t *input,
                size_t input_elements,
                uint32_t batch,
                uint32_t frames,
                uint32_t height,
                uint32_t width) const {
        if (!encoder_) {
            throw std::invalid_argument("video VAE instance has no encoder");
        }
        if (!output || !input) {
            throw std::invalid_argument("missing video VAE input/output buffer");
        }
        if (height % 32u || width % 32u) {
            throw std::invalid_argument(
                "video VAE encoder height/width must be divisible by 32");
        }
        const uint32_t latent_frames = (frames + 7u) / 8u;
        const uint32_t latent_height = height / 32u;
        const uint32_t latent_width = width / 32u;
        const size_t expected_input = checked_product(
            {batch, 3u, frames, height, width});
        const size_t expected_output = checked_product(
            {batch, latent_frames, latent_height, latent_width, 128u});
        if (input_elements != expected_input) {
            throw std::invalid_argument("video VAE encoder input element count mismatch");
        }
        if (output_elements != expected_output) {
            throw std::invalid_argument("video VAE encoder output element count mismatch");
        }

        const mx::bfloat16_t *typed_input =
            reinterpret_cast<const mx::bfloat16_t *>(input);
        mx::array value(
            typed_input,
            {static_cast<int32_t>(batch), 3,
             static_cast<int32_t>(frames), static_cast<int32_t>(height),
             static_cast<int32_t>(width)});
        value = mx::transpose(value, {0, 2, 3, 4, 1});
        dump_stage("encode_00_pixels", value);
        value = patchify_spatial(value);
        dump_stage("encode_01_patchify", value);
        value = convolution(value, "encoder.conv_in.conv", true);
        dump_stage("encode_02_conv_in", value);

        value = residual_stage(value, "encoder.down_blocks", 0, 4, true);
        value = space_to_depth_downsample(value, 1, 256, 1, 2, 2);
        value = residual_stage(value, "encoder.down_blocks", 2, 6, true);
        value = space_to_depth_downsample(value, 3, 512, 2, 1, 1);
        value = residual_stage(value, "encoder.down_blocks", 4, 4, true);
        value = space_to_depth_downsample(value, 5, 1024, 2, 2, 2);
        value = residual_stage(value, "encoder.down_blocks", 6, 2, true);
        value = space_to_depth_downsample(value, 7, 1024, 2, 2, 2);
        value = residual_stage(value, "encoder.down_blocks", 8, 2, true);
        dump_stage("encode_03_down_blocks", value);

        value = convolution(
            silu(pixel_norm(value)), "encoder.conv_out.conv", true);
        value = mx::slice(
            value, {0, 0, 0, 0, 0},
            {value.shape(0), value.shape(1), value.shape(2), value.shape(3),
             128});
        const mx::array mean = mx::reshape(
            weight("per_channel_statistics.mean-of-means"),
            {1, 1, 1, 1, 128});
        const mx::array standard_deviation = mx::reshape(
            weight("per_channel_statistics.std-of-means"),
            {1, 1, 1, 1, 128});
        value = (value - mean) / standard_deviation;
        dump_stage("encode_04_normalized", value);
        value = mx::contiguous(value);
        if (value.dtype() != mx::bfloat16) {
            value = mx::astype(value, mx::bfloat16);
        }
        mx::eval(value);
        if (value.size() != output_elements) {
            throw std::runtime_error("video VAE encoder output shape mismatch");
        }
        std::memcpy(output, value.data<uint16_t>(),
                    output_elements * sizeof(uint16_t));
    }

    void decode(uint16_t *output,
                size_t output_elements,
                const uint16_t *input,
                size_t input_elements,
                uint32_t batch,
                uint32_t frames,
                uint32_t height,
                uint32_t width) const {
        if (encoder_) {
            throw std::invalid_argument("video VAE instance has no decoder");
        }
        if (!output || !input) {
            throw std::invalid_argument("missing video VAE input/output buffer");
        }
        const size_t expected_input = checked_product(
            {batch, frames, height, width, 128});
        const uint32_t output_frames = frames * 8u - 7u;
        const uint32_t output_height = height * 32u;
        const uint32_t output_width = width * 32u;
        const size_t expected_output = checked_product(
            {batch, 3, output_frames, output_height, output_width});
        if (input_elements != expected_input) {
            throw std::invalid_argument("video VAE input element count mismatch");
        }
        if (output_elements != expected_output) {
            throw std::invalid_argument("video VAE output element count mismatch");
        }

        const mx::bfloat16_t *typed_input =
            reinterpret_cast<const mx::bfloat16_t *>(input);
        mx::array value(
            typed_input,
            {static_cast<int32_t>(batch), static_cast<int32_t>(frames),
             static_cast<int32_t>(height), static_cast<int32_t>(width), 128});
        dump_stage("00_input", value);
        const mx::array mean = mx::reshape(
            weight("per_channel_statistics.mean-of-means"),
            {1, 1, 1, 1, 128});
        const mx::array standard_deviation = mx::reshape(
            weight("per_channel_statistics.std-of-means"),
            {1, 1, 1, 1, 128});
        value = value * standard_deviation + mean;
        dump_stage("01_denorm", value);
        value = convolution(value, "decoder.conv_in.conv", false);
        dump_stage("02_conv_in", value);

        value = residual_stage(value, 0, 2);
        dump_stage("03_res0", value);
        value = depth_to_space(value, 1, 512, 2, 2);
        dump_stage("04_up1", value);
        value = residual_stage(value, 2, 2);
        dump_stage("05_res2", value);
        value = depth_to_space(value, 3, 512, 2, 2);
        dump_stage("06_up3", value);
        value = residual_stage(value, 4, 4);
        dump_stage("07_res4", value);
        value = depth_to_space(value, 5, 256, 1, 2);
        dump_stage("08_up5", value);
        value = residual_stage(value, 6, 6);
        dump_stage("09_res6", value);
        value = depth_to_space(value, 7, 128, 2, 1);
        dump_stage("10_up7", value);
        value = residual_stage(value, 8, 4);
        dump_stage("11_res8", value);

        value = silu(pixel_norm(value));
        dump_stage("12_pre_out", value);
        value = convolution(value, "decoder.conv_out.conv", false);
        dump_stage("13_conv_out", value);
        value = unpatchify(value);
        dump_stage("14_output", value);
        value = mx::contiguous(value);
        if (value.dtype() != mx::bfloat16) {
            value = mx::astype(value, mx::bfloat16);
        }
        mx::eval(value);
        if (value.size() != output_elements) {
            throw std::runtime_error("video VAE output shape mismatch");
        }
        std::memcpy(output, value.data<uint16_t>(),
                    output_elements * sizeof(uint16_t));
    }

 private:
    const mx::array &weight(const std::string &name) const {
        const auto found = weights_.find(name);
        if (found == weights_.end()) {
            throw std::runtime_error("video VAE weight lookup failed: " + name);
        }
        return found->second;
    }

    static mx::array silu(const mx::array &value) {
        return value * mx::sigmoid(value);
    }

    static mx::array pixel_norm(const mx::array &value) {
        return mx::fast::rms_norm(value, std::nullopt, 1.0e-8f);
    }

    mx::array convolution(const mx::array &input,
                          const std::string &prefix,
                          bool causal) const {
        const int32_t batch = input.shape(0);
        const int32_t frames = input.shape(1);
        const int32_t height = input.shape(2);
        const int32_t width = input.shape(3);
        const int32_t channels = input.shape(4);
        mx::array first = mx::slice(
            input, {0, 0, 0, 0, 0},
            {batch, 1, height, width, channels});
        mx::array padded = causal ?
            mx::concatenate({first, first, input}, 1) :
            mx::concatenate(
                {first, input,
                 mx::slice(input, {0, frames - 1, 0, 0, 0},
                           {batch, frames, height, width, channels})},
                1);
        padded = mx::pad(
            padded,
            {{0, 0}, {0, 0}, {1, 1}, {1, 1}, {0, 0}},
            mx::array(0.0f, input.dtype()));
        mx::array result = mx::conv3d(
            padded, weight(prefix + ".weight"),
            std::tuple<int, int, int>{1, 1, 1},
            std::tuple<int, int, int>{0, 0, 0});
        const int32_t output_channels = result.shape(4);
        return result + mx::reshape(
            weight(prefix + ".bias"),
            {1, 1, 1, 1, output_channels});
    }

    mx::array residual_block(const mx::array &input,
                             const std::string &stages_prefix,
                             uint32_t stage,
                             uint32_t block,
                             bool causal) const {
        const std::string base = stages_prefix + "." + std::to_string(stage) +
            ".res_blocks." + std::to_string(block);
        mx::array value = convolution(
            silu(pixel_norm(input)), base + ".conv1.conv", causal);
        value = convolution(
            silu(pixel_norm(value)), base + ".conv2.conv", causal);
        return value + input;
    }

    mx::array residual_stage(mx::array value,
                             uint32_t stage,
                             uint32_t blocks) const {
        for (uint32_t block = 0; block < blocks; ++block) {
            value = residual_block(
                value, "decoder.up_blocks", stage, block, false);
        }
        return value;
    }

    mx::array residual_stage(mx::array value,
                             const std::string &stages_prefix,
                             uint32_t stage,
                             uint32_t blocks,
                             bool causal) const {
        for (uint32_t block = 0; block < blocks; ++block) {
            value = residual_block(
                value, stages_prefix, stage, block, causal);
        }
        return value;
    }

    mx::array depth_to_space(const mx::array &input,
                             uint32_t block,
                             uint32_t output_channels,
                             uint32_t spatial_factor,
                             uint32_t temporal_factor) const {
        const std::string prefix =
            "decoder.up_blocks." + std::to_string(block) + ".conv.conv";
        mx::array value = convolution(input, prefix, false);
        const int32_t batch = value.shape(0);
        const int32_t frames = value.shape(1);
        const int32_t height = value.shape(2);
        const int32_t width = value.shape(3);
        value = mx::reshape(
            value,
            {batch, frames, height, width,
             static_cast<int32_t>(output_channels),
             static_cast<int32_t>(temporal_factor),
             static_cast<int32_t>(spatial_factor),
             static_cast<int32_t>(spatial_factor)});
        value = mx::transpose(value, {0, 1, 5, 2, 6, 3, 7, 4});
        const int32_t expanded_frames =
            frames * static_cast<int32_t>(temporal_factor);
        const int32_t expanded_height =
            height * static_cast<int32_t>(spatial_factor);
        const int32_t expanded_width =
            width * static_cast<int32_t>(spatial_factor);
        value = mx::reshape(
            value,
            {batch, expanded_frames, expanded_height, expanded_width,
             static_cast<int32_t>(output_channels)});
        if (temporal_factor > 1) {
            value = mx::slice(
                value,
                {0, 1, 0, 0, 0},
                {batch, expanded_frames, expanded_height, expanded_width,
                 static_cast<int32_t>(output_channels)});
        }
        return value;
    }

    static mx::array patchify_spatial(const mx::array &input) {
        const int32_t batch = input.shape(0);
        const int32_t frames = input.shape(1);
        const int32_t height = input.shape(2);
        const int32_t width = input.shape(3);
        const int32_t channels = input.shape(4);
        mx::array value = mx::reshape(
            input, {batch, frames, height / 4, 4, width / 4, 4, channels});
        value = mx::transpose(value, {0, 1, 2, 4, 6, 5, 3});
        return mx::reshape(
            value, {batch, frames, height / 4, width / 4,
                    channels * 16});
    }

    static mx::array space_to_depth(const mx::array &input,
                                    uint32_t temporal_factor,
                                    uint32_t height_factor,
                                    uint32_t width_factor) {
        const int32_t batch = input.shape(0);
        const int32_t frames = input.shape(1);
        const int32_t height = input.shape(2);
        const int32_t width = input.shape(3);
        const int32_t channels = input.shape(4);
        mx::array value = mx::reshape(
            input,
            {batch, frames / static_cast<int32_t>(temporal_factor),
             static_cast<int32_t>(temporal_factor),
             height / static_cast<int32_t>(height_factor),
             static_cast<int32_t>(height_factor),
             width / static_cast<int32_t>(width_factor),
             static_cast<int32_t>(width_factor), channels});
        value = mx::transpose(value, {0, 1, 3, 5, 7, 2, 4, 6});
        return mx::reshape(
            value,
            {batch, frames / static_cast<int32_t>(temporal_factor),
             height / static_cast<int32_t>(height_factor),
             width / static_cast<int32_t>(width_factor),
             channels * static_cast<int32_t>(temporal_factor) *
                 static_cast<int32_t>(height_factor) *
                 static_cast<int32_t>(width_factor)});
    }

    mx::array space_to_depth_downsample(
            mx::array input, uint32_t block, uint32_t output_channels,
            uint32_t temporal_factor, uint32_t height_factor,
            uint32_t width_factor) const {
        if (temporal_factor > 1u) {
            mx::array first = mx::slice(
                input, {0, 0, 0, 0, 0},
                {input.shape(0), 1, input.shape(2), input.shape(3),
                 input.shape(4)});
            input = mx::concatenate({first, input}, 1);
        }
        mx::array skip = space_to_depth(
            input, temporal_factor, height_factor, width_factor);
        const uint32_t group_size =
            static_cast<uint32_t>(input.shape(4)) * temporal_factor *
            height_factor * width_factor / output_channels;
        if (group_size > 1u) {
            skip = mx::reshape(
                skip,
                {skip.shape(0), skip.shape(1), skip.shape(2), skip.shape(3),
                 static_cast<int32_t>(output_channels),
                 static_cast<int32_t>(group_size)});
            skip = mx::mean(skip, -1);
        }
        const std::string prefix =
            "encoder.down_blocks." + std::to_string(block) + ".conv.conv";
        mx::array branch = convolution(input, prefix, true);
        branch = space_to_depth(
            branch, temporal_factor, height_factor, width_factor);
        return branch + skip;
    }

    static mx::array unpatchify(const mx::array &input) {
        const int32_t batch = input.shape(0);
        const int32_t frames = input.shape(1);
        const int32_t height = input.shape(2);
        const int32_t width = input.shape(3);
        mx::array value = mx::reshape(
            input, {batch, frames, height, width, 3, 4, 4});
        value = mx::transpose(value, {0, 4, 1, 2, 6, 3, 5});
        return mx::reshape(
            value, {batch, 3, frames, height * 4, width * 4});
    }

    static void dump_stage(const char *name, const mx::array &value) {
        const char *directory = ltx_runtime_getenv("LTX_MLX_VAE_DUMP_DIR");
        if (!directory || !directory[0]) return;
        std::filesystem::create_directories(directory);
        mx::array materialized = mx::contiguous(mx::astype(value, mx::float32));
        mx::eval(materialized);
        const std::filesystem::path path =
            std::filesystem::path(directory) / (std::string(name) + ".f32");
        std::ofstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("cannot create VAE stage dump: " +
                                     path.string());
        }
        stream.write(
            reinterpret_cast<const char *>(materialized.data<float>()),
            static_cast<std::streamsize>(materialized.nbytes()));
        if (!stream) {
            throw std::runtime_error("cannot write VAE stage dump: " +
                                     path.string());
        }
    }

    std::unordered_map<std::string, mx::array> weights_;
    uint32_t weight_tensors_ = 0;
    uint64_t weight_bytes_ = 0;
    bool encoder_ = false;
};

}  // namespace

struct ltx_mlx_video_vae {
    std::unique_ptr<VideoVAE> implementation;
};

extern "C" ltx_mlx_video_vae *ltx_mlx_video_vae_create(
    const char *checkpoint_path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    try {
        auto result = std::make_unique<ltx_mlx_video_vae>();
        result->implementation =
            std::make_unique<VideoVAE>(checkpoint_path, false);
        return result.release();
    } catch (const std::exception &exception) {
        set_error(error, error_size, "create MLX video VAE: %s",
                  exception.what());
        return nullptr;
    }
}

extern "C" ltx_mlx_video_vae *ltx_mlx_video_vae_create_encoder(
    const char *checkpoint_path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    try {
        auto result = std::make_unique<ltx_mlx_video_vae>();
        result->implementation =
            std::make_unique<VideoVAE>(checkpoint_path, true);
        return result.release();
    } catch (const std::exception &exception) {
        set_error(error, error_size, "create MLX video VAE encoder: %s",
                  exception.what());
        return nullptr;
    }
}

extern "C" void ltx_mlx_video_vae_free(ltx_mlx_video_vae *vae) {
    delete vae;
}

extern "C" int ltx_mlx_video_vae_get_info(
    const ltx_mlx_video_vae *vae, ltx_mlx_video_vae_info *info) {
    if (!vae || !vae->implementation || !info) return 0;
    info->weight_tensors = vae->implementation->weight_tensors();
    info->weight_bytes = vae->implementation->weight_bytes();
    return 1;
}

extern "C" int ltx_mlx_video_vae_decode_tokens_bf16(
    ltx_mlx_video_vae *vae,
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
    if (!vae || !vae->implementation) {
        set_error(error, error_size, "missing MLX video VAE instance");
        return 0;
    }
    try {
        vae->implementation->decode(
            output, output_elements, input, input_elements,
            batch, frames, height, width);
        return 1;
    } catch (const std::exception &exception) {
        set_error(error, error_size, "decode MLX video VAE: %s",
                  exception.what());
        return 0;
    }
}

extern "C" int ltx_mlx_video_vae_encode_pixels_bf16(
    ltx_mlx_video_vae *vae,
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
    if (!vae || !vae->implementation) {
        set_error(error, error_size, "missing MLX video VAE instance");
        return 0;
    }
    try {
        vae->implementation->encode(
            output, output_elements, input, input_elements,
            batch, frames, height, width);
        return 1;
    } catch (const std::exception &exception) {
        set_error(error, error_size, "encode MLX video VAE: %s",
                  exception.what());
        return 0;
    }
}

extern "C" void ltx_mlx_video_vae_clear_cache(void) {
    mx::clear_cache();
}
