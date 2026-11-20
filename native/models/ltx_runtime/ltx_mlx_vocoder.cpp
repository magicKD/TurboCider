#include "ltx_mlx_vocoder.h"
#include "ltx_mlx_vocoder_layers.hpp"

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace mx = mlx::core;

namespace {

using ltx::mlx_vocoder::VocoderLayers;
using ltx::mlx_vocoder::checked_product;
using ltx::mlx_vocoder::dump_stage;
using ltx::mlx_vocoder::load_checkpoint;
using ltx::mlx_vocoder::set_error;

constexpr const char *kBase = "vocoder.vocoder";
constexpr std::array<int, 6> kRates = {5, 2, 2, 2, 2, 2};
constexpr std::array<int, 6> kUpsampleKernels = {11, 4, 4, 4, 4, 4};

class BaseVocoder {
public:
    explicit BaseVocoder(const char *checkpoint_path) {
        const auto loaded = load_checkpoint(checkpoint_path, "missing vocoder checkpoint path");
        auto materialized = layers_.prepare_weights(loaded.first, kRates.size());
        mx::eval(materialized);
        layers_.detach_weights();
        if (layers_.weight_tensors() != 667u) {
            throw std::runtime_error("unexpected base vocoder tensor count");
        }
    }

    const VocoderLayers &layers() const { return layers_; }

    void decode(float *output, size_t output_elements,
                const uint16_t *input, size_t input_elements,
                uint32_t batch, uint32_t mel_time,
                const char *dump_directory = nullptr) const {
        if (!output || !input) {
            throw std::invalid_argument("missing base vocoder input/output buffer");
        }
        const uint32_t samples = mel_time * 160u;
        const size_t expected_input = checked_product(
            {batch, 2u, mel_time, 64u}, "vocoder tensor element count overflow");
        const size_t expected_output = checked_product(
            {batch, samples, 2u}, "vocoder tensor element count overflow");
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
        dump_stage(dump_directory, "00_mel_btm", value, "vocoder");
        value = layers_.convolution(value, std::string(kBase) + ".conv_pre", 1, 3, 1, true);
        dump_stage(dump_directory, "01_conv_pre", value, "vocoder");
        for (int stage = 0; stage < 6; ++stage) {
            value = layers_.upsample_stage(
                value, stage, kRates[stage], kUpsampleKernels[stage]);
            dump_stage(dump_directory,
                       ("02_stage_" + std::to_string(stage)).c_str(),
                       value, "vocoder");
        }
        value = layers_.activation(value, std::string(kBase) + ".act_post");
        dump_stage(dump_directory, "08_act_post", value, "vocoder");
        value = layers_.convolution(
            value, std::string(kBase) + ".conv_post", 1, 3, 1, false);
        value = mx::tanh(value);
        dump_stage(dump_directory, "09_waveform_16k", value, "vocoder");
        value = mx::contiguous(value);
        mx::eval(value);
        if (value.dtype() != mx::float32 || value.size() != output_elements) {
            throw std::runtime_error("base vocoder output shape/dtype mismatch");
        }
        std::memcpy(output, value.data<float>(), output_elements * sizeof(float));
    }

private:
    VocoderLayers layers_{kBase, "base vocoder"};
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
    const auto &layers = vocoder->implementation->layers();
    info->weight_tensors = layers.weight_tensors();
    info->source_weight_bytes = layers.source_weight_bytes();
    info->resident_weight_bytes = layers.resident_weight_bytes();
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
