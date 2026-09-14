#include "native.hpp"

#include "model.hpp"
#include "../../backends/mlx.hpp"
#include "../ltx_runtime/ltx.h"
#include "../ltx_runtime/ltx_mlx_upsampler.h"
#include "../ltx_runtime/ltx_rng.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

using tc::Tensor;

uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>(bits >> 16u);
}

float bf16_to_f32(uint16_t value) {
    const uint32_t bits = uint32_t(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

Tensor host_bf16(const uint16_t *values, size_t elements,
                 const tc::mx::Shape &shape) {
    tc::require(values != nullptr, "LTX MLX host tensor is null");
    size_t expected = 1;
    for (auto dimension : shape)
        expected *= static_cast<size_t>(dimension);
    tc::require(expected == elements, "LTX MLX host tensor size mismatch");
    std::vector<float> converted(elements);
    for (size_t index = 0; index < elements; ++index)
        converted[index] = bf16_to_f32(values[index]);
    return tc::mx::astype(
        Tensor(converted.data(), shape, tc::mx::float32), tc::mx::bfloat16);
}

void write_bf16(const Tensor &value, uint16_t *output, size_t elements) {
    tc::require(output != nullptr, "LTX MLX output tensor is null");
    auto f32 = tc::mx::astype(value, tc::mx::float32);
    tc::mx::eval(f32);
    tc::require(f32.size() == elements, "LTX MLX output size mismatch");
    const auto *values = f32.data<float>();
    for (size_t index = 0; index < elements; ++index)
        output[index] = f32_to_bf16(values[index]);
}

Tensor make_f32(const std::vector<float> &values, const tc::mx::Shape &shape) {
    return Tensor(values.data(), shape, tc::mx::float32);
}

/* Optional step-level fixtures for numerical parity debugging.  This is
 * deliberately opt-in: the normal denoiser never performs host reads or
 * filesystem I/O inside the schedule.  The files contain the exact BF16
 * state after each scheduler boundary, matching the ABI buffers consumed by
 * the C/Metal implementation. */
class StepDump {
    std::filesystem::path directory_;
    int stage_ = 0;

    void write(const char *kind, size_t step, const Tensor &value) const {
        if (directory_.empty()) return;
        auto f32 = tc::mx::astype(value, tc::mx::float32);
        tc::mx::eval(f32);
        const auto path = directory_ /
            ("stage" + std::to_string(stage_) + "_step" +
             (step < 10 ? "0" : "") + std::to_string(step) + "_" + kind +
             ".bf16");
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        tc::require(output.good(), "cannot open LTX MLX step dump: " +
                    path.string());
        const auto *values = f32.data<float>();
        std::vector<uint16_t> encoded(f32.size());
        for (size_t i = 0; i < encoded.size(); ++i)
            encoded[i] = f32_to_bf16(values[i]);
        output.write(reinterpret_cast<const char *>(encoded.data()),
                     static_cast<std::streamsize>(encoded.size() *
                                                  sizeof(uint16_t)));
        tc::require(output.good(), "cannot write LTX MLX step dump: " +
                    path.string());
    }

  public:
    StepDump(int stage, const ltx_workload &workload) : stage_(stage) {
        const char *configured = std::getenv("TURBOCIDER_LTX_MLX_DUMP_STEPS");
        if (!configured || !*configured) return;
        directory_ = std::filesystem::path(configured);
        std::filesystem::create_directories(directory_);
        const auto metadata = directory_ / "metadata.txt";
        if (!std::filesystem::exists(metadata)) {
            std::ofstream output(metadata, std::ios::trunc);
            if (output.good())
                output << "stage=" << stage_ << "\n"
                       << "video_rows=" << workload.stage1_video_tokens << ","
                       << workload.stage2_video_tokens << "\n"
                       << "audio_rows=" << workload.audio_tokens << "\n";
        }
    }

    bool enabled() const { return !directory_.empty(); }
    void initial(const Tensor &video, const Tensor &audio) const {
        write("video", 0, video);
        write("audio", 0, audio);
    }
    void state(size_t step, const Tensor &video, const Tensor &audio) const {
        write("video", step, video);
        write("audio", step, audio);
    }
};

bool report(ltx_native_progress progress, void *opaque,
            const char *phase, int current, int total,
            char *error, size_t error_size) {
    if (!progress)
        return true;
    // ltx_native_progress returns 0 to continue and non-zero to cancel.
    // Keep the wrapper's contract in the same polarity: true means continue,
    // false means the request was cancelled.  The previous inverted check
    // made a cancelled MLX request continue and made a successful callback
    // look like a cancellation.
    if (progress(phase, current, total, opaque) == 0)
        return true;
    if (error && error_size)
        std::snprintf(error, error_size, "generation cancelled");
    return false;
}

struct AncestralScales {
    float ratio = 0.0f;
    float signal_scale = 1.0f;
    float noise_scale = 0.0f;
};

AncestralScales ancestral_scales(float sigma, float sigma_next) {
    const double sigma_d = sigma;
    const double sigma_next_d = sigma_next;
    const double sigma_down = sigma_next_d *
        (1.0 + (sigma_next_d / sigma_d - 1.0));
    const double ratio = sigma_down / sigma_d;
    const double alpha_next = 1.0 - sigma_next_d;
    const double alpha_down = 1.0 - sigma_down;
    const double signal_scale = alpha_next / alpha_down;
    const double variance = sigma_next_d * sigma_next_d -
        sigma_down * sigma_down * alpha_next * alpha_next /
            (alpha_down * alpha_down);
    return {static_cast<float>(ratio), static_cast<float>(signal_scale),
            static_cast<float>(std::sqrt(std::max(variance, 0.0)))};
}

} // namespace

struct ltx_mlx_denoiser {
    ltx_workload workload{};
    tc::ltx_mlx::Transformer transformer;
    uint32_t cache_capacity = 48;
    double last_run_seconds = 0.0;
};

extern "C" ltx_mlx_denoiser *ltx_mlx_create(
    const ltx_mlx_options *options, ltx_native_progress progress,
    void *opaque, char *error, size_t error_size) {
    if (!options || !options->checkpoint || options->fps != 24u) {
        std::snprintf(error, error_size,
                      "LTX MLX requires checkpoint and 24 fps");
        return nullptr;
    }
    try {
        tc::configure_streams();
        auto result = std::make_unique<ltx_mlx_denoiser>();
        if (!ltx_workload_init(&result->workload, options->width,
                               options->height, options->frames, options->fps,
                               error, error_size))
            return nullptr;
        tc::ltx_mlx::TransformerRunOptions run_options;
        run_options.block_cache_capacity = options->block_cache_capacity ?
            options->block_cache_capacity : 48u;
        run_options.convrot_group_size =
            options->convrot_group_size == 32u ||
                    options->convrot_group_size == 64u ||
                    options->convrot_group_size == 128u
                ? static_cast<int>(options->convrot_group_size)
                : 64;
        run_options.block_count = 48;
        run_options.force_eval_each_block = options->force_eval_each_block != 0;
        run_options.av_ca_timestep_scale_multiplier =
            options->av_ca_timestep_scale_multiplier > 0.0f ?
                options->av_ca_timestep_scale_multiplier : 1.0f;
        if (!report(progress, opaque, "ltx_mlx_load", 0, 1, error, error_size))
            return nullptr;
        result->transformer.load(options->checkpoint, run_options);
        result->cache_capacity = run_options.block_cache_capacity;
        if (!report(progress, opaque, "ltx_mlx_load", 1, 1, error, error_size))
            return nullptr;
        return result.release();
    } catch (const std::exception &exception) {
        if (error && error_size)
            std::snprintf(error, error_size, "%s", exception.what());
        return nullptr;
    }
}

extern "C" void ltx_mlx_free(ltx_mlx_denoiser *ctx) {
    delete ctx;
}

extern "C" int ltx_mlx_get_info(const ltx_mlx_denoiser *ctx,
                                 ltx_mlx_info *info) {
    if (!ctx || !info) return 0;
    const auto &metrics = ctx->transformer.cache_metrics();
    *info = {};
    info->block_count = 48;
    info->block_cache_capacity = ctx->cache_capacity;
    info->pinned_block_count =
        static_cast<uint32_t>(ctx->transformer.pinned_block_count());
    info->refill_slot_count =
        static_cast<uint32_t>(ctx->transformer.refill_slot_count());
    info->top_weight_bytes = ctx->transformer.top_weight_bytes();
    info->cache_loads = metrics.loads;
    info->cache_hits = metrics.hits;
    info->cache_evictions = metrics.evictions;
    info->cache_load_bytes = metrics.loaded_bytes;
    info->slot_allocations = metrics.slot_allocations;
    info->slot_refills = metrics.slot_refills;
    info->cache_load_seconds = metrics.load_seconds;
    info->resident_bytes = metrics.resident_bytes;
    info->peak_resident_bytes = metrics.peak_resident_bytes;
    info->last_run_seconds = ctx->last_run_seconds;
    return 1;
}

extern "C" int ltx_mlx_run(
    ltx_mlx_denoiser *ctx, int stage, uint64_t seed,
    uint16_t *video, size_t video_elements,
    uint16_t *audio, size_t audio_elements,
    const uint16_t *video_text, const uint16_t *audio_text,
    const uint16_t *mask, uint32_t text_rows,
    const uint16_t *first_frame, float strength,
    ltx_native_progress progress, void *opaque,
    char *error, size_t error_size) {
    if (!ctx || (stage != 1 && stage != 2) || !video || !audio ||
        !video_text || !audio_text || !text_rows || text_rows > 4096u ||
        first_frame || strength != 0.0f) {
        std::snprintf(error, error_size,
                      "invalid LTX MLX stage input; image conditioning is not enabled yet");
        return 0;
    }
    try {
        const uint32_t video_rows = stage == 1 ?
            static_cast<uint32_t>(ctx->workload.stage1_video_tokens) :
            static_cast<uint32_t>(ctx->workload.stage2_video_tokens);
        const uint32_t audio_rows = ctx->workload.audio_tokens;
        if (video_elements != static_cast<size_t>(video_rows) * 128u ||
            audio_elements != static_cast<size_t>(audio_rows) * 128u) {
            std::snprintf(error, error_size, "LTX MLX latent tensor size mismatch");
            return 0;
        }
        const size_t video_positions_count = static_cast<size_t>(video_rows) * 3u;
        const size_t audio_positions_count = static_cast<size_t>(audio_rows);
        std::vector<float> video_positions(video_positions_count);
        std::vector<float> audio_positions(audio_positions_count);
        if (!ltx_compute_video_positions(
                video_positions.data(), video_positions.size(),
                ctx->workload.latent_frames,
                stage == 1 ? ctx->workload.stage1_latent_height :
                             ctx->workload.stage2_latent_height,
                stage == 1 ? ctx->workload.stage1_latent_width :
                             ctx->workload.stage2_latent_width,
                static_cast<float>(ctx->workload.fps), error, error_size) ||
            !ltx_compute_audio_positions(
                audio_positions.data(), audio_positions.size(), audio_rows,
                error, error_size))
            return 0;

        auto video_state = host_bf16(video, video_elements,
                                     {1, static_cast<int>(video_rows), 128});
        auto audio_state = host_bf16(audio, audio_elements,
                                     {1, static_cast<int>(audio_rows), 128});
        auto video_text_state = host_bf16(
            video_text, static_cast<size_t>(text_rows) * 4096u,
            {1, static_cast<int>(text_rows), 4096});
        auto audio_text_state = host_bf16(
            audio_text, static_cast<size_t>(text_rows) * 2048u,
            {1, static_cast<int>(text_rows), 2048});
        std::optional<Tensor> text_mask;
        if (mask) {
            text_mask = host_bf16(mask, text_rows,
                                  {1, 1, 1, static_cast<int>(text_rows)});
        }
        auto video_position_state = make_f32(video_positions,
                                             {1, static_cast<int>(video_rows), 3});
        auto audio_position_state = make_f32(audio_positions,
                                             {1, static_cast<int>(audio_rows), 1});

        ltx_rng video_rng{}, audio_rng{};
        ltx_rng_seed(&video_rng, seed + (stage == 1 ? 10000u : 2u), 0u);
        ltx_rng_seed(&audio_rng, seed + 2u, 0u);
        size_t sigma_count = 0;
        const float *sigmas = stage == 1 ?
            ltx_distilled_stage1_sigmas(&sigma_count) :
            ltx_distilled_stage2_sigmas(&sigma_count);
        tc::require(sigma_count >= 2u, "LTX MLX schedule is empty");
        if (stage == 2) {
            // Match run_denoise_schedule(): Stage 1 consumes the caller's
            // already seeded BF16 noise directly. Stage 2 renoises the
            // normalized upsampled latent at sigma[0] before deterministic
            // Euler. Round both the generated noise and resulting state at
            // the same BF16 boundary as C/Metal.
            std::vector<float> video_noise(video_elements);
            std::vector<float> audio_noise(audio_elements);
            ltx_rng_fill_normal_f32(&video_rng, video_noise.data(), video_elements);
            ltx_rng_fill_normal_f32(&audio_rng, audio_noise.data(), audio_elements);
            auto video_noise_tensor = tc::mx::astype(make_f32(
                video_noise, {1, static_cast<int>(video_rows), 128}),
                tc::mx::bfloat16);
            auto audio_noise_tensor = tc::mx::astype(make_f32(
                audio_noise, {1, static_cast<int>(audio_rows), 128}),
                tc::mx::bfloat16);
            const float sigma0 = sigmas[0];
            video_state = tc::mx::astype(
                video_state * (1.0f - sigma0) + video_noise_tensor * sigma0,
                tc::mx::bfloat16);
            audio_state = tc::mx::astype(
                audio_state * (1.0f - sigma0) + audio_noise_tensor * sigma0,
                tc::mx::bfloat16);
        }
        tc::mx::eval(video_state, audio_state);
        StepDump step_dump(stage, ctx->workload);
        if (step_dump.enabled())
            step_dump.initial(video_state, audio_state);
        const auto started = std::chrono::steady_clock::now();
        for (size_t step = 0; step + 1u < sigma_count; ++step) {
            if (!report(progress, opaque, "ltx_mlx_step",
                        static_cast<int>(step), static_cast<int>(sigma_count - 1u),
                        error, error_size))
                return 0;
            const float sigma = sigmas[step];
            const float sigma_next = sigmas[step + 1u];
            auto velocity = ctx->transformer.forward(
                video_state, audio_state, sigma,
                video_text_state, audio_text_state,
                video_position_state, audio_position_state,
                {}, {}, text_mask);
            if (stage == 1) {
                std::vector<float> video_noise(video_elements);
                std::vector<float> audio_noise(audio_elements);
                ltx_rng_fill_normal_f32(&video_rng, video_noise.data(), video_elements);
                // The C Stage-1 ABI intentionally passes video_rng for both
                // ancestral noise tensors; preserve that exact stream.
                ltx_rng_fill_normal_f32(&video_rng, audio_noise.data(), audio_elements);
                const auto scales = ancestral_scales(sigma, sigma_next);
                auto video_x0 = video_state + velocity.first * (-sigma);
                auto audio_x0 = audio_state + velocity.second * (-sigma);
                auto video_deterministic = video_state * scales.ratio +
                    video_x0 * (1.0f - scales.ratio);
                auto audio_deterministic = audio_state * scales.ratio +
                    audio_x0 * (1.0f - scales.ratio);
                video_state = video_deterministic * scales.signal_scale +
                    make_f32(video_noise, {1, static_cast<int>(video_rows), 128}) *
                        scales.noise_scale;
                audio_state = audio_deterministic * scales.signal_scale +
                    make_f32(audio_noise, {1, static_cast<int>(audio_rows), 128}) *
                        scales.noise_scale;
            } else {
                const float dt = sigma_next - sigma;
                video_state = tc::euler_step(video_state, velocity.first, dt);
                audio_state = tc::euler_step(audio_state, velocity.second, dt);
            }
            tc::mx::eval(video_state, audio_state);
            if (step_dump.enabled())
                step_dump.state(step + 1u, video_state, audio_state);
        }
        write_bf16(video_state, video, video_elements);
        write_bf16(audio_state, audio, audio_elements);
        ctx->last_run_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        report(progress, opaque, "ltx_mlx_step",
               static_cast<int>(sigma_count - 1u),
               static_cast<int>(sigma_count - 1u), error, error_size);
        return true;
    } catch (const std::exception &exception) {
        if (error && error_size)
            std::snprintf(error, error_size, "%s", exception.what());
        return 0;
    }
}

extern "C" int ltx_mlx_upsample_stage2(
    uint16_t *output, size_t output_elements,
    const uint16_t *input, size_t input_elements,
    const char *upsampler_checkpoint, const char *video_vae_checkpoint,
    uint32_t frames, uint32_t height, uint32_t width,
    char *error, size_t error_size) {
    if (!output || !input || !upsampler_checkpoint || !video_vae_checkpoint ||
        !frames || !height || !width) {
        std::snprintf(error, error_size, "invalid LTX MLX upsample arguments");
        return 0;
    }
    try {
        tc::configure_streams();
        auto loaded = tc::mx::load_safetensors(video_vae_checkpoint);
        const auto find_stat = [&](const char *first, const char *second,
                                   const char *third) -> Tensor {
            for (const char *name : {first, second, third}) {
                auto found = loaded.first.find(name);
                if (found != loaded.first.end())
                    return found->second;
            }
            throw std::runtime_error("video VAE latent statistics are missing");
        };
        auto mean = find_stat("per_channel_statistics.mean-of-means",
                              "per_channel_statistics._mean_of_means",
                              "vae_encoder.per_channel_statistics._mean_of_means");
        auto standard_deviation = find_stat(
            "per_channel_statistics.std-of-means",
            "per_channel_statistics._std_of_means",
            "vae_encoder.per_channel_statistics._std_of_means");
        mean = tc::mx::astype(mean, tc::mx::bfloat16);
        standard_deviation = tc::mx::astype(standard_deviation, tc::mx::bfloat16);
        const size_t input_expected = static_cast<size_t>(frames) * height * width * 128u;
        const size_t output_expected = static_cast<size_t>(frames) * (height * 2u) *
            (width * 2u) * 128u;
        tc::require(input_elements == input_expected && output_elements == output_expected,
                    "LTX MLX upsample tensor geometry mismatch");
        auto normalized = Tensor(
            reinterpret_cast<const tc::mx::bfloat16_t *>(input),
            {1, static_cast<int>(frames), static_cast<int>(height),
             static_cast<int>(width), 128});
        auto denormalized = normalized * standard_deviation + mean;
        tc::mx::eval(denormalized);
        std::vector<uint16_t> denormalized_host(input_elements);
        std::memcpy(denormalized_host.data(),
                    denormalized.data<tc::mx::bfloat16_t>(),
                    input_elements * sizeof(uint16_t));
        std::vector<uint16_t> upsampled(output_elements);
        char nested_error[1024] = {};
        auto *upsampler = ltx_mlx_upsampler_create(
            upsampler_checkpoint, nested_error, sizeof(nested_error));
        if (!upsampler) {
            std::snprintf(error, error_size, "%s", nested_error);
            return 0;
        }
        const int ok = ltx_mlx_upsampler_run_tokens_bf16(
            upsampler, upsampled.data(), upsampled.size(),
            denormalized_host.data(), denormalized_host.size(), 1u, frames,
            height, width, nested_error, sizeof(nested_error));
        ltx_mlx_upsampler_free(upsampler);
        if (!ok) {
            std::snprintf(error, error_size, "%s", nested_error);
            return 0;
        }
        auto upsampled_tensor = Tensor(
            reinterpret_cast<const tc::mx::bfloat16_t *>(upsampled.data()),
            {1, static_cast<int>(frames), static_cast<int>(height * 2u),
             static_cast<int>(width * 2u), 128});
        auto renormalized = (upsampled_tensor - mean) / standard_deviation;
        tc::mx::eval(renormalized);
        std::memcpy(output, renormalized.data<tc::mx::bfloat16_t>(),
                    output_elements * sizeof(uint16_t));
        return 1;
    } catch (const std::exception &exception) {
        if (error && error_size)
            std::snprintf(error, error_size, "%s", exception.what());
        return 0;
    }
}
