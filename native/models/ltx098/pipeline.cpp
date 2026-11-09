#include "pipeline.hpp"
#include "adain.hpp"
#include "resize.hpp"
#include "scheduler.hpp"
#include "../../components/diffusion/seeded_noise.hpp"
#include "../ltx_runtime/ltx_mlx_upsampler.h"

#include <array>
#include <chrono>
#include <cmath>
#include <memory>

namespace tc::ltx098 {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
constexpr std::array<float, 9> sigmas{1.f, .9937f, .9875f, .9812f, .975f, .9094f, .725f, .4219f, 0.f};
constexpr std::array<float, 8> first_sigmas{1.f, .9937f, .9875f, .9812f, .975f, .9094f, .725f, 0.f};
constexpr std::array<float, 4> second_sigmas{.9094f, .725f, .4219f, 0.f};

void materialize_text_weights(Weights &weights) {
    // The T5 checkpoint has tied embedding aliases. The encoder consumes the
    // shared key only; retaining the duplicate would allocate a second table.
    if (weights.has("encoder.embed_tokens.weight")) weights.erase("encoder.embed_tokens.weight");
    auto names = weights.sorted_keys();
    std::vector<Tensor> arrays;
    arrays.reserve(names.size());
    for (const auto &name : names) arrays.push_back(mx::astype(weights.at(name), mx::bfloat16));
    weights.bind_arrays(names, arrays);
    weights.materialize();
}
} // namespace

Pipeline::Pipeline(std::filesystem::path root) : root_(std::filesystem::absolute(std::move(root))) {
    require(std::filesystem::is_regular_file(root_ / "ltxv-2b-0.9.8-distilled.safetensors") &&
                std::filesystem::is_regular_file(root_ / "text_encoders/t5xxl_fp16.safetensors"),
            "LTX 0.9.8 checkpoint or T5-XXL text encoder is missing");
}

std::pair<Tensor, GenerateMetrics> Pipeline::generate(const Tokens &tokens, const Tensor &pixels,
                                                      const GenerateOptions &options, const Event &event,
                                                      std::atomic<bool> &cancelled) {
    require(options.width >= 64 && options.height >= 64 && options.width % 32 == 0 && options.height % 32 == 0 &&
                options.frames >= 9 && options.frames % 8 == 1 && options.fps > 0 &&
                (options.steps == 8 || options.steps == 10) &&
                (!options.resize_to_base || options.steps == 10) &&
                std::isfinite(options.strength) && options.strength >= 0 && options.strength <= 1 &&
                pixels.shape() == mx::Shape{1, options.height, options.width, 3},
            "invalid LTX 0.9.8 video/input geometry");
    require(tokens.valid > 0 && size_t(tokens.valid) <= tokens.ids.size() && tokens.ids.size() >= 128 && tokens.ids.size() <= 512,
            "invalid LTX 0.9.8 text tokens");
    GenerateMetrics metrics;
    const bool multiscale = options.steps == 10;
    const int first_h = multiscale ? int(options.height * .6666666) / 32 * 32 : options.height;
    const int first_w = multiscale ? int(options.width * .6666666) / 32 * 32 : options.width;
    const int second_h = multiscale ? 2 * first_h : options.height;
    const int second_w = multiscale ? 2 * first_w : options.width;
    require(first_h >= 32 && first_w >= 32, "invalid LTX 0.9.8 multiscale geometry");
    const auto upscaler_path = root_ / "ltxv-spatial-upscaler-0.9.8.safetensors";
    if (multiscale)
        require(std::filesystem::is_regular_file(upscaler_path),
                "LTX 0.9.8 10-step multiscale requires ltxv-spatial-upscaler-0.9.8.safetensors in the model folder");
    metrics.stage1_width = first_w; metrics.stage1_height = first_h;
    metrics.stage2_width = second_w; metrics.stage2_height = second_h;
    metrics.output_width = options.resize_to_base ? options.width : second_w;
    metrics.output_height = options.resize_to_base ? options.height : second_h;
    metrics.text_tokens = int(tokens.ids.size()); metrics.valid_text_tokens = tokens.valid;
    const auto checkpoint_path = root_ / "ltxv-2b-0.9.8-distilled.safetensors";
    checkpoint(cancelled);
    auto start = Clock::now();
    if (!text_encoder_) {
        event("text_load", 0, 1);
        text_weights_.load_file(root_ / "text_encoders/t5xxl_fp16.safetensors");
        materialize_text_weights(text_weights_);
        components::UMT5Config config;
        config.shared_relative_attention_bias = true; config.zero_padding_output = false;
        config.comfy_t5_sdpa = true;
        config.comfy_t5_fp32 = true;
        text_encoder_ = std::make_unique<components::UMT5Encoder>(text_weights_, config);
        event("text_load", 1, 1);
    }
    if (!cached_text_ || cached_ids_ != tokens.ids || cached_valid_ != tokens.valid) {
        cached_text_ = text_encoder_->encode(tokens, event, cancelled);
        cached_ids_ = tokens.ids; cached_valid_ = tokens.valid;
        cached_projection_.reset();
        cached_cross_attention_.clear();
    } else {
        metrics.text_cache_hit = true;
        event("t5_prompt_cache", 1, 1);
    }
    metrics.text_seconds = elapsed(start);
    if (options.trace) options.trace("text", *cached_text_);
    checkpoint(cancelled);
    start = Clock::now();
    if (!image_encoder_) {
        image_encoder_ = std::make_unique<VAEEncoder>();
        image_encoder_->load(checkpoint_path);
    }
    event("first_frame_vae_encode", 0, 1);
    auto input_video = mx::expand_dims(pixels, 1);
    auto first_pixels = multiscale ? resize_bilinear(input_video, first_h, first_w) : input_video;
    auto first = image_encoder_->encode(mx::astype(first_pixels, mx::bfloat16));
    mx::eval(first);
    if (options.trace) options.trace("first_frame", first);
    event("first_frame_vae_encode", 1, 1);
    metrics.image_seconds = elapsed(start);
    checkpoint(cancelled);
    start = Clock::now();
    if (!transformer_) {
        transformer_ = std::make_unique<Transformer>();
        transformer_->load(checkpoint_path);
    }
    metrics.transformer_load_seconds = elapsed(start);
    start = Clock::now();
    if (!cached_projection_) {
        auto projected = transformer_->prepare_text(mx::astype(*cached_text_, mx::bfloat16));
        mx::eval(projected);
        auto cross_attention = transformer_->prepare_cross_attention(projected);
        cached_projection_ = std::move(projected);
        cached_cross_attention_ = std::move(cross_attention);
    } else {
        metrics.cross_attention_cache_hit = true;
    }
    metrics.cross_attention_prepare_seconds = elapsed(start);
    const auto &text = *cached_projection_;
    if (options.trace) options.trace("projected_text", text);
    std::vector<float> mask_values(tokens.ids.size(), -10000.f);
    std::fill(mask_values.begin(), mask_values.begin() + tokens.valid, 0.f);
    auto mask = mx::astype(Tensor(mask_values.data(), {1, 1, 1, int(mask_values.size())}, mx::float32), mx::bfloat16);
    const int f = (options.frames + 7) / 8, h = first_h / 32, w = first_w / 32;
    const size_t first_count = size_t(128) * f * h * w;
    const size_t second_count = multiscale ? size_t(128) * f * (second_h / 32) * (second_w / 32) : 0;
    auto random = components::wan_initial_noise(first_count + second_count, options.seed);
    auto noise = mx::transpose(Tensor(random.data(), {1, 128, f, h, w}, mx::float32), {0, 2, 3, 4, 1});
    auto state = noise;
    auto first_f32 = mx::astype(first, mx::float32);
    mx::eval(state);
    if (options.trace) options.trace("initial_noise", state);
    auto sample = [&](auto schedule, bool original) {
        auto began = Clock::now();
        if (original) state = original_condition(state, first_f32, options.strength);
        for (size_t index = 0; index + 1 < schedule.size(); ++index) {
            checkpoint(cancelled);
            event("denoise", metrics.denoise_steps, options.steps);
            auto model_input = original ? state : conditioned_input(state, first_f32, options.strength);
            const auto trace_index = std::to_string(metrics.denoise_steps);
            if (options.trace) options.trace("model_input_" + trace_index, model_input);
            const float time_strength = original ? original_first_frame_strength(options.strength, schedule[index]) : options.strength;
            auto prediction = transformer_->forward_cached(mx::astype(model_input, mx::bfloat16), text, schedule[index],
                                                            float(options.fps), true, time_strength, mask,
                                                            cached_cross_attention_);
            if (options.trace) options.trace("prediction_" + trace_index, prediction);
            auto velocity = mx::astype(prediction, mx::float32);
            state = original ? original_euler_step(state, velocity, options.strength, schedule[index], schedule[index + 1])
                             : euler_step(state, model_input, velocity, first_f32,
                                          options.strength, schedule[index], schedule[index + 1]);
            mx::eval(state);
            if (options.trace) options.trace("step_" + trace_index, state);
            ++metrics.denoise_steps;
        }
        metrics.denoise_seconds += elapsed(began);
    };
    if (!multiscale) {
        sample(sigmas, false);
    } else {
        sample(first_sigmas, true);
        checkpoint(cancelled);
        start = Clock::now();
        event("latent_upsample", 0, 1);
        auto unnormalized = mx::contiguous(image_encoder_->denormalize_latents(mx::astype(state, mx::bfloat16)));
        mx::eval(unnormalized);
        std::vector<uint16_t> upsampled(second_count);
        std::array<char, 2048> error{};
        std::unique_ptr<ltx_mlx_upsampler, decltype(&ltx_mlx_upsampler_free)> upsampler(
            ltx_mlx_upsampler_create_ltx098(upscaler_path.c_str(), error.data(), error.size()), &ltx_mlx_upsampler_free);
        require(bool(upsampler), std::string("cannot load LTX 0.9.8 latent upscaler: ") + error.data());
        require(ltx_mlx_upsampler_run_tokens_bf16(upsampler.get(), upsampled.data(), upsampled.size(),
                    unnormalized.data<uint16_t>(), first_count, 1, f, h, w, error.data(), error.size()),
                std::string("LTX 0.9.8 latent upscaler failed: ") + error.data());
        auto high = Tensor(reinterpret_cast<const mx::bfloat16_t *>(upsampled.data()),
                           {1, f, second_h / 32, second_w / 32, 128});
        high = mx::astype(image_encoder_->normalize_latents(high), mx::float32);
        mx::eval(high);
        if (options.trace) options.trace("upsampled_latent", high);
        high = mx::astype(adain_latents(high, state), mx::float32);
        mx::eval(high);
        if (options.trace) options.trace("adain_latent", high);
        upsampler.reset();
        event("latent_upsample", 1, 1);
        metrics.upscale_seconds = elapsed(start);
        checkpoint(cancelled);
        start = Clock::now();
        event("first_frame_vae_encode", 0, 1);
        auto second_pixels = resize_bilinear(input_video, second_h, second_w);
        first_f32 = mx::astype(image_encoder_->encode(mx::astype(second_pixels, mx::bfloat16)), mx::float32);
        mx::eval(first_f32);
        if (options.trace) options.trace("second_first_frame", first_f32);
        event("first_frame_vae_encode", 1, 1);
        metrics.image_seconds += elapsed(start);
        auto second_noise = mx::transpose(Tensor(random.data() + first_count,
                    {1, 128, f, second_h / 32, second_w / 32}, mx::float32), {0, 2, 3, 4, 1});
        state = high * Tensor(1.f - second_sigmas[0], mx::float32) +
                second_noise * Tensor(second_sigmas[0], mx::float32);
        if (options.trace) options.trace("second_initial_state", state);
        sample(second_sigmas, true);
    }
    event("denoise", metrics.denoise_steps, options.steps);
    checkpoint(cancelled);
    start = Clock::now();
    if (!decoder_) {
        decoder_ = std::make_unique<VAEDecoder>();
        decoder_->load(checkpoint_path);
    }
    metrics.decoder_load_seconds = elapsed(start);
    start = Clock::now();
    event("video_vae", 0, 1);
    mx::random::KeySequence keys(options.seed ^ 0x9e3779b97f4a7c15ull);
    auto decode_noise = mx::random::normal(state.shape(), mx::bfloat16, keys.next());
    if (options.trace) options.trace("decode_noise", decode_noise);
    auto result = decoder_->decode(mx::astype(state, mx::bfloat16), decode_noise);
    if (options.resize_to_base) result = resize_bilinear(result, options.height, options.width);
    mx::eval(result);
    if (options.trace) options.trace("pixels", result);
    require(result.shape() == mx::Shape{1, options.frames, metrics.output_height, metrics.output_width, 3},
            "LTX 0.9.8 decoder produced unexpected video geometry");
    event("video_vae", 1, 1);
    metrics.decode_seconds = elapsed(start);
    checkpoint(cancelled);
    return {std::move(result), metrics};
}

} // namespace tc::ltx098
