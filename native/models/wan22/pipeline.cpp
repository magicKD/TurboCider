#include "pipeline.hpp"
#include "latent.hpp"
#include "tae_super.hpp"
#include "../../components/diffusion/seeded_noise.hpp"

#include <array>

namespace tc::wan22 {
namespace {
Tensor condition_first_frame(const Tensor &value, const Tensor &first) {
    return mx::concatenate({first, slice_axis(value, 2, 1, value.shape(2))}, 2);
}
} // namespace

Pipeline::Pipeline(std::filesystem::path transformer_checkpoint, std::filesystem::path vae_checkpoint)
    : transformer_checkpoint_(std::move(transformer_checkpoint)), vae_checkpoint_(std::move(vae_checkpoint)) {
    require(transformer_checkpoint_.is_absolute() && std::filesystem::is_regular_file(transformer_checkpoint_) &&
                vae_checkpoint_.is_absolute() && std::filesystem::is_regular_file(vae_checkpoint_),
            "Wan2.2 pipeline requires absolute model checkpoint paths");
}

GenerateResult Pipeline::generate(const Tensor &pixels, const Tensor &text, const GenerateOptions &options,
                                  const Event &event, std::atomic<bool> &cancelled, const Trace &trace) const {
    require(options.frames >= 5 && (options.frames - 1) % 4 == 0 && pixels.ndim() == 5 &&
                pixels.shape(0) == 1 && pixels.shape(1) == 3 && pixels.shape(2) == 1 &&
                pixels.shape(3) >= 32 && pixels.shape(4) >= 32 && pixels.shape(3) % 32 == 0 &&
                pixels.shape(4) % 32 == 0 && pixels.dtype() == mx::bfloat16 && text.ndim() == 3 &&
                text.shape(0) == 1 && text.shape(2) == 4096 && text.dtype() == mx::bfloat16,
            "invalid Wan2.2 I2V request geometry or dtype");
    checkpoint(cancelled);
    if (!options.tiny_decoder_checkpoint.empty())
        require(options.tiny_decoder_checkpoint.is_absolute() &&
                    std::filesystem::is_regular_file(options.tiny_decoder_checkpoint),
                "TAEW2.2-Super checkpoint must be an absolute regular file");
    const int f = (options.frames - 1) / 4 + 1, h = pixels.shape(3) / 16, w = pixels.shape(4) / 16;
    const mx::Shape shape{1, 48, f, h, w};
    const auto encode_started = Clock::now();
    Tensor clean = [&]() {
        VAEEncoder encoder;
        encoder.load(vae_checkpoint_);
        auto result = normalize_vae_latents(encoder.encode_first_frame(pixels));
        mx::eval(result);
        return result;
    }();
    const double encode_seconds = std::chrono::duration<double>(Clock::now() - encode_started).count();
    mx::clear_cache();
    if (trace) trace("first_latent", clean);
    components::WanNoiseGenerator generator(options.seed);
    auto draw = [&]() {
        auto values = generator.normal(size_t(48) * f * h * w);
        return Tensor(values.data(), shape, mx::float32);
    };
    auto latents = draw();
    // FastVideo MLX reference's shift=5 warped DMD schedule; this is not
    // equivalent to an ordinary three-step Euler sampler.
    constexpr std::array<float, 3> sigmas{1.f, .9396722939424031f, .8452072538860105f};
    const auto denoise_started = Clock::now();
    {
        Transformer transformer;
        transformer.load(transformer_checkpoint_);
        auto context = transformer.prepare_text(text);
        mx::eval(context);
        for (int step = 0; step < 3; ++step) {
            checkpoint(cancelled);
            event("wan22_denoise", step, 3);
            auto input = condition_first_frame(latents, clean);
            if (trace) trace("model_input_" + std::to_string(step), input);
            auto prediction = transformer.forward(mx::astype(input, mx::bfloat16), context,
                                                  sigmas[step] * 1000.f, event, cancelled);
            if (trace) trace("prediction_" + std::to_string(step), prediction);
            auto denoised = condition_first_frame(input - mx::astype(prediction, mx::float32) * sigmas[step], clean);
            if (trace) trace("denoised_" + std::to_string(step), denoised);
            if (step < 2) {
                auto fresh = draw();
                latents = (1.f - sigmas[step + 1]) * denoised + sigmas[step + 1] * fresh;
            } else latents = denoised;
            mx::eval(latents);
            if (trace) trace("step_" + std::to_string(step), latents);
        }
    }
    const double denoise_seconds = std::chrono::duration<double>(Clock::now() - denoise_started).count();
    mx::clear_cache();
    checkpoint(cancelled);
    auto raw_latents = denormalize_vae_latents(latents);
    mx::eval(raw_latents);
    if (trace) trace("raw_latents", raw_latents);
    const auto decode_started = Clock::now();
    auto decoded = [&]() -> Tensor {
        if (options.tiny_decoder_checkpoint.empty()) {
            VAEDecoder decoder;
            decoder.load(vae_checkpoint_);
            return decoder.decode(raw_latents, event, &cancelled);
        }
        TAESuperDecoder decoder(options.tiny_decoder_checkpoint);
        // Match the full decoder's BF16 VAE-space latent and convert it back
        // to the normalized diffusion space expected by the upstream TAE.
        auto normalized = normalize_vae_latents(raw_latents);
        auto pixels01 = decoder.decode(normalized, event, &cancelled);
        return mx::astype(pixels01, mx::float32) * 2.f - 1.f;
    }();
    mx::eval(decoded);
    const double decode_seconds = std::chrono::duration<double>(Clock::now() - decode_started).count();
    checkpoint(cancelled);
    event("wan22_denoise", 3, 3);
    return {raw_latents, decoded, encode_seconds, denoise_seconds, decode_seconds};
}

} // namespace tc::wan22
