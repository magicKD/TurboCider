#include "pipeline.hpp"
#include "../../components/diffusion/seeded_noise.hpp"

#include <cmath>

namespace tc::wan {

Pipeline::Pipeline(const std::filesystem::path &dit_root,
                   const std::filesystem::path &text_root,
                   const std::filesystem::path &taehv_safetensors,
                   const Event &event, std::atomic<bool> &cancelled)
    : text_encoder_(text_weights_), decoder_(taehv_safetensors, 16), text_root_(text_root),
      dit_root_(dit_root) {
    dit_weights_.load(dit_root, event, cancelled);
    // Text weights are deliberately loaded through the shared Weights loader;
    // directory-backed shard discovery is deterministic and symlink-safe.
    require(text_root_.is_absolute() && std::filesystem::is_directory(text_root_),
            "Wan text root must be an absolute component directory");
}

std::pair<Tensor, Tensor> Pipeline::rotary(const GenerateOptions &options) const {
    require(options.frames >= 5 && options.frames % 4 == 1 && options.height % 16 == 0 &&
                options.width % 16 == 0, "Wan output dimensions violate latent geometry");
    const std::array<int, 3> latent = {options.frames / 4 + 1, options.height / 8, options.width / 8};
    const std::array<int, 3> patch = {1, 2, 2};
    auto grid = components::wan_patch_grid(latent, patch);
    return components::wan_rotary(grid);
}

Tensor Pipeline::noise(const GenerateOptions &options) const {
    const int frames = options.frames / 4 + 1, height = options.height / 8, width = options.width / 8;
    auto values = components::wan_initial_noise(size_t(16) * frames * height * width, options.seed);
    return mx::astype(Tensor(values.data(), {1, 16, frames, height, width}, mx::float32), mx::float16);
}

Tensor Pipeline::generate(const Tokens &tokens, const GenerateOptions &options,
                          const Event &event, std::atomic<bool> &cancelled) {
    checkpoint(cancelled);
    require(options.frames >= 5 && options.frames <= 81 && options.frames % 4 == 1 &&
                options.height >= 16 && options.height <= 480 && options.height % 16 == 0 &&
                options.width >= 16 && options.width <= 832 && options.width % 16 == 0 &&
                options.fps == 16 && options.timesteps == std::array<float, 3>{1000.f, 757.f, 522.f},
            "unsupported native Wan QAD request geometry or schedule");
    std::unique_ptr<HybridFFN> hybrid;
    if (!options.hybrid_manifest.empty()) {
        require(!options.compile_dit, "Wan hybrid and whole-DiT compilation are mutually exclusive");
        require(options.frames == 81 && options.height == 480 && options.width == 832,
                "Wan hybrid artifacts require exactly 832x480x81 output geometry");
        hybrid = load_hybrid(options.hybrid_manifest, dit_root_, dit_weights_, event, cancelled);
    }
    if (!cached_conditioning_ || cached_ids_ != tokens.ids || cached_valid_ != tokens.valid) {
        // UMT5 is large and has no role during diffusion or VAE decoding.
        // Retain only the evaluated conditioning, not the text checkpoint.
        try {
            text_weights_.load(text_root_, event, cancelled);
            auto encoded = mx::astype(text_encoder_.encode(tokens, event, cancelled), mx::float16);
            mx::eval(encoded);
            checkpoint(cancelled);
            cached_conditioning_ = std::move(encoded);
            cached_ids_ = tokens.ids;
            cached_valid_ = tokens.valid;
            text_weights_.clear();
        } catch (...) {
            text_weights_.clear();
            throw;
        }
    } else event("umt5_prompt_cache", 1, 1);
    auto text = *cached_conditioning_;
    auto frequencies = rotary(options);
    auto latents = noise(options);
    mx::random::KeySequence renoise_keys(options.seed);
    DiT dit(dit_weights_);
    auto sigmas = components::wan_qad_sigmas({options.timesteps.begin(), options.timesteps.end()});
    for (size_t index = 0; index < sigmas.size(); ++index) {
        checkpoint(cancelled);
        event("wan_denoise", int(index), int(sigmas.size()));
        auto timestep = Tensor(options.timesteps[index], mx::float32);
        auto prediction = hybrid
            ? dit.forward_hybrid(latents, text, mx::reshape(timestep, {1}), frequencies.first,
                                 frequencies.second,
                                 [&](int block, const Tensor &input, const Tensor &residual, const Tensor &gate) {
                                     checkpoint(cancelled);
                                     return hybrid->predict(block, input, residual, gate);
                                 }, event, cancelled)
            : options.compile_dit
            ? dit.forward_compiled(latents, text, mx::reshape(timestep, {1}), frequencies.first,
                                   frequencies.second, event, cancelled)
            : dit.forward(latents, text, mx::reshape(timestep, {1}), frequencies.first,
                          frequencies.second, event, cancelled);
        std::optional<float> next = index + 1 < sigmas.size() ? std::optional<float>(sigmas[index + 1]) : std::nullopt;
        std::optional<Tensor> fresh;
        if (next) {
            fresh = mx::random::normal(latents.shape(), mx::float32, renoise_keys.next());
        }
        latents = mx::astype(components::wan_dmd_step(mx::astype(latents, mx::float32),
            mx::astype(prediction, mx::float32), sigmas[index], next, fresh), mx::float16);
        mx::eval(latents);
    }
    auto video = decoder_.decode_ntchw(mx::transpose(latents, {0, 2, 1, 3, 4}), event, cancelled);
    mx::eval(video);
    event("wan_generate", 1, 1);
    return video;
}

} // namespace tc::wan
