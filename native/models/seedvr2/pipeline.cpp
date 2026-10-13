#include "pipeline.hpp"

namespace tc::seedvr2 {
namespace {
double seconds(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}
Tensor time_slice(const Tensor &x, int start, int stop) {
    return mx::slice(x, {0, 0, start, 0, 0}, {x.shape(0), x.shape(1), stop, x.shape(3), x.shape(4)});
}
} // namespace

void Pipeline::load(const std::filesystem::path &root, const Event &event, std::atomic<bool> &cancelled) {
    if (loaded_) return;
    checkpoint(cancelled);
    try {
        if (event) event("seedvr2_load", 0, 2);
        vae_.load(root / "vae/seedvr2_ema_vae_fp16.safetensors");
        checkpoint(cancelled);
        if (event) event("seedvr2_load", 1, 2);
        dit_.load(root / "diffusion_models/seedvr2_3b_fp16.safetensors");
        checkpoint(cancelled);
        if (event) event("seedvr2_load", 2, 2);
        loaded_ = true;
    } catch (...) {
        clear();
        throw;
    }
}
Restoration Pipeline::restore(const Tensor &pixels, const Tensor &noise, const Event &event,
                              std::atomic<bool> &cancelled, const std::filesystem::path &dump, const FFNRoute &ffn) {
    require(loaded_, "SeedVR2 pipeline is not loaded");
    require(pixels.ndim() == 5 && pixels.shape(0) == 1 && pixels.shape(1) == 3 &&
            pixels.shape(3) % 16 == 0 && pixels.shape(4) % 16 == 0,
            "SeedVR2 requires resized batch-one RGB BCTHW pixels and dimensions divisible by sixteen");
    const int original_frames = pixels.shape(2), frames = padded_frames(original_frames);
    auto input = pixels;
    if (frames != original_frames) input = mx::concatenate({pixels,
        mx::repeat(time_slice(pixels, original_frames - 1, original_frames), frames - original_frames, 2)}, 2);
    require(noise.shape() == mx::Shape{1, 16, (frames - 1) / 4 + 1, pixels.shape(3) / 8, pixels.shape(4) / 8} &&
            noise.dtype() == mx::float32, "SeedVR2 requires matching FP32 sampling noise");
    const auto started = Clock::now();
    auto condition = vae_.encode(input, event, cancelled);
    const auto encoded = Clock::now();
    auto velocity = dit_.forward(noise, condition, 1000.f, event, cancelled, dump, ffn);
    // Flow/Euler schedule [1,0], CFG=1, condition-noise scale=0. Model inputs
    // use BF16 but sampling retains FP32 noise, just like the stock sampler.
    auto latent = noise - mx::astype(velocity, mx::float32);
    mx::eval(latent);
    const auto denoised = Clock::now();
    auto restored = vae_.decode(latent, event, cancelled);
    restored = time_slice(restored, 0, original_frames);
    mx::eval(restored);
    const auto decoded = Clock::now();
    checkpoint(cancelled);
    return {restored, condition, velocity, latent, seconds(started, encoded),
            seconds(encoded, denoised), seconds(denoised, decoded)};
}
} // namespace tc::seedvr2
