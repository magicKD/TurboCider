#include "pipeline.hpp"

#include <mlx/random.h>

namespace tc::h3_mlx {
namespace {
double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void validate_noise(const Tensor &noise, int rows, int features,
                    const std::string &name) {
    require(noise.ndim() == 2 && noise.shape(0) == rows &&
                noise.shape(1) == features,
            "invalid H3 MLX " + name + " noise geometry");
    require(noise.dtype() == mx::float32,
            "H3 MLX parity noise must be FP32");
}
} // namespace

Tensor unpatchify_video_rows(const Tensor &rows, const PackedLayout &layout,
                             const DiTConfig &config) {
    const int patch_t = config.patch_size[0];
    const int patch_h = config.patch_size[1];
    const int patch_w = config.patch_size[2];
    require(layout.video_latent_frames % patch_t == 0 &&
                layout.latent_height % patch_h == 0 &&
                layout.latent_width % patch_w == 0,
            "H3 MLX video latent geometry is not patch divisible");
    require(rows.ndim() == 2 &&
                rows.shape(0) == int(layout.video_indices.size()) &&
                rows.shape(1) == config.patch_dim(),
            "invalid H3 MLX video rows for unpatchify");
    auto value = mx::reshape(rows,
        {1, layout.video_latent_frames / patch_t,
         layout.latent_height / patch_h, layout.latent_width / patch_w,
         config.latent_channels, patch_t, patch_h, patch_w});
    value = mx::transpose(value, {0, 4, 1, 5, 2, 6, 3, 7});
    return mx::contiguous(mx::reshape(value,
        {1, config.latent_channels, layout.video_latent_frames,
         layout.latent_height, layout.latent_width}));
}

Tensor unpack_audio_rows(const Tensor &rows, const PackedLayout &layout,
                         const DiTConfig &config) {
    require(rows.ndim() == 2 &&
                rows.shape(0) == int(layout.audio_indices.size()) &&
                rows.shape(1) == config.audio_latent_channels,
            "invalid H3 MLX audio rows for unpack");
    auto value = mx::reshape(rows,
        {audio_channels, layout.audio_latents, config.audio_latent_channels});
    return mx::contiguous(mx::transpose(value, {0, 2, 1}));
}

void Pipeline::load(const std::filesystem::path &root, const Event &event,
                    std::atomic<bool> &cancelled) {
    if (dit_ && loaded_root_ == root)
        return;
    unload();
    auto started = Clock::now();
    try {
        checkpoint_.load(root, event, cancelled);
        dit_.emplace(checkpoint_);
        loaded_root_ = root;
        last_load_seconds_ = seconds_since(started);
    } catch (...) {
        unload();
        throw;
    }
}

void Pipeline::unload() {
    dit_.reset();
    checkpoint_.clear();
    loaded_root_.clear();
    last_load_seconds_ = 0;
    mx::clear_cache();
}

DenoiseResult Pipeline::denoise(const Tensor &text_rows,
                                const std::vector<int32_t> &text_tags,
                                const DenoiseOptions &options,
                                const Event &event,
                                std::atomic<bool> &cancelled) {
    require(dit_.has_value(), "H3 MLX DiT pipeline is not loaded");
    require(options.width > 0 && options.height > 0 &&
                options.width % 32 == 0 && options.height % 32 == 0,
            "H3 MLX dimensions must be positive multiples of 32");
    require(options.fps == 24, "H3 MLX requires 24 fps");
    require(options.steps == 4, "FastH3 MLX requires exactly four steps");
    require(options.frames >= 1, "H3 MLX frame count must be positive");
    require(text_rows.ndim() == 2 && text_rows.shape(0) > 0 &&
                text_rows.shape(1) == checkpoint_.config().text_dim,
            "invalid H3 MLX conditioner rows");
    require(text_tags.empty() || int(text_tags.size()) == text_rows.shape(0),
            "H3 MLX conditioner tags do not match text rows");

    const int frames = align_frames(options.frames);
    const int latent_frames = video_latent_frames(frames);
    const int latent_height = options.height / 16;
    const int latent_width = options.width / 16;
    const int audio_latents = audio_latent_frames(frames, options.fps);
    auto layout = build_packed_layout(text_rows.shape(0), latent_frames,
                                      latent_height, latent_width,
                                      audio_latents,
                                      checkpoint_.config().patch_size,
                                      text_tags);
    const int video_rows = int(layout.video_indices.size());
    const int audio_rows = int(layout.audio_indices.size());

    auto noise_keys = mx::random::split(mx::random::key(options.seed));
    Tensor video = options.video_noise ? *options.video_noise :
        mx::random::normal({video_rows, checkpoint_.config().patch_dim()},
                           mx::float32, noise_keys.first);
    Tensor audio = options.audio_noise ? *options.audio_noise :
        mx::random::normal({audio_rows, checkpoint_.config().audio_latent_channels},
                           mx::float32, noise_keys.second);
    validate_noise(video, video_rows, checkpoint_.config().patch_dim(), "video");
    validate_noise(audio, audio_rows,
                   checkpoint_.config().audio_latent_channels, "audio");
    mx::eval({video, audio});

    auto video_scheduler = Scheduler::create(video_shift, options.steps);
    auto audio_scheduler = Scheduler::create(audio_shift, options.steps);
    checkpoint_.set_affine_dq_gemm_min_rows(options.affine_dq_gemm_min_rows);
    options.vsa.validate(checkpoint_.config().num_layers);
    require(!options.vsa.enabled || checkpoint_.identity().vsa_capable,
            "H3 VSA requires a VSA-capable checkpoint; dense-only checkpoint rejected");
    checkpoint_.reset_dispatch_metrics();
    mx::reset_peak_memory();
    auto started = Clock::now();
    std::optional<Tensor> first_video_velocity;
    std::optional<Tensor> first_audio_velocity;
    std::optional<Tensor> first_video_sample;
    std::optional<Tensor> first_audio_sample;
    std::vector<DenoiseStepCapture> steps;
    if (options.capture_steps) steps.reserve(options.steps);
    DiTDebugCapture debug;
    std::optional<VSAStats> vsa_stats;
    if (options.vsa.enabled) {
        vsa_stats.emplace();
        vsa_stats->capture_debug = options.capture_debug;
    }
    for (int step = 0; step < options.steps; ++step) {
        tc::checkpoint(cancelled);
        event("h3_mlx_denoise", step, options.steps);
        auto row_timesteps = build_row_timesteps(
            layout, video_scheduler.timesteps[step],
            audio_scheduler.timesteps[step]);
        auto velocity = dit_->forward(
            video, audio, text_rows, layout, row_timesteps, event, cancelled,
            step, options.vsa, vsa_stats ? &*vsa_stats : nullptr,
            options.capture_debug && step == 0 ? &debug : nullptr);
        if (step == 0 && options.capture_first_velocity) {
            first_video_velocity = velocity.video;
            first_audio_velocity = velocity.audio;
            mx::eval(*first_video_velocity, *first_audio_velocity);
        }
        if (options.capture_steps)
            mx::eval(velocity.video, velocity.audio);
        video = scheduler_step(video, velocity.video, video_scheduler, step);
        audio = scheduler_step(audio, velocity.audio, audio_scheduler, step);
        mx::eval({video, audio});
        if (step == 0 && options.capture_first_velocity) {
            first_video_sample = video;
            first_audio_sample = audio;
        }
        if (options.capture_steps) {
            steps.push_back({std::move(velocity.video), std::move(velocity.audio),
                             video, audio, row_timesteps.unique,
                             row_timesteps.inverse});
        }
    }
    tc::checkpoint(cancelled);
    event("h3_mlx_denoise", options.steps, options.steps);

    DenoiseMetrics metrics;
    metrics.load_seconds = last_load_seconds_;
    metrics.denoise_seconds = seconds_since(started);
    metrics.peak_bytes = mx::get_peak_memory();
    metrics.quantized_matmul_calls = checkpoint_.quantized_matmul_calls();
    metrics.dequantized_gemm_calls = checkpoint_.dequantized_gemm_calls();
    metrics.affine_dq_gemm_min_rows = checkpoint_.affine_dq_gemm_min_rows();
    metrics.vsa = std::move(vsa_stats);
    if (metrics.vsa)
        metrics.vsa->configured_sparsity = options.vsa.sparsity;
    return {std::move(video), std::move(audio), std::move(first_video_velocity),
            std::move(first_audio_velocity), std::move(first_video_sample),
            std::move(first_audio_sample), std::move(steps), std::move(debug),
            std::move(layout), metrics};
}

} // namespace tc::h3_mlx
