#pragma once

#include "dit.hpp"

namespace tc::h3_mlx {

struct DenoiseOptions {
    int width = 832;
    int height = 480;
    int frames = 124;
    int fps = 24;
    int steps = 4;
    uint64_t seed = 0;
    int affine_dq_gemm_min_rows = 768;
    std::optional<Tensor> video_noise;
    std::optional<Tensor> audio_noise;
    bool capture_first_velocity = false;
    bool capture_steps = false;
    bool capture_debug = false;
    VSAConfig vsa;
};

struct DenoiseMetrics {
    double load_seconds = 0;
    double denoise_seconds = 0;
    uint64_t peak_bytes = 0;
    uint64_t quantized_matmul_calls = 0;
    uint64_t dequantized_gemm_calls = 0;
    int affine_dq_gemm_min_rows = 768;
    std::optional<VSAStats> vsa;
};

struct DenoiseStepCapture {
    Tensor video_velocity;
    Tensor audio_velocity;
    Tensor video_sample;
    Tensor audio_sample;
    std::vector<float> unique_timesteps;
    std::vector<int32_t> inverse_timesteps;
};

struct DenoiseResult {
    Tensor video_rows;
    Tensor audio_rows;
    std::optional<Tensor> first_video_velocity;
    std::optional<Tensor> first_audio_velocity;
    std::optional<Tensor> first_video_sample;
    std::optional<Tensor> first_audio_sample;
    std::vector<DenoiseStepCapture> steps;
    DiTDebugCapture debug;
    PackedLayout layout;
    DenoiseMetrics metrics;
};

Tensor unpatchify_video_rows(const Tensor &, const PackedLayout &,
                             const DiTConfig &);
Tensor unpack_audio_rows(const Tensor &, const PackedLayout &,
                         const DiTConfig &);

class Pipeline {
    Checkpoint checkpoint_;
    std::optional<DiT> dit_;
    std::filesystem::path loaded_root_;
    double last_load_seconds_ = 0;

  public:
    void load(const std::filesystem::path &, const Event &, std::atomic<bool> &);
    void unload();
    bool loaded() const { return dit_.has_value(); }
    const Checkpoint &checkpoint() const { return checkpoint_; }
    DenoiseResult denoise(const Tensor &text_rows,
                          const std::vector<int32_t> &text_tags,
                          const DenoiseOptions &, const Event &,
                          std::atomic<bool> &);
};

} // namespace tc::h3_mlx
