#pragma once

#include "transformer.hpp"
#include "vae.hpp"
#include "schedule.hpp"

namespace tc::wan22 {

struct GenerateOptions {
    int frames = 97;
    uint64_t seed = 42;
    Distillation distillation = Distillation::fastvideo_3;
    bool cache_cross_attention = false;
    // Empty keeps the full VAE. Super is an explicit, approximate decoder;
    // first-frame conditioning always uses the full VAE encoder.
    std::filesystem::path tiny_decoder_checkpoint;
};

struct GenerateResult {
    Tensor latents; // Raw VAE-space BF16 [1,48,(frames+3)/4,H/16,W/16].
    Tensor pixels;  // [-1,1]-style [1,3,frames,H,W].
    double first_frame_vae_seconds = 0;
    double transformer_and_denoise_seconds = 0;
    double video_vae_seconds = 0;
};

// Native, image-conditioned distilled paths. Text is the already encoded
// BF16 UMT5 output; tokenization, media export and App registration remain
// outside this component.
class Pipeline {
    std::filesystem::path transformer_checkpoint_, vae_checkpoint_;

  public:
    using Trace = std::function<void(const std::string &, const Tensor &)>;
    Pipeline(std::filesystem::path transformer_checkpoint, std::filesystem::path vae_checkpoint);
    GenerateResult generate(const Tensor &pixels, const Tensor &text, const GenerateOptions &,
                            const Event &, std::atomic<bool> &, const Trace &trace = {}) const;
};

} // namespace tc::wan22
