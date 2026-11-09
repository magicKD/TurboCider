#pragma once

#include "../../components/text/umt5.hpp"
#include "transformer.hpp"
#include "vae.hpp"

namespace tc::ltx098 {

struct GenerateOptions {
    int width = 512, height = 512, frames = 121, fps = 24;
    uint64_t seed = 42;
    float strength = 1.f;
    // 8 is the established Comfy-style single-resolution sampler; 10 is
    // Lightricks' 7 + 3 multiscale schedule with the 0.9.8 latent upscaler.
    int steps = 8;
    bool resize_to_base = false;
    // Development-only observer; public requests never install it.
    std::function<void(const std::string &, const Tensor &)> trace;
};

struct GenerateMetrics {
    double text_seconds = 0, image_seconds = 0, transformer_load_seconds = 0;
    double cross_attention_prepare_seconds = 0;
    double denoise_seconds = 0, upscale_seconds = 0, decoder_load_seconds = 0, decode_seconds = 0;
    bool text_cache_hit = false, cross_attention_cache_hit = false;
    int text_tokens = 0, valid_text_tokens = 0, denoise_steps = 0;
    int stage1_width = 0, stage1_height = 0, stage2_width = 0, stage2_height = 0;
    int output_width = 0, output_height = 0;
};

// GPU-only resident 0.9.8 pipeline. It owns its components and prompt cache;
// video export and public model admission are handled by the session layer.
class Pipeline {
    std::filesystem::path root_;
    Weights text_weights_;
    std::unique_ptr<components::UMT5Encoder> text_encoder_;
    std::unique_ptr<VAEEncoder> image_encoder_;
    std::unique_ptr<Transformer> transformer_;
    std::unique_ptr<VAEDecoder> decoder_;
    std::optional<Tensor> cached_text_;
    std::optional<Tensor> cached_projection_;
    std::vector<CrossAttentionKV> cached_cross_attention_;
    std::vector<int> cached_ids_;
    int cached_valid_ = 0;

  public:
    explicit Pipeline(std::filesystem::path root);
    // Input pixels [1,H,W,3] are normalized to [-1,1]. Output pixels are
    // unclamped BF16 [1,frames,H,W,3]. Both are evaluated before returning.
    std::pair<Tensor, GenerateMetrics> generate(const Tokens &, const Tensor &pixels,
                                                const GenerateOptions &, const Event &,
                                                std::atomic<bool> &cancelled);
};

} // namespace tc::ltx098
