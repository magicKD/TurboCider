#pragma once

#include "../../components/diffusion/wan.hpp"
#include "../../components/text/umt5.hpp"
#include "../../components/vae/taehv.hpp"
#include "dit.hpp"
#include "hybrid.hpp"

namespace tc::wan {

struct GenerateOptions {
    int frames = 81;
    int height = 480;
    int width = 832;
    int fps = 16;
    uint64_t seed = 42;
    bool compile_dit = false;
    // Explicit opt-in; never discover artifacts from cwd or environment.
    std::filesystem::path hybrid_manifest;
    std::array<float, 3> timesteps = {1000.f, 757.f, 522.f};
};

// Native orchestration boundary. It accepts already-tokenized text and emits
// NTCHW RGB tensors; media encoding remains the separate native media API.
class Pipeline {
    Checkpoint dit_weights_;
    Weights text_weights_;
    components::UMT5Encoder text_encoder_;
    components::TAEHVDecoder decoder_;
    std::filesystem::path text_root_;
    std::filesystem::path dit_root_;
    std::vector<int> cached_ids_;
    int cached_valid_ = 0;
    std::optional<Tensor> cached_conditioning_;

    std::pair<Tensor, Tensor> rotary(const GenerateOptions &) const;
    Tensor noise(const GenerateOptions &) const;

  public:
    Pipeline(const std::filesystem::path &dit_root,
             const std::filesystem::path &text_root,
             const std::filesystem::path &taehv_safetensors,
             const Event &, std::atomic<bool> &);
    Tensor generate(const Tokens &, const GenerateOptions &, const Event &,
                    std::atomic<bool> &);
};

} // namespace tc::wan
