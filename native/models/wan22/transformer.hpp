#pragma once

#include "block.hpp"

namespace tc::wan22 {

struct TransformerConfig {
    int channels = 48;
    int hidden = 3072;
    int heads = 24;
    int intermediate = 14336;
    int layers = 30;
    int text_dim = 4096;
    int frequency_dim = 256;
    float epsilon = 1e-6f;
};

// Resident dense BF16 TI2V transformer. A separate session/pipeline owns
// VAE conditioning, DMD sampling, video export, cancellation and caching.
class Transformer {
    Weights top_;
    std::vector<std::unique_ptr<Block>> blocks_;
    TransformerConfig config_;

  public:
    using Trace = std::function<void(const std::string &, const Tensor &)>;
    void load(const std::filesystem::path &, TransformerConfig = {});
    Tensor prepare_text(const Tensor &, const Trace &trace = {}) const;
    Tensor forward(const Tensor &latents, const Tensor &prepared_text, float timestep,
                   const Event &, std::atomic<bool> &, const Trace &trace = {}) const;
};

} // namespace tc::wan22
