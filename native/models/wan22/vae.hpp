#pragma once

#include "../../backends/mlx.hpp"
#include <unordered_set>

namespace tc::wan22 {

// Wan2.2's full-quality VAE, limited for now to encoding the clean first
// frame. It is distinct from both the Wan2.1 16-channel and Qwen-image VAEs.
class VAEEncoder {
    Weights weights_;
    Tensor conv(const Tensor &, const std::string &, int padding = 1, int stride = 1) const;
    Tensor normalize(const Tensor &, const std::string &) const;
    Tensor residual(const Tensor &, const std::string &) const;
    Tensor middle(const Tensor &) const;

  public:
    void load(const std::filesystem::path &checkpoint);
    // BF16 pixels [1,3,1,H,W] in [-1,1] -> *unnormalized* VAE means
    // [1,48,1,H/16,W/16]. Latent-format normalization is a separate step.
    Tensor encode_first_frame(const Tensor &) const;
};

// Chunked causal decoder for the same 48-channel VAE. Cache state is local
// to one decode invocation, so concurrent requests cannot share histories.
class VAEDecoder {
    using Cache = std::unordered_map<std::string, Tensor>;
    Weights weights_;
    Tensor causal_conv(const Tensor &, const std::string &, Cache &, bool cache = true) const;
    Tensor spatial_conv(const Tensor &, const std::string &, int padding = 1) const;
    Tensor normalize(const Tensor &, const std::string &) const;
    Tensor residual(const Tensor &, const std::string &, Cache &) const;
    Tensor middle(const Tensor &, Cache &) const;
    Tensor upsample(const Tensor &, int stage, Cache &, std::unordered_set<std::string> &) const;

  public:
    void load(const std::filesystem::path &checkpoint);
    // BF16 unnormalized VAE latents [1,48,F,H,W] -> unclamped BF16 pixels
    // [1,3,4F-3,16H,16W]. No latent-format transform is performed here.
    Tensor decode(const Tensor &, const Event &event = {}, std::atomic<bool> *cancelled = nullptr) const;
};

} // namespace tc::wan22
