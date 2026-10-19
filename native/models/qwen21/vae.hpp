#pragma once
#include "../../backends/mlx.hpp"

namespace tc::qwen21 {
// Single-frame RGBA VAE. Public tensors are NCHW; decode returns all four
// channels in model range, without silently dropping alpha or quantizing pixels.
class VAE {
  public:
    enum class DecodeConvolution { Original, FP16WithFP32State };
    explicit VAE(const Weights &, DecodeConvolution = DecodeConvolution::Original,
                 bool fused_decode_norm = false);
    Tensor decode(const Tensor &, const Event &, std::atomic<bool> &) const;
    struct Decoded {
        Tensor pixels;
        bool fp16_nonfinite_fallback = false;
    };
    // A nonfinite optional FP16 attempt cannot become a published image.
    // Retry the COMPLETE decoder with original convolution precision; retain
    // the independently selected FP32 normalization recipe. Other errors and
    // cancellation propagate, rather than being disguised as precision fallback.
    Decoded decode_checked(const Tensor &, const Event &, std::atomic<bool> &) const;
    Tensor encode(const Tensor &, const Event &, std::atomic<bool> &,
                  std::unordered_map<std::string, Tensor> *trace = nullptr) const;
    static std::string canonical_key(const std::string &);

  private:
    std::unordered_map<std::string, Tensor> values_;
    // Optional decoder-only compute view. Original checkpoint arrays, encoder,
    // normalization, attention, shortcuts, bias and residual state stay intact.
    std::unordered_map<std::string, Tensor> decode_fp16_weights_;
    bool fused_decode_norm_ = false;
    Tensor conv(const Tensor &, const std::string &, int padding = 1, int stride = 1) const;
    Tensor normalize(const Tensor &, const std::string &) const;
    Tensor normalized_activation(const Tensor &, const std::string &) const;
    Tensor residual(const Tensor &, const std::string &) const;
    Tensor middle(const Tensor &, const std::string &) const;
    Tensor mean_, std_;
};
} // namespace tc::qwen21
