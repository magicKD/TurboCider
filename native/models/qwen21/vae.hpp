#pragma once
#include "../../backends/mlx.hpp"

namespace tc::qwen21 {
// Single-frame RGBA VAE. Public tensors are NCHW; decode returns all four
// channels in model range, without silently dropping alpha or quantizing pixels.
class VAE {
  public:
    explicit VAE(const Weights &);
    Tensor decode(const Tensor &, const Event &, std::atomic<bool> &) const;
    Tensor encode(const Tensor &, const Event &, std::atomic<bool> &,
                  std::unordered_map<std::string, Tensor> *trace = nullptr) const;
    static std::string canonical_key(const std::string &);

  private:
    std::unordered_map<std::string, Tensor> values_;
    Tensor conv(const Tensor &, const std::string &, int padding = 1, int stride = 1) const;
    Tensor normalize(const Tensor &, const std::string &) const;
    Tensor residual(const Tensor &, const std::string &) const;
    Tensor middle(const Tensor &, const std::string &) const;
    Tensor mean_, std_;
};
} // namespace tc::qwen21
