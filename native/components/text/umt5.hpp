#pragma once

#include "../../backends/mlx.hpp"
#include "../../core/tokenizer.hpp"

namespace tc::components {

struct UMT5Config {
    int layers = 24;
    int hidden = 4096;
    int heads = 64;
    int head_dim = 64;
    int intermediate = 10240;
    int buckets = 32;
    int max_distance = 128;
    float epsilon = 1e-6f;
};

// Bidirectional encoder only; no decoder, sampling, tokenizer or process state.
// Weight ownership remains with the model session. Wan uses BF16 UMT5 and
// converts the final conditioning to FP16 at the DiT boundary.
class UMT5Encoder {
    const Weights &weights_;
    UMT5Config config_;
    Tensor weight(const std::string &) const;
    Tensor project(const Tensor &, const std::string &) const;
    Tensor normalize(const Tensor &, const std::string &) const;

  public:
    using Trace = std::function<void(const std::string &, const Tensor &)>;
    static Tensor activation(const Tensor &);
    explicit UMT5Encoder(const Weights &, UMT5Config = {});
    Tensor embed(const Tokens &) const;
    Tensor relative_buckets(int count) const;
    Tensor block(const Tensor &, const Tensor &buckets, int valid, int layer,
                 const Trace &trace = {}) const;
    Tensor finish(const Tensor &, int valid) const;
    Tensor encode(const Tokens &, const Event &, std::atomic<bool> &) const;
};

} // namespace tc::components
