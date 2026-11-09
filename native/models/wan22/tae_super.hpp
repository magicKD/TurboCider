#pragma once

#include "../../backends/mlx.hpp"

namespace tc::wan22 {

// Experimental TAEW2.2-Super decoder. Input is the normalized diffusion
// latent [1,48,F,H,W], NOT the full VAE's raw latent space. Output pixels
// are [1,3,4F-3,16H,16W] in [0,1]. This is an approximate decoder.
class TAESuperDecoder {
    std::unordered_map<std::string, Tensor> weights_;
    Tensor conv(const Tensor &, const std::string &, int padding = 0, int groups = 1) const;
    Tensor memory_block(const Tensor &, int index) const;
    Tensor temporal_grow(const Tensor &, int index, int stride) const;

  public:
    explicit TAESuperDecoder(const std::filesystem::path &checkpoint);
    Tensor decode(const Tensor &, const Event &event = {}, std::atomic<bool> *cancelled = nullptr) const;
};

} // namespace tc::wan22
