#pragma once

#include "../../backends/mlx.hpp"

namespace tc::components {

// Native TAEHV decoder. The Python .pth -> safetensors conversion is an
// offline step; this class only consumes an immutable converted checkpoint.
class TAEHVDecoder {
    std::unordered_map<std::string, Tensor> weights_;
    int latent_channels_;
    int trim_frames_ = 3;

    Tensor conv(const Tensor &, const std::string &, int stride = 1) const;
    Tensor conv1(const Tensor &, const std::string &) const;
    Tensor memory_block(const Tensor &, const Tensor &, int) const;
    Tensor temporal_grow(const Tensor &, int, int) const;
    Tensor past(const Tensor &, int batch) const;

  public:
    TAEHVDecoder(const std::filesystem::path &, int latent_channels);
    Tensor decode_ntchw(const Tensor &, const Event &, std::atomic<bool> &) const;
};

} // namespace tc::components
