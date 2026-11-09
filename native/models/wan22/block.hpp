#pragma once

#include "../../backends/mlx.hpp"

namespace tc::wan22 {

struct BlockConfig {
    int hidden = 3072;
    int heads = 24;
    int intermediate = 14336;
    float epsilon = 1e-6f;
};

// One Wan2.2 TI2V block using the original Diffusers safetensors keys.
// Two modulation groups let the clean first latent frame use timestep zero
// while all remaining frames share the current diffusion timestep.
class Block {
    Weights weights_;
    BlockConfig config_;

  public:
    using Trace = std::function<void(const std::string &, const Tensor &)>;
    void load(const std::filesystem::path &, int index, BlockConfig = {});
    Tensor forward(const Tensor &hidden, const Tensor &context, const Tensor &modulation,
                   const Tensor &cosine, const Tensor &sine, int first_frame_tokens,
                   const Trace &trace = {}) const;
};

} // namespace tc::wan22
