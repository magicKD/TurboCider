#pragma once
#include "../../backends/mlx.hpp"
namespace tc::z_image {
// Shared Comfy VAE path. Returns a lazy NCHW tensor; the caller retains weights
// and input until GPU completion. Includes the model's BF16 input boundary.
Tensor decode_vae(const Tensor &latent, const Weights &weights);
}
