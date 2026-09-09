#pragma once

#include "../../backends/mlx.hpp"
#include <array>
#include <limits>

namespace tc::components {

struct WanPatchGrid {
    int frames;
    int height;
    int width;
    int tokens() const {
        require(frames > 0 && height > 0 && width > 0 &&
                    int64_t(frames) * height <= std::numeric_limits<int>::max() / width,
                "invalid or overflowing Wan patch grid");
        return frames * height * width;
    }
};

WanPatchGrid wan_patch_grid(const std::array<int, 3> &latent_shape,
                            const std::array<int, 3> &patch);
Tensor wan_timestep_embedding(const Tensor &, int frequency_dim, int max_period = 10000);
Tensor wan_dmd_step(const Tensor &noise_input, const Tensor &pred_noise,
                    float sigma, std::optional<float> next_sigma,
                    const std::optional<Tensor> &next_noise = {});
float wan_shifted_sigma(float timestep);
std::vector<float> wan_qad_sigmas(const std::vector<float> &timesteps);
std::pair<Tensor, Tensor> wan_rotary(const WanPatchGrid &);

} // namespace tc::components
