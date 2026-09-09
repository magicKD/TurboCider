#include "wan.hpp"

#include <cmath>
#include <limits>

namespace tc::components {

std::pair<Tensor, Tensor> wan_rotary(const WanPatchGrid &grid) {
    const int tokens = grid.tokens();
    const std::array<int, 3> dimensions{44, 42, 42};
    std::vector<Tensor> cosines, sines;
    auto positions = mx::arange(tokens, mx::int32);
    for (int axis = 0; axis < 3; ++axis) {
        const int inner = axis == 0 ? grid.height * grid.width : (axis == 1 ? grid.width : 1);
        const int extent = axis == 0 ? grid.frames : (axis == 1 ? grid.height : grid.width);
        auto coordinates = mx::remainder(mx::floor_divide(positions, Tensor(inner)), Tensor(extent));
        auto frequency = 1.f / mx::power(Tensor(10000.f),
            mx::arange(0, dimensions[axis], 2, mx::float32) / float(dimensions[axis]));
        auto angle = mx::expand_dims(mx::astype(coordinates, mx::float32), 1) * frequency;
        cosines.push_back(mx::repeat(mx::cos(angle), 2, 1));
        sines.push_back(mx::repeat(mx::sin(angle), 2, 1));
    }
    return {mx::concatenate(cosines, 1), mx::concatenate(sines, 1)};
}

WanPatchGrid wan_patch_grid(const std::array<int, 3> &latent_shape,
                            const std::array<int, 3> &patch) {
    for (int i = 0; i < 3; ++i)
        require(latent_shape[i] > 0 && patch[i] > 0 && latent_shape[i] % patch[i] == 0,
                "Wan latent shape is not divisible by patch size");
    return {latent_shape[0] / patch[0], latent_shape[1] / patch[1],
            latent_shape[2] / patch[2]};
}

Tensor wan_timestep_embedding(const Tensor &timestep, int frequency_dim, int max_period) {
    require(frequency_dim > 0 && max_period > 1 && frequency_dim % 2 == 0,
            "invalid Wan timestep embedding geometry");
    auto half = frequency_dim / 2;
    auto frequencies = mx::exp(-std::log(float(max_period)) *
                                mx::arange(0, half, mx::float32) / float(half));
    auto args = mx::astype(mx::expand_dims(timestep, -1), mx::float32) * frequencies;
    return mx::concatenate({mx::cos(args), mx::sin(args)}, -1);
}

Tensor wan_dmd_step(const Tensor &noise_input, const Tensor &pred_noise,
                    float sigma, std::optional<float> next_sigma,
                    const std::optional<Tensor> &next_noise) {
    require(std::isfinite(sigma) && sigma >= 0.f && sigma <= 1.f,
            "invalid Wan DMD sigma");
    require(noise_input.shape() == pred_noise.shape() && noise_input.dtype() == pred_noise.dtype(),
            "Wan DMD prediction shape/dtype mismatch");
    auto clean = noise_input - Tensor(sigma, pred_noise.dtype()) * pred_noise;
    if (!next_sigma) return clean;
    require(*next_sigma >= 0.f && *next_sigma <= 1.f && next_noise,
            "Wan DMD re-noising requires next sigma and noise");
    require(next_noise->shape() == clean.shape() && next_noise->dtype() == clean.dtype(),
            "Wan DMD re-noising shape/dtype mismatch");
    return Tensor(1.f - *next_sigma, clean.dtype()) * clean +
           Tensor(*next_sigma, next_noise->dtype()) * *next_noise;
}

float wan_shifted_sigma(float timestep) {
#pragma clang fp contract(off)
    require(std::isfinite(timestep) && timestep >= 0.f && timestep <= 1000.f,
            "invalid Wan training timestep");
    const float sigma = timestep / 1000.f;
    return 8.f * sigma / (1.f + 7.f * sigma);
}

std::vector<float> wan_qad_sigmas(const std::vector<float> &timesteps) {
    require(!timesteps.empty(), "Wan QAD schedule cannot be empty");
    std::vector<float> result;
    result.reserve(timesteps.size());
    for (float timestep : timesteps) {
        require(std::isfinite(timestep) && timestep >= 0.f && timestep <= 1000.f,
                "invalid Wan QAD timestep");
        double nearest = std::numeric_limits<double>::infinity();
        float selected = 0.f;
        // QAD timesteps index the shifted training table. Applying the shift
        // directly to 757/522 is NOT the scheduler's nearest-timestep lookup.
        // Descending iteration also preserves argmin's first-index tie rule.
        for (int training = 1000; training >= 1; --training) {
            const float sigma = wan_shifted_sigma(float(training));
            const float shifted_timestep = sigma * 1000.f;
            const double distance = std::abs(double(shifted_timestep) - double(timestep));
            if (distance < nearest) { nearest = distance; selected = sigma; }
        }
        result.push_back(selected);
    }
    return result;
}

} // namespace tc::components
