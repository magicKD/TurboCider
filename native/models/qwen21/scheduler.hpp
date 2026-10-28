#pragma once
#include "../../backends/mlx.hpp"
#include "viggle_adapter.hpp"
#include "checkpoint_recipe.hpp"

namespace tc::qwen21 {
inline Tensor turbo_sigmas(int width,int height,int steps=8) {
    require(width>0 && height>0 && width%16==0 && height%16==0 && steps==8,
        "Qwen21 Turbo requires aligned canvas and its authored eight-step schedule");
    return Tensor(turbo_sigma_nodes.data(),{int(turbo_sigma_nodes.size())},mx::float32);
}
inline Tensor sigmas(int width, int height, int steps) {
    require(width > 0 && height > 0 && width % 16 == 0 && height % 16 == 0 && steps > 0,
            "invalid Qwen21 schedule dimensions/steps");
    // Explicit degenerate Euler schedule avoids the terminal stretch 0/0.
    if (steps == 1) return Tensor(std::vector<float>{1.f, 0.f}.data(), {2}, mx::float32);
    double slope = (0.9 - 0.5) / (8192.0 - 256.0);
    float mu = float(slope * double(width) * double(height) / 256.0 + 0.5 - slope * 256.0);
    auto initial = mx::linspace(1.f, 1.f / steps, steps, mx::float32);
    auto e = mx::exp(Tensor(mu));
    auto shifted = e / (e + (Tensor(1.f) / initial - 1.f));
    auto complement = 1.f - shifted;
    auto scale = slice_axis(complement, 0, steps - 1, steps) / .98f;
    return mx::concatenate({1.f - complement / scale, mx::zeros({1}, mx::float32)});
}

// Viggle v0.2.1 student: shift the authored six/five raw sigma nodes with
// the same resolution-dependent Qwen21 mu, but do not apply the base model's
// terminal 0.02 stretch (the adapter ships shift_terminal: null).
inline Tensor viggle_v021_sigmas(int width, int height, int steps = 6) {
    require(width > 0 && height > 0 && width % 16 == 0 && height % 16 == 0,
            "invalid Viggle Qwen21 schedule dimensions");
    require(!viggle_v021_sampler_identity(steps).empty(), "Viggle Qwen21 sampler requires five or six steps");
    double slope = (0.9 - 0.5) / (8192.0 - 256.0);
    float mu = float(slope * double(width) * double(height) / 256.0 + 0.5 - slope * 256.0);
    // Five-step usage removes .9375; it is not the first five six-step nodes,
    // a base Euler linspace, or the old four-step v0.1 adapter's schedule.
    const float six[] = {1.f, .9375f, .875f, .75f, .5f, .25f};
    const float five[] = {1.f, .875f, .75f, .5f, .25f};
    auto raw = Tensor(steps == 5 ? five : six, {steps}, mx::float32);
    auto e = mx::exp(Tensor(mu));
    auto shifted = e / (e + (Tensor(1.f) / raw - 1.f));
    return mx::concatenate({shifted, mx::zeros({1}, mx::float32)});
}
} // namespace tc::qwen21
