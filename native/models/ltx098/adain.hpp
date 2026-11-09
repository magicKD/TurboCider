#pragma once

#include "../../backends/mlx.hpp"

namespace tc::ltx098 {

// Lightricks' multiscale pipeline matches each channel's mean and sample
// standard deviation to the first-pass latent after spatial upsampling.
// Inputs/outputs are BFHWC. Statistics are accumulated in FP32 and rounded
// to BF16 like torch.std_mean on the official BF16 model. The epsilon guard
// prevents a degenerate constant channel from producing NaN/Inf.
inline Tensor adain_latents(const Tensor &upsampled, const Tensor &reference) {
    require(upsampled.ndim() == 5 && reference.ndim() == 5 &&
                upsampled.shape(0) == 1 && reference.shape(0) == 1 &&
                upsampled.shape(1) == reference.shape(1) &&
                upsampled.shape(4) == 128 && reference.shape(4) == 128 &&
                upsampled.shape(1) * upsampled.shape(2) * upsampled.shape(3) > 1 &&
                reference.shape(1) * reference.shape(2) * reference.shape(3) > 1,
            "LTX 0.9.8 AdaIN expects matching nondegenerate BFHWC latents");
    auto stats = [](const Tensor &input) {
        auto value = mx::astype(mx::astype(input, mx::bfloat16), mx::float32);
        const std::vector<int> axes{1, 2, 3};
        auto mean = mx::mean(value, axes, true);
        auto centered = value - mean;
        const int count = value.shape(1) * value.shape(2) * value.shape(3);
        auto variance = mx::mean(centered * centered, axes, true) * Tensor(float(count) / float(count - 1), mx::float32);
        auto deviation = mx::sqrt(variance);
        return std::pair<Tensor, Tensor>{mx::astype(mean, mx::bfloat16), mx::astype(deviation, mx::bfloat16)};
    };
    auto [source_mean, source_std] = stats(reference);
    auto [target_mean, target_std] = stats(upsampled);
    auto value = mx::astype(upsampled, mx::bfloat16) - target_mean;
    mx::eval(value);
    value = value / mx::maximum(target_std, Tensor(1.0e-6f, mx::bfloat16));
    mx::eval(value);
    value = value * source_std;
    mx::eval(value);
    return value + source_mean;
}

} // namespace tc::ltx098
