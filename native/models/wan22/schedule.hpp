#pragma once

#include <array>
#include <span>

namespace tc::wan22 {

enum class Distillation { fastvideo_3, quanhaol_4 };

inline std::span<const float> denoising_sigmas(Distillation variant) {
    static constexpr std::array fastvideo{1.f, .9396722939424031f, .8452072538860105f};
    // Shift 8 applied to the four training-table indices 1000, 750, 500, 250.
    static constexpr std::array quanhaol{1.f, .96f, 8.f / 9.f, 8.f / 11.f};
    return variant == Distillation::quanhaol_4 ? std::span<const float>(quanhaol) : std::span<const float>(fastvideo);
}

inline float transformer_timestep(Distillation variant, int step) {
    if (variant == Distillation::quanhaol_4) {
        // The published I2V path narrows timesteps to BF16 before integer conversion.
        static constexpr std::array values{1000.f, 960.f, 888.f, 728.f};
        return values.at(step);
    }
    return denoising_sigmas(variant)[step] * 1000.f;
}

} // namespace tc::wan22
