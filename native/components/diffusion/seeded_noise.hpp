#pragma once

#include "../../core/common.hpp"
#include <cmath>
#include <cstdint>
#include <random>

namespace tc::components {

// Seed-compatible with the reference's contiguous FP32 torch.randn on ARM64.
// Use the standard-library MT19937 engine, low 24-bit uniforms, and the
// reference's eight-pair Box-Muller layout. No PyTorch code/library is linked.
// Wan latent element counts are always multiples of 16.
class WanNoiseGenerator {
    std::mt19937 engine_;

  public:
    explicit WanNoiseGenerator(uint64_t seed) : engine_(static_cast<uint32_t>(seed)) {}

    std::vector<float> normal(size_t count) {
        require(count >= 16 && count % 16 == 0, "Wan noise requires a multiple of 16 elements");
        std::vector<float> values(count);
        for (auto &value : values) value = float(engine_() & 0xffffffu) * (1.f / 16777216.f);
        for (size_t offset = 0; offset < count; offset += 16) {
            for (int pair = 0; pair < 8; ++pair) {
                const float radius = std::sqrt(-2.f * std::log(1.f - values[offset + pair]));
                const float theta = float(6.283185307179586476925286766559 * double(values[offset + pair + 8]));
                values[offset + pair] = radius * std::cos(theta);
                values[offset + pair + 8] = radius * std::sin(theta);
            }
        }
        return values;
    }
};

inline std::vector<float> wan_initial_noise(size_t count, uint64_t seed) {
    return WanNoiseGenerator(seed).normal(count);
}

} // namespace tc::components
