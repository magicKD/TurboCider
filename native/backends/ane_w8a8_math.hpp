#pragma once
#include "ane_runtime.hpp"
#include "../core/gguf_decode.hpp"
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <span>

namespace tc::ane {
// Versioned recipe: input signs then Sylvester H, R=D*H/sqrt(B).
// Both X and W use the same orthogonal R. NOT Comfy H256 ordering.
inline constexpr const char *w8a8_recipe = "sylvester-dh-b128-b512-rne-norm-f16-v2";
inline int rotation_sign(uint64_t seed, uint32_t index) {
    uint64_t x = seed + uint64_t(index) * 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return ((x ^ (x >> 31)) & 1) ? -1 : 1;
}
inline void rotate_block(std::span<float> values, uint64_t seed) {
    if (values.size() != 128 && values.size() != 512) throw std::invalid_argument("W8 rotation requires H128/H512");
    for (uint32_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) throw std::invalid_argument("W8 nonfinite source");
        values[i] *= float(rotation_sign(seed, i));
    }
    for (size_t half = 1; half < values.size(); half *= 2)
        for (size_t base = 0; base < values.size(); base += 2 * half)
            for (size_t i = 0; i < half; ++i) {
                const float a = values[base + i], b = values[base + i + half];
                values[base + i] = a + b; values[base + i + half] = a - b;
            }
    const float norm = 1.f / std::sqrt(float(values.size()));
    for (uint32_t i = 0; i < values.size(); ++i) {
        values[i] *= norm;
        if (!std::isfinite(values[i])) throw std::invalid_argument("W8 rotation overflow");
    }
}
inline uint16_t normalized_scale(float peak) {
    if (!std::isfinite(peak) || peak < 0) throw std::invalid_argument("W8 invalid peak");
    // Zero rows have q=0 and an exact nonzero safe scale. Tiny nonzero rows
    // use the smallest representable normalized FP16 scale, never divide by0.
    const float scale = peak == 0 ? 128.f : std::max(peak / 127.f * 128.f, 0x1p-24f);
    return gguf::float_to_fp16_rne(scale);
}
inline int8_t quantize_rotated(float value, uint16_t normalized) {
    const float scale = gguf::fp16_to_float(normalized);
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0) throw std::invalid_argument("W8 invalid quantizer input");
    const float x = std::clamp((value / scale) * 128.f, -127.f, 127.f);
    const float lower = std::floor(x), fraction = x - lower;
    const int integer = int(lower);
    return int8_t(integer + (fraction > .5f || (fraction == .5f && (integer & 1))));
}
}
