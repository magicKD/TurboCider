#include "conditioner_math.hpp"

#include <cmath>

namespace tc::h3_mlx {

// FastVideo creates the text MRoPE table with NumPy float32 ufuncs.  NumPy's
// contiguous ARM64 path uses the Cody-Waite reduction and FMA polynomials
// below, which are intentionally not bitwise identical to scalar libm.
Float32SinCos numpy_float32_sin_cos(float input) {
    constexpr float two_over_pi = 0x1.45f306p-1f;
    constexpr float pio2_high = -0x1.921fb0p+00f;
    constexpr float pio2_medium = -0x1.5110b4p-22f;
    constexpr float pio2_low = -0x1.846988p-48f;
    constexpr float rounding_magic = 0x1.800000p+23f;

    float quadrant = input * two_over_pi;
    quadrant = (quadrant + rounding_magic) - rounding_magic;
    float reduced = std::fma(quadrant, pio2_high, input);
    reduced = std::fma(quadrant, pio2_medium, reduced);
    reduced = std::fma(quadrant, pio2_low, reduced);
    const float squared = reduced * reduced;

    float cosine = std::fma(0x1.98e616p-16f, squared,
                            -0x1.6c06dcp-10f);
    cosine = std::fma(cosine, squared, 0x1.55553cp-05f);
    cosine = std::fma(cosine, squared, -0x1.000000p-01f);
    cosine = std::fma(cosine, squared, 0x1.000000p+00f);

    float sine = std::fma(0x1.7d3bbcp-19f, squared,
                          -0x1.a06bbap-13f);
    sine = std::fma(sine, squared, 0x1.11119ap-07f);
    sine = std::fma(sine, squared, -0x1.555556p-03f);
    sine = std::fma(sine, squared, 0.f);
    sine = std::fma(sine, reduced, reduced);

    const int base_quadrant = static_cast<int>(quadrant);
    auto select = [&](int mapped_quadrant) {
        float value = (mapped_quadrant & 1) == 0 ? sine : cosine;
        return (mapped_quadrant & 2) == 2 ? -value : value;
    };
    return {select(base_quadrant), select(base_quadrant + 1)};
}

} // namespace tc::h3_mlx
