#include "ltx_rng.h"

#include <math.h>
#include <string.h>

static uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

uint32_t ltx_rng_next_u32(ltx_rng *rng) {
    uint64_t old_state = rng->state;
    rng->state =
        old_state * UINT64_C(6364136223846793005) + rng->increment;
    uint32_t xor_shifted =
        (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
    uint32_t rotation = (uint32_t)(old_state >> 59u);
    return (xor_shifted >> rotation) |
        (xor_shifted << ((0u - rotation) & 31u));
}

void ltx_rng_seed(ltx_rng *rng, uint64_t seed, uint64_t stream) {
    if (!rng) return;
    rng->state = 0;
    rng->increment = (stream << 1u) | 1u;
    rng->spare_normal = 0.0f;
    rng->has_spare_normal = 0;
    (void)ltx_rng_next_u32(rng);
    rng->state += seed;
    (void)ltx_rng_next_u32(rng);
}

static float uniform_signed_open(ltx_rng *rng) {
    double unit =
        ((double)ltx_rng_next_u32(rng) + 0.5) / 4294967296.0;
    return (float)(unit * 2.0 - 1.0);
}

float ltx_rng_next_normal_f32(ltx_rng *rng) {
    if (!rng) return 0.0f;
    if (rng->has_spare_normal) {
        rng->has_spare_normal = 0;
        return rng->spare_normal;
    }
    float first = 0.0f;
    float second = 0.0f;
    float radius2 = 0.0f;
    do {
        first = uniform_signed_open(rng);
        second = uniform_signed_open(rng);
        radius2 = first * first + second * second;
    } while (radius2 <= 0.0f || radius2 >= 1.0f);
    float scale = sqrtf(-2.0f * logf(radius2) / radius2);
    rng->spare_normal = second * scale;
    rng->has_spare_normal = 1;
    return first * scale;
}

void ltx_rng_fill_normal_f32(ltx_rng *rng, float *values, size_t count) {
    if (!rng || (!values && count)) return;
    for (size_t index = 0; index < count; index++)
        values[index] = ltx_rng_next_normal_f32(rng);
}

void ltx_rng_fill_normal_bf16(ltx_rng *rng, uint16_t *values,
                              size_t count) {
    if (!rng || (!values && count)) return;
    for (size_t index = 0; index < count; index++)
        values[index] = f32_to_bf16(ltx_rng_next_normal_f32(rng));
}
