#ifndef LTX_RNG_H
#define LTX_RNG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t state;
    uint64_t increment;
    float spare_normal;
    int has_spare_normal;
} ltx_rng;

void ltx_rng_seed(ltx_rng *rng, uint64_t seed, uint64_t stream);
uint32_t ltx_rng_next_u32(ltx_rng *rng);
float ltx_rng_next_normal_f32(ltx_rng *rng);
void ltx_rng_fill_normal_f32(ltx_rng *rng, float *values, size_t count);
void ltx_rng_fill_normal_bf16(ltx_rng *rng, uint16_t *values, size_t count);

#ifdef __cplusplus
}
#endif

#endif
