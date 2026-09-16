/* Native production attention core benchmark; no weights or Python backend.
 * Timings include RoPE, packing, summaries/routing, attention and gate/unpack.
 * Random-tensor approximation error is NOT a video quality metric. */
#include "../../native/models/ltx_runtime/ltx_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t rng = 42;
static float random_float(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((float)(rng >> 8) / 16777216.0f - 0.5f) * 3.4641016f;
}
static uint16_t bf16(float value) {
    uint32_t bits; memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
static float f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result; memcpy(&result, &bits, sizeof(result)); return result;
}
static unsigned route_bit(const uint32_t *route, size_t logical_index, unsigned blocks) {
    size_t row = logical_index / blocks;
    unsigned block = (unsigned)(logical_index % blocks);
    return (route[row * ((blocks + 31u) / 32u) + block / 32u] >> (block & 31u)) & 1u;
}
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *values, unsigned n) {
    qsort(values, n, sizeof(*values), compare_double);
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) * .5;
}

/* Small-shape scalar oracle reads the packed BF16 inputs and actual routes.
 * It checks the attention math independently of both Metal implementations;
 * route-pattern correctness is checked separately below. */
static double oracle_error(ltx_gpu_buffer **b, unsigned n, unsigned h,
                           unsigned blocks, unsigned mode) {
    const uint16_t *q = ltx_gpu_buffer_contents(b[8]);
    const uint16_t *k = ltx_gpu_buffer_contents(b[9]);
    const uint16_t *v = ltx_gpu_buffer_contents(b[10]);
    const uint16_t *kc = ltx_gpu_buffer_contents(b[13]);
    const uint16_t *vs = ltx_gpu_buffer_contents(b[14]);
    const uint32_t *routes = ltx_gpu_buffer_contents(b[16]);
    const uint16_t *out = ltx_gpu_buffer_contents(b[7]);
    const uint16_t *gate = ltx_gpu_buffer_contents(b[5]);
    double error2 = 0, norm2 = 0;
    for (unsigned head = 0; head < h; ++head) {
        for (unsigned row = 0; row < n; ++row) {
            double maximum = -INFINITY, denominator = 0, numerator[128] = {0};
            for (unsigned block = 0; block < blocks; ++block) {
                int exact = route_bit(routes, (head * blocks + row / 64) * blocks + block, blocks);
                if (!exact && (mode == 1 || mode == 4)) continue;
                unsigned length = n - block * 64 < 64 ? n - block * 64 : 64;
                for (unsigned token = 0; token < (exact ? length : 1); ++token) {
                    size_t offset = exact ? ((size_t)head * n + block * 64 + token) * 128 :
                                           ((size_t)head * blocks + block) * 128;
                    const uint16_t *key = (exact ? k : kc) + offset;
                    const uint16_t *value = (exact ? v : vs) + offset;
                    double score = 0;
                    for (unsigned d = 0; d < 128; ++d)
                        score += (double)f32(q[((size_t)head * n + row)*128+d]) * f32(key[d]);
                    score /= sqrt(128.0);
                    double next = fmax(score, maximum);
                    double correction = exp(maximum - next), probability = exp(score - next);
                    denominator = denominator * correction + probability * (exact ? 1 : length);
                    for (unsigned d = 0; d < 128; ++d)
                        numerator[d] = numerator[d] * correction + probability * f32(value[d]);
                    maximum = next;
                }
            }
            for (unsigned d = 0; d < 128; ++d) {
                double expected = numerator[d] / denominator * f32(gate[(size_t)row*h+head]);
                double actual = f32(out[((size_t)row * h + head)*128+d]);
                error2 += (actual-expected)*(actual-expected);
                norm2 += expected*expected;
            }
        }
    }
    return sqrt(error2 / fmax(norm2, 1e-30));
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 10 && argc != 11 && argc != 12) {
        fprintf(stderr, "usage: %s ABSOLUTE_SHADERS ROWS HEADS RUNS TAU [MODE RADIUS ANCHOR_STRIDE TOKENS_PER_FRAME [KEEP_BLOCKS [FIXTURE_DIRECTORY]]]\n", argv[0]);
        return 2;
    }
    unsigned n = (unsigned)strtoul(argv[2], NULL, 10);
    unsigned h = (unsigned)strtoul(argv[3], NULL, 10);
    unsigned runs = (unsigned)strtoul(argv[4], NULL, 10);
    float tau = strtof(argv[5], NULL);
    ltx_sparse_pattern pattern = {0, 1, 0, 0, 0};
    if (argc >= 10) {
        pattern.mode = (unsigned)strtoul(argv[6], NULL, 10);
        pattern.radius = (unsigned)strtoul(argv[7], NULL, 10);
        pattern.anchor_stride = (unsigned)strtoul(argv[8], NULL, 10);
        pattern.tokens_per_frame = (unsigned)strtoul(argv[9], NULL, 10);
        if (argc >= 11) pattern.keep_blocks = (unsigned)strtoul(argv[10], NULL, 10);
    }
    if (!n || n > 16384 || !h || h > 64 || !runs || runs > 100 || !isfinite(tau))
        return 2;
    unsigned blocks = (n + 63) / 64;
    size_t elements = (size_t)n * h * 128;
    size_t summaries = (size_t)blocks * h * 128;
    size_t routes = (size_t)blocks * blocks * h;
    char error[2048] = {0};
    ltx_gpu *gpu = ltx_gpu_create(argv[1], error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 1; }
    enum {Q,K,V,COS,SIN,GATE,DENSE,SOL,PQ,PK,PV,PO,QC,KC,VS,TH,RO,COUNT};
    ltx_gpu_buffer *b[COUNT] = {0};
    size_t bytes[COUNT] = {
        elements*2,elements*2,elements*2,elements,elements,(size_t)n*h*2,
        elements*2,elements*2,elements*2,elements*2,elements*2,elements*2,
        summaries*4,summaries*2,summaries*2,(size_t)h*blocks*4,
        (size_t)ltx_sparse_route_scratch_words(n,h)*sizeof(uint32_t)
    };
    int status = 1;
    for (unsigned i = 0; i < COUNT; ++i) {
        b[i] = ltx_gpu_buffer_new(gpu, bytes[i], error, sizeof(error));
        if (!b[i]) goto done;
        memset(ltx_gpu_buffer_contents(b[i]), 0, bytes[i]);
    }
    for (unsigned j = Q; j <= V; ++j) {
        uint16_t *data = ltx_gpu_buffer_contents(b[j]);
        for (size_t i = 0; i < elements; ++i) data[i] = bf16(random_float());
    }
    for (size_t i = 0; i < elements / 2; ++i) {
        ((uint16_t *)ltx_gpu_buffer_contents(b[COS]))[i] = bf16(cosf((float)(i % 97) * .01f));
        ((uint16_t *)ltx_gpu_buffer_contents(b[SIN]))[i] = bf16(sinf((float)(i % 97) * .01f));
    }
    for (size_t i = 0; i < (size_t)n*h; ++i)
        ((uint16_t *)ltx_gpu_buffer_contents(b[GATE]))[i] = bf16(.7f);
    if (argc == 12) {
        const char *names[] = {"query.bf16", "key.bf16", "value.bf16",
                              "cosine.bf16", "sine.bf16", "gate.bf16"};
        for (unsigned i = 0; i < 6; ++i) {
            char path[4096];
            int length = snprintf(path, sizeof(path), "%s/%s", argv[11], names[i]);
            if (length < 0 || (size_t)length >= sizeof(path)) {
                snprintf(error, sizeof(error), "fixture path too long"); goto done;
            }
            FILE *file = fopen(path, "rb");
            if (!file) { snprintf(error, sizeof(error), "cannot read fixture %s", names[i]); goto done; }
            size_t count = fread(ltx_gpu_buffer_contents(b[i]), 1, bytes[i], file);
            int extra = fgetc(file), failed = ferror(file), closed = fclose(file);
            if (count != bytes[i] || extra != EOF || failed || closed) {
                snprintf(error, sizeof(error), "fixture size/read mismatch for %s", names[i]); goto done;
            }
        }
    }
    double timings[2][100];
    for (unsigned iteration = 0; iteration < runs + 2; ++iteration) {
        for (unsigned position = 0; position < 2; ++position) {
            unsigned candidate = (position + iteration) % 2;
            if (candidate && n <= 257)
                memset(ltx_gpu_buffer_contents(b[RO]), 0xa5, bytes[RO]);
            double start = now();
            int ok = candidate ? (argc == 6 ? ltx_gpu_self_attention_core_sol_bf16(
                gpu,b[SOL],b[Q],b[K],b[V],b[COS],b[SIN],b[GATE],
                b[PQ],b[PK],b[PV],b[PO],b[QC],b[KC],b[VS],b[TH],b[RO],
                n,h,128,1.f/sqrtf(128.f),tau,0,0,0,0,error,sizeof(error)) :
                ltx_gpu_self_attention_core_sparse_bf16(
                gpu,b[SOL],b[Q],b[K],b[V],b[COS],b[SIN],b[GATE],
                b[PQ],b[PK],b[PV],b[PO],b[QC],b[KC],b[VS],b[TH],b[RO],
                n,h,128,1.f/sqrtf(128.f),tau,0,0,0,0,&pattern,error,sizeof(error))) :
                ltx_gpu_self_attention_core_mps_bf16(
                gpu,b[DENSE],b[Q],b[K],b[V],b[COS],b[SIN],b[GATE],
                n,h,128,1.f/sqrtf(128.f),error,sizeof(error));
            if (!ok) goto done;
            if (iteration >= 2) timings[candidate][iteration-2] = now()-start;
        }
    }
    double difference = 0, norm = 0, other = 0, dot = 0, maximum = 0;
    double head_difference[64] = {0}, head_norm[64] = {0}, head_maximum[64] = {0};
    uint16_t *dense = ltx_gpu_buffer_contents(b[DENSE]);
    uint16_t *sol = ltx_gpu_buffer_contents(b[SOL]);
    for (size_t i = 0; i < elements; ++i) {
        double x = f32(dense[i]), y = f32(sol[i]);
        if (!isfinite(x) || !isfinite(y)) {
            snprintf(error, sizeof(error), "non-finite attention output"); goto done;
        }
        difference += (x-y)*(x-y); norm += x*x; other += y*y; dot += x*y;
        maximum = fmax(maximum, fabs(x-y));
        /* Core outputs are row-head-dim, unlike the packed head-row-dim inputs. */
        unsigned head = (unsigned)((i / 128u) % h);
        head_difference[head] += (x-y)*(x-y);
        head_norm[head] += x*x;
        head_maximum[head] = fmax(head_maximum[head], fabs(x-y));
    }
    size_t exact = 0;
    uint32_t *route = ltx_gpu_buffer_contents(b[RO]);
    for (size_t i = 0; i < routes; ++i) exact += route_bit(route, i, blocks);
    if (blocks % 32u) {
        unsigned words = (blocks + 31u) / 32u;
        uint32_t valid = (1u << (blocks % 32u)) - 1u;
        for (size_t i = 0; i < (size_t)h * blocks; ++i) {
            if (route[i * words + words - 1u] & ~valid) {
                snprintf(error, sizeof(error), "nonzero packed route padding"); goto done;
            }
        }
    }
    /* For structured modes, verify every materialized route independently. */
    if (pattern.mode == 1 || pattern.mode == 2 || ((pattern.mode == 4 || pattern.mode == 5) && n <= 257)) {
        for (unsigned qb = 0; qb < blocks; ++qb) {
            for (unsigned kb = 0; kb < blocks; ++kb) {
                int expected = abs((int)qb - (int)kb) <= (int)pattern.radius;
                if (pattern.tokens_per_frame) {
                    unsigned qf = qb*64/pattern.tokens_per_frame;
                    unsigned ql = ((qb+1)*64 < n ? (qb+1)*64-1 : n-1)/pattern.tokens_per_frame;
                    unsigned kf = kb*64/pattern.tokens_per_frame;
                    unsigned kl = ((kb+1)*64 < n ? (kb+1)*64-1 : n-1)/pattern.tokens_per_frame;
                    expected = kf <= ql+pattern.radius && qf <= kl+pattern.radius;
                }
                expected |= pattern.anchor_stride && kb % pattern.anchor_stride == 0;
                for (unsigned head = 0; head < h; ++head) {
                    int selected = expected;
                    if (pattern.mode == 4 || pattern.mode == 5) {
                        const float *qc = ltx_gpu_buffer_contents(b[QC]);
                        const uint16_t *kc = ltx_gpu_buffer_contents(b[KC]);
                        float scores[256];
                        for (unsigned key_block = 0; key_block < blocks; ++key_block) {
                            float score = 0;
                            for (unsigned d = 0; d < 128; ++d)
                                score = fmaf(qc[(head*blocks+qb)*128+d],
                                             f32(kc[(head*blocks+key_block)*128+d]), score);
                            scores[key_block] = score;
                        }
                        unsigned rank = 0;
                        for (unsigned key_block = 0; key_block < blocks; ++key_block)
                            rank += scores[key_block] > scores[kb] ||
                                    (scores[key_block] == scores[kb] && key_block < kb);
                        selected |= rank < pattern.keep_blocks;
                    }
                    if (route_bit(route, (head*blocks+qb)*blocks+kb, blocks) != (unsigned)selected) {
                        snprintf(error, sizeof(error), "incorrect structured route"); goto done;
                    }
                }
            }
        }
    }
    double oracle = n <= 257 ? oracle_error(b,n,h,blocks,pattern.mode) : -1;
    if (!isfinite(oracle) || oracle > .03) {
        snprintf(error, sizeof(error), "scalar-oracle relative L2 failed: %.9g", oracle); goto done;
    }
    double dense_time = median(timings[0], runs), sol_time = median(timings[1], runs);
    printf("{\"rows\":%u,\"heads\":%u,\"dim\":128,\"runs\":%u,\"tau\":%.7g,"
           "\"mode\":%u,\"radius\":%u,\"anchor_stride\":%u,\"tokens_per_frame\":%u,\"keep_blocks\":%u,"
           "\"dense_seconds\":%.9g,\"sol_seconds\":%.9g,\"speedup\":%.9g,"
           "\"exact_block_fraction\":%.9g,\"relative_l2\":%.9g,\"cosine\":%.9g,"
           "\"max_abs\":%.9g,\"scalar_oracle_relative_l2\":%.9g,\"captured_input\":%s,\"per_head\":[",n,h,runs,tau,
           pattern.mode,pattern.radius,pattern.anchor_stride,pattern.tokens_per_frame,pattern.keep_blocks,
           dense_time,sol_time,dense_time/sol_time,
           (double)exact/routes,sqrt(difference/fmax(norm,1e-30)),
           dot/sqrt(fmax(norm*other,1e-60)),maximum,oracle,argc == 12 ? "true" : "false");
    for (unsigned head = 0; head < h; ++head) {
        size_t head_exact = 0, head_routes = (size_t)blocks * blocks;
        for (size_t i = 0; i < head_routes; ++i)
            head_exact += route_bit(route, head * head_routes + i, blocks);
        printf("%s{\"head\":%u,\"squared_error\":%.12g,\"reference_squared_norm\":%.12g,"
               "\"max_abs\":%.9g,\"exact_block_fraction\":%.9g,\"relative_l2\":",
               head ? "," : "", head, head_difference[head], head_norm[head],
               head_maximum[head], (double)head_exact / head_routes);
        if (head_norm[head] > 0)
            printf("%.9g}", sqrt(head_difference[head] / head_norm[head]));
        else
            printf("null}"); /* Relative error is undefined for a zero-energy head. */
    }
    printf("],\"route_layout\":\"packed-u32-v1\",\"route_scratch_bytes\":%zu}\n", bytes[RO]);
    status = 0;
done:
    if (status) fprintf(stderr, "%s\n", error);
    for (unsigned i = 0; i < COUNT; ++i) ltx_gpu_buffer_free(b[i]);
    ltx_gpu_free(gpu);
    return status;
}
