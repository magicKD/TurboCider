#include "h3_quant_cache.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 7) return 2;
    char *tail = NULL;
    uint64_t identity = strtoull(argv[2], &tail, 0);
    if (!tail || *tail) return 2;
    unsigned blocks = (unsigned)strtoul(argv[3], NULL, 10);
    uint32_t hidden = (uint32_t)strtoul(argv[4], NULL, 10);
    uint32_t inner = (uint32_t)strtoul(argv[5], NULL, 10);
    uint32_t ffn = (uint32_t)strtoul(argv[6], NULL, 10);
    h3_quant_cache cache;
    char error[512];
    if (!h3_quant_cache_open(&cache, argv[1], identity, blocks,
                             hidden, inner, ffn, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    uint64_t expected_weights =
        (uint64_t)inner * 3u * hidden + (uint64_t)hidden * inner +
        (uint64_t)ffn * 2u * hidden + (uint64_t)hidden * ffn;
    uint64_t expected_scales =
        (uint64_t)inner * 3u + hidden + (uint64_t)ffn * 2u + hidden;
    if (cache.block_bytes != expected_weights + expected_scales * 4u ||
        !h3_quant_cache_layer_at(&cache, blocks - 1) ||
        h3_quant_cache_layer_at(&cache, blocks)) {
        h3_quant_cache_close(&cache);
        return 1;
    }
    printf("PASS: H3 quantized cache blocks=%u bytes=%" PRIu64 "\n",
           blocks, cache.block_bytes);
    h3_quant_cache_close(&cache);
    return 0;
}
