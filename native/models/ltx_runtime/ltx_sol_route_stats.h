#ifndef LTX_SOL_ROUTE_STATS_H
#define LTX_SOL_ROUTE_STATS_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t exact_blocks;
    uint64_t total_blocks;
    uint64_t exact_token_pairs;
    uint64_t total_token_pairs;
} ltx_sol_route_counts;

/* Layout matches ltx_sol_build_routes: [head, query_block, key_word].
 * Count only valid tokens in the final 64-token tile, never padded lanes. */
static inline int ltx_sol_count_routes(const uint32_t *routes,
        size_t word_count, uint32_t rows, uint32_t heads,
        ltx_sol_route_counts *out) {
    if (!routes || !out || !rows || rows > 16384u || !heads || heads > 1024u)
        return 0;
    uint32_t blocks = (rows + 63u) / 64u;
    uint32_t words = (blocks + 31u) / 32u;
    if (word_count != (uint64_t)heads * blocks * words) return 0;
    *out = (ltx_sol_route_counts){0};
    out->total_blocks = (uint64_t)heads * blocks * blocks;
    out->total_token_pairs = (uint64_t)heads * rows * rows;
    for (uint32_t head = 0; head < heads; ++head) {
        for (uint32_t query = 0; query < blocks; ++query) {
            uint32_t query_tokens = rows - query * 64u;
            if (query_tokens > 64u) query_tokens = 64u;
            size_t base = ((size_t)head * blocks + query) * words;
            for (uint32_t key = 0; key < blocks; ++key) {
                if (!((routes[base + key / 32u] >> (key & 31u)) & 1u)) continue;
                uint32_t key_tokens = rows - key * 64u;
                if (key_tokens > 64u) key_tokens = 64u;
                ++out->exact_blocks;
                out->exact_token_pairs += (uint64_t)query_tokens * key_tokens;
            }
        }
    }
    return 1;
}

#endif
