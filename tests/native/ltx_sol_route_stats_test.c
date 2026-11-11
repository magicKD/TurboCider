#include "ltx_sol_route_stats.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    ltx_sol_route_counts counts;
    /* Two heads, 65 tokens: head 0 keeps both diagonal tiles (4096+1
     * token pairs); head 1 keeps the two off-diagonal tiles (64+64). */
    uint32_t routes[] = {1u, 2u, 2u, 1u};
    assert(ltx_sol_count_routes(routes, 4, 65, 2, &counts));
    assert(counts.exact_blocks == 4 && counts.total_blocks == 8);
    assert(counts.exact_token_pairs == 4225 && counts.total_token_pairs == 8450);
    routes[2] = routes[3] = 0;
    assert(ltx_sol_count_routes(routes, 4, 65, 2, &counts));
    assert(counts.exact_blocks == 2 && counts.exact_token_pairs == 4097);
    /* Ignore padding bits in the final route word. */
    uint32_t dense[] = {~0u, ~0u};
    assert(ltx_sol_count_routes(dense, 2, 65, 1, &counts));
    assert(counts.exact_token_pairs == 4225 && counts.exact_blocks == 4);
    uint32_t singleton = 1;
    assert(ltx_sol_count_routes(&singleton, 1, 1, 1, &counts));
    assert(counts.exact_token_pairs == 1 && counts.total_token_pairs == 1);
    /* Cross a 32-bit route-word boundary and both query/key tail cases. */
    uint32_t wide[68] = {0};
    wide[1] = 2u;
    wide[66] = 1u;
    assert(ltx_sol_count_routes(wide, 68, 2113, 1, &counts));
    assert(counts.exact_blocks == 2 && counts.exact_token_pairs == 128);
    assert(!ltx_sol_count_routes(dense, 1, 65, 1, &counts));
    assert(!ltx_sol_count_routes(dense, 2, 0, 1, &counts));
    assert(!ltx_sol_count_routes(dense, 2, 16385, 1, &counts));
    puts("PASS: Sol route density, unequal tail tile, head layout and padding bits");
}
