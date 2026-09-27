#include "../../native/models/qwen21/diagnostic_options.hpp"
#include <cassert>

using tc::qwen21::W8A8CallBudget;
using tc::qwen21::expected_w8a8_calls;

int main() {
    using namespace tc::qwen21;
    assert(tiled_prefill_layer_count("0") == 0);
    assert(tiled_prefill_layer_count("1") == 32);
    assert(tiled_prefill_layer_count("16") == 16);
    assert(tiled_prefill_layer_count("17") == -1);
    assert(db_cache_threshold(nullptr) == 0.08f);
    assert(db_cache_threshold("0.25") == 0.25f);
    assert(db_cache_threshold("nan") < 0);
    assert(db_cache_threshold("0.51") < 0);
    assert(db_cache_max_consecutive(nullptr) == 2);
    assert(db_cache_max_consecutive("1") == 1);
    assert(db_cache_max_consecutive("4") == 4);
    assert(db_cache_max_consecutive("8") == 8);
    assert(db_cache_max_consecutive("0") < 0);
    assert(db_cache_max_consecutive("9") < 0);
    assert(db_cache_max_consecutive("4oops") < 0);

    W8A8CallBudget budget{
        .steps = 20, .decode_layers = 32, .db_skipped_layers = 24,
        .total_tokens = 4096, .tile_rows = 1024,
    };
    assert(expected_w8a8_calls(budget) == 608);
    budget.db_cached_steps = 8;
    assert(expected_w8a8_calls(budget) == 416);
    budget.steps = 40;
    budget.db_cached_steps = 21;
    assert(expected_w8a8_calls(budget) == 744);

    budget.steps = 5;
    budget.db_cached_steps = 0;
    budget.final_reuse_layers = 16;
    budget.penultimate_reuse_layers = 16;
    assert(expected_w8a8_calls(budget) == 96);
    budget.final_reuse_layers = budget.penultimate_reuse_layers = 0;
    budget.tiled_prefill_layers = 16;
    assert(expected_w8a8_calls(budget) == 192);
    budget.last_target_only = true;
    assert(expected_w8a8_calls(budget) == 189);

    budget.prefix_hit = budget.tiled_prefix_reuse = true;
    budget.prefix_tokens = 2100; // incomplete prefix tile
    assert(expected_w8a8_calls(budget) == 159);
    budget.prefix_target_only = true;
    assert(expected_w8a8_calls(budget) == 144);
    budget.prefix_target_only = false;
    budget.prefix_tokens = 2048; // aligned prefix, no tile tail
    assert(expected_w8a8_calls(budget) == 144);

    budget.tiled_prefill_layers = 0;
    budget.decode_layers = 29; // 3/5/7 full GPU fallback
    assert(expected_w8a8_calls(budget) == 116);
    budget.decode_layers = 32;
    budget.decode_tiles = 2; // 1536-row rectangle on a 1024-row graph
    assert(expected_w8a8_calls(budget) == 256);
}
