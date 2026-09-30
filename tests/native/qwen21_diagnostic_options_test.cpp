#include "../../native/models/qwen21/diagnostic_options.hpp"
#include <array>
#include <cassert>
#include <optional>

using tc::qwen21::W8A8CallBudget;
using tc::qwen21::expected_w8a8_calls;

namespace {
constexpr std::array cache_environment{
    "TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC",
    "TURBOCIDER_QWEN21_DBCACHE_THRESHOLD",
    "TURBOCIDER_QWEN21_DBCACHE_MAX_CONSECUTIVE",
    "TURBOCIDER_QWEN21_DBCACHE_FRONT_BLOCKS",
    "TURBOCIDER_QWEN21_DBCACHE_WARMUP_STEPS",
};
struct CacheEnvironment {
    std::array<std::optional<std::string>, cache_environment.size()> saved;
    CacheEnvironment() {
        for (size_t i = 0; i < saved.size(); ++i) {
            if (const auto *value = std::getenv(cache_environment[i])) saved[i] = value;
            assert(unsetenv(cache_environment[i]) == 0);
        }
    }
    ~CacheEnvironment() {
        for (size_t i = 0; i < saved.size(); ++i) {
            if (saved[i]) setenv(cache_environment[i], saved[i]->c_str(), 1);
            else unsetenv(cache_environment[i]);
        }
    }
};

void test_cache_options() {
    using tc::qwen21::db_cache_options;
    CacheEnvironment environment;
    tc::Request request;
    const auto disabled = db_cache_options(request);
    assert(!disabled.enabled && !disabled.diagnostic && disabled.mode == "off");
    assert(disabled.threshold == .08f && disabled.max_consecutive == 2);
    assert(disabled.front_blocks == 8 && disabled.warmup_steps == 8);
    for (const auto &mode : {"conservative", "balanced", "fast"}) {
        request.qwen21_dit_cache = mode;
        const auto options = db_cache_options(request);
        assert(options.enabled && !options.diagnostic && options.mode == mode);
        assert(options.threshold == (options.mode == "conservative" ? .15f : .25f));
        assert(options.max_consecutive == (options.mode == "conservative" ? 1 :
                                          options.mode == "balanced" ? 2 : 4));
        assert(options.front_blocks == 8 && options.warmup_steps == 8);
    }
    auto rejected = [&] {
        try { (void)db_cache_options(request); }
        catch (const std::invalid_argument &) { return true; }
        return false;
    };
    request.qwen21_dit_cache = "balanced";
    assert(setenv(cache_environment[0], "0", 1) == 0);
    assert(db_cache_options(request).enabled);
    assert(setenv(cache_environment[0], "1", 1) == 0);
    assert(rejected());
    assert(unsetenv(cache_environment[0]) == 0);
    for (const auto *name : {cache_environment[1], cache_environment[2],
                             cache_environment[3], cache_environment[4]}) {
        assert(setenv(name, "0", 1) == 0);
        assert(rejected()); // Even a zero numeric override is not a preset input.
        assert(setenv(name, "8", 1) == 0);
        assert(rejected()); // Matching preset geometry is still an override.
        assert(unsetenv(name) == 0);
    }
    request.qwen21_dit_cache = "unknown";
    assert(rejected());
    request.qwen21_dit_cache = "off";
    assert(setenv(cache_environment[0], "1", 1) == 0);
    assert(setenv(cache_environment[1], "0.25", 1) == 0);
    assert(setenv(cache_environment[2], "4", 1) == 0);
    const auto diagnostic = db_cache_options(request);
    assert(diagnostic.enabled && diagnostic.diagnostic && diagnostic.mode == "off");
    assert(diagnostic.threshold == .25f && diagnostic.max_consecutive == 4);
    assert(diagnostic.front_blocks == 8 && diagnostic.warmup_steps == 8);
    assert(setenv(cache_environment[1], "0.24", 1) == 0);
    assert(setenv(cache_environment[2], "3", 1) == 0);
    assert(setenv(cache_environment[3], "1", 1) == 0);
    assert(setenv(cache_environment[4], "4", 1) == 0);
    const auto sglang_candidate = db_cache_options(request);
    assert(sglang_candidate.enabled && sglang_candidate.diagnostic);
    assert(sglang_candidate.threshold == .24f && sglang_candidate.max_consecutive == 3);
    assert(sglang_candidate.front_blocks == 1 && sglang_candidate.warmup_steps == 4);
    request.qwen21_dit_cache_explicit = true;
    assert(!db_cache_options(request).enabled); // Explicit Off overrides every diagnostic knob.
    assert(setenv(cache_environment[3], "invalid", 1) == 0);
    assert(!db_cache_options(request).enabled); // A stale malformed env cannot defeat Off.
    request.qwen21_dit_cache_explicit = false;
    assert(rejected());
    assert(setenv(cache_environment[3], "1", 1) == 0);
    assert(setenv(cache_environment[4], "5", 1) == 0);
    assert(rejected());
    assert(setenv(cache_environment[4], "4", 1) == 0);
    assert(setenv(cache_environment[0], "0", 1) == 0);
    assert(rejected()); // Retain the established max-requires-opt-in contract.
    assert(unsetenv(cache_environment[2]) == 0);
    assert(rejected()); // Front/warmup also require the diagnostic flag.
    assert(unsetenv(cache_environment[3]) == 0);
    assert(unsetenv(cache_environment[4]) == 0);
    assert(!db_cache_options(request).enabled);
    assert(setenv(cache_environment[1], "nan", 1) == 0);
    assert(rejected());
    assert(unsetenv(cache_environment[1]) == 0);
    assert(setenv(cache_environment[0], "true", 1) == 0);
    assert(rejected());
}
} // namespace

int main() {
    using namespace tc::qwen21;
    test_cache_options();
    assert(!option_enabled(nullptr));
    assert(!option_enabled("0"));
    assert(!option_enabled(""));
    assert(!option_enabled("true"));
    assert(option_enabled("1"));
    assert(binary_option_or_unset(nullptr));
    assert(binary_option_or_unset("0"));
    assert(binary_option_or_unset("1"));
    assert(!binary_option_or_unset("true"));
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
    assert(db_cache_front_blocks(nullptr) == 8);
    assert(db_cache_front_blocks("1") == 1);
    assert(db_cache_front_blocks("8") == 8);
    assert(db_cache_front_blocks("2") < 0);
    assert(db_cache_front_blocks("01") < 0);
    assert(db_cache_warmup_steps(nullptr) == 8);
    assert(db_cache_warmup_steps("4") == 4);
    assert(db_cache_warmup_steps("8") == 8);
    assert(db_cache_warmup_steps("0") < 0);
    assert(db_cache_warmup_steps("4oops") < 0);

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
