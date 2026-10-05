#include "../../native/backends/ane_calibration.hpp"
#include <iostream>

using namespace tc::ane::calibration;

static void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class Function> static void rejected(Function function) {
    bool failed = false;
    try { function(); } catch (const std::exception &) { failed = true; }
    check(failed, "invalid calibration was accepted");
}
static Metadata metadata() {
    Metadata result;
    result.request_id = "cpu-fixture-request"; result.model_id = "fixture-model";
    result.model_fingerprint = "canonical-stat-fixture-identity";
    result.identity_kind = "canonical-stat-identity"; result.recipe = "source-runtime-recipe-v1";
    result.loras = {{"fixture-adapter", "fixture-sha256", .12345678901234567}};
    result.seed = 42; result.width = 512; result.height = 512; result.total_steps = 6;
    result.reference_size = 512; result.reference_count = 1; result.execution_route = "cpu-test-only";
    return result;
}
static Config config() { Config c; c.enabled = true; c.layers = {0}; c.steps = {0}; return c; }
static Point point() { return {0, 0, "prefill", {{"text", 0, 16}, {"reference", 16, 129}}}; }

int main(int argc, char **argv) {
    try {
        check(argc == 2, "expected fixture parent");
        int reads = 0;
        Sampler disabled;
        check(!disabled.observe_rows({}, 0, 0, [&](size_t, size_t, auto) { ++reads; }), "default capture enabled");
        check(reads == 0 && disabled.input_bytes() == 0 && disabled.statistics_bytes() == 0, "disabled capture touched input");
        auto bad_metadata = metadata(); bad_metadata.identity_kind.clear();
        rejected([&] { Sampler invalid(config(), bad_metadata); });
        bad_metadata = metadata(); bad_metadata.execution_route.clear();
        rejected([&] { Sampler invalid(config(), bad_metadata); });
        auto c = config(); c.layers = {0, 0};
        rejected([&] { Sampler invalid(c, metadata()); });
        c = config(); c.steps = {6};
        rejected([&] { Sampler invalid(c, metadata()); });
        c = config(); c.input_budget = 8;
        Sampler tiny(c, metadata());
        rejected([&] { tiny.observe_rows(point(), 129, 8, [&](size_t, size_t, auto) { ++reads; }); });
        check(reads == 0, "over-budget capture transferred input");
        Sampler sampler(config(), metadata());
        const std::vector<float> weight_max(8, 2.f);
        check(sampler.needs_weight_channel_max(0) && !sampler.needs_weight_channel_max(1), "weight selection mismatch");
        check(sampler.set_weight_channel_max(0, weight_max, "selected-base-layer-0"), "weight statistics not accepted");
        check(!sampler.needs_weight_channel_max(0), "weight statistics require another scan");
        check(!sampler.set_weight_channel_max(0, weight_max, "selected-base-layer-0"), "weight statistics duplicated");
        rejected([&] { sampler.set_weight_channel_max(0, weight_max, "different-source"); });
        auto invalid_point = point(); invalid_point.regions[1].begin = 17;
        rejected([&] { sampler.observe_rows(invalid_point, 129, 8, [&](size_t, size_t, auto) { ++reads; }); });
        check(reads == 0, "invalid region transferred input");
        rejected([&] { sampler.observe_rows(point(), 129, 8, [](size_t, size_t, auto out) {
            std::fill(out.begin(), out.end(), uint16_t(0x7e00));
        }); });
        check(sampler.records().empty() && sampler.input_bytes() == 0 && sampler.statistics_bytes() == 32,
              "failed capture published partial statistics");
        check(sampler.observe_rows(point(), 129, 8, [&](size_t first, size_t count, auto out) {
            ++reads; check(count <= 128 && out.size() == count * 8, "unbounded reader batch");
            std::fill(out.begin(), out.end(), uint16_t(0x3c00));
            if (first <= 97 && first + count > 97) out[(97 - first) * 8 + 3] = 0x4c00;
        }), "capture missing");
        check(reads == 2 && sampler.complete(), "capture did not scan all rows");
        const auto &record = sampler.records()[0];
        check(record.rows_observed == 129 && record.sample_rows.size() == 64 && record.channel_max[3] == 16,
              "incomplete full-row statistics");
        check(std::find(record.sample_rows.begin(), record.sample_rows.end(), 97) != record.sample_rows.end(),
              "outlier absent from samples");
        check(sampler.input_bytes() == 64 * 8 * 2 && sampler.statistics_bytes() == 2 * 8 * 4,
              "capture accounting mismatch");
        check(!sampler.observe_rows(point(), 129, 8, [&](size_t, size_t, auto) { ++reads; }) && reads == 2,
              "duplicate point transferred input");
        sampler.write(std::filesystem::path(argv[1]) / "fixture");
        rejected([&] { sampler.write(std::filesystem::path(argv[1]) / "fixture"); });
        check(decode_fp16(0x0001) == std::ldexp(1.f, -24) && decode_fp16(0xbc00) == -1,
              "FP16 host decoder mismatch");
        rejected([] { decode_fp16(0x7c00); });

        // Local real hidden geometry, CPU constants only: full 3x3 coverage is
        // 4.5 MiB input and 192 KiB combined input/weight statistics.
        c = Config{}; c.enabled = true;
        Sampler bounded(c, metadata());
        const std::vector<float> real_max(4096, 1.f);
        for (const auto layer : c.layers) {
            bounded.set_weight_channel_max(layer, real_max, "selected-layer-" + std::to_string(layer));
            for (const auto step : c.steps)
                bounded.observe_rows({layer, step, "denoise", {{"target", 0, 129}}}, 129, 4096,
                    [](size_t, size_t count, auto out) {
                        check(count <= 128, "large unbounded transfer");
                        std::fill(out.begin(), out.end(), uint16_t(0x3c00));
                    });
        }
        check(bounded.complete() && bounded.input_bytes() == 9 * 64 * 4096 * 2 &&
              bounded.statistics_bytes() == 12 * 4096 * 4, "real geometry exceeds expected bounded payload");
        std::cout << "PASS request-local CPU calibration disabled/bounded/regions/outliers/identity/rollback\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
