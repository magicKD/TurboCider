#include "streaming/layout.hpp"
#include "../core/json_keys.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace tc;
using namespace tc::streaming;
static constexpr uint64_t mib = 1ull << 20;

static StreamingConfig config(uint32_t k = 2, uint32_t g = 3, uint32_t p = 2) {
    StreamingConfig c;
    c.enabled = true; c.schema_version = 1; c.selection = "manual"; c.retention = "request";
    c.stages["denoiser"] = {"streamed", g, k, p, k - 1, 1};
    return c;
}
static Descriptor descriptor(uint32_t n = 10) {
    Descriptor d{"synthetic", "checkpoint-revision-1", "fake-v1", {}};
    StageDescriptor s;
    s.id = "denoiser"; s.adapter_revision = "test-v1"; s.max_group_size = 16;
    s.pass_count = 2;
    for (uint32_t b = 0; b < n; ++b)
        s.blocks.push_back({b, "weights", {{"w", "w" + std::to_string(b), 128 * mib, 64}}});
    d.stages.push_back(std::move(s));
    return d;
}
template<class F> static void rejects(F fn, const std::string &fragment) {
    bool failed = false;
    try { fn(); } catch (const std::invalid_argument &e) {
        failed = true;
        if (std::string(e.what()).find(fragment) == std::string::npos) {
            std::cerr << "wrong rejection: " << e.what() << ", wanted " << fragment << '\n';
            std::abort();
        }
    }
    assert(failed);
}
int main() {
    reject_duplicate_json_keys(R"({"a":1,"b":{"a":2},"c":[{"x":1},{"x":2}]})");
    rejects([]{reject_duplicate_json_keys(R"({"slot_count":2,"slot_count":3})");}, "duplicate JSON");
    rejects([]{reject_duplicate_json_keys(R"({"enabled":true,"enabl\u0065d":false})");}, "duplicate JSON");
    rejects([]{reject_duplicate_json_keys("{\"é\":1,\"\\u00e9\":2}");}, "duplicate JSON");
    rejects([]{reject_duplicate_json_keys("{\"😀\":1,\"\\ud83d\\ude00\":2}");}, "duplicate JSON");
    auto c = config(); auto d = descriptor();
    const auto plan = compile_layout(c, d);
    const auto &s = plan.stages.at(0);
    assert(s.prefix_bytes == 256 * mib && s.peak_pool_bytes == 768 * mib);
    assert(s.groups.size() == 3 && s.groups[0].blocks == std::vector<uint32_t>({2,3,4}));
    assert(s.groups[1].blocks == std::vector<uint32_t>({5,6,7}));
    assert(s.groups[2].blocks == std::vector<uint32_t>({8,9}));
    assert(s.groups[0].slot == 0 && s.groups[1].slot == 1 && s.groups[2].slot == 0);
    assert(s.suffix_content_bytes_per_pass * s.pass_count == 2048 * mib);
    assert(!s.source_read_bytes_per_pass && !plan.materializations_complete);
    assert(plan.digest.size() == 64 && compile_layout(c, d).canonical == plan.canonical);
    c.provenance["enabled"] = "profile";
    assert(compile_layout(c, d).digest == plan.digest);
    auto q = c; q.stages["denoiser"].io_workers = 2;
    assert(compile_layout(q, d).digest != plan.digest);

    // Field maxima, not max of group total: 640MiB rather than 384MiB.
    auto hetero = descriptor(2);
    hetero.stages[0].blocks[0].fields = {{"a","a0",320*mib,1},{"b","b0",64*mib,1}};
    hetero.stages[0].blocks[1].fields = {{"a","a1",64*mib,1},{"b","b1",320*mib,1}};
    assert(compile_layout(config(1,1,0), hetero).stages[0].peak_pool_bytes == 640*mib);
    auto overflow = descriptor(2);
    overflow.stages[0].blocks[0].fields[0].bytes = std::numeric_limits<uint64_t>::max();
    rejects([&]{compile_layout(config(1,1,0), overflow);}, "overflow");

    auto bad = c; bad.stages["denoiser"].slot_count = 0;
    rejects([&]{compile_layout(bad,d);}, "slot_count");
    bad = c; bad.stages["denoiser"].resident_prefix_blocks = 10;
    rejects([&]{compile_layout(bad,d);}, "prefix");
    rejects([&]{compile_layout(config(3,3,5),d);}, "slot_count_exceeds_groups");
    bad = c; bad.stages["denoiser"].prefetch_distance = 2;
    rejects([&]{compile_layout(bad,d);}, "prefetch_distance");
    bad = c; bad.stages["denoiser"].io_workers = 0;
    rejects([&]{compile_layout(bad,d);}, "io_workers");
    bad = c; bad.stages["typo"] = bad.stages["denoiser"];
    rejects([&]{compile_layout(bad,d);}, "unknown stage");
    auto dup = d; dup.stages[0].blocks[1].id = 0;
    rejects([&]{compile_layout(c,dup);}, "duplicate block");
    dup = d; dup.stages[0].blocks[1].fields[0].storage_id = "w0";
    assert(compile_layout(c,dup).stages[0].prefix_bytes == 128*mib);
    dup.stages[0].blocks[2].fields[0].storage_id = "w0";
    rejects([&]{compile_layout(c,dup);}, "shared storage");
    dup = d; dup.stages[0].blocks[4].safe_boundary_after = false;
    rejects([&]{compile_layout(c,dup);}, "unsafe group");
    dup = d; dup.stages[0].blocks[2].fields[0].name = "different-binding";
    rejects([&]{compile_layout(c,dup);}, "signature mismatch");
    dup = d;
    for (size_t b = 6; b < 10; ++b) dup.stages[0].blocks[b].layout_class = "other";
    auto multi = compile_layout(config(2,2,2), dup);
    assert(multi.stages[0].pools.size() == 2);
    assert(multi.stages[0].groups[2].slot == 0 && multi.stages[0].groups[2].pool == 1);
    assert(multi.stages[0].peak_pool_bytes == 512*mib);
    dup.stages[0].multi_pool_policy = MultiPoolPolicy::retain_all;
    auto retained_multi = compile_layout(config(2,2,2), dup);
    assert(retained_multi.stages[0].multi_pool_policy ==
           MultiPoolPolicy::retain_all);
    assert(retained_multi.stages[0].peak_pool_bytes == 1024*mib);
    assert(retained_multi.digest != multi.digest);
    assert(multi.digest == compile_layout(config(2,2,2), [&] {
        auto serial = dup;
        serial.stages[0].multi_pool_policy = MultiPoolPolicy::serial;
        return serial;
    }()).digest);

    // Resident is a distinct layout, with no slot fields.
    StreamingStageConfig resident; resident.residency = "resident";
    bad = c; bad.stages["denoiser"] = resident;
    auto rp = compile_layout(bad,d);
    assert(rp.stages[0].pools.empty() && rp.stages[0].prefix_bytes == 1280*mib);
    auto inherited = bad; inherited.stages.clear();
    auto fixed = d; fixed.stages[0].fixed_policy = resident;
    assert(compile_layout(inherited,fixed).stages[0].inherited);
    rejects([&]{compile_layout(inherited,d);}, "missing stage");

    // Presence and overlay, including an explicit zero prefix.
    StreamingConfig overrides; overrides.stages["denoiser"].resident_prefix_blocks = 0;
    auto merged = c; overlay_streaming_config(merged, overrides);
    assert(merged.stages["denoiser"].resident_prefix_blocks == 0);
    overrides = {}; overrides.stages["denoiser"].slot_count = 1;
    merged = c; overlay_streaming_config(merged, overrides);
    rejects([&]{validate_streaming_config(merged);}, "prefetch_distance");
    overrides = {}; overrides.stages["denoiser"] = resident;
    merged = c; overlay_streaming_config(merged, overrides);
    validate_streaming_config(merged); assert(!merged.stages["denoiser"].slot_count);
    overrides = {}; overrides.enabled = false;
    overlay_streaming_config(merged, overrides); assert(!merged.active());

    // Exhaustive small layouts, independently verify coverage and capacities.
    size_t cases = 0;
    for (uint32_t n = 1; n <= 24; ++n)
        for (uint32_t p = 0; p < n; ++p)
            for (uint32_t g = 1; g <= 5; ++g)
                for (uint32_t k = 1; k <= 3; ++k) {
                    auto cfg = config(k,g,p); auto desc = descriptor(n);
                    const auto groups = (n-p+g-1)/g;
                    if (groups < k) {
                        rejects([&]{compile_layout(cfg,desc);}, "slot_count_exceeds_groups");
                        continue;
                    }
                    const auto result = compile_layout(cfg,desc);
                    const auto &stage = result.stages[0];
                    uint32_t next = p;
                    for (const auto &group : stage.groups) {
                        assert(group.slot == group.id%k);
                        for (auto block : group.blocks) assert(block == next++);
                        assert(group.bytes <= stage.pools[0].slots[group.slot].capacity_bytes);
                    }
                    assert(next == n && stage.prefix_bytes == p*128*mib);
                    assert(stage.suffix_content_bytes_per_pass == (n-p)*128*mib);
                    ++cases;
                }
    std::cout << "PASS streaming layout/config; " << cases << " exhaustive valid layouts\n";
}
