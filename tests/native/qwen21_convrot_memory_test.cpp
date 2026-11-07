#include "../../native/models/qwen21/memory_policy.hpp"
#include "../../native/runtime/residency.hpp"
#include <cassert>
#include <iostream>
#include <limits>

int main() {
    using namespace tc;
    constexpr uint64_t gib = uint64_t(1) << 30;
    ane::StreamingMemoryObservation observation{{true,24*gib,gib,2*gib,0,0},8*gib,true};
    assert(qwen21::convrot_stream_cache_headroom(observation)==2*gib);
    observation.memory.free_bytes=6*gib;
    assert(qwen21::convrot_stream_cache_headroom(observation)==4*gib);
    observation.pressure_normal=false;
    assert(qwen21::convrot_stream_cache_headroom(observation)==0);
    observation.pressure_normal=true;observation.file_backed_non_speculative_bytes.reset();
    assert(qwen21::convrot_stream_cache_headroom(observation)==0);
    observation.file_backed_non_speculative_bytes=0;observation.memory.free_bytes=6*gib;
    assert(qwen21::convrot_stream_cache_headroom(observation)==0);
    observation.memory.free_bytes=7*gib;
    assert(qwen21::convrot_stream_cache_headroom(observation)==gib/2);
    // Exact lengths of the user's verified ModelScope Turbo/encoder/VAE files.
    const std::array<uint64_t, 3> int8_sources{7256787829, 9350798360, 675509688};
    Request request;
    request.model = "qwen-image-2.1";
    request.model_variant = "qwen-image-2.1-turbo";
    request.width = request.height = 512;
    request.steps = 8;
    const auto rejects = [](auto operation) {
        try { operation(); } catch (const std::invalid_argument &) { return true; }
        return false;
    };
    auto plan_for = [&](const Request &r, const std::array<uint64_t, 3> &sizes) {
        ExecutionPlan plan;
        plan.request = r;
        plan.memory_estimate_bytes = qwen21::convrot_resident_memory_estimate(r, sizes);
        return plan;
    };
    auto plan = plan_for(request, int8_sources);
    assert(*plan.memory_estimate_bytes == 21041192261);
    ResidencyPolicy::validate_budget(plan, 24 * gib);
    assert(rejects([&] { ResidencyPolicy::validate_budget(plan, *plan.memory_estimate_bytes + 4 * gib - 1); }));
    ResidencyPolicy::validate_budget(plan, *plan.memory_estimate_bytes + 4 * gib);
    plan.request.memory_budget_bytes = *plan.memory_estimate_bytes - 1;
    assert(rejects([&] { ResidencyPolicy::validate_budget(plan, 24 * gib); }));
    // Merely having one ConvRot component must not discount a BF16 companion.
    auto mixed = int8_sources;
    mixed[0] = 14230280616; // actual Turbo BF16 DiT length
    assert(rejects([&] { ResidencyPolicy::validate_budget(plan_for(request, mixed), 24 * gib); }));
    mixed = int8_sources;
    mixed[1] = 18 * gib; // representative complete BF16 encoder source
    assert(rejects([&] { ResidencyPolicy::validate_budget(plan_for(request, mixed), 24 * gib); }));
    const std::array<uint64_t, 3> bf16_sources{14230280616, 18 * gib, 675509688};
    assert(rejects([&] { ResidencyPolicy::validate_budget(plan_for(request, bf16_sources), 24 * gib); }));
    ResidencyPolicy::validate_budget(plan_for(request, bf16_sources), 48 * gib);
    for (int dimension : {768, 1024, 2048}) {
        auto larger = request;
        larger.width = larger.height = dimension;
        const auto expanded = plan_for(larger, int8_sources);
        assert(*expanded.memory_estimate_bytes > *plan.memory_estimate_bytes);
        assert(rejects([&] { ResidencyPolicy::validate_budget(expanded, 24 * gib); }));
        ResidencyPolicy::validate_budget(expanded, *expanded.memory_estimate_bytes + 4 * gib);
    }
    for (size_t count : {size_t(1), size_t(2), size_t(10)}) {
        auto edit = request;
        edit.inputs.resize(count);
        auto expanded = plan_for(edit, int8_sources);
        assert(*expanded.memory_estimate_bytes == 21041192261 + count * (5 * gib / 2));
        assert(rejects([&] { ResidencyPolicy::validate_budget(expanded, 24 * gib); }));
    }
    for (int dimension : {-1, 0, 32, 65, 4097, std::numeric_limits<int>::max()}) {
        auto invalid = request;
        invalid.width = dimension;
        assert(rejects([&] { (void)plan_for(invalid, int8_sources); }));
        invalid = request;
        invalid.height = dimension;
        assert(rejects([&] { (void)plan_for(invalid, int8_sources); }));
    }
    auto excessive_pixels = request;
    excessive_pixels.width = excessive_pixels.height = 4096;
    assert(rejects([&] { (void)plan_for(excessive_pixels, int8_sources); }));
    for (int index = 0; index < 3; ++index) {
        auto empty = int8_sources;
        empty[index] = 0;
        assert(rejects([&] { (void)plan_for(request, empty); }));
    }
    constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
    assert(rejects([&] { (void)plan_for(request, {maximum, 1, 1}); }));
    assert(rejects([&] { (void)plan_for(request, {maximum - 2, 1, 1}); }));
    const uint64_t overhead = gib + uint64_t(512) * 512 * 10240;
    // Source/temporary/canvas arithmetic can fit while the separate 4 GiB
    // system reserve would wrap. That request must still be rejected.
    assert(rejects([&] { (void)plan_for(request, {maximum - overhead - 2, 1, 1}); }));
    const auto representable = plan_for(request, {maximum - overhead - 4 * gib - 2, 1, 1});
    assert(*representable.memory_estimate_bytes == maximum - 4 * gib);
    assert(rejects([&] { ResidencyPolicy::validate_budget(representable, 24 * gib); }));
    std::cout << "PASS ConvRot source-aware memory: real INT8 512 fits 24 GiB with reserve; BF16/canvas/references/budget/invalid/overflow guarded\n";
}
