#include "runtime/residency.hpp"
#include "runtime/acceleration.hpp"
#include "runtime/execution.hpp"
#include "core/tokenizer.hpp"
#include "platform/apple/platform.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace tc;
    Request request;
    assert(hybrid_case(request, 1044, "Apple M4 Pro", 48ull << 30));
    assert(hybrid_case(request, 1044, "Apple M4 Max", 64ull << 30));
    assert(!hybrid_case(request, 1089, "Apple M4 Pro", 48ull << 30));
    assert(!hybrid_case(request, 1044, "Apple M3", 48ull << 30));
    assert(!hybrid_case(request, 1044, "Apple M4 Pro", 24ull << 30));
    request.width = request.height = 1024;
    assert(hybrid_case(request, 4116, "Apple M4 Pro", 48ull << 30)->bucket == 4160);
    assert(!hybrid_case(request, 4161, "Apple M4 Pro", 48ull << 30));
    request.width = request.height = 256;
    assert(!hybrid_case(request, 276, "Apple M4 Pro", 48ull << 30));
    request.width = request.height = 512;
    request.operation = "image.transform";
    assert(!hybrid_case(request, 1044, "Apple M4 Pro", 48ull << 30));
    request.operation = "image.generate"; request.steps = 1;
    assert(!hybrid_case(request, 1044, "Apple M4 Pro", 48ull << 30));
    request.steps = 4; request.residency = "component_staged";
    assert(!hybrid_case(request, 1044, "Apple M4 Pro", 48ull << 30));
    request.residency = "resident";
    request.loras.push_back({"adapter.safetensors", 1.f, "transformer"});
    assert(!hybrid_case(request, 1044, "Apple M4 Max", 64ull << 30));
    request.loras.clear();
    request.model = "z-image-turbo"; request.width = request.height = 1024; request.steps = 9;
    assert(hybrid_case(request, 4128, "Apple M4 Max", 64ull << 30));
    assert(!hybrid_case(request, 4160, "Apple M4 Max", 64ull << 30));
    request.model = "flux2-klein-4b"; request.width = request.height = 512; request.steps = 4;
    auto resident = ResidencyPolicy::for_request(request, 48ull << 30);
    assert(resident.retain_images_during_text && !resident.release_after_denoise);
    assert(!ResidencyPolicy::for_request(request, 16ull << 30).retain_images_during_text);
    request.memory_budget_bytes = 20ull << 30;
    assert(!ResidencyPolicy::for_request(request, 48ull << 30).retain_images_during_text);
    request.residency = "component_staged";
    auto staged = ResidencyPolicy::for_request(request, 48ull << 30);
    assert(!staged.retain_images_during_text && staged.release_before_denoise &&
           staged.release_after_denoise && staged.release_after_decode);
    auto rejects = [](auto operation) {
        try {
            operation();
        } catch (const std::invalid_argument &) {
            return true;
        }
        return false;
    };
    ExecutionPlan plan{request, {}, 16ull << 30};
    ResidencyPolicy::validate_budget(plan, 20ull << 30);
    assert(rejects([&] { ResidencyPolicy::validate_budget(plan, (20ull << 30) - 1); }));
    plan.request.memory_budget_bytes = (16ull << 30) - 1;
    assert(rejects([&] { ResidencyPolicy::validate_budget(plan, 48ull << 30); }));
    plan.memory_estimate_bytes.reset();
    assert(rejects([&] { ResidencyPolicy::validate_budget(plan, 48ull << 30); }));
    std::atomic<bool> cancel{true};
    bool cancelled = false;
    try {
        checkpoint(cancel);
    } catch (const Cancelled &) {
        cancelled = true;
    }
    assert(cancelled);
    // Core runtime can compile/link/run without MLX, Foundation, or a GPU.
    std::cout << "PASS: native C++ boundaries, residency and budget thresholds, cancellation\n";
}
