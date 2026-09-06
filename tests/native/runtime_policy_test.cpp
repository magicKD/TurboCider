#include "runtime/residency.hpp"
#include "runtime/execution.hpp"
#include "core/tokenizer.hpp"
#include "platform/apple/platform.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace tc;
    Request request;
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
