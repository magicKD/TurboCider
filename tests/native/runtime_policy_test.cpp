#include "runtime/residency.hpp"
#include "runtime/acceleration.hpp"
#include "runtime/execution.hpp"
#include "runtime/lora_identity.hpp"
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
    auto h3_stream = make_block_residency_plan(
        16ull << 30, 4ull << 30, 770725376ull, 50u, 0u, 2u, false, false);
    assert(h3_stream.pinned_blocks ==
           static_cast<unsigned>(((16ull << 30) - (4ull << 30) -
                                   2ull * 770725376ull) / 770725376ull));
    assert(h3_stream.streamed_blocks == 50u - h3_stream.pinned_blocks &&
           h3_stream.refill_slots == 2u && !h3_stream.fully_resident);
    auto ltx_stream = make_block_residency_plan(
        8ull << 30, 4ull << 30, 768ull << 20, 48u, 0u, 3u, true, true);
    assert(ltx_stream.pinned_blocks == 2u &&
           ltx_stream.streamed_blocks == 46u &&
           ltx_stream.refill_slots == 3u);
    auto ltx_resident = make_block_residency_plan(
        40ull << 30, 4ull << 30, 768ull << 20, 48u, 0u, 3u, true, true);
    assert(ltx_resident.fully_resident && ltx_resident.pinned_blocks == 48u &&
           ltx_resident.refill_slots == 0u);
    LoRAAsset adapter{"relative/adapter.safetensors", 0.8f, "refiner"};
    auto lora = make_verified_lora_identity(
        adapter, "/models/adapter.safetensors", 1234u,
        std::string(64, 'a'));
    auto lora_key = lora.cache_key("ltx-in-memory-v1");
    assert(lora_key.find("|path=27:/models/adapter.safetensors") !=
               std::string::npos &&
           lora_key.find("|bytes=1234") != std::string::npos &&
           lora_key.find("|role=7:refiner") != std::string::npos &&
           lora_key.find("|strength_bits=3f4ccccd") != std::string::npos);
    auto transformer_lora = lora;
    transformer_lora.role = "transformer";
    assert(transformer_lora.cache_key("ltx-in-memory-v1") != lora_key);
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
