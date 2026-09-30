#include "runtime/residency.hpp"
#include "runtime/acceleration.hpp"
#include "runtime/execution.hpp"
#include "runtime/lora_identity.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen21/memory_policy.hpp"
#include "platform/apple/platform.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace tc;
    const auto &optimized = device_optimizations("Apple M5 Pro", 24ull << 30);
    assert(std::string_view(optimized.id) == "m5pro24-v1");
    assert(optimized.z_image_suffix_streaming && optimized.z_image_hybrid_segments &&
           optimized.z_image_memory_lifecycle && optimized.z_image_smallest_partition &&
           optimized.external_automatic_partitions && optimized.coreml_output_copy &&
           optimized.z_image_int8_streaming);
    assert(optimized.qwen21_layer_streaming);
    Request qwen_gpu;
    qwen_gpu.model = "qwen-image-2.1";
    qwen_gpu.width = qwen_gpu.height = 512;
    qwen_gpu.steps = 40;
    qwen_gpu.residency = "component_staged";
    Request qwen_ane = qwen_gpu;
    qwen_ane.execution = "gpu_ane";
    qwen_ane.qwen21_w8a8 = true;
    qwen_ane.allow_approximation = true;
    qwen_ane.ane_manifest = "verified-at-execution.json";
    const DeviceInfo qwen_device{"Apple M5 Pro", 24ull << 30};
    assert(qwen21::layer_staged_t2i(qwen_gpu, qwen_device));
    assert(qwen21::layer_staged_t2i(qwen_ane, qwen_device));
    assert(qwen21::layer_staged_hybrid_t2i(qwen_ane, qwen_device));
    for (const auto &mutate : std::vector<std::function<void(Request &)>>{
            [](Request &r) { r.residency = "resident"; },
            [](Request &r) { r.width = r.height = 1024; },
            [](Request &r) { r.operation = "image.edit"; },
            [](Request &r) { InputAsset input; input.kind = "image"; input.role = "reference";
                             input.path = "ref.png"; r.inputs.push_back(input); },
            [](Request &r) { r.loras.push_back({"adapter.safetensors"}); },
            [](Request &r) { r.prompt_enhance = true; }}) {
        for (auto request : {qwen_gpu, qwen_ane}) {
            mutate(request);
            assert(!qwen21::layer_staged_t2i(request, qwen_device));
        }
    }
    for (const auto &mutate : std::vector<std::function<void(Request &)>>{
            [](Request &r) { r.qwen21_gpu_w8a16 = true; },
            [](Request &r) { r.qwen21_w8a8 = false; },
            [](Request &r) { r.allow_approximation = false; },
            [](Request &r) { r.steps = 1; },
            [](Request &r) { r.ane_manifest.clear(); },
            [](Request &r) { r.qwen21_gpu_full_ffn_blocks = {3, 5, 7}; }}) {
        auto request = qwen_ane; mutate(request);
        assert(!qwen21::layer_staged_t2i(request, qwen_device));
    }
    for (const char *mode : {"lora_suffix", "lora_gate_up", "lora_fused", "runtime", "runtime_qkv"}) {
        auto request = qwen_ane; request.hybrid_mlp_mode = mode;
        assert(!qwen21::layer_staged_t2i(request, qwen_device));
    }
    auto base_fused = qwen_ane; base_fused.hybrid_mlp_mode = "base_fused";
    assert(qwen21::layer_staged_hybrid_t2i(base_fused, qwen_device));
    const auto &m4_copy = device_optimizations("Apple M4 Pro", 48ull << 30);
    assert(std::string_view(m4_copy.id) == "m4pro48-coreml-copy-v1");
    assert(!m4_copy.qwen21_layer_streaming);
    assert(!qwen21::layer_staged_t2i(qwen_gpu, {"Apple M4 Pro", 48ull << 30}));
    assert(!qwen21::layer_staged_t2i(qwen_ane, {"Apple M4 Pro", 48ull << 30}));
    assert(m4_copy.coreml_output_copy && !m4_copy.z_image_suffix_streaming &&
           !m4_copy.z_image_hybrid_segments && !m4_copy.z_image_memory_lifecycle &&
           !m4_copy.z_image_smallest_partition && !m4_copy.external_automatic_partitions &&
           !m4_copy.z_image_int8_streaming);
    auto rejects_stream = [](auto operation) {
        try { operation(); }
        catch (const std::invalid_argument &) { return true; }
        return false;
    };
    assert(optimized.supports_z_image_streaming(true));
    assert(optimized.z_image_stream_prefetch(true, true, 6ull << 30) == 2);
    assert(optimized.z_image_stream_prefetch(true, true, 8ull << 30) == 1);
    assert(optimized.z_image_stream_prefetch(true, true, 10ull << 30) == 1);
    assert(optimized.z_image_stream_prefetch(true, false, 6ull << 30) == 1);
    assert(optimized.z_image_stream_prefetch(false, true, 6ull << 30) == 1);
    for (const char *value : {"1", "2", "4", "8"})
        assert(optimized.z_image_stream_prefetch(true, true, 6ull << 30, value) == unsigned(value[0] - '0'));
    for (const char *value : {"", "0", "9", "12", "auto"})
        assert(rejects_stream([&] { optimized.z_image_stream_prefetch(true, true, 6ull << 30, value); }));
    for (const auto &device : std::vector<DeviceInfo>{
             {"Apple M4 Pro", 24ull << 30}, {"Apple M4 Pro", 64ull << 30},
             {"Apple M4 Max", 64ull << 30}, {"Apple M5", 24ull << 30},
             {"Apple M5 Max", 24ull << 30}, {"Apple M5 Pro", 48ull << 30},
             {"Apple M5 Pro", 16ull << 30}, {"Apple M5 Pro", 64ull << 30},
             {"Apple M5 Pro", (24ull << 30) - 1}, {"Apple M5 Pro", (24ull << 30) + 1},
             {"Apple M5 Pro (unknown)", 24ull << 30}, {"unavailable", 0}}) {
        const auto &legacy = device.optimizations();
        assert(std::string_view(legacy.id) == "legacy");
        assert(!legacy.z_image_suffix_streaming && !legacy.z_image_hybrid_segments &&
               !legacy.z_image_memory_lifecycle && !legacy.z_image_smallest_partition &&
               !legacy.external_automatic_partitions && !legacy.coreml_output_copy &&
               !legacy.z_image_int8_streaming);
        assert(!legacy.qwen21_layer_streaming);
        assert(!qwen21::layer_staged_t2i(qwen_gpu, device));
        assert(!qwen21::layer_staged_t2i(qwen_ane, device));
        assert(!qwen21::layer_staged_hybrid_t2i(qwen_ane, device));
        assert(legacy.supports_z_image_streaming(false));
        assert(!legacy.supports_z_image_streaming(true));
        assert(legacy.z_image_stream_prefetch(false, false, 6ull << 30) == 1);
        for (bool hybrid : {false, true}) {
            assert(rejects_stream([&] { legacy.z_image_stream_prefetch(true, hybrid, 6ull << 30); }));
            for (const char *value : {"1", "2", "4", "8"})
                assert(rejects_stream([&] { legacy.z_image_stream_prefetch(true, hybrid, 6ull << 30, value); }));
        }
        assert(rejects_stream([&] { legacy.z_image_stream_prefetch(false, false, 6ull << 30, "4"); }));
    }
    Request request;
    assert(hybrid_case(request, 1044, "Apple M4 Pro", 48ull << 30));
    assert(hybrid_case(request, 1044, "Apple M4 Max", 64ull << 30));
    const auto *m5 = hybrid_case(request, 1044, "Apple M5 Pro", 24ull << 30);
    assert(m5 && m5->bucket == 1088);
    assert(has_hybrid_measurement(request.model, "Apple M5 Pro", 24ull << 30));
    assert(hybrid_partition_matches(*m5, 9216, 0, 6144));
    assert(!hybrid_partition_matches(*m5, 9216, 0, 3072));
    assert(!hybrid_partition_matches(*m5, 9216, 0, 9216));
    assert(!hybrid_partition_matches(*m5, 9216, 1, 6144));
    assert(!hybrid_partition_matches(*m5, 10240, 0, 6144));
    assert(!hybrid_case(request, 1024, "Apple M5 Pro", 24ull << 30));
    assert(!hybrid_case(request, 1089, "Apple M5 Pro", 24ull << 30));
    assert(!hybrid_case(request, 1044, "Apple M5 Pro", 48ull << 30));
    assert(!hybrid_case(request, 1044, "Apple M5", 24ull << 30));
    assert(!hybrid_case(request, 1044, "Apple M5 Max", 64ull << 30));
    assert(!hybrid_case(request, 1089, "Apple M4 Pro", 48ull << 30));
    assert(!hybrid_case(request, 1044, "Apple M3", 48ull << 30));
    assert(!hybrid_case(request, 1044, "Apple M4 Pro", 24ull << 30));
    request.width = request.height = 1024;
    assert(!hybrid_case(request, 4116, "Apple M5 Pro", 24ull << 30));
    assert(hybrid_case(request, 4116, "Apple M4 Pro", 48ull << 30)->bucket == 4160);
    assert(!hybrid_case(request, 4161, "Apple M4 Pro", 48ull << 30));
    request.width = request.height = 256;
    assert(!hybrid_case(request, 276, "Apple M4 Pro", 48ull << 30));
    request.width = request.height = 512;
    request.operation = "image.transform";
    assert(!hybrid_case(request, 1044, "Apple M5 Pro", 24ull << 30));
    assert(!hybrid_case(request, 1044, "Apple M4 Pro", 48ull << 30));
    request.operation = "image.generate"; request.steps = 1;
    assert(!hybrid_case(request, 1044, "Apple M5 Pro", 24ull << 30));
    assert(!hybrid_case(request, 1044, "Apple M4 Pro", 48ull << 30));
    request.steps = 4; request.residency = "component_staged";
    assert(!hybrid_case(request, 1044, "Apple M5 Pro", 24ull << 30));
    assert(!hybrid_case(request, 1044, "Apple M4 Pro", 48ull << 30));
    request.residency = "resident";
    request.loras.push_back({"adapter.safetensors", 1.f, "transformer"});
    assert(!hybrid_case(request, 1044, "Apple M5 Pro", 24ull << 30));
    assert(!hybrid_case(request, 1044, "Apple M4 Max", 64ull << 30));
    request.loras.clear();
    request.model = "z-image-turbo"; request.width = request.height = 1024; request.steps = 9;
    assert(!has_hybrid_measurement(request.model, "Apple M5 Pro", 24ull << 30));
    assert(!hybrid_case(request, 4128, "Apple M5 Pro", 24ull << 30));
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
    ExecutionPlan plan{request, {}, 16ull << 30, {}};
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
