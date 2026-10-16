#include "runtime/residency.hpp"
#include "runtime/acceleration.hpp"
#include "runtime/execution.hpp"
#include "runtime/lora_identity.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen21/memory_policy.hpp"
#include "platform/apple/platform.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <optional>

namespace {
constexpr const char *staged_incompatible_flags[] = {
    "TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV",
    "TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC",
    "TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC",
    "TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC",
    "TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC",
    "TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION",
    "TURBOCIDER_QWEN21_METAL_QK_ROPE",
    "TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC",
    "TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN",
    "TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN",
    "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC",
    "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC",
    "TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC",
    "TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE",
    "TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC",
    "TURBOCIDER_QWEN21_LORA_GATE_UP_DIAGNOSTIC",
    "TURBOCIDER_QWEN21_PROFILE_STEPS",
    "TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS",
    "TURBOCIDER_QWEN21_PROFILE_GPU_OPS",
    "TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS",
    "TURBOCIDER_QWEN21_PROFILE_RUNTIME_LORA_FFN",
    "TURBOCIDER_QWEN21_PROFILE_TILED_FFN",
    "TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS",
    "TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME",
    "TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU",
    "TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU",
    "TURBOCIDER_QWEN21_BF16_STREAMING"
};
constexpr const char *staged_incompatible_paths[] = {
    "TURBOCIDER_QWEN21_ENCODER_RUNTIME_MANIFEST",
    "TURBOCIDER_QWEN21_ENCODER_CALIBRATION_OUT",
    "TURBOCIDER_QWEN21_ENCODER_S1_PROFILE",
    "TURBOCIDER_QWEN21_NATIVE_ENCODER_ANE_CACHE",
    "TURBOCIDER_QWEN21_NATIVE_PREFILL_ANE_CACHE",
    "TURBOCIDER_QWEN21_PREFILL_GPU_FFN_BLOCKS"
};
struct PolicyEnvironment {
    std::vector<std::pair<std::string, std::optional<std::string>>> saved;
    void clear(const char *name) {
        const char *value = std::getenv(name);
        saved.emplace_back(name, value ? std::optional<std::string>(value) : std::nullopt);
        assert(unsetenv(name) == 0);
    }
    PolicyEnvironment() {
        for (const char *name : staged_incompatible_flags) clear(name);
        for (const char *name : staged_incompatible_paths) clear(name);
        for (const char *name : {"TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS",
                "TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS", "TURBOCIDER_QWEN21_STAGED_ATTN_BITS",
                "TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE", "TURBOCIDER_QWEN21_ENCODER_BACKEND",
                "TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE", "TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC",
                "TURBOCIDER_QWEN21_DBCACHE_THRESHOLD", "TURBOCIDER_QWEN21_DBCACHE_MAX_CONSECUTIVE",
                "TURBOCIDER_QWEN21_DBCACHE_FRONT_BLOCKS", "TURBOCIDER_QWEN21_DBCACHE_WARMUP_STEPS"})
            clear(name);
    }
    ~PolicyEnvironment() {
        for (const auto &[name, value] : saved) {
            if (value) setenv(name.c_str(), value->c_str(), 1);
            else unsetenv(name.c_str());
        }
    }
};
}

int main() {
    using namespace tc;
    PolicyEnvironment policy_environment;
    auto rejects_policy = [](auto operation) {
        try { operation(); }
        catch (const std::invalid_argument &) { return true; }
        return false;
    };
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
    assert(qwen21::qualified_staged_optimizations(qwen_ane, qwen_device));
    const auto defaults = qwen21::staged_optimization_options(qwen_ane, qwen_device);
    assert(defaults.suffix_bits == 4 && defaults.attention_bits == 4 && defaults.qk_norm_rope);
    assert(qwen21::staged_quantized_precision({0, 0, false}).empty());
    assert(qwen21::staged_quantized_precision({0, 0, true}).empty());
    assert(qwen21::staged_quantized_precision({4, 4, true}) ==
           "bf16_prefill+w4_attention_bf16_io+w4_ffn_suffix_fp16_io+w8a8_mlp_prefix_fp16_io");
    assert(qwen21::staged_quantized_precision({8, 0, false}) ==
           "bf16_prefill+bf16_attention+w8_ffn_suffix_fp16_io+w8a8_mlp_prefix_fp16_io");
    assert(qwen21::staged_quantized_precision({0, 8, false}) ==
           "bf16_prefill+w8_attention_bf16_io+bf16_ffn_suffix+w8a8_mlp_prefix_fp16_io");
    assert(qwen_ane.qwen21_dit_cache == "off" && !qwen_ane.qwen21_dit_cache_explicit);
    const auto gpu_defaults = qwen21::staged_optimization_options(qwen_gpu, qwen_device);
    assert(gpu_defaults.suffix_bits == 0 && gpu_defaults.attention_bits == 0 && !gpu_defaults.qk_norm_rope);
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
    assert(qwen21::qualified_staged_optimizations(base_fused, qwen_device));
    for (const char *mode : {"conservative", "balanced", "fast"}) {
        auto cached = base_fused;
        cached.qwen21_dit_cache = mode;
        cached.qwen21_dit_cache_explicit = true;
        assert(qwen21::qualified_staged_optimizations(cached, qwen_device));
        assert(qwen21::layer_staged_hybrid_t2i(cached, qwen_device));
        assert(qwen21::layer_staged_t2i(cached, qwen_device));
        assert(qwen21::staged_optimization_options(cached, qwen_device).suffix_bits == 4);
        assert(cached.qwen21_dit_cache == mode && cached.qwen21_dit_cache_explicit);
        for (int steps : {2, 19, 41}) {
            auto unsupported = cached; unsupported.steps = steps;
            assert(!qwen21::qualified_staged_optimizations(unsupported, qwen_device));
            assert(!qwen21::layer_staged_hybrid_t2i(unsupported, qwen_device));
        }
    }
    for (int steps : {2, 19, 20, 21, 39, 40, 41}) {
        auto request = qwen_ane; request.steps = steps;
        assert(qwen21::layer_staged_hybrid_t2i(request, qwen_device));
        assert(qwen21::qualified_staged_optimizations(request, qwen_device) == (steps >= 20 && steps <= 40));
    }
    for (const auto &mutate : std::vector<std::function<void(Request &)>>{
            [](Request &r) { r.model = "z-image-turbo"; },
            [](Request &r) { r.residency = "resident"; },
            [](Request &r) { r.execution = "gpu"; },
            [](Request &r) { r.execution = "auto"; },
            [](Request &r) { r.width = 256; },
            [](Request &r) { r.height = 1024; },
            [](Request &r) { r.operation = "image.edit"; },
            [](Request &r) { InputAsset input; input.kind = "image"; input.role = "reference";
                             input.path = "ref.png"; r.inputs.push_back(input); },
            [](Request &r) { r.loras.push_back({"adapter.safetensors"}); },
            [](Request &r) { r.prompt_enhance = true; },
            [](Request &r) { r.prompt_enhance_edit_experimental = true; },
            [](Request &r) { r.prompt_enhancer_path = "enhancer"; },
            [](Request &r) { r.ane_manifest.clear(); },
            [](Request &r) { r.encoder_ane_manifest = "encoder.json"; },
            [](Request &r) { r.qwen21_w8a8 = false; },
            [](Request &r) { r.qwen21_gpu_w8a16 = true; },
            [](Request &r) { r.qwen21_gpu_full_ffn_blocks = {1}; },
            [](Request &r) { r.allow_approximation = false; },
            [](Request &r) { r.hybrid_mlp_mode = "runtime"; },
            [](Request &r) { r.hybrid_mlp_mode = "lora_suffix"; },
            [](Request &r) { r.qwen21_denoiser_backend = "native-metal-graph"; },
            [](Request &r) { r.qwen21_native_ane_cache = "/cache/native"; },
            [](Request &r) { r.dump = "/tmp/dump"; },
            [](Request &r) { r.streaming.enabled = true; },
            [](Request &r) { r.streaming_requested.emplace().enabled = true; },
            [](Request &r) { r.streaming_selector.emplace().enabled = true; },
            [](Request &r) { r.streaming_selector_requested.emplace().enabled = true; },
            [](Request &r) { r.streaming_offload = true; },
            [](Request &r) { r.memory_constrained.enabled = true; },
            [](Request &r) { r.memory_budget_bytes = 20ull << 30; },
            [](Request &r) { r.quantized_execution.enabled = true; },
            [](Request &r) { r.compile_gpu = true; },
            [](Request &r) { r.vsa = true; }}) {
        auto request = qwen_ane; mutate(request);
        assert(!qwen21::qualified_staged_optimizations(request, qwen_device));
        const auto options = qwen21::staged_optimization_options(request, qwen_device);
        assert(options.suffix_bits == 0 && options.attention_bits == 0 && !options.qk_norm_rope);
        for (const char *name : {"TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS", "TURBOCIDER_QWEN21_STAGED_ATTN_BITS"}) {
            assert(setenv(name, "4", 1) == 0);
            assert(rejects_policy([&] { qwen21::staged_optimization_options(request, qwen_device); }));
            assert(unsetenv(name) == 0);
        }
        request.qwen21_dit_cache = "fast";
        assert(!qwen21::layer_staged_hybrid_t2i(request, qwen_device));
    }
    auto cached = qwen_ane; cached.qwen21_dit_cache = "fast";
    for (const char *name : staged_incompatible_flags) {
        assert(setenv(name, "1", 1) == 0);
        assert(!qwen21::qualified_staged_optimizations(qwen_ane, qwen_device));
        assert(!qwen21::qualified_staged_optimizations(base_fused, qwen_device));
        assert(!qwen21::layer_staged_hybrid_t2i(cached, qwen_device));
        const auto options = qwen21::staged_optimization_options(qwen_ane, qwen_device);
        assert(options.suffix_bits == 0 && options.attention_bits == 0 && !options.qk_norm_rope);
        assert(setenv(name, "0", 1) == 0);
        assert(qwen21::qualified_staged_optimizations(qwen_ane, qwen_device));
        assert(unsetenv(name) == 0);
    }
    for (const char *name : staged_incompatible_paths) {
        assert(setenv(name, "/diagnostic/path", 1) == 0);
        assert(!qwen21::qualified_staged_optimizations(qwen_ane, qwen_device));
        assert(!qwen21::layer_staged_hybrid_t2i(cached, qwen_device));
        assert(unsetenv(name) == 0);
    }
    for (const auto &[name, value] : std::vector<std::pair<const char *, const char *>>{
            {"TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC", "16"},
            {"TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION", "2"},
            {"TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE", "16"},
            {"TURBOCIDER_QWEN21_ENCODER_BACKEND", "metal"},
            {"TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE", "prefill"}}) {
        assert(setenv(name, value, 1) == 0);
        assert(!qwen21::qualified_staged_optimizations(qwen_ane, qwen_device));
        assert(!qwen21::layer_staged_hybrid_t2i(cached, qwen_device));
        assert(unsetenv(name) == 0);
    }
    assert(setenv("TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC", "1", 1) == 0);
    assert(qwen21::qualified_staged_optimizations(qwen_ane, qwen_device));
    assert(qwen21::layer_staged_hybrid_t2i(qwen_ane, qwen_device));
    assert(setenv("TURBOCIDER_QWEN21_DBCACHE_FRONT_BLOCKS", "1", 1) == 0);
    assert(!qwen21::qualified_staged_optimizations(qwen_ane, qwen_device));
    assert(!qwen21::layer_staged_hybrid_t2i(qwen_ane, qwen_device));
    assert(setenv("TURBOCIDER_QWEN21_DBCACHE_FRONT_BLOCKS", "8", 1) == 0);
    assert(setenv("TURBOCIDER_QWEN21_DBCACHE_WARMUP_STEPS", "4", 1) == 0);
    assert(!qwen21::qualified_staged_optimizations(qwen_ane, qwen_device));
    assert(!qwen21::layer_staged_hybrid_t2i(qwen_ane, qwen_device));
    // Explicit request Off remains authoritative even with incompatible
    // process cache settings; enabling staged defaults never selects a cache.
    auto cache_off = qwen_ane; cache_off.qwen21_dit_cache_explicit = true;
    assert(!qwen21::db_cache_options(cache_off).enabled);
    assert(qwen21::qualified_staged_optimizations(cache_off, qwen_device));
    assert(setenv("TURBOCIDER_QWEN21_DBCACHE_WARMUP_STEPS", "8", 1) == 0);
    assert(qwen21::qualified_staged_optimizations(qwen_ane, qwen_device));
    for (const char *name : {"TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_DBCACHE_FRONT_BLOCKS", "TURBOCIDER_QWEN21_DBCACHE_WARMUP_STEPS"})
        assert(unsetenv(name) == 0);
    assert(setenv("TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS", "0", 1) == 0);
    const auto disabled = qwen21::staged_optimization_options(cached, qwen_device);
    assert(disabled.suffix_bits == 0 && disabled.attention_bits == 0 && !disabled.qk_norm_rope);
    assert(qwen21::qualified_staged_optimizations(cached, qwen_device));
    assert(qwen21::layer_staged_hybrid_t2i(cached, qwen_device));
    for (const char *bits : {"0", "4", "8"}) {
        assert(setenv("TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS", bits, 1) == 0);
        assert(setenv("TURBOCIDER_QWEN21_STAGED_ATTN_BITS", bits, 1) == 0);
        const auto options = qwen21::staged_optimization_options(cached, qwen_device);
        assert(options.suffix_bits == bits[0] - '0' && options.attention_bits == bits[0] - '0');
        assert(!options.qk_norm_rope);
        assert(qwen21::qualified_staged_optimizations(cached, qwen_device));
    }
    assert(unsetenv("TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS") == 0);
    assert(unsetenv("TURBOCIDER_QWEN21_STAGED_ATTN_BITS") == 0);
    assert(setenv("TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE", "1", 1) == 0);
    assert(qwen21::staged_optimization_options(qwen_ane, qwen_device).qk_norm_rope);
    assert(qwen21::staged_optimization_options(qwen_gpu, qwen_device).qk_norm_rope);
    assert(setenv("TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS", "1", 1) == 0);
    assert(setenv("TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE", "0", 1) == 0);
    const auto qk_disabled = qwen21::staged_optimization_options(qwen_ane, qwen_device);
    assert(qk_disabled.suffix_bits == 4 && qk_disabled.attention_bits == 4 && !qk_disabled.qk_norm_rope);
    assert(unsetenv("TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE") == 0);
    assert(setenv("TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS", "0", 1) == 0);
    assert(setenv("TURBOCIDER_QWEN21_STAGED_ATTN_BITS", "8", 1) == 0);
    const auto mixed = qwen21::staged_optimization_options(qwen_ane, qwen_device);
    assert(mixed.suffix_bits == 0 && mixed.attention_bits == 8 && mixed.qk_norm_rope);
    assert(unsetenv("TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS") == 0);
    assert(unsetenv("TURBOCIDER_QWEN21_STAGED_ATTN_BITS") == 0);
    assert(unsetenv("TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS") == 0);
    for (const char *name : {"TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS", "TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE"}) {
        for (const char *value : {"", "2", "true", "-1"}) {
            assert(setenv(name, value, 1) == 0);
            assert(rejects_policy([&] { qwen21::staged_optimization_options(qwen_ane, qwen_device); }));
        }
        assert(unsetenv(name) == 0);
    }
    for (const char *name : {"TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS", "TURBOCIDER_QWEN21_STAGED_ATTN_BITS"}) {
        for (const char *value : {"", "1", "2", "16", "04", "-4", "4x", "auto"}) {
            assert(setenv(name, value, 1) == 0);
            assert(rejects_policy([&] { qwen21::staged_optimization_options(qwen_ane, qwen_device); }));
        }
        assert(unsetenv(name) == 0);
    }
    const auto &m4_copy = device_optimizations("Apple M4 Pro", 48ull << 30);
    assert(std::string_view(m4_copy.id) == "m4pro48-coreml-copy-v1");
    assert(!m4_copy.qwen21_layer_streaming);
    assert(!qwen21::layer_staged_t2i(qwen_gpu, {"Apple M4 Pro", 48ull << 30}));
    assert(!qwen21::layer_staged_t2i(qwen_ane, {"Apple M4 Pro", 48ull << 30}));
    assert(!qwen21::qualified_staged_optimizations(qwen_ane, {"Apple M4 Pro", 48ull << 30}));
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
        assert(!qwen21::qualified_staged_optimizations(qwen_ane, device));
        const auto options = qwen21::staged_optimization_options(qwen_ane, device);
        assert(options.suffix_bits == 0 && options.attention_bits == 0 && !options.qk_norm_rope);
        for (const char *name : {"TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS", "TURBOCIDER_QWEN21_STAGED_ATTN_BITS"}) {
            for (const char *bits : {"4", "8"}) {
                assert(setenv(name, bits, 1) == 0);
                assert(rejects_policy([&] { qwen21::staged_optimization_options(qwen_ane, device); }));
            }
            assert(setenv(name, "0", 1) == 0);
            const auto explicit_off = qwen21::staged_optimization_options(qwen_ane, device);
            assert(explicit_off.suffix_bits == 0 && explicit_off.attention_bits == 0 && !explicit_off.qk_norm_rope);
            assert(unsetenv(name) == 0);
        }
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
