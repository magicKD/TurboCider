#include "ane_backend.hpp"
#include "ane_fp16_value_config.hpp"
#include <chrono>
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
#include "private/ane_executor.hpp"
#include "private/ane_w8_executor.hpp"
#endif

namespace tc::ane {
BuiltExecutor build_runtime_executor(const std::filesystem::path &manifest, size_t budget,
                                    GraphGeometry expected, BackendPolicy policy,
                                    std::optional<int> calibrated_channels) {
    BuiltExecutor result;
    const bool fp16_values=configured_fp16_bf16_values();
    if(fp16_values && (!policy.allow_private || policy.preferred!=BackendPreference::Private ||
                      expected.kind!=Kind::SwiGLU || expected.require_lora_inputs))
        throw std::runtime_error("FP16 BF16 values require explicitly authorized Private base-only SwiGLU");
    if (policy.preferred == BackendPreference::Off) { result.fallback_reason = "ANE disabled by backend policy"; return result; }
    if (policy.preferred == BackendPreference::Private && !policy.allow_private)
        throw std::runtime_error("private ANE requires explicit authorization");
    const bool try_private = policy.allow_private &&
        (policy.preferred == BackendPreference::Auto || policy.preferred == BackendPreference::Private);
    const bool bf16_boundaries=configured_convrot_bf16_boundaries();
    if(bf16_boundaries && (!try_private || expected.require_lora_inputs))
        throw std::runtime_error("BF16 value boundaries require authorized base-only Private Comfy recipe");
    const char *group=std::getenv("TURBOCIDER_PRIVATE_ANE_A8_GROUP_SIZE");
    if(group && std::string(group)!="0" && std::string(group)!="256")
        throw std::runtime_error("private ANE A8 group size requires 0 or 256");
    const bool requested_group=group && std::string(group)=="256";
    if(bf16_boundaries && requested_group)throw std::runtime_error("BF16 value boundaries cannot combine with grouped A8");
    const char *scope=std::getenv("TURBOCIDER_PRIVATE_ANE_A8_GROUP_SCOPE");
    if(scope && std::string(scope)!="input" && std::string(scope)!="hidden" && std::string(scope)!="both")
        throw std::runtime_error("private ANE A8 group scope requires input, hidden or both");
    if(scope && !requested_group)throw std::runtime_error("A8 group scope requires explicit group size 256");
    if(requested_group && !try_private)throw std::runtime_error("group A8 requires an authorized private backend");
    const int channels = resolved_private_channel_count(expected.width, calibrated_channels);
    if (channels && !try_private) throw std::runtime_error("channel split requires an authorized private W8A8 backend");
    if (try_private) {
        // Malformed manifests/geometries remain configuration failures, not
        // capability fallback. Only the native graph consumes this shape.
        auto shape = runtime_template_shape(manifest);
        if (shape.kind != expected.kind || shape.hidden != expected.hidden || shape.width != expected.width ||
            (expected.require_lora_inputs && !shape.lora_inputs))
            throw std::runtime_error("runtime ANE graph does not match model FFN geometry");
        if (calibrated_channels && channels == 0) {
            result.fallback_reason = "native channel calibration selected optimized GPU-only";
            return result;
        }
        try {
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
            const std::string path = std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH") ? std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH") : "fp16";
            if (path != "fp16" && path != "w8a8" && path != "convrot_w8a8")
                throw std::runtime_error("TURBOCIDER_PRIVATE_ANE_DATA_PATH requires fp16, w8a8 or convrot_w8a8");
            if(requested_group && path!="convrot_w8a8")throw std::runtime_error("group A8 requires convrot_w8a8 data path");
            if(bf16_boundaries && path!="convrot_w8a8")throw std::runtime_error("BF16 value boundaries require convrot_w8a8 data path");
            if(fp16_values && path!="fp16")throw std::runtime_error("FP16 BF16 values require the fp16 data path");
            if (channels && path == "fp16") throw std::runtime_error("channel split requires the W8A8 data path");
            if (channels) shape.width = channels; // base template still validated against the FULL model
            if (calibrated_channels) shape.lora_inputs = expected.require_lora_inputs;
            std::unique_ptr<Executor> graph = path != "fp16" ? std::unique_ptr<Executor>(std::make_unique<PrivateW8Graph>(shape,budget,
                std::filesystem::path{},path=="convrot_w8a8"?W8Basis::ComfyH256:W8Basis::SylvesterDH)) :
                std::unique_ptr<Executor>(std::make_unique<PrivateGraph>(shape,budget));
            std::string error;
            const auto start = std::chrono::steady_clock::now();
            if (!graph->self_test(error)) throw CapabilityError("private ANE self-test failed: " + error);
            result.self_test_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            result.executor = std::move(graph); result.self_test_passed = true; return result;
#else
            throw CapabilityError("private ANE omitted from this distribution build");
#endif
        } catch (const CapabilityError &error) {
            if (policy.preferred == BackendPreference::Private) throw;
            result.fallback_reason = error.what();
        } catch (const MemoryBudgetError &error) {
            if (policy.preferred == BackendPreference::Private) throw;
            result.fallback_reason = error.what();
        }
    }
    result.executor = std::make_unique<RuntimeGraph>(manifest, budget, false, false, expected);
    return result;
}
}
