#include "ane_backend.hpp"
#include <chrono>
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
#include "private/ane_executor.hpp"
#include "private/ane_w8_executor.hpp"
#endif

namespace tc::ane {
BuiltExecutor build_runtime_executor(const std::filesystem::path &manifest, size_t budget,
                                    GraphGeometry expected, BackendPolicy policy) {
    BuiltExecutor result;
    if (policy.preferred == BackendPreference::Off) { result.fallback_reason = "ANE disabled by backend policy"; return result; }
    if (policy.preferred == BackendPreference::Private && !policy.allow_private)
        throw std::runtime_error("private ANE requires explicit authorization");
    const bool try_private = policy.allow_private &&
        (policy.preferred == BackendPreference::Auto || policy.preferred == BackendPreference::Private);
    const char *group=std::getenv("TURBOCIDER_PRIVATE_ANE_A8_GROUP_SIZE");
    if(group && std::string(group)!="0" && std::string(group)!="256")
        throw std::runtime_error("private ANE A8 group size requires 0 or 256");
    const bool requested_group=group && std::string(group)=="256";
    if(requested_group && !try_private)throw std::runtime_error("group A8 requires an authorized private backend");
    const int channels = private_channel_count(expected.width);
    if (channels && !try_private) throw std::runtime_error("channel split requires an authorized private W8A8 backend");
    if (try_private) {
        // Malformed manifests/geometries remain configuration failures, not
        // capability fallback. Only the native graph consumes this shape.
        auto shape = runtime_template_shape(manifest);
        if (shape.kind != expected.kind || shape.hidden != expected.hidden || shape.width != expected.width ||
            (expected.require_lora_inputs && !shape.lora_inputs))
            throw std::runtime_error("runtime ANE graph does not match model FFN geometry");
        try {
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
            const std::string path = std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH") ? std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH") : "fp16";
            if (path != "fp16" && path != "w8a8" && path != "convrot_w8a8")
                throw std::runtime_error("TURBOCIDER_PRIVATE_ANE_DATA_PATH requires fp16, w8a8 or convrot_w8a8");
            if(requested_group && path!="convrot_w8a8")throw std::runtime_error("group A8 requires convrot_w8a8 data path");
            if (channels && path == "fp16") throw std::runtime_error("channel split requires the W8A8 data path");
            if (channels) shape.width = channels; // base template still validated against the FULL model
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
