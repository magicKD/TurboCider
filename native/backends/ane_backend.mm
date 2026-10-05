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
    const int channels = private_channel_count(expected.width);
    if (channels && !try_private) throw std::runtime_error("channel split requires an authorized private W8A8 backend");
    auto descriptor = runtime_template_descriptor(manifest);
    if (!descriptor.has_public_artifact() && policy.preferred == BackendPreference::Public)
        throw std::runtime_error("private runtime ANE shape descriptor requires a private backend; it has no Public Core ML artifact");
    // Both descriptor types validate the complete model before any capability
    // fallback or smaller channel split. A malformed config must not be hidden.
    auto shape = descriptor.shape;
    if (shape.kind != expected.kind || shape.hidden != expected.hidden || shape.width != expected.width ||
        (expected.require_lora_inputs && !shape.lora_inputs))
        throw std::runtime_error("runtime ANE graph does not match model FFN geometry");
    if (try_private) {
        // Malformed manifests/geometries remain configuration failures, not
        // capability fallback. Only the native graph consumes this shape.
        [[maybe_unused]] std::filesystem::path private_cache;
        if (const char *raw = std::getenv("TURBOCIDER_PRIVATE_ANE_CACHE_DIR")) {
            private_cache = raw;
            if (private_cache.empty() || !private_cache.is_absolute() || private_cache.native().size() > 4096)
                throw std::runtime_error("TURBOCIDER_PRIVATE_ANE_CACHE_DIR requires a bounded absolute directory path");
        }
        try {
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
            const std::string path = std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH") ? std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH") : "fp16";
            if (path != "fp16" && path != "w8a8") throw std::runtime_error("TURBOCIDER_PRIVATE_ANE_DATA_PATH requires fp16 or w8a8");
            if (channels && path != "w8a8") throw std::runtime_error("channel split requires the W8A8 data path");
            if (channels) shape.width = channels; // base template still validated against the FULL model
            std::unique_ptr<Executor> graph = path == "w8a8" ? std::unique_ptr<Executor>(std::make_unique<PrivateW8Graph>(shape,budget,private_cache)) :
                std::unique_ptr<Executor>(std::make_unique<PrivateGraph>(shape,budget,private_cache));
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
    if (!descriptor.has_public_artifact()) {
        if (!result.fallback_reason.empty()) result.fallback_reason += "; ";
        result.fallback_reason += "private runtime ANE shape descriptor has no Public Core ML artifact; using GPU";
        return result;
    }
    result.executor = std::make_unique<RuntimeGraph>(manifest, budget, false, false, expected);
    return result;
}
}
