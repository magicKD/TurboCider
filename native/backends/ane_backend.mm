#include "ane_backend.hpp"
#include "ane_fp16_value_config.hpp"
#include "ane_public_w8.hpp"
#include "ane_w8_functions.hpp"
#include <chrono>
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
#include "private/ane_executor.hpp"
#include "private/ane_w8_executor.hpp"
#endif

namespace tc::ane {
BuiltExecutor build_runtime_executor(const std::filesystem::path &manifest, size_t budget,
                                    GraphGeometry expected, BackendPolicy policy,
                                    std::optional<int> calibrated_channels,std::optional<int> channel_override) {
    return build_runtime_executor(manifest,budget,expected,policy,calibrated_channels,channel_override,std::nullopt);
}
BuiltExecutor build_runtime_executor(const std::filesystem::path &manifest, size_t budget,
                                    GraphGeometry expected, BackendPolicy policy,
                                    std::optional<int> calibrated_channels,std::optional<int> channel_override,
                                    std::optional<int> bucket_override, int fixed_bucket_rows) {
    BuiltExecutor result;
    const bool fixed_program=configured_fixed_program();
    const auto projection=configured_w8_projection();
    const auto activation=configured_w8_activation();
    const auto function_rows=configured_w8_function_rows();
    if(!function_rows.empty() && (!fixed_program || policy.preferred!=BackendPreference::Private || !policy.allow_private ||
        expected.kind!=Kind::SwiGLU || expected.require_lora_inputs || calibrated_channels || bucket_override || fixed_bucket_rows ||
        configured_request_rows() || projection!=W8Projection::Descriptor ||
        !std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH") || std::string_view(std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH"))!="w8a8"))
        throw std::runtime_error("W8 function family requires fixed authorized Private base SwiGLU descriptor channels without alternate row/projection policy");
    validate_w8_activation_backend(activation,projection,fixed_program,policy,
        std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH")?std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH"):"fp16",expected.require_lora_inputs);
    validate_w8_projection_backend(projection,fixed_program,policy,
        std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH")?std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH"):"fp16",expected.require_lora_inputs);
    validate_fixed_program_backend(fixed_program,policy,
        std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH")?std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH"):"fp16");
    if(fixed_program && (expected.kind!=Kind::SwiGLU || configured_request_rows() || calibrated_channels))
        throw std::runtime_error("fixed program requires descriptor SwiGLU without request-row or calibration reconfiguration");
    if(bucket_override && (fixed_bucket_rows || configured_request_rows()))
        throw std::runtime_error("Private bucket modes are mutually exclusive");
    if(bucket_override && (*bucket_override<32 || *bucket_override>8192 || *bucket_override%32 ||
            policy.preferred!=BackendPreference::Private || !policy.allow_private || expected.kind!=Kind::SwiGLU || calibrated_channels))
        throw std::runtime_error("Private bucket override requires rows32..8192 aligned32, explicit authorized SwiGLU and no calibration");
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
    const int channels = resolved_private_channel_count(expected.width, calibrated_channels,channel_override);
    if(!function_rows.empty() && (channels<=0 || fp16_values || bf16_boundaries || requested_group))
        throw std::runtime_error("W8 functions require fixed positive channels and the ungrouped Sylvester recipe");
    if(bucket_override && channels<=0)throw std::runtime_error("Private bucket override requires fixed positive intermediate channels");
    if (channels && !try_private) throw std::runtime_error("channel split requires an authorized private W8A8 backend");
    if(runtime_template_w8a8(manifest)) {
        if(fixed_bucket_rows)throw std::runtime_error("Public W8A8 template does not support a private fixed bucket override");
        if(policy.preferred!=BackendPreference::Public || channels || fp16_values || bf16_boundaries || requested_group)
            throw std::runtime_error("Public W8A8 template requires Public policy and its explicit row/basis ABI");
        result.executor=std::make_unique<PublicW8Graph>(manifest,budget,expected);
        return result;
    }
    auto descriptor = runtime_template_descriptor(manifest);
    if (!descriptor.has_public_artifact() && policy.preferred == BackendPreference::Public)
        throw std::runtime_error("private runtime ANE shape descriptor requires a private backend; it has no Public Core ML artifact");
    // Both descriptor types validate the complete model before any capability
    // fallback or smaller channel split. A malformed config must not be hidden.
    auto shape = descriptor.shape;
    if(!function_rows.empty()) {
        if(descriptor.has_public_artifact())throw std::runtime_error("W8 functions require a shape-only private descriptor");
        (void)checked_w8_function_rows(shape,function_rows);
    }
    if(activation!=W8Activation::ExpDivide && (descriptor.has_public_artifact() || shape.lora_inputs))
        throw std::runtime_error("experimental W8 activation requires a base-only shape descriptor");
    if(projection!=W8Projection::Descriptor && (descriptor.has_public_artifact() || shape.lora_inputs))
        throw std::runtime_error("experimental W8 projection requires a base-only shape descriptor");
    if (shape.kind != expected.kind || shape.hidden != expected.hidden || shape.width != expected.width ||
        (expected.require_lora_inputs && !shape.lora_inputs))
        throw std::runtime_error("runtime ANE graph does not match model FFN geometry");
    if(fixed_bucket_rows) {
        if(!try_private || descriptor.has_public_artifact() || channels<=0 || configured_request_rows() ||
                fixed_bucket_rows<512 || fixed_bucket_rows>2048 || fixed_bucket_rows%128)
            throw std::runtime_error("fixed bucket override requires Private channels, descriptor rows and bounded128 geometry");
        shape.rows=fixed_bucket_rows;
    }
    if (configured_request_rows()) {
        if (!try_private || descriptor.has_public_artifact() || channels <= 0)
            throw std::runtime_error("request rows require Private channel splitting and a shape-only descriptor");
        shape.rows = private_request_bucket(expected.request_rows);
    }
    if (try_private) {
        // Malformed manifests/geometries remain configuration failures, not
        // capability fallback. Only the native graph consumes this shape.
        if(bucket_override)shape.rows=*bucket_override;
        if (calibrated_channels && channels == 0) {
            result.fallback_reason = "native channel calibration selected optimized GPU-only";
            return result;
        }
        [[maybe_unused]] std::filesystem::path private_cache;
        if (const char *raw = std::getenv("TURBOCIDER_PRIVATE_ANE_CACHE_DIR")) {
            private_cache = raw;
            if (private_cache.empty() || !private_cache.is_absolute() || private_cache.native().size() > 4096)
                throw std::runtime_error("TURBOCIDER_PRIVATE_ANE_CACHE_DIR requires a bounded absolute directory path");
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
            std::unique_ptr<Executor> graph = !function_rows.empty()?std::unique_ptr<Executor>(std::make_unique<PrivateW8Graph>(shape,budget,
                private_cache,W8Basis::SylvesterDH,function_rows)):path != "fp16" ? std::unique_ptr<Executor>(std::make_unique<PrivateW8Graph>(shape,budget,
                private_cache,path=="convrot_w8a8"?W8Basis::ComfyH256:W8Basis::SylvesterDH)) :
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
