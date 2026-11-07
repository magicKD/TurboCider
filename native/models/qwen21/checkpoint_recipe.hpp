#pragma once
#include "../../core/contracts.hpp"
#include <array>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string_view>

namespace tc::qwen21 {
// Qwen/Qwen-Image-2.1-Turbo@fc13ba1986fcb905854dc8de2f52eee395f49b5b:
// model_index.json and scheduler/scheduler_config.json. CFG=1, shift=1,
// use_dynamic_shifting=false, shift_terminal=null. FINAL sigma nodes,
// independent of canvas size, not the shifted Viggle adapter schedule.
inline constexpr std::array<float,9> turbo_sigma_nodes{
    1.f,.978453f,.95418f,.926626f,.89508f,.845148f,.704534f,.414568f,0.f};
inline constexpr std::string_view turbo_sampler_identity="qwen21-turbo-authored-fixed8-cfg1-v1";
inline bool turbo_checkpoint(const std::filesystem::path& path) {
    const auto name=path.filename().string();
    return name=="qwen_image_2.1_turbo_bf16.safetensors" || name=="qwen_image_2.1_turbo_int8_convrot.safetensors";
}
inline bool convrot_checkpoint(const std::filesystem::path& path) {
    const auto name=path.filename().string();
    return name=="qwen_image_2.1_int8_convrot.safetensors" || name=="qwen_image_2.1_turbo_int8_convrot.safetensors";
}
inline bool configured_convrot_packed() {
    const char* raw=std::getenv("TURBOCIDER_QWEN21_CONVROT_PACKED");
    if(raw && std::string_view(raw)!="0" && std::string_view(raw)!="1")
        throw std::invalid_argument("Qwen21 ConvRot packing requires0 or1");
    return !raw || std::string_view(raw)=="1";
}
// A single switch selects both components; component overrides allow the
// encoder and DiT to be measured independently against the same checkpoint.
// The runtime supplies the Metal capability: M5+ defaults to W8A8, while
// older GPUs retain floating activations. Overrides are useful for A/B runs.
inline bool configured_convrot_w8a8(bool encoder=false,bool hardware_supported=false) {
    const auto read=[](const char* key,const char* raw) {
        if(raw && std::string_view(raw)!="0" && std::string_view(raw)!="1")
            throw std::invalid_argument(std::string(key)+" requires 0 or 1");
        return raw && std::string_view(raw)=="1";
    };
    const char* common=std::getenv("TURBOCIDER_QWEN21_CONVROT_W8A8");
    const bool enabled=common ? read("TURBOCIDER_QWEN21_CONVROT_W8A8",common) : hardware_supported;
    const char* key=encoder ? "TURBOCIDER_QWEN21_CONVROT_W8A8_ENCODER" :
                             "TURBOCIDER_QWEN21_CONVROT_W8A8_DIT";
    const char* component=std::getenv(key);
    return component ? read(key,component) : enabled;
}
inline std::filesystem::path transformer_checkpoint_source(const std::filesystem::path& root) {
    // Existing base roots retain their preference. A separate Turbo/ConvRot
    // root uses the ORIGINAL filename and shared read-only companions. Never
    // rename Turbo to base: schedule and source must agree. A filename chooses
    // a recipe, not a content/quality verification certificate.
    for(const char* name:{"qwen_image_2.1_bf16.safetensors","qwen_image_2.1_int8_convrot.safetensors",
        "qwen_image_2.1_turbo_bf16.safetensors","qwen_image_2.1_turbo_int8_convrot.safetensors","qwen-image-2.1-Q4_K_M.gguf"}) {
        const auto path=root/"diffusion_models"/name;
        if(std::filesystem::is_regular_file(path))return path;
    }
    throw std::invalid_argument("missing supported Qwen Image 2.1 base/Turbo/ConvRot/GGUF checkpoint");
}
inline void validate_turbo_request(const Request& r) {
    const auto check=[](bool ok,const char* error){if(!ok)throw std::invalid_argument(error);};
    check(r.steps==8 && r.loras.empty(),"Qwen21 Turbo requires its authored eight-step schedule and no base/student LoRA");
    check(r.qwen21_denoiser_backend=="mlx" && (r.execution=="gpu" || r.execution=="auto" ||
        (r.execution=="gpu_ane" && r.hybrid_mlp_mode=="runtime")),"Qwen21 Turbo supports ordinary MLX GPU or explicit runtime FFN ANE only");
    check(!r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 && r.qwen21_gpu_full_ffn_blocks.empty() &&
        r.qwen21_dit_cache=="off" && !r.prompt_enhance && !r.streaming.active() && !r.memory_constrained.enabled && !r.memory_budget_bytes,
        "Turbo excludes frozen base manifests, temporal cache, PE and constrained/streaming recipes until separately validated");
    check((r.operation=="image.generate" && r.inputs.empty()) || (r.operation=="image.edit" && !r.inputs.empty() && r.inputs.size()<=2),
        "Turbo initial route supports generation or one/two-reference editing");
}
inline constexpr uint64_t convrot_stream_weight_budget_bytes = uint64_t(4) << 30;

inline bool convrot_stream_ane_requested(const Request &r) {
    if (r.residency != "streamed" || r.execution != "gpu_ane" || r.hybrid_mlp_mode != "runtime")
        return false;
    const char *value = std::getenv("TURBOCIDER_QWEN21_CONVROT_STREAM_ANE");
    if (!value || std::string_view(value) != "1") return false;
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    return true;
#else
    throw std::invalid_argument("Qwen21 ConvRot streaming Runtime ANE requires a private diagnostic build");
#endif
}

inline void validate_convrot_stream_request(const Request &r) {
    const bool canvas = r.width >= 64 && r.height >= 64 && r.width <= 512 && r.height <= 512 &&
                        r.width % 32 == 0 && r.height % 32 == 0;
    if (!canvas || r.steps < 2 || r.steps > 40 || r.operation != "image.generate" || !r.inputs.empty())
        throw std::invalid_argument("Qwen21 ConvRot streaming supports text-to-image up to 512x512 with 2...40 steps (Turbo uses exactly 8)");
    const bool runtime_ane = convrot_stream_ane_requested(r);
    const bool ordinary_gpu = (r.execution == "gpu" || r.execution == "auto") &&
                              r.hybrid_mlp_mode == "auto" && r.ane_manifest.empty();
    const bool diagnostic = runtime_ane && r.width == 512 && r.height == 512 && r.steps == 8 && !r.ane_manifest.empty();
    if (!r.allow_approximation || (!ordinary_gpu && !diagnostic) || r.qwen21_denoiser_backend != "mlx" ||
        !r.encoder_ane_manifest.empty() || !r.loras.empty() ||
        r.prompt_enhance || r.qwen21_dit_cache != "off" || r.qwen21_w8a8 || r.qwen21_gpu_w8a16 ||
        r.qwen21_dit_w8a8 || r.qwen21_dit_smoothquant_convrot || !r.qwen21_gpu_full_ffn_blocks.empty() ||
        r.streaming.active() || r.streaming_offload || r.memory_constrained.enabled || r.memory_budget_bytes)
        throw std::invalid_argument("Qwen21 ConvRot streaming requires approximate MLX GPU generation without LoRA, ANE, prompt enhancement or another memory/cache recipe");
}

inline void validate_checkpoint_recipe(const Request& r,const std::filesystem::path& source,bool w8a8_available=false) {
    const bool turbo=turbo_checkpoint(source);
    if(r.model_variant!="auto" && r.model_variant!=(turbo ? "qwen-image-2.1-turbo" : "qwen-image-2.1"))
        throw std::invalid_argument("Qwen21 requested model_variant differs from selected checkpoint; use a separate matching model root");
    if(turbo)validate_turbo_request(r);
    if (r.residency == "streamed") validate_convrot_stream_request(r);
    if(convrot_checkpoint(source)) {
        const auto check=[](bool ok,const char* error){if(!ok)throw std::invalid_argument(error);};
        const bool packed=configured_convrot_packed();
        const bool w8a8=configured_convrot_w8a8(false,w8a8_available);
        check(r.allow_approximation && (r.residency=="resident" || r.residency=="streamed") && r.loras.empty() && r.qwen21_denoiser_backend=="mlx" &&
            !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 && r.qwen21_dit_cache=="off" &&
            !r.streaming.active() && !r.memory_constrained.enabled && !r.prompt_enhance,
            "Qwen21 ConvRot initial route requires explicit approximation, MLX and no base/student LoRA, frozen or temporal recipe");
        const bool streamed_runtime_ane = convrot_stream_ane_requested(r);
        check(!w8a8 || streamed_runtime_ane || ((r.execution=="gpu" || r.execution=="auto") &&
            r.ane_manifest.empty() && r.encoder_ane_manifest.empty()),
            "ConvRot Metal W8A8 requires ordinary GPU execution without ANE manifests");
        if(r.execution=="gpu_ane" && !streamed_runtime_ane) {
            const auto selected=[](const char* key,std::string_view value){const char* v=std::getenv(key);return v&&std::string_view(v)==value;};
            check(packed && r.hybrid_mlp_mode=="runtime" && r.residency=="resident" && !r.ane_manifest.empty() && r.encoder_ane_manifest.empty() &&
                selected("TURBOCIDER_ANE_BACKEND","private") && selected("TURBOCIDER_ALLOW_PRIVATE_ANE","1") &&
                selected("TURBOCIDER_PRIVATE_ANE_DATA_PATH","convrot_w8a8") && selected("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","1") &&
                selected("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN","1"),
                "Qwen21 ConvRot ANE requires packed original F32 row coefficients, explicit fixed Private Comfy-H256 runtime/F32 join and GPU encoder");
            for(const char* key:{"TURBOCIDER_QWEN21_PRIVATE_QKV","TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH",
                "TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT","TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_SNAPSHOT"}) {
                const char* value=std::getenv(key);check(!value || std::string_view(value)=="0",
                    "ConvRot initial runtime does not borrow dense-BF16 QKV or prefix-cache qualification");
            }
        }
    }
}
} // namespace tc::qwen21
