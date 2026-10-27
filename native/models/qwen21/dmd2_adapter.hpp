#pragma once
#include "../../core/common.hpp"
#include <array>
#include <string_view>

namespace tc::qwen21 {
// linoyts/Qwen-Image-2.1-DMD2-2steps, immutable upstream revision
// 95c7e9c2b32607d5464d6b74a86d66c49ce87e03. The sampler is part of the
// distillation contract, not the base model's shifted/terminal-stretched Euler.
inline constexpr std::string_view dmd2_filename = "Qwen-Image-2.1-DMD2-2steps.safetensors";
inline constexpr std::string_view dmd2_sha256 =
    "3a50c28d2f70170a06011e7b5d93a12418edbc4a0fc4de4acee8d9d227a38c21";
inline constexpr std::string_view dmd2_sampler_identity = "qwen21-dmd2-fixed-1-0.5-0-cfg1-v1";
inline constexpr std::array<float, 3> dmd2_sigma_nodes = {1.f, .5f, 0.f};
inline constexpr size_t dmd2_projection_count = 128;

inline bool dmd2_requested(const Request &r) {
    return r.qwen21_sampler == "dmd2" ||
        (r.qwen21_sampler == "auto" && r.loras.size() == 1 &&
         std::filesystem::path(r.loras.front().path).filename().string() == dmd2_filename);
}

inline void validate_dmd2_request(const Request &r) {
    require(r.model == "qwen-image-2.1" && r.model_variant != "qwen-image-2.1-turbo" &&
                r.steps == 2 && r.frames == 1 && !r.audio && r.allow_approximation &&
                r.execution == "gpu" && r.hybrid_mlp_mode == "auto" &&
                r.qwen21_denoiser_backend == "mlx" && r.ane_manifest.empty() &&
                r.encoder_ane_manifest.empty() && !r.prompt_enhance &&
                !r.prompt_enhance_edit_experimental && !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
                r.qwen21_dit_cache == "off" && r.qwen21_gpu_full_ffn_blocks.empty() &&
                !r.memory_constrained.enabled && !r.memory_budget_bytes && !r.streaming.active() &&
                !r.streaming_selector && (r.residency == "resident" || r.residency == "component_staged") &&
                r.loras.size() == 1 && r.loras[0].role == "transformer" &&
                r.loras[0].strength == 1.f && r.lora_strategy == "inference_time" &&
                r.width >= 64 && r.height >= 64 && r.width <= 2048 && r.height <= 2048 &&
                r.width % 32 == 0 && r.height % 32 == 0 && int64_t(r.width) * r.height <= 1048576 &&
                (r.qwen21_reference_size == 1024 || r.qwen21_reference_size == 512) &&
                ((r.operation == "image.generate" && r.inputs.empty()) ||
                 (r.operation == "image.edit" && !r.inputs.empty() && r.inputs.size() <= 3)),
            "DMD2 requires the original Qwen Image 2.1 base, exactly two fixed-sigma steps, "
            "one pinned attention-only LoRA at strength 1, explicit approximation, GPU/MLX, "
            "at most 1MP and 0...3 references without another sampler/ANE/streaming recipe");
}
} // namespace tc::qwen21
