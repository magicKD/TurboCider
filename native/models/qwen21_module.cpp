#include "qwen21/pipeline.hpp"

namespace tc {
ModelModule qwen21_module() {
    return {"qwen-image-2.1",
        [] { return Recipe{"qwen-image-2.1", {{"text_encode", {}}, {"denoise", {"text_encode"}, 40},
                                            {"vae_decode", {"denoise"}}, {"export", {"vae_decode"}}}, true}; },
        [](const Request &r) {
            require(r.operation == "image.generate" || r.operation == "image.edit", "unsupported Qwen21 operation");
            require(r.frames == 1 && !r.audio, "Qwen21 produces one RGBA image without audio");
            require(r.width % 32 == 0 && r.height % 32 == 0 && int64_t(r.width) * r.height <= 8388608,
                    "Qwen21 dimensions must be multiples of 32 within 8 megapixels");
            require(r.model_variant == "auto" || r.model_variant == "qwen-image-2.1", "incorrect Qwen21 variant");
            require(r.encoder_ane_manifest.empty(), "Qwen21 encoder ANE is not implemented");
            require(r.qwen21_reference_size == 1024 ||
                        (r.qwen21_reference_size == 256 && r.allow_approximation &&
                         r.operation == "image.edit" && !r.inputs.empty() && r.inputs.size() <= 3),
                    "Qwen21 256px reference resize requires explicit approximation and 1...3 edit images");
            require(!r.qwen21_gpu_w8a16 || r.qwen21_w8a8,
                    "Qwen21 W8A16 GPU suffix requires explicit W8A8 Core ML opt-in");
            require(!r.qwen21_w8a8 || r.steps >= 2,
                    "Qwen21 W8A8 needs at least one cached decode step");
            require(r.qwen21_gpu_full_ffn_blocks.empty() ||
                        (r.qwen21_w8a8 && r.qwen21_gpu_full_ffn_blocks == std::vector<int>{3, 5, 7}),
                    "Qwen21 full GPU fallback requires explicit W8A8 and validated layers 3,5,7");
            if (r.execution == "gpu_ane") {
                require(r.allow_approximation && !r.ane_manifest.empty() &&
                            r.width == 512 && r.height == 512 &&
                            ((r.operation == "image.generate" && r.inputs.empty() &&
                              r.qwen21_reference_size == 1024 && r.qwen21_gpu_full_ffn_blocks.empty()) ||
                             (r.qwen21_w8a8 && r.operation == "image.edit" &&
                              r.qwen21_reference_size == 256 &&
                              r.inputs.size() >= 1 && r.inputs.size() <= 3 &&
                              r.qwen21_gpu_full_ffn_blocks == std::vector<int>{3, 5, 7})),
                        "Qwen21 gpu_ane requires 512px text-to-image or explicit W8A8 1...3-reference edit with 256px references and GPU fallback 3,5,7");
            } else {
                require(r.ane_manifest.empty() && r.encoder_ane_manifest.empty() &&
                            !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
                            r.qwen21_gpu_full_ffn_blocks.empty(),
                        "Qwen21 ANE manifests require execution=gpu_ane");
            }
            require(r.residency == "resident" || r.residency == "component_staged", "Qwen21 supports resident or component_staged residency");
            require(!r.streaming_offload && r.quantized_cache.empty() && r.loras.empty(), "Qwen21 streaming/quantized cache/LoRA are not implemented");
            if (r.operation == "image.generate") require(r.inputs.empty(), "Qwen21 image.generate takes no images");
            else {
                require(!r.inputs.empty() && r.inputs.size() <= 10, "Qwen21 image.edit requires 1...10 references");
                for (const auto &input : r.inputs)
                    require(input.kind == "image" && input.role == "reference", "Qwen21 editing requires image/reference inputs");
            }
        },
        [](const std::filesystem::path &root) { return std::make_unique<qwen21::Session>(root); },
        [] {
            ModelDescriptor d;
            d.id = "qwen-image-2.1"; d.name = "Qwen Image 2.1 (experimental)"; d.executable = true;
            d.operations = d.executor_operations = {"image.generate", "image.edit"};
            d.inputs = {"text", "image"}; d.roles = {"reference"}; d.max_images = 10; d.output = "image";
            d.steps = 40; d.frames = 1; d.width = d.height = 1024; d.default_audio = false;
            d.default_residency = "component_staged"; d.backend = "mlx_cpp_metal";
            d.supports_gpu_ane = true;
            d.runtime_dependency = "bundled-native-mlx-cpp";
            d.candidate_limitations = {
                "experimental: actual edit material fidelity is under investigation; not quality-qualified",
                "BF16 Comfy checkpoint plus official processor/tokenizer.json required",
                "reference images are resized to approximately 1024 squared pixels with 32-aligned dimensions",
                "RGBA is preserved; App masks are visual references, not hard pixel-preserving inpainting",
                "native PE-T2I is optional and slow; PE-I2I requires explicit prompt_enhance_edit_experimental with FP32 vision, supported 8-bit files, and is not quality-qualified; BF16 visual parity remains unaccepted",
                "experimental gpu_ane: explicit verified FP16 512x512 text-to-image or W8A8 512x512 edit with 1...3 references scaled to 256 and full GPU FFN blocks 3,5,7; device placement and warm E2E not qualified"
            };
            return d;
        }};
}
} // namespace tc
