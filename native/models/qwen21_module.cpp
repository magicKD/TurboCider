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
            if (r.execution == "gpu_ane") {
                require(r.allow_approximation && !r.ane_manifest.empty() &&
                            r.operation == "image.generate" && r.width == 512 && r.height == 512 &&
                            r.inputs.empty(),
                        "Qwen21 gpu_ane is experimental and restricted to 512x512 text-to-image");
            } else {
                require(r.ane_manifest.empty() && r.encoder_ane_manifest.empty(),
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
                "experimental gpu_ane: explicit verified FP16 prefix4096 manifest, 512x512 text-to-image only; end-to-end performance and runtime device placement not qualified"
            };
            return d;
        }};
}
} // namespace tc
