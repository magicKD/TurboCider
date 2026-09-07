#include "../runtime/session.hpp"
#include "flux2/flux.hpp"
namespace tc {
ModelModule flux_module() {
    return {"flux2-klein-4b",
            [] {
                return Recipe{"flux2-klein-4b",
                              {{"text_encode", {}},
                               {"denoise", {"text_encode"}, 4},
                               {"vae_decode", {"denoise"}},
                               {"export", {"vae_decode"}}},
                              true};
            },
            [](const Request &r) {
                require(r.width % 16 == 0 && r.height % 16 == 0,
                        "FLUX dimensions must be multiples of 16");
                require(r.frames == 1, "FLUX requires frames=1");
                require(r.operation == "image.generate" || r.operation == "image.transform" ||
                            r.operation == "image.edit",
                        "unsupported FLUX operation");
                require(r.model_variant == "auto" || r.model_variant == "flux2-klein-4b",
                        "model_variant does not match the selected FLUX module");
                if (r.operation == "image.generate")
                    require(r.inputs.empty(), "image.generate does not accept image inputs");
                else {
                    require(!r.inputs.empty() && r.inputs.size() <= 8,
                            "FLUX requires 1...8 images");
                    if (r.operation == "image.transform")
                        require(r.inputs.size() == 1, "image.transform requires one init_image");
                    for (auto &input : r.inputs)
                        require(input.kind == "image" &&
                                    input.role == (r.operation == "image.transform" ? "init_image"
                                                                                    : "reference"),
                                "incorrect FLUX image role");
                }
                require(r.residency == "resident" || r.residency == "component_staged",
                        "FLUX block streaming is not supported");
                require(r.loras.size() <= 8, "at most eight LoRA adapters may be active");
                for (const auto &lora : r.loras) {
                    require(lora.role == "transformer" || lora.role == "text_encoder",
                            "unsupported FLUX LoRA role");
                    require(lora.strength >= -8.0f && lora.strength <= 8.0f,
                            "LoRA strength must be -8...8");
                }
            },
            [](const std::filesystem::path &root) { return std::make_unique<Flux>(root, "flux2-klein-4b"); },
            [] {
                ModelDescriptor d;
                d.id = "flux2-klein-4b"; d.name = "FLUX.2 Klein 4B"; d.executable = true;
                d.operations = {"image.generate", "image.transform", "image.edit"};
                d.executor_operations = d.operations; d.inputs = {"text", "image"};
                d.roles = {"init_image", "reference"}; d.max_images = 8; d.output = "image";
                d.steps = 4; d.frames = 1; d.width = 512; d.height = 512; d.default_audio = false;
                d.supports_lora = true; d.runtime_lora = true; d.lora_mode = "load-time-baked";
                d.supports_gpu_ane = true; d.backend = "mlx_cpp_metal";
                d.runtime_dependency = "bundled-native-mlx-cpp";
                d.parallel_strategy = "compiled GPU attention/MLP complement overlaps a Core ML ANE MLP prefix";
                d.candidate_limitations = {
                    "compiled fused GPU blocks are automatic for the validated 4B geometry",
                    "automatic GPU+ANE remains restricted to exact validated device and manifest geometry",
                    "LoRA GPU+ANE requires an artifact bound to the exact adapter path, content, role and strength"
                };
                return d;
            }};
}
ModelModule flux9_module() {
    return {
        "flux2-klein-9b",
        [] { return Recipe{"flux2-klein-9b",
                           {{"text_encode", {}}, {"denoise", {"text_encode"}, 4},
                            {"vae_decode", {"denoise"}}, {"export", {"vae_decode"}}},
                           true}; },
        [](const Request &r) {
            require(r.width % 16 == 0 && r.height % 16 == 0,
                    "FLUX dimensions must be multiples of 16");
            require(r.frames == 1, "FLUX requires frames=1");
            require(r.operation == "image.generate" || r.operation == "image.transform" ||
                        r.operation == "image.edit", "unsupported FLUX operation");
            require(r.model_variant == "auto" || r.model_variant == "flux2-klein-9b",
                    "model_variant does not match the selected FLUX module");
            if (r.operation == "image.generate")
                require(r.inputs.empty(), "image.generate does not accept image inputs");
            else {
                require(!r.inputs.empty() && r.inputs.size() <= 8, "FLUX requires 1...8 images");
                if (r.operation == "image.transform")
                    require(r.inputs.size() == 1, "image.transform requires one init_image");
                for (const auto &input : r.inputs)
                    require(input.kind == "image" &&
                                input.role == (r.operation == "image.transform" ? "init_image" : "reference"),
                            "incorrect FLUX image role");
            }
            require(r.execution != "gpu_ane", "FLUX.2 Klein 9B GPU+ANE partition is not validated");
            require(r.residency == "resident" || r.residency == "component_staged",
                    "FLUX block streaming is not supported");
            require(r.loras.size() <= 8, "at most eight LoRA adapters may be active");
            for (const auto &lora : r.loras) {
                require(lora.role == "transformer" || lora.role == "text_encoder",
                        "unsupported FLUX LoRA role");
                require(lora.strength >= -8.0f && lora.strength <= 8.0f,
                        "LoRA strength must be -8...8");
            }
        },
        [](const std::filesystem::path &root) { return std::make_unique<Flux>(root, "flux2-klein-9b"); },
        [] {
            ModelDescriptor d;
            d.id = "flux2-klein-9b"; d.name = "FLUX.2 Klein 9B";
            d.executable = true; d.operations = {"image.generate", "image.transform", "image.edit"};
            d.inputs = {"text", "image"}; d.roles = {"init_image", "reference"}; d.max_images = 8;
            d.output = "image"; d.steps = 4; d.frames = 1; d.width = 512; d.height = 512;
            d.supports_lora = true; d.runtime_lora = true; d.lora_mode = "load-time-baked";
            d.backend = "mlx_cpp_metal"; d.default_residency = "resident";
            d.default_audio = false; d.candidate_limitations = {"9B runtime requires a matching 9B checkpoint and remains GPU-only"};
            return d;
        }
    };
}
} // namespace tc
