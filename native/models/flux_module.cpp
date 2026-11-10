#include "../runtime/session.hpp"
#include "image_recipe.hpp"
#include "flux2/flux.hpp"
#include <cstdlib>
#include <cstring>
namespace tc {
namespace {
void validate_flux_request(const Request &request, const char *variant) {
    require(request.width % 16 == 0 && request.height % 16 == 0,
            "FLUX dimensions must be multiples of 16");
    require(request.frames == 1, "FLUX requires frames=1");
    require(request.operation == "image.generate" ||
                request.operation == "image.transform" ||
                request.operation == "image.edit",
            "unsupported FLUX operation");
    require(request.model_variant == "auto" || request.model_variant == variant,
            "model_variant does not match the selected FLUX module");
}

void validate_flux_inputs(const Request &request) {
    if (request.operation == "image.generate") {
        require(request.inputs.empty(), "image.generate does not accept image inputs");
    } else {
        require(!request.inputs.empty() && request.inputs.size() <= 8,
                "FLUX requires 1...8 images");
        if (request.operation == "image.transform")
            require(request.inputs.size() == 1, "image.transform requires one init_image");
        for (const auto &input : request.inputs)
            require(input.kind == "image" &&
                        input.role == (request.operation == "image.transform" ? "init_image"
                                                                              : "reference"),
                    "incorrect FLUX image role");
    }
}

void validate_flux_residency_and_loras(const Request &request) {
    require(request.residency == "resident" || request.residency == "component_staged",
            "FLUX block streaming is not supported");
    require(request.loras.size() <= 8, "at most eight LoRA adapters may be active");
    for (const auto &lora : request.loras) {
        require(lora.role == "transformer" || lora.role == "text_encoder",
                "unsupported FLUX LoRA role");
        require(lora.strength >= -8.0f && lora.strength <= 8.0f,
                "LoRA strength must be -8...8");
    }
}
} // namespace

ModelModule flux_module() {
    return {"flux2-klein-4b",
            [] {
                return model_detail::image_recipe("flux2-klein-4b", 4);
            },
            [](const Request &r) {
                validate_flux_request(r, "flux2-klein-4b");
                validate_flux_inputs(r);
                validate_flux_residency_and_loras(r);
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
                d.lora_strategies = {"in_memory_merge"};
                d.default_lora_strategy = "in_memory_merge";
                d.supports_gpu_ane = true; d.supports_encoder_gpu_ane = true;
                d.backend = "mlx_cpp_metal";
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
        [] { return model_detail::image_recipe("flux2-klein-9b", 4); },
        [](const Request &r) {
            validate_flux_request(r, "flux2-klein-9b");
            if ([[maybe_unused]] const char *predecode = std::getenv("TURBOCIDER_FLUX9_PREDECODE_AFFINE")) {
#ifdef TURBOCIDER_ENABLE_QUANTIZED_EXECUTION_EXPERIMENTS
                require(std::strcmp(predecode, "1") == 0 && r.execution == "gpu" &&
                            r.compile_gpu && r.allow_approximation && r.residency == "resident" &&
                            !r.streaming.active() && r.operation == "image.generate" &&
                            r.width == 512 && r.height == 512 && r.steps == 4 &&
                            !r.dynamic_text && r.loras.empty(),
                        "FLUX9B affine predecode requires exact512/4 base GPU experimental opt-in");
#else
                require(false, "FLUX9B affine predecode is experimental-only");
#endif
            }
            validate_flux_inputs(r);
            if (r.execution == "gpu_ane") {
#ifdef TURBOCIDER_ENABLE_QUANTIZED_EXECUTION_EXPERIMENTS
                const char *candidate = std::getenv("TURBOCIDER_FLUX9_EXPERIMENTAL_HYBRID");
                require(candidate && std::strcmp(candidate, "1") == 0 &&
                            r.allow_approximation && !r.ane_manifest.empty() &&
                            r.operation == "image.generate" && r.width == 512 &&
                            r.height == 512 && r.steps == 4 && !r.dynamic_text && r.loras.empty(),
                        "FLUX9B hybrid requires explicit source-bound512/4 base experimental opt-in");
#else
                require(false, "FLUX.2 Klein 9B GPU+ANE partition is not validated");
#endif
            }
            validate_flux_residency_and_loras(r);
        },
        [](const std::filesystem::path &root) { return std::make_unique<Flux>(root, "flux2-klein-9b"); },
        [] {
            ModelDescriptor d;
            d.id = "flux2-klein-9b"; d.name = "FLUX.2 Klein 9B";
            d.executable = true; d.operations = {"image.generate", "image.transform", "image.edit"};
            d.inputs = {"text", "image"}; d.roles = {"init_image", "reference"}; d.max_images = 8;
            d.output = "image"; d.steps = 4; d.frames = 1; d.width = 512; d.height = 512;
            d.supports_lora = true; d.runtime_lora = true; d.lora_mode = "load-time-baked";
            d.lora_strategies = {"in_memory_merge"};
            d.default_lora_strategy = "in_memory_merge";
            d.supports_encoder_gpu_ane = true;
            d.backend = "mlx_cpp_metal"; d.default_residency = "resident";
            d.default_audio = false;
            d.candidate_limitations = {
                "9B runtime requires a matching 9B checkpoint",
                "the 9B denoiser remains GPU-only; an explicit Qwen3 encoder GPU+ANE manifest is experimental"
            };
            return d;
        }
    };
}
} // namespace tc
