#include "../runtime/session.hpp"
#include "z_image/z_image.hpp"

namespace tc {

ModelModule z_image_module() {
    return {
        "z-image-turbo",
        [] {
            return Recipe{"z-image-turbo",
                          {{"text_encode", {}},
                           {"denoise", {"text_encode"}, 9},
                           {"vae_decode", {"denoise"}},
                           {"export", {"vae_decode"}}},
                          true};
        },
        [](const Request &r) {
            require(r.operation == "image.generate",
                    "Z-Image-Turbo currently supports image.generate only");
            require(r.inputs.empty(), "Z-Image-Turbo does not accept image inputs");
            require(r.frames == 1, "Z-Image-Turbo requires frames=1");
            require(!r.audio, "Z-Image-Turbo does not produce audio");
            require(r.width % 16 == 0 && r.height % 16 == 0,
                    "Z-Image dimensions must be multiples of 16");
            require(r.steps >= 1 && r.steps <= 50,
                    "Z-Image-Turbo steps must be 1...50 (default 9)");
            require(r.model_variant == "auto" || r.model_variant == "z-image-turbo",
                    "model_variant does not match Z-Image-Turbo");
            if (r.execution == "gpu_ane") {
                require(r.allow_approximation,
                        "Z-Image GPU+ANE requires allow_approximation=true");
                if (!r.loras.empty())
                    require(r.lora_strategy == "in_memory_merge",
                            "Z-Image GPU+ANE LoRA requires lora_strategy=in_memory_merge");
            }
            require(r.residency == "resident",
                    "Z-Image component-staged residency is not implemented");
            require(r.loras.size() <= 8, "at most eight Z-Image LoRA adapters may be active");
            for (const auto &lora : r.loras) {
                require(lora.role == "transformer",
                        "Z-Image LoRA role must be transformer");
                require(lora.strength >= -8.0f && lora.strength <= 8.0f,
                        "LoRA strength must be -8...8");
            }
        },
        [](const std::filesystem::path &root) { return std::make_unique<ZImage>(root); },
        [] {
            ModelDescriptor d;
            d.id = "z-image-turbo";
            d.name = "Z-Image Turbo";
            d.executable = true;
            d.operations = {"image.generate"};
            d.executor_operations = d.operations;
            d.inputs = {"text"};
            d.output = "image";
            d.steps = 9;
            d.frames = 1;
            d.width = 1024;
            d.height = 1024;
            d.default_audio = false;
            d.default_residency = "resident";
            d.weight_validation_pending = false;
            d.supports_lora = true;
            d.runtime_lora = true;
            d.lora_mode = "in-memory-delta";
            d.lora_strategies = {"in_memory_merge", "inference_time"};
            d.default_lora_strategy = "in_memory_merge";
            d.supports_gpu_ane = true;
            d.backend = "mlx_cpp_metal";
            d.runtime_dependency = "bundled-native-mlx-cpp";
            d.parallel_strategy = "GPU computes attention first, then the compiled MLP suffix overlaps the Core ML ANE gated-MLP prefix; base 4096-channel M4 Max route is automatic";
            d.candidate_limitations = {
                "text-to-image only",
                "inference_time LoRA is an explicit GPU path and is not yet performance-qualified",
                "automatic GPU+ANE is limited to the base model on Apple M4 Max 64 GB with the measured 4096-channel 32-block manifest",
                "the repeated warm 1024x1024 base workload measured about 1.21x end-to-end versus the optimized GPU path",
                "LoRA GPU+ANE remains explicit and requires an artifact bound to the exact adapter path, content, role and strength"
            };
            return d;
        }
    };
}

} // namespace tc
