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
            require(r.steps == 9, "Z-Image-Turbo requires its 9-step schedule");
            require(r.model_variant == "auto" || r.model_variant == "z-image-turbo",
                    "model_variant does not match Z-Image-Turbo");
            require(r.execution != "gpu_ane",
                    "Z-Image GPU+ANE partition is not validated; use gpu or auto");
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
            d.weight_validation_pending = true;
            d.supports_lora = true;
            d.runtime_lora = true;
            d.lora_mode = "in-memory-delta";
            d.supports_gpu_ane = false;
            d.backend = "mlx_cpp_metal";
            d.runtime_dependency = "bundled-native-mlx-cpp";
            d.parallel_strategy = "single Metal GPU stream; ANE partition pending";
            d.candidate_limitations = {
                "text-to-image only",
                "GPU+ANE partition remains fail-closed until parity and speedup are validated",
                "full real-weight image parity remains pending"
            };
            return d;
        }
    };
}

} // namespace tc
