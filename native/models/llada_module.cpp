#include "../runtime/session.hpp"

namespace tc {

std::unique_ptr<ModelSession> create_llada_image(const std::filesystem::path &);

ModelModule llada_module() {
    return {
        "llada-image-turbo",
        [] {
            return Recipe{"llada-image-turbo",
                          {{"text_encode", {}},
                           {"denoise", {"text_encode"}, 4},
                           {"vae_decode", {"denoise"}},
                           {"export", {"vae_decode"}}},
                          true};
        },
        [](const Request &r) {
            require(r.operation == "image.generate",
                    "native LLaDA-Image-Turbo currently supports image.generate only");
            require(r.frames == 1, "LLaDA-Image-Turbo requires frames=1");
            require(!r.audio, "LLaDA-Image-Turbo does not produce audio");
            require(r.steps == 4, "LLaDA-Image-Turbo requires its 4-step schedule");
            require(r.residency == "resident",
                    "LLaDA-Image-Turbo requires resident execution");
            require(r.width % 16 == 0 && r.height % 16 == 0,
                    "LLaDA text dimensions must be multiples of 16");
            require(r.inputs.empty(), "LLaDA image.generate does not accept image inputs");
            require(r.loras.empty(),
                    "LLaDA-Image-Turbo LoRA is not yet supported by the native executor");
            if (r.execution == "gpu_ane") {
                require(r.allow_approximation,
                        "LLaDA GPU+ANE requires allow_approximation=true");
                require(r.operation == "image.generate",
                        "LLaDA GPU+ANE currently supports text-to-image only");
            }
        },
        create_llada_image,
        [] {
            ModelDescriptor d;
            d.id = "llada-image-turbo";
            d.name = "LLaDA-Image-Turbo";
            d.executable = true;
            d.operations = {"image.generate"};
            d.executor_operations = d.operations;
            d.inputs = {"text"};
            d.output = "image";
            d.steps = 4;
            d.frames = 1;
            d.width = 1024;
            d.height = 1024;
            d.default_audio = false;
            d.default_residency = "resident";
            d.weight_validation_pending = false;
            d.supports_lora = false;
            d.runtime_lora = false;
            d.lora_mode = "unsupported";
            d.supports_gpu_ane = true;
            d.backend = "native C++/MLX Metal; optional Core ML ANE FFN partition";
            d.runtime_dependency = "bundled native MLX C++ plus Apple Core ML frameworks";
            d.parallel_strategy = "native MLX attention plus GPU FFN suffix, with a checkpoint-bound Core ML FFN prefix; explicit approximation opt-in";
            d.candidate_limitations = {
                "text-to-image uses the in-process native C++/MLX engine",
                "image editing is not exposed until the native reference-image path is complete",
                "GPU+ANE remains explicit until repeated 1024x1024 E2E parity and speedup are proven",
                "LoRA is not exposed until a LLaDA adapter format is identified"
            };
            return d;
        }
    };
}

} // namespace tc
