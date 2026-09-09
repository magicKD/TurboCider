#include "../runtime/session.hpp"

namespace tc {

std::unique_ptr<ModelSession> create_z_image_gguf(const std::filesystem::path &);

ModelModule z_image_gguf_module() {
    return {
        "z-image-turbo-gguf",
        [] {
            return Recipe{"z-image-turbo-gguf",
                          {{"native_server_load", {}},
                           {"text_encode", {"native_server_load"}},
                           {"denoise", {"text_encode"}, 9},
                           {"vae_decode", {"denoise"}},
                           {"export", {"vae_decode"}}},
                          true};
        },
        [](const Request &r) {
            require(r.operation == "image.generate",
                    "Z-Image GGUF currently supports image.generate only");
            require(r.inputs.empty(), "Z-Image GGUF does not accept image inputs");
            require(r.frames == 1, "Z-Image GGUF requires frames=1");
            require(!r.audio, "Z-Image GGUF does not produce audio");
            require(r.width % 16 == 0 && r.height % 16 == 0,
                    "Z-Image GGUF dimensions must be multiples of 16");
            require(r.steps == 9, "Z-Image-Turbo GGUF requires its 9-step schedule");
            require(r.execution == "gpu" || r.execution == "auto" ||
                        r.execution == "gpu_ane",
                    "Z-Image GGUF execution must be gpu, auto or gpu_ane");
            if (r.execution == "gpu_ane")
                require(r.allow_approximation,
                        "Z-Image GGUF GPU+ANE requires allow_approximation=true");
            if (r.execution == "gpu_ane" && !r.loras.empty())
                require(r.lora_strategy == "in_memory_merge",
                        "Z-Image GGUF GPU+ANE LoRA requires lora_strategy=in_memory_merge");
            require(r.residency == "resident" || r.residency == "streaming",
                    "Z-Image GGUF residency must be resident or streaming");
            if (r.streaming_offload || r.residency == "streaming") {
                require(r.memory_budget_bytes >= (1ull << 30),
                        "Z-Image GGUF streaming offload requires memory_budget_bytes >= 1 GiB");
                require(r.execution != "gpu_ane",
                        "Z-Image GGUF streaming offload is GPU-only");
                require(effective_lora_strategy(r) != "in_memory_merge",
                        "Z-Image GGUF streaming offload cannot use in_memory_merge; use inference_time");
            }
            require(r.loras.size() <= 8, "at most eight Z-Image GGUF LoRAs may be active");
            for (const auto &lora : r.loras) {
                require(lora.role == "transformer",
                        "Z-Image GGUF LoRA role must be transformer");
                require(lora.strength >= -8.0f && lora.strength <= 8.0f,
                        "LoRA strength must be -8...8");
            }
        },
        create_z_image_gguf,
        [] {
            ModelDescriptor d;
            d.id = "z-image-turbo-gguf";
            d.name = "Z-Image Turbo GGUF";
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
            d.supports_lora = true;
            d.runtime_lora = true;
            d.lora_mode = "sd-cpp-request-time-or-native-low-rank";
            d.lora_strategies = {"inference_time", "in_memory_merge"};
            d.default_lora_strategy = "inference_time";
            d.supports_gpu_ane = true;
            d.backend = "stable-diffusion-cpp-metal or native MLX GGUF";
            d.runtime_dependency = "managed-stable-diffusion-cpp and bundled-native-mlx-cpp";
            d.parallel_strategy = "sd.cpp Metal for broad quantization coverage; Q8_0/Q4_0/Q4_1 native MLX can split the FFN between Metal and Core ML ANE";
            d.candidate_limitations = {
                "requires a compatible sd-server binary",
                "GGUF transformer, VAE and Qwen3 text encoder remain separate files",
                "mixed K-quants remain on sd.cpp Metal; native GPU+ANE currently accepts Q8_0/Q4_0/Q4_1 or floating GGUF",
                "in_memory_merge requires a native-compatible Q8_0/Q4_0/Q4_1 or floating GGUF checkpoint",
                "inference_time uses sd.cpp for mixed K-quants and can use the packed native low-rank branch for native-compatible GGUF when TURBOCIDER_Z_GGUF_NATIVE_GPU is enabled",
                "streaming residency uses the pinned sd.cpp CPU-staged layer-prefetch path, keeps text/VAE disk-backed, and requires an explicit memory budget",
                "GPU+ANE remains explicit until checkpoint-bound artifacts pass paired performance and quality gates",
                "quality and speed gates require the downloaded Q4 checkpoint and paired reference run"
            };
            return d;
        }};
}

} // namespace tc
