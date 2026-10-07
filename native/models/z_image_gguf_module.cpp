#include "../runtime/session.hpp"

namespace tc {

std::unique_ptr<ModelSession> create_z_image_gguf(const std::filesystem::path &);

ModelModule z_image_gguf_module() {
    return {
        "z-image-turbo-gguf",
        [] {
            return Recipe{"z-image-turbo-gguf",
                          {{"text_encode", {}},
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
            require(r.steps >= 1 && r.steps <= 50,
                    "Z-Image-Turbo GGUF steps must be 1...50 (default 9)");
            require(r.execution == "gpu" || r.execution == "auto" ||
                        r.execution == "gpu_ane",
                    "Z-Image GGUF execution must be gpu, auto or gpu_ane");
            require(r.hybrid_mlp_mode == "auto" || r.hybrid_mlp_mode == "base_fused" ||
                        r.hybrid_mlp_mode == "lora_merged" || r.hybrid_mlp_mode == "runtime",
                    "unsupported Z-Image GGUF hybrid_mlp_mode");
            if (r.hybrid_mlp_mode == "runtime")
                require(r.execution == "gpu_ane" && r.allow_approximation &&
                            !r.ane_manifest.empty() && (r.loras.empty() || r.lora_strategy=="inference_time") &&
                            r.encoder_ane_manifest.empty(),
                        "Z-Image GGUF runtime-weight FFN requires explicit GPU/ANE, inference-time LoRA and no encoder ANE");
            else if (r.hybrid_mlp_mode != "auto")
                require(r.execution == "gpu_ane" && r.allow_approximation && !r.ane_manifest.empty() &&
                            (r.hybrid_mlp_mode == "base_fused" ? r.loras.empty() :
                             (!r.loras.empty() && r.lora_strategy == "in_memory_merge")),
                        "Z-Image GGUF explicit frozen hybrid needs a matching base or merged-adapter request");
            if (r.execution == "gpu_ane")
                require(r.allow_approximation,
                        "Z-Image GGUF GPU+ANE requires allow_approximation=true");
            if (r.execution == "gpu_ane" && !r.loras.empty() && r.hybrid_mlp_mode!="runtime")
                require(r.lora_strategy == "in_memory_merge",
                        "Z-Image GGUF GPU+ANE LoRA requires lora_strategy=in_memory_merge");
            require(r.residency == "resident" && !r.streaming_offload,
                    "native MLX GGUF currently supports resident execution only");
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
            d.lora_mode = "native-low-rank-or-in-memory-merge";
            d.lora_strategies = {"inference_time", "in_memory_merge"};
            d.default_lora_strategy = "inference_time";
            d.supports_gpu_ane = true;
            d.supports_encoder_gpu_ane = true;
            d.backend = "mlx_cpp_metal_gguf";
            d.runtime_dependency = "bundled-native-mlx-cpp";
            d.parallel_strategy = "Q8_0/Q4_0/Q4_1 native MLX can split the FFN between Metal and Core ML ANE";
            d.candidate_limitations = {
                "GGUF transformer, VAE and Qwen3 text encoder remain separate files",
                "native MLX currently accepts Q8_0/Q4_0/Q4_1 or floating GGUF; mixed K-quants are unsupported",
                "in_memory_merge requires a native-compatible Q8_0/Q4_0/Q4_1 or floating GGUF checkpoint",
                "inference_time uses the native packed low-rank branch",
                "streaming residency is not yet supported by the native MLX GGUF executor",
                "frozen GPU+ANE uses checkpoint-bound artifacts; runtime-weight FFN uses shape-bound artifacts",
                "runtime-weight FFN is explicit, resident, without encoder ANE; inference-time LoRA uses GPU corrections/hidden output and packed GPU projections stay native",
                "experimental raw GGUF ANE input uses an additional bounded six-matrix source window; CPU-direct GPU masters remain MLX affine, not a complete raw-resident model",
                "Q4/Q8 runtime quality and speed require separate paired whole-model qualification"
            };
            return d;
        }};
}

} // namespace tc
