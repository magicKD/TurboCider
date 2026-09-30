#include "../runtime/session.hpp"
#include "z_image/z_image.hpp"
#include "../platform/apple/platform.hpp"

namespace tc {

ModelModule z_image_module() {
    return {
        "z-image-turbo",
        [] {
            return Recipe{"z-image-turbo",
                          {{"text_encode", {}},
                           {"denoise", {"text_encode"}, 8},
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
            require(r.hybrid_mlp_mode == "auto" || r.hybrid_mlp_mode == "base_fused" ||
                        r.hybrid_mlp_mode == "lora_merged" || r.hybrid_mlp_mode == "lora_suffix" ||
                        r.hybrid_mlp_mode == "lora_fused" || r.hybrid_mlp_mode == "runtime",
                    "unsupported Z-Image hybrid_mlp_mode");
            if (r.hybrid_mlp_mode == "runtime")
                require(r.execution == "gpu_ane" && r.allow_approximation &&
                            !r.ane_manifest.empty() &&
                            (r.loras.empty() || r.lora_strategy == "inference_time") &&
                            r.residency == "resident" && r.encoder_ane_manifest.empty(),
                        "Z-Image runtime-weight FFN requires explicit resident GPU/ANE, unmerged runtime LoRA and no encoder ANE");
            if (r.hybrid_mlp_mode != "auto" && r.hybrid_mlp_mode != "runtime")
                require(r.execution == "gpu_ane" && r.allow_approximation &&
                            !r.ane_manifest.empty() &&
                            (r.hybrid_mlp_mode == "base_fused" ? r.loras.empty() :
                             (r.hybrid_mlp_mode == "lora_merged" ?
                                (!r.loras.empty() && r.lora_strategy == "in_memory_merge") :
                                (((r.hybrid_mlp_mode == "lora_fused" && r.loras.empty()) ||
                                  (!r.loras.empty() && r.lora_strategy == "inference_time")) &&
                                 r.residency == "resident" && r.width == 512 &&
                                 r.height == 512 && r.steps == 8))),
                        "Z-Image explicit hybrid needs a matching base, merged-adapter, or resident 512px eight-step runtime LoRA request");
            require(r.model_variant == "auto" || r.model_variant == "z-image-turbo",
                    "model_variant does not match Z-Image-Turbo");
            if (r.execution == "gpu_ane") {
                require(r.allow_approximation,
                        "Z-Image GPU+ANE requires allow_approximation=true");
                if (!r.loras.empty() && r.hybrid_mlp_mode != "lora_suffix" &&
                    r.hybrid_mlp_mode != "lora_fused" && r.hybrid_mlp_mode != "runtime")
                    require(r.lora_strategy == "in_memory_merge",
                            "Z-Image GPU+ANE LoRA requires lora_strategy=in_memory_merge");
            }
            require(r.residency == "resident" || r.residency == "streamed",
                    "Z-Image supports resident or streamed residency");
            require(!r.streaming_offload, "use residency=streamed for native Z-Image streaming");
            if (r.residency == "streamed") {
                require(r.execution == "gpu" || r.execution == "gpu_ane",
                        "Z-Image streaming requires explicit GPU or GPU+ANE execution");
                require(r.execution != "gpu_ane" || device_info().optimizations().z_image_suffix_streaming,
                        "Z-Image GPU+ANE streaming is only enabled for the measured M5 Pro 24 GiB device profile");
                require(r.loras.empty(), "Z-Image streaming with LoRA is not yet supported");
                require(r.memory_budget_bytes == 0 || r.memory_budget_bytes >= (6ull << 30),
                        "Z-Image streaming budget must be at least 6 GiB");
            }
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
            d.steps = 8;
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
            d.supports_encoder_gpu_ane = true;
            d.backend = "mlx_cpp_metal";
            d.runtime_dependency = "bundled-native-mlx-cpp";
            d.parallel_strategy = "GPU computes attention first, then the compiled MLP suffix overlaps the Core ML ANE gated-MLP prefix; base 4096-channel M4 Max route is automatic";
            d.candidate_limitations = {
                "text-to-image only",
                "streamed residency is experimental: Comfy BF16, explicit GPU, no LoRA; INT8 ConvRot streaming, multi-layer prefetch and GPU+ANE compact suffix streaming require Apple M5 Pro with exactly 24 GiB",
                "inference_time LoRA defaults to GPU; the explicit 4096-channel frozen-base GPU/Core ML graph has a measured 512px eight-step speedup for one adapter, not a general quality or hardware guarantee",
                "automatic GPU+ANE is limited to the base model on Apple M4 Max 64 GB with the measured 4096-channel 32-block manifest",
                "the repeated warm 1024x1024 base workload measured about 1.21x end-to-end versus the optimized GPU path",
                "merged LoRA GPU+ANE requires an artifact bound to the exact adapter; lora_suffix omits ANE-prefix LoRA; explicit lora_fused reuses one base-only graph with zero deltas for base and runtime pre-SiLU/down-LoRA corrections for adapters (experimental)"
            };
            return d;
        },
        [](const Request &r) {
            const bool gpu = r.execution == "gpu" && r.ane_manifest.empty();
            const bool hybrid = r.execution == "gpu_ane" && !r.ane_manifest.empty() &&
                                device_info().optimizations().z_image_suffix_streaming;
            require(r.encoder_ane_manifest.empty() && (gpu || hybrid),
                    "streaming_route_unsupported: Z-Image manual streaming requires GPU "
                    "or measured-device GPU+ANE without encoder ANE");
        }
    };
}

} // namespace tc
