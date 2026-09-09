#include "../runtime/session.hpp"

namespace tc {
std::unique_ptr<ModelSession> create_wan(const std::filesystem::path &);

ModelModule wan_module() {
    return {
        "wan2.1-1.3b-qad",
        [] {
            return Recipe{"wan2.1-1.3b-qad",
                          {{"text_encode", {}},
                           {"dit", {"text_encode"}, 3},
                           {"video_vae", {"dit"}},
                           {"export", {"video_vae"}}},
                          true};
        },
        [](const Request &r) {
            require(r.operation == "video.generate", "Wan supports video.generate only");
            require(r.inputs.empty(), "Wan does not accept media inputs");
            require(r.width == 832 && r.height == 480, "Wan requires 832x480 output");
            require(r.frames >= 5 && r.frames <= 81 && r.frames % 4 == 1, "Wan requires frames=4n+1 in 5...81");
            require(r.fps == 16, "Wan requires 16 fps");
            require(r.steps == 3, "Wan QAD requires three steps");
            require(r.residency == "resident", "Wan requires resident execution");
            require(r.loras.size() <= 1, "Wan supports one transformer LoRA");
            if (!r.loras.empty()) {
                require(r.loras.front().role == "transformer",
                        "Wan LoRA role must be transformer");
                require(r.loras.front().strength > 0.0f && r.loras.front().strength <= 4.0f,
                        "Wan LoRA strength must be in (0, 4]");
                require(r.execution == "gpu",
                        "Wan premerged LoRA currently supports GPU execution only");
            }
            require(!r.audio, "Wan native QAD emits silent video only");
            require(r.noise_path.empty() && r.dump.empty() && !r.streaming_offload,
                    "Wan does not support noise overrides, tensor dumps or streaming offload");
            if (r.execution == "gpu_ane") {
                require(!r.ane_manifest.empty(), "Wan gpu_ane requires an explicit ANE manifest");
                require(r.frames == 81, "Wan hybrid requires exactly 81 frames");
                require(!r.compile_gpu, "Wan hybrid and whole-DiT compilation are mutually exclusive");
            }
        },
        create_wan,
        [] {
            ModelDescriptor d;
            d.id = "wan2.1-1.3b-qad"; d.name = "Wan 2.1 1.3B QAD";
            d.executable = true; d.operations = {"video.generate"};
            d.executor_operations = d.operations; d.inputs = {"text"};
            d.output = "video"; d.steps = 3; d.frames = 81;
            d.width = 832; d.height = 480; d.fps = 16;
            d.default_audio = false; d.default_residency = "resident";
            d.supports_lora = true; d.runtime_lora = false;
            d.lora_mode = "premerged-manifest";
            d.lora_strategies = {"disk_premerge"};
            d.default_lora_strategy = "disk_premerge";
            d.request_lora_identity_validation = true; d.supports_gpu_ane = true;
            d.backend = "mlx_cpp_metal+coreml";
            d.runtime_dependency = "native";
            d.parallel_strategy = "MLX GPU residual FFN + Core ML ANE FFN split";
            d.candidate_limitations = {
                "full-video quality parity and hardware performance medians remain pending",
                "requires offline-converted vae/taew2_1.safetensors in the model package",
                "LoRA requires a separately merged and provenance-verified checkpoint",
                "premerged LoRA currently runs on GPU only"
            };
            return d;
        }
    };
}
} // namespace tc
