#include "../runtime/session.hpp"

namespace tc {
std::unique_ptr<ModelSession> create_fastmetal(const std::filesystem::path &);

ModelModule fastmetal_module() {
    return {
        "fastmetal-1.3b-qad",
        [] {
            return Recipe{"fastmetal-1.3b-qad",
                          {{"text_encode", {}},
                           {"dit", {"text_encode"}, 3},
                           {"video_vae", {"dit"}},
                           {"export", {"video_vae"}}},
                          true};
        },
        [](const Request &r) {
            require(r.operation == "video.generate", "FastMetal supports video.generate only");
            require(r.inputs.empty(), "FastMetal does not accept media inputs");
            require(r.width == 832 && r.height == 480, "FastMetal requires 832x480 output");
            require(r.frames >= 5 && r.frames % 4 == 1, "FastMetal requires frames=4n+1");
            require(r.fps == 16, "FastMetal requires 16 fps");
            require(r.steps == 3, "FastMetal QAD requires three steps");
            require(r.residency == "resident", "FastMetal requires resident execution");
            require(r.loras.size() <= 1, "FastMetal supports one transformer LoRA");
            if (!r.loras.empty()) {
                require(r.loras.front().role == "transformer",
                        "FastMetal LoRA role must be transformer");
                require(r.loras.front().strength > 0.0f && r.loras.front().strength <= 4.0f,
                        "FastMetal LoRA strength must be in (0, 4]");
                require(r.execution == "gpu",
                        "FastMetal premerged LoRA currently supports GPU execution only");
            }
            if (r.execution == "gpu_ane")
                require(!r.ane_manifest.empty(),
                        "FastMetal gpu_ane requires an explicit ANE manifest");
        },
        create_fastmetal,
        [] {
            ModelDescriptor d;
            d.id = "fastmetal-1.3b-qad"; d.name = "FastMetal 1.3B QAD";
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
            d.backend = "mlx+ane-bridge";
            d.runtime_dependency = "explicit-python-fastvideo-worker";
            d.parallel_strategy = "MLX GPU residual FFN + Core ML ANE FFN split";
            d.candidate_limitations = {
                "broader hardware performance medians remain pending",
                "LoRA requires a separately merged and provenance-verified checkpoint",
                "premerged LoRA currently runs on GPU only"
            };
            return d;
        }
    };
}
} // namespace tc
