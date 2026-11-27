#include "../runtime/session.hpp"
#include "relayvsr/pipeline.hpp"

namespace tc {
#ifdef TURBOCIDER_HAS_RELAYVSR
std::unique_ptr<ModelSession> create_relayvsr(const std::filesystem::path &);
#endif

ModelModule relayvsr_module() {
#ifdef TURBOCIDER_HAS_RELAYVSR
    constexpr bool enabled = true;
    auto create = create_relayvsr;
#else
    constexpr bool enabled = false;
    auto create = [](const std::filesystem::path &) -> std::unique_ptr<ModelSession> {
        throw std::runtime_error("RelayVSR 2× requires a build with LibTorch 2.14.1");
    };
#endif
    return {
        rv::kModelID,
        [] {
            return Recipe{rv::kModelID, {
                {"media_import", {}}, {"relayvsr_restore", {"media_import"}},
                {"export", {"relayvsr_restore"}}}, enabled};
        },
        [](const Request &r) {
            const bool video = r.operation == "video.upscale";
            require(video || r.operation == "image.upscale", "RelayVSR supports image.upscale and video.upscale");
            require(r.inputs.size() == 1 && r.inputs[0].kind == (video ? "video" : "image") &&
                        r.inputs[0].role == "source" && !r.inputs[0].path.empty() &&
                        r.inputs[0].audio_path.empty(),
                    "RelayVSR requires one source image or video");
            require(r.prompt.empty() && r.steps == 1 && r.noise_path.empty() && r.dump.empty(),
                    "RelayVSR uses embedded conditioning and one restoration step");
            require(video || (!r.audio && r.frames <= 1), "RelayVSR image output is a single PNG");
            require(r.frames >= 0 && r.fps >= 0 && r.fps <= 120,
                    "RelayVSR frame count and FPS must be valid source values");
            require((r.execution == "gpu" || r.execution == "auto") && r.residency == "streamed" &&
                        !r.streaming.specified() && !r.streaming_selector && !r.streaming_offload &&
                        !r.memory_constrained.enabled && !r.memory_budget_bytes &&
                        r.ane_manifest.empty() && r.encoder_ane_manifest.empty() && r.quantized_cache.empty() &&
                        !r.compile_gpu && !r.warmup_iterations,
                    "RelayVSR uses built-in GPU block streaming");
        },
        create,
        [] {
            ModelDescriptor d;
            d.id = rv::kModelID;
            d.name = "RelayVSR 2× INT8";
            d.executable = enabled;
            d.operations = d.executor_operations = {"image.upscale", "video.upscale"};
            d.inputs = {"image", "video"};
            d.roles = {"source"};
            d.max_images = 1;
            d.output = "image+video";
            d.steps = 1;
            d.width = d.height = d.frames = 0;
            d.default_audio = false;
            d.default_residency = "streamed";
            d.backend = "metal_tensorops_mps";
            d.runtime_dependency = "libtorch";
            d.audio_output = true;
            d.audio_capability = "source_audio_passthrough";
            d.candidate_limitations = {
                "fixed 2× output; source sides up to 2048 pixels",
                "model canvas fits a 640×512 landscape or 512×640 portrait envelope",
                "AVFoundation preserves video presentation timestamps and optionally copies source audio"
            };
            if (!enabled) d.candidate_limitations.push_back("Build with LibTorch 2.14.1 to enable RelayVSR");
            return d;
        }
    };
}
} // namespace tc
