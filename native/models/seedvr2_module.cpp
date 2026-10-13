#include "../runtime/session.hpp"

namespace tc {
std::unique_ptr<ModelSession> create_seedvr2(const std::filesystem::path &);

ModelModule seedvr2_module() {
    return {
        "seedvr2-3b",
        [] { return Recipe{"seedvr2-3b", {
            {"media_import", {}}, {"vae_encode", {"media_import"}},
            {"denoise", {"vae_encode"}}, {"vae_decode", {"denoise"}},
            {"export", {"vae_decode"}}}, true}; },
        [](const Request &r) {
            const bool video = r.operation == "video.upscale";
            require(video || r.operation == "image.upscale", "SeedVR2 supports image.upscale and video.upscale only");
            require(r.inputs.size() == 1 && r.inputs[0].kind == (video ? "video" : "image") &&
                    r.inputs[0].role == "source" && !r.inputs[0].path.empty(),
                    "SeedVR2 requires exactly one source image/video input");
            require(r.prompt.empty(), "SeedVR2 uses embedded conditioning; text prompts are not supported");
            require(r.width % 16 == 0 && r.height % 16 == 0, "SeedVR2 output dimensions must be multiples of sixteen");
            require(r.steps == 1, "SeedVR2 requires exactly one flow/Euler step");
            require(video ? r.frames >= 2 && r.frames <= 121 : r.frames == 1,
                    "SeedVR2 image output is one frame; video output requires the complete source frame count in 2...121");
            require(!video || (r.fps >= 1 && r.fps <= 120), "SeedVR2 requires an integer source FPS in 1...120");
            require(!r.audio, "SeedVR2 currently exports silent video; audio preservation is not yet supported");
            require(r.execution == "gpu" || r.execution == "auto", "SeedVR2 GPU+ANE has not yet been integrated");
            require(r.ane_manifest.empty() && r.encoder_ane_manifest.empty(), "SeedVR2 does not accept unqualified ANE manifests");
            require(r.residency == "resident" && !r.streaming_offload && !r.streaming.active() &&
                    !r.streaming_selector && !r.memory_constrained.enabled && !r.quantized_execution.active(),
                    "SeedVR2 currently requires resident, non-streaming execution");
            require(!r.compile_gpu, "SeedVR2 explicit GPU graph compilation is not yet integrated");
            require(r.loras.empty() && r.quantized_cache.empty() && !r.vsa,
                    "SeedVR2 does not support LoRA, quantized caches or VSA overrides");
            require(r.warmup_iterations == 0, "SeedVR2 request warmup overrides are not yet supported");
        },
        create_seedvr2,
        [] {
            ModelDescriptor d;
            d.id = "seedvr2-3b"; d.name = "SeedVR2 3B — Image / Video Super-resolution";
            d.executable = true; d.operations = {"image.upscale", "video.upscale"};
            d.executor_operations = d.operations; d.inputs = {"image", "video"};
            d.roles = {"source"}; d.max_images = 1; d.output = "image+video";
            d.steps = 1; d.frames = 1; d.width = 1024; d.height = 1024; d.fps = 24;
            d.default_audio = false; d.default_residency = "resident";
            d.backend = "mlx_cpp_metal"; d.runtime_dependency = "native";
            d.candidate_limitations = {
                "stable FP16/FP32 profile passes original-parameter FP32 component quality; full-request speed and hybrid qualification are pending",
                "original FP16 checkpoint coefficients and projections; FP32 residuals/normalization/time/velocity and FP16 causal VAE",
                "raw restoration without optional color transfer; source image alpha is reapplied",
                "silent CFR videos, complete declared frame count and matching integer source FPS",
                "GPU+ANE and explicit GPU graph compilation remain unavailable"
            };
            return d;
        }
    };
}
} // namespace tc
