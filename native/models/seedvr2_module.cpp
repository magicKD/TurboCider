#include "../runtime/session.hpp"
#include "seedvr2/hybrid.hpp"

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
            if (r.seedvr2_vae && r.seedvr2_vae->encoder == "streaming") {
                require(video,"SeedVR2 streaming VAE encoder requires video.upscale");
                require(r.dump.empty(),"SeedVR2 streaming encoder cannot retain a complete input tensor dump; select buffered for tensor dumps");
            }
            if (r.seedvr2_vae && r.seedvr2_vae->decoder == "streaming") {
                require(video, "SeedVR2 streaming VAE decoder requires video.upscale");
                require(r.dump.empty(), "SeedVR2 streaming decoder cannot materialize a full pixel dump; select buffered for tensor dumps");
            }
            require(r.encoder_ane_manifest.empty(), "SeedVR2 has no separate encoder ANE route");
            if (r.execution == "gpu_ane") {
                require(r.allow_approximation && !r.ane_manifest.empty(), "SeedVR2 manual GPU+ANE requires approximation consent and a qualified manifest");
                seedvr2::validate_public_hybrid_manifest(r.ane_manifest);
            } else {
                require((r.execution == "gpu" || r.execution == "auto") && r.ane_manifest.empty(),
                        "SeedVR2 GPU/auto must not silently consume a hybrid manifest");
            }
            require((r.residency == "resident" || r.residency == "component_staged") && !r.streaming_offload && !r.streaming.active() &&
                    !r.streaming_selector && !r.memory_constrained.enabled && !r.quantized_execution.active(),
                    "SeedVR2 requires resident or component_staged, non-streaming execution");
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
            d.supports_gpu_ane = true;
            d.parallel_strategy = "GPU attention/text FFN/VAE and video-FFN suffix; Core ML CPU+NE original-FP16 prefix";
            d.candidate_limitations = {
                "stable GPU profile and exact original-FP16 Core ML prefix bank pass original-parameter FP32 image/t25-video component quality; complete-request hybrid performance is hardware/workload-specific",
                "optional component_staged releases DiT during VAE phases and reloads it for sampling; retains VAE and includes reload costs in timings",
                "original FP16 checkpoint coefficients and projections; FP32 residuals/normalization/time/velocity and FP16 causal VAE",
                "raw restoration without optional color transfer; source image alpha is reapplied",
                "silent CFR videos, complete declared frame count and matching integer source FPS",
                "manual GPU+ANE accepts only the qualified all-32 original-FP16 bank; auto remains GPU-only; CPU+NE configuration is not physical occupancy evidence",
                "explicit GPU graph compilation remains unavailable"
            };
            return d;
        }
    };
}
} // namespace tc
