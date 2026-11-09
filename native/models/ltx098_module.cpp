#include "../runtime/session.hpp"

#include <cmath>

namespace tc {
std::unique_ptr<ModelSession> create_ltx098(const std::filesystem::path &);

ModelModule ltx098_module() {
    return {"ltx-video-0.9.8-distilled",
        [] {
            return Recipe{"ltx-video-0.9.8-distilled",
                {{"text_encode", {}}, {"first_frame_vae_encode", {}},
                 {"denoise", {"text_encode", "first_frame_vae_encode"}, 8},
                 {"video_vae", {"denoise"}}, {"export", {"video_vae"}}}, true};
        },
        [](const Request &r) {
            require(r.operation == "video.image" && r.inputs.size() == 1 && r.inputs[0].kind == "image" &&
                        r.inputs[0].role == "first_frame", "LTX 0.9.8 requires one first-frame image");
            require(std::isfinite(r.inputs[0].strength) && r.inputs[0].strength >= 0.f && r.inputs[0].strength <= 1.f,
                    "LTX 0.9.8 first-frame strength must be in [0,1]");
            require(r.width >= 64 && r.height >= 64 && r.width <= 1024 && r.height <= 1024 &&
                        r.width % 32 == 0 && r.height % 32 == 0,
                    "LTX 0.9.8 dimensions must be multiples of 32 in 64...1024");
            require(r.frames >= 9 && r.frames <= 121 && r.frames % 8 == 1 && r.fps == 24 && (r.steps == 8 || r.steps == 10),
                    "LTX 0.9.8 requires 9...121 frames (8n+1), 24 fps and either eight single-scale or ten multiscale steps");
            require(!r.ltx098_resize_to_base || r.steps == 10,
                    "LTX 0.9.8 resize-to-base is only available for the ten-step multiscale workflow");
            const int output_width = r.steps == 10 ? 2 * (int(double(r.width) * .6666666) / 32 * 32) : r.width;
            const int output_height = r.steps == 10 ? 2 * (int(double(r.height) * .6666666) / 32 * 32) : r.height;
            const uint64_t max_tokens = r.steps == 10 ? 6400 : 4096;
            require(uint64_t(output_width / 32) * uint64_t(output_height / 32) * uint64_t((r.frames + 7) / 8) <= max_tokens &&
                        uint64_t(r.width) * uint64_t(r.height) * uint64_t(r.frames) <= uint64_t(512) * 512 * 121 &&
                        uint64_t(output_width) * uint64_t(output_height) * uint64_t(r.frames) <= uint64_t(640) * 640 * 121,
                    "LTX 0.9.8 geometry exceeds the current GPU working-set bound");
            require(r.execution == "gpu" && r.residency == "resident" && !r.audio && !r.compile_gpu &&
                        r.loras.empty() && r.profile.empty() && r.ane_manifest.empty() && r.encoder_ane_manifest.empty() &&
                        !r.streaming.active() && !r.streaming_selector && !r.memory_constrained.enabled &&
                        !r.memory_budget_bytes && !r.streaming_offload && r.noise_path.empty() && r.dump.empty(),
                    "LTX 0.9.8 currently requires resident GPU-only video without LoRA, ANE, streaming or noise overrides");
            require(r.ltx_backend == "auto" && !r.ltx_video_attention_batch && !r.ltx_sol_stage1 && !r.ltx_sol_stage2 &&
                        !r.ltx_sparse_mode && !r.ltx_stage2_text_rows,
                    "LTX 2.5-specific accelerators cannot be applied to LTX 0.9.8");
        },
        create_ltx098,
        [] {
            ModelDescriptor d;
            d.id = "ltx-video-0.9.8-distilled"; d.name = "LTX-Video 0.9.8 2B Distilled (experimental)";
            d.executable = true; d.operations = {"video.image"}; d.executor_operations = d.operations;
            d.inputs = {"text", "image"}; d.roles = {"first_frame"}; d.max_images = 1;
            d.output = "video"; d.steps = 10; d.frames = 121; d.width = 512; d.height = 512; d.fps = 24;
            d.default_audio = false; d.default_residency = "resident";
            d.backend = "mlx_cpp_metal"; d.runtime_dependency = "native";
            d.parallel_strategy = "resident BF16 GPU transformer and video VAE";
            d.candidate_limitations = {
                "initial GPU candidate; broader prompt/seed/video quality qualification remains pending",
                "resident GPU only; ANE, LoRA and streaming are not yet supported",
                "at most 121 frames; 8-step single-scale or 7+3-step multiscale with the optional 0.9.8 upscaler",
                "10-step output keeps the upscaled dimensions, which can exceed the requested base canvas"
            };
            return d;
        }};
}
} // namespace tc
