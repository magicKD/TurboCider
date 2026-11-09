#include "../runtime/session.hpp"
#include "../platform/apple/platform.hpp"

#include <cmath>

namespace tc {
std::unique_ptr<ModelSession> create_wan22(const std::filesystem::path &);

ModelModule wan22_module() {
    return {"wan2.2-ti2v-5b-fastwan",
        [] {
            return Recipe{"wan2.2-ti2v-5b-fastwan",
                {{"text_encode", {}}, {"first_frame_vae_encode", {}},
                 {"denoise", {"text_encode", "first_frame_vae_encode"}, 3},
                 {"video_vae", {"denoise"}}, {"export", {"video_vae"}}}, true};
        },
        [](const Request &r) {
            require(r.operation == "video.image" && r.inputs.size() == 1 && r.inputs[0].kind == "image" &&
                        r.inputs[0].role == "first_frame" && r.inputs[0].strength == 1.f,
                    "Wan2.2 requires one first-frame image with strength 1");
            require(r.width >= 128 && r.height >= 128 && r.width <= 768 && r.height <= 768 &&
                        r.width % 32 == 0 && r.height % 32 == 0 && r.frames >= 5 && r.frames <= 97 &&
                        r.frames % 4 == 1 && r.fps == 24 && r.steps == 3,
                    "Wan2.2 experimental GPU route requires 128...768 dimensions divisible by 32, 5...97 frames (4n+1), 24 fps and three steps");
            if (r.width > 512 || r.height > 512)
                require(device_info().physical_memory >= (64ull << 30),
                        "Wan2.2 resolutions above 512 currently require at least 64 GiB unified memory");
            require(r.execution == "gpu" && r.residency == "resident" && !r.audio && !r.compile_gpu &&
                        r.loras.empty() && r.profile.empty() && r.ane_manifest.empty() && r.encoder_ane_manifest.empty() &&
                        !r.streaming.active() && !r.streaming_selector && !r.memory_constrained.enabled &&
                        !r.memory_budget_bytes && !r.streaming_offload && r.noise_path.empty() && r.dump.empty(),
                    "Wan2.2 currently requires resident GPU-only silent video without LoRA, ANE or streaming");
            require(r.ltx_backend == "auto" && !r.ltx_video_attention_batch && !r.ltx_sol_stage1 &&
                        !r.ltx_sol_stage2 && !r.ltx_sparse_mode && !r.ltx_stage2_text_rows,
                    "LTX-specific options cannot be applied to Wan2.2");
            if (r.tiny_vae) {
                require(r.tiny_vae->encoder == "original" && !r.tiny_vae->preview &&
                            (r.tiny_vae->decoder == "original" || r.tiny_vae->decoder == "tiny"),
                        "Wan2.2 Tiny VAE supports final decoding only");
                require(r.tiny_vae->decoder != "tiny" || r.allow_approximation,
                        "Wan2.2 Super decoding requires allow_approximation=true");
            }
        },
        create_wan22,
        [] {
            ModelDescriptor d;
            d.id = "wan2.2-ti2v-5b-fastwan"; d.name = "Wan 2.2 TI2V 5B FastWan (experimental)";
            d.executable = true; d.operations = {"video.image"}; d.executor_operations = d.operations;
            d.inputs = {"text", "image"}; d.roles = {"first_frame"}; d.max_images = 1;
            d.output = "video"; d.steps = 3; d.frames = 97; d.width = 512; d.height = 512; d.fps = 24;
            d.default_audio = false; d.default_residency = "resident";
            d.backend = "mlx_cpp_metal"; d.runtime_dependency = "native";
            d.parallel_strategy = "staged BF16 GPU transformer and causal video VAE";
            d.candidate_limitations = {
                "experimental FastVideo full-attention three-step DMD checkpoint; not the four-step quanhaol model",
                "one first-frame image locked at strength 1; silent video, 24 fps, at most 97 frames",
                "up to 768x768 is experimental; resolutions above 512 require at least 64 GiB unified memory",
                "broader visual quality and hardware qualification remain pending",
                "model phases reload per request; ANE, LoRA and streaming are not supported"
            };
            return d;
        }};
}
} // namespace tc
