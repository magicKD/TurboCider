#include "../runtime/session.hpp"
#include "../platform/apple/platform.hpp"
#include "wan22/identifiers.hpp"
#include "wan22/geometry.hpp"

#include <cmath>

namespace tc {
std::unique_ptr<ModelSession> create_wan22(const std::filesystem::path &, bool quanhaol);

static ModelModule make_wan22_module(bool quanhaol) {
    const std::string id(quanhaol ? wan22::quanhaol_id : wan22::fastwan_id);
    const int steps = quanhaol ? 4 : 3;
    return {id,
        [id, steps] {
            return Recipe{id,
                {{"text_encode", {}}, {"first_frame_vae_encode", {}},
                 {"denoise", {"text_encode", "first_frame_vae_encode"}, steps},
                 {"video_vae", {"denoise"}}, {"export", {"video_vae"}}}, true};
        },
        [steps](const Request &r) {
            require(r.operation == "video.image" && r.inputs.size() == 1 && r.inputs[0].kind == "image" &&
                        r.inputs[0].role == "first_frame" && r.inputs[0].strength == 1.f,
                    "Wan2.2 requires one first-frame image with strength 1");
            require(wan22::canvas_geometry(r.width, r.height, r.frames).has_value() &&
                        r.fps == 24 && r.steps == steps,
                    "Wan2.2 requires positive even output dimensions within integer geometry bounds, 5...97 frames (4n+1), 24 fps and " + std::to_string(steps) + " fixed steps");
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
        [quanhaol](const std::filesystem::path &root) { return create_wan22(root, quanhaol); },
        [id, steps, quanhaol] {
            ModelDescriptor d;
            d.id = id; d.name = quanhaol ? "Wan 2.2 TI2V 5B Turbo 4-step (experimental)" :
                                         "Wan 2.2 TI2V 5B FastWan (experimental)";
            d.executable = true; d.operations = {"video.image"}; d.executor_operations = d.operations;
            d.inputs = {"text", "image"}; d.roles = {"first_frame"}; d.max_images = 1;
            d.output = "video"; d.steps = steps; d.frames = 97; d.width = 512; d.height = 512; d.fps = 24;
            d.default_audio = false; d.default_residency = "resident";
            d.backend = "mlx_cpp_metal"; d.runtime_dependency = "native";
            d.parallel_strategy = "staged BF16 GPU transformer and causal video VAE";
            d.candidate_limitations = {
                quanhaol ? "experimental quanhaol four-step checkpoint; Kijai FP16 conversion; verify model license" :
                            "experimental FastVideo full-attention three-step DMD checkpoint; not the four-step quanhaol model",
                "one first-frame image locked at strength 1; silent video, 24 fps, at most 97 frames",
                "even output dimensions are padded to a multiple-of-32 model canvas and cropped without resizing; large sizes may exhaust memory",
                "broader visual quality and hardware qualification remain pending",
                "model phases reload per request; ANE, LoRA and streaming are not supported"
            };
            return d;
        }};
}
ModelModule wan22_module() { return make_wan22_module(false); }
ModelModule wan22_quanhaol_module() { return make_wan22_module(true); }
} // namespace tc
