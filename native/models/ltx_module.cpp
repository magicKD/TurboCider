#include "../runtime/session.hpp"
#include <cmath>
namespace tc {
std::unique_ptr<ModelSession> create_ltx_native_candidate(const std::filesystem::path &);
ModelModule ltx_module() {
    return {"ltx-2.5-distilled",
            [] {
                return Recipe{"ltx-2.5-distilled",
                              {{"text_encode", {}},
                               {"connector", {"text_encode"}},
                               {"av_stage1", {"connector"}, 8},
                               {"latent_upsample", {"av_stage1"}},
                               {"av_stage2", {"latent_upsample"}, 3},
                               {"video_vae", {"av_stage2"}},
                               {"audio_vae_vocoder", {"av_stage2"}},
                               {"mux", {"video_vae", "audio_vae_vocoder"}}},
                              true};
            },
            [](const Request &r) {
                require(r.width % 64 == 0 && r.height % 64 == 0,
                        "LTX two-stage dimensions must be multiples of 64");
                require(r.steps == 11, "LTX requires the 8+3 schedule");
                require(r.frames >= 9 && r.frames % 8 == 1, "LTX requires frames=8n+1");
                require(r.operation == "video.generate" || r.operation == "video.image",
                        "unsupported LTX operation");
                if (r.operation == "video.generate")
                    require(r.inputs.empty(), "video.generate does not accept media inputs");
                else
                    require(r.inputs.size() == 1 && r.inputs[0].kind == "image" &&
                                r.inputs[0].role == "first_frame",
                            "LTX image-to-video requires one first_frame");
                require(r.fps == 24, "LTX distilled contract requires 24 fps");
                require(r.residency == "resident" ||
                            r.residency == "component_staged" ||
                            r.residency == "streamed",
                        "unsupported LTX residency");
                require(r.ltx_backend == "auto" ||
                            r.ltx_backend == "c_metal" ||
                            r.ltx_backend == "cpp_mlx",
                        "LTX backend must be auto, c_metal, or cpp_mlx");
                if (r.ltx_backend == "cpp_mlx")
                    require(r.operation == "video.generate",
                            "LTX C++/MLX currently supports text-to-video only");
                if (r.ltx_video_attention_batch) {
                    require(r.ltx_backend != "cpp_mlx",
                            "LTX Video attention batching requires C/Metal");
                    require(r.execution != "gpu_ane",
                            "LTX Video attention batching is GPU-only");
                    require(!r.ltx_sol_stage1 && !r.ltx_sol_stage2,
                            "LTX Video attention batching cannot be combined with Sol");
                }
                const bool ltx_approximation =
                    r.ltx_sol_stage1 || r.ltx_sol_stage2 ||
                    r.ltx_stage2_text_rows != 0;
                if (ltx_approximation) {
                    require(r.allow_approximation,
                            "LTX Sol/text pruning requires allow_approximation=true");
                    require(r.ltx_backend != "cpp_mlx",
                            "LTX Sol/text pruning currently requires the C/Metal backend");
                }
                require(std::isfinite(r.ltx_sol_tau) &&
                            r.ltx_sol_tau >= -2.0 && r.ltx_sol_tau <= 3.0,
                        "LTX Sol tau must be in [-2, 3]");
                require(r.ltx_sol_dense_edge_blocks >= 0 &&
                            r.ltx_sol_dense_edge_blocks <= 24,
                        "LTX Sol dense edge blocks must be 0...24");
                require(r.ltx_sol_dense_edge_steps >= 0 &&
                            r.ltx_sol_dense_edge_steps <= 16,
                        "LTX Sol dense edge steps must be 0...16");
                require(r.ltx_stage2_text_rows >= 0 &&
                            r.ltx_stage2_text_rows <= 4096,
                        "LTX Stage-2 text rows must be 0...4096");
                if (r.ltx_stage2_text_rows)
                    require(r.ltx_stage2_text_rows >= 128,
                            "LTX Stage-2 text rows must retain at least one register group");
                const uint64_t latent_frames =
                    static_cast<uint64_t>((r.frames - 1) / 8 + 1);
                const uint64_t stage1_rows = latent_frames *
                    static_cast<uint64_t>(r.width / 64) *
                    static_cast<uint64_t>(r.height / 64);
                const uint64_t stage2_rows = latent_frames *
                    static_cast<uint64_t>(r.width / 32) *
                    static_cast<uint64_t>(r.height / 32);
                if (r.ltx_sol_stage1)
                    require(stage1_rows <= 16384,
                            "LTX Stage-1 Sol supports at most 16384 video rows");
                if (r.ltx_sol_stage2)
                    require(stage2_rows <= 16384,
                            "LTX Stage-2 Sol supports at most 16384 video rows");
                if (r.residency == "streamed") {
                    require(r.operation == "video.generate",
                            "LTX block streaming currently supports text-to-video only");
                    require(!r.audio,
                            "LTX block streaming currently supports video-only output");
                    require(r.execution != "gpu_ane",
                            "LTX block streaming currently requires GPU execution");
                    require(!r.memory_budget_bytes ||
                                r.memory_budget_bytes >= (8ull << 30),
                            "LTX streamed memory budget must be at least 8 GiB");
                }
                require(r.loras.size() <= 1, "LTX supports one transformer LoRA adapter");
                for (const auto &lora : r.loras) {
                    require(lora.role == "transformer" || lora.role == "refiner",
                            "LTX LoRA role must be transformer or refiner");
                    require(lora.strength > 0.0f && lora.strength <= 4.0f,
                            "LTX LoRA strength must be in (0, 4]");
                }
            },
            create_ltx_native_candidate,
            [] {
                ModelDescriptor d;
                d.id = "ltx-2.5-distilled"; d.name = "LTX 2.5 Distilled";
                d.executable = true;
                d.operations = {"video.generate", "video.image"};
                d.executor_operations = {"video.generate", "video.image"};
                d.inputs = {"text", "image"}; d.roles = {"first_frame"}; d.max_images = 1;
                d.output = "video"; d.steps = 11; d.frames = 97; d.width = 704; d.height = 448; d.fps = 24;
                d.default_audio = false; d.default_residency = "component_staged";
                d.supports_lora = true; d.runtime_lora = false; d.lora_mode = "premerged-manifest";
                d.lora_strategies = {"disk_premerge"};
                d.default_lora_strategy = "disk_premerge";
                d.request_lora_identity_validation = true; d.supports_gpu_ane = true;
                d.backend = "metal+mlx_cpp";
                d.runtime_dependency = "bundled-native-ltx-runtime";
                d.audio_capability = "latent_to_48khz_aac_candidate";
                d.parallel_strategy = "dual Metal queues for AV plus ANE MLP/KV/QKV partitions";
                d.candidate_limitations = {
                    "broader prompt-suite qualification remains pending",
                    "native Audio VAE/base-vocoder/BWE and AVFoundation AAC mux require the pinned ModelScope provenance manifest; the 5-second multi-prompt quality suite remains pending",
                    "LoRA requires an offline-premerged checkpoint and verified sidecar manifest",
                    "streamed residency uses a budget-selected resident prefix and up to three reusable look-ahead refill slots on GPU; budgets that fit all 48 blocks automatically avoid refill I/O; the budget targets the denoiser working set, not whole-process peak RSS, and hybrid streaming remains gated"
                };
                d.native_gemma4_candidate = true; d.native_conditioning_connector = true;
                d.native_i2v_clean_prefix = true; d.native_gpu_ane_profile = true;
                d.native_audio_output_candidate = true; d.native_audio_vae_candidate = true;
                d.native_base_vocoder_candidate = true; d.audio_output = true;
                return d;
            }};
}
} // namespace tc
