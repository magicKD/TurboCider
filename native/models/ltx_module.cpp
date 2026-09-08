#include "../runtime/session.hpp"
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
                require(r.residency == "resident" || r.residency == "component_staged",
                        "LTX block streaming is not yet supported");
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
                d.executor_operations = {"video.generate"};
                d.inputs = {"text", "image"}; d.roles = {"first_frame"}; d.max_images = 1;
                d.output = "video"; d.steps = 11; d.frames = 97; d.width = 704; d.height = 448; d.fps = 24;
                d.default_audio = false; d.default_residency = "component_staged";
                d.supports_lora = true; d.runtime_lora = true; d.lora_mode = "runtime-bake-cache";
                d.lora_strategies = {"disk_premerge"};
                d.default_lora_strategy = "disk_premerge";
                d.request_lora_identity_validation = true; d.supports_gpu_ane = true;
                d.backend = "metal+mlx_cpp";
                d.runtime_dependency = "bundled-native-ltx-runtime";
                d.audio_capability = "latent_to_48khz_aac_candidate";
                d.parallel_strategy = "dual Metal queues for AV plus ANE MLP/KV/QKV partitions";
                d.candidate_limitations = {
                    "broader prompt-suite qualification remains pending",
                    "native Audio VAE/base-vocoder/BWE and AVFoundation AAC mux are validated on local fixtures; end-to-end native Session parity still pending",
                    "LoRA is identity-bound through a runtime cache or verified sidecar manifest"
                };
                d.native_gemma4_candidate = true; d.native_conditioning_connector = true;
                d.native_i2v_clean_prefix = true; d.native_gpu_ane_profile = true;
                d.native_audio_output_candidate = true; d.native_audio_vae_candidate = true;
                d.native_base_vocoder_candidate = true; d.audio_output = false;
                return d;
            }};
}
} // namespace tc
