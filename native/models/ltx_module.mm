#include "runtime.hpp"
namespace tc {

std::unique_ptr<ModelSession> create_ltx_native_candidate(
    const std::filesystem::path& root);

ModelModule ltx_module() {
    return {
        "ltx-2.5-distilled",
        [] {
            return Recipe{
                "ltx-2.5-distilled",
                {
                    {"text_encode", {}},
                    {"connector", {"text_encode"}},
                    {"av_stage1", {"connector"}, 8},
                    {"latent_upsample", {"av_stage1"}},
                    {"av_stage2", {"latent_upsample"}, 3},
                    {"video_vae", {"av_stage2"}},
                    {"audio_vae_vocoder", {"av_stage2"}},
                    {"mux", {"video_vae", "audio_vae_vocoder"}},
                },
                true,
            };
        },
        [](const Request& request) {
            require(request.width % 64 == 0 && request.height % 64 == 0,
                    "LTX two-stage dimensions must be multiples of 64");
            require(request.steps == 11, "LTX requires the 8+3 schedule");
            require(request.frames >= 9 && request.frames % 8 == 1,
                    "LTX requires frames=8n+1");
            require(request.operation == "video.generate" ||
                        request.operation == "video.image",
                    "unsupported LTX operation");
            if (request.operation == "video.generate") {
                require(request.inputs.empty(),
                        "video.generate does not accept media inputs");
            } else {
                require(request.inputs.size() == 1 &&
                            request.inputs[0].kind == "image" &&
                            request.inputs[0].role == "first_frame",
                        "LTX image-to-video requires one first_frame");
            }
            require(request.fps == 24,
                    "LTX distilled contract requires 24 fps");
            require(request.residency == "resident" ||
                        request.residency == "component_staged",
                    "LTX block streaming is not yet supported");
            require(request.loras.size() <= 1,
                    "LTX supports one transformer LoRA adapter");
            for (const auto& lora : request.loras) {
                require(lora.role == "transformer" || lora.role == "refiner",
                        "LTX LoRA role must be transformer or refiner");
                require(lora.strength > 0.0f && lora.strength <= 4.0f,
                        "LTX LoRA strength must be in (0, 4]");
            }
        },
        create_ltx_native_candidate,
        [] {
            return @{
                @"id": @"ltx-2.5-distilled",
                @"name": @"LTX 2.5 Distilled",
                /* Video-only text-to-video is a public native executor.  The
                 * audio and I2V branches remain capability-gated below. */
                @"executor": @YES,
                @"executor_operations": @[@"video.generate"],
                @"native_video_candidate": @YES,
                @"native_gemma4_candidate": @YES,
                @"native_conditioning_connector": @YES,
                @"native_i2v_clean_prefix": @YES,
                @"native_gpu_ane_profile": @YES,
                @"native_audio_vae_candidate": @YES,
                @"native_base_vocoder_candidate": @YES,
                @"native_bwe_candidate": @YES,
                @"native_audio_output_candidate": @YES,
                @"audio_output": @NO,
                @"audio_capability": @"latent_to_48khz_aac_candidate",
                @"audio_assets": @[
                    @"ltx-2.5-audio-vae-vocoder-bf16.safetensors",
                    @"ltx-2.5-audio-vae-bf16.safetensors",
                ],
                @"default_audio": @NO,
                @"default_residency": @"component_staged",
                @"candidate_limitations": @[
                    @"native Gemma4 parity is validated for the connected video-only path; broader prompt-suite qualification remains pending",
                    @"native I2V remains an internal candidate until end-to-end numerical parity is complete",
                    @"native Audio VAE/base-vocoder/BWE and AVFoundation AAC mux are validated on local fixtures; end-to-end native Session parity still pending",
                    @"audio assets require a provenance manifest before public readiness",
                    @"LoRA is baked on first use into an identity-bound runtime cache; a sidecar manifest remains accepted",
                    @"audio and I2V remain capability-gated; video-only public executor is enabled",
                ],
                @"operations": @[@"video.generate", @"video.image"],
                @"inputs": @[@"text", @"image"],
                @"roles": @[@"first_frame"],
                @"max_images": @1,
                @"output": @"video",
                @"default_steps": @11,
                @"default_frames": @97,
                @"default_width": @704,
                @"default_height": @448,
                @"supports_lora": @YES,
                @"runtime_lora": @YES,
                @"lora_mode": @"runtime-bake-cache",
                @"lora_cache": @"content-addressed-evictable",
                @"request_lora_identity_validation": @YES,
                @"supports_gpu_ane": @YES,
                @"parallel_strategy":
                    @"dual Metal queues for AV plus ANE MLP/KV/QKV partitions",
            };
        },
    };
}

}  // namespace tc
