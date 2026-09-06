#include "../runtime/session.hpp"
namespace tc {
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
                              false};
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
            },
            {},
            [] {
                return ModelDescriptor{"ltx-2.5-distilled",
                                       "LTX 2.5 Distilled",
                                       false,
                                       {"video.generate", "video.image"},
                                       {"text", "image"},
                                       {"first_frame"},
                                       1,
                                       "video",
                                       11,
                                       97,
                                       704,
                                       448,
                                       false};
            }};
}
} // namespace tc
