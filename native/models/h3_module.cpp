#include "../runtime/session.hpp"
namespace tc {
std::unique_ptr<ModelSession> create_h3(const std::filesystem::path &);
ModelModule h3_module() {
    return {"minimax-h3-turbo",
            [] {
                return Recipe{"minimax-h3-turbo",
                              {{"text_encode", {}},
                               {"reference_encode", {}},
                               {"conditioning", {"text_encode", "reference_encode"}},
                               {"av_denoise", {"conditioning"}, 4},
                               {"video_decode", {"av_denoise"}},
                               {"audio_decode", {"av_denoise"}},
                               {"mux", {"video_decode", "audio_decode"}}},
                              false};
            },
            [](const Request &r) {
                require(r.width % 32 == 0 && r.height % 32 == 0 &&
                            int64_t(r.width) * r.height <= 768 * 1344,
                        "H3 canvas must be multiples of 32 within 768*1344 pixels");
                require(r.frames >= 22 && r.frames <= 362 && (r.frames - 5) % 17 == 0,
                        "H3 frames must be 5+17n, between 22 and 362");
                require(r.fps == 24, "H3 requires 24 fps");
                require(r.residency == "resident" || r.residency == "component_staged" ||
                            r.residency == "streamed",
                        "unsupported H3 residency");
                require(r.steps == 4, "H3 Turbo requires four steps");
                require(r.operation == "video.generate" || r.operation == "video.keyframes" ||
                            r.operation == "video.reference",
                        "unsupported H3 operation");
                bool first = false, last = false, refs = false;
                for (auto &input : r.inputs) {
                    require(input.kind == "image" || input.kind == "video" || input.kind == "audio",
                            "unsupported H3 input");
                    if (input.role == "first_frame") {
                        require(input.kind == "image" && !first,
                                "duplicate or invalid first frame");
                        first = true;
                    } else if (input.role == "last_frame") {
                        require(input.kind == "image" && !last, "duplicate or invalid last frame");
                        last = true;
                    } else {
                        require(input.role == "reference", "unsupported H3 input role");
                        refs = true;
                    }
                }
                require(!(refs && (first || last)),
                        "H3 first/last frames are incompatible with reference mode");
                if (r.operation == "video.generate")
                    require(r.inputs.empty(), "video.generate requires no media inputs");
                if (r.operation == "video.keyframes")
                    require(first || last, "video.keyframes requires first/last frame");
                if (r.operation == "video.reference")
                    require(refs, "video.reference requires references");
            },
            {},
            [] {
                return ModelDescriptor{"minimax-h3-turbo",
                                       "MiniMax H3 Turbo",
                                       false,
                                       {"video.generate", "video.keyframes", "video.reference"},
                                       {"text", "image", "video", "audio"},
                                       {"first_frame", "last_frame", "reference"},
                                       0,
                                       "video",
                                       4,
                                       22,
                                       512,
                                       512,
                                       true};
            }};
}
} // namespace tc
