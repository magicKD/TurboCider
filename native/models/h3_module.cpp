#include "../runtime/session.hpp"
namespace tc {
std::unique_ptr<ModelSession> create_h3(const std::filesystem::path &);
namespace {
constexpr uint64_t h3_streaming_minimum_bytes =
    (4ull << 30) + 2ull * 770725376ull;
}
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
                              true};
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
                require(!r.memory_budget_bytes || r.residency == "streamed",
                        "H3 memory budget requires streamed residency");
                require(!r.memory_budget_bytes ||
                            r.memory_budget_bytes >= h3_streaming_minimum_bytes,
                        "H3 streamed memory budget is below the activation and two-slot minimum");
                require(r.steps == 4, "H3 Turbo requires four steps");
                require(r.loras.size() <= 1, "H3 supports one Turbo LoRA adapter");
                if (!r.loras.empty()) {
                    require(r.loras.front().role == "transformer", "H3 LoRA role must be transformer");
                    require(r.loras.front().strength > 0.0f && r.loras.front().strength <= 4.0f,
                            "H3 LoRA strength must be in (0, 4]");
                }
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
            create_h3,
            [] {
                ModelDescriptor d;
                d.id = "minimax-h3-turbo"; d.name = "MiniMax H3 Turbo";
                d.executable = true;
                d.operations = {"video.generate", "video.keyframes", "video.reference"};
                d.inputs = {"text", "image", "video", "audio"};
                d.roles = {"first_frame", "last_frame", "reference"};
                d.output = "video"; d.steps = 4; d.frames = 22; d.width = 512; d.height = 512;
                d.fps = 24; d.supports_lora = true; d.runtime_lora = true;
                d.default_audio = true; d.default_residency = "resident";
                d.lora_mode = "runtime-bake-cache"; d.supports_gpu_ane = true;
                d.lora_strategies = {"disk_premerge"};
                d.default_lora_strategy = "disk_premerge";
                d.request_lora_identity_validation = true;
                d.backend = "metal_mps_native";
                d.parallel_strategy = "GPU denoise + ANE MLP/QKV with explicit join";
                d.executor_operations = d.operations;
                d.weight_validation_pending = false;
                return d;
            }};
}
} // namespace tc
