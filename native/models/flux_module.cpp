#include "../runtime/session.hpp"
#include "flux2/flux.hpp"
namespace tc {
ModelModule flux_module() {
    return {"flux2-klein-4b",
            [] {
                return Recipe{"flux2-klein-4b",
                              {{"text_encode", {}},
                               {"denoise", {"text_encode"}, 4},
                               {"vae_decode", {"denoise"}},
                               {"export", {"vae_decode"}}},
                              true};
            },
            [](const Request &r) {
                require(r.width % 16 == 0 && r.height % 16 == 0,
                        "FLUX dimensions must be multiples of 16");
                require(r.frames == 1, "FLUX requires frames=1");
                require(r.operation == "image.generate" || r.operation == "image.transform" ||
                            r.operation == "image.edit",
                        "unsupported FLUX operation");
                if (r.operation == "image.generate")
                    require(r.inputs.empty(), "image.generate does not accept image inputs");
                else {
                    require(!r.inputs.empty() && r.inputs.size() <= 8,
                            "FLUX requires 1...8 images");
                    if (r.operation == "image.transform")
                        require(r.inputs.size() == 1, "image.transform requires one init_image");
                    for (auto &input : r.inputs)
                        require(input.kind == "image" &&
                                    input.role == (r.operation == "image.transform" ? "init_image"
                                                                                    : "reference"),
                                "incorrect FLUX image role");
                }
                require(r.residency == "resident" || r.residency == "component_staged",
                        "FLUX block streaming is not supported");
            },
            [](const std::filesystem::path &root) { return std::make_unique<Flux>(root); },
            [] {
                return ModelDescriptor{"flux2-klein-4b",
                                       "FLUX.2 Klein 4B",
                                       true,
                                       {"image.generate", "image.transform", "image.edit"},
                                       {"text", "image"},
                                       {"init_image", "reference"},
                                       8,
                                       "image",
                                       4,
                                       1,
                                       512,
                                       512,
                                       false};
            }};
}
} // namespace tc
