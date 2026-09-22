#pragma once
#include "text_encoder.hpp"
#include "vision.hpp"

namespace tc::qwen21 {
struct VisionInput {
    Tensor patches;
    int grid_height, grid_width;
};
// Unit-range NHWC RGB. Resize uses half-pixel bilinear sampling, without
// antialiasing, matching Comfy's Qwen3-VL processor (not the VAE processor).
VisionInput vision_input(const Tensor &, int min_pixels = 3136, int max_pixels = 12845056);
// PE-I2I processor: uint8 NHWC RGB/RGBA, with RGB channels retained (alpha
// dropped, not composited) exactly as the official Pillow convert("RGB").
// Includes the 1024^2 Lanczos source cap, then bicubic smart resize.
VisionInput pe_vision_input(const Tensor &, int min_pixels = 65536, int max_pixels = 16777216);
// Official area/aspect sizing (32-aligned). Lanczos3 filtering in premultiplied
// alpha prevents hidden transparent colors bleeding into visible boundaries.
Tensor resize_reference(const Tensor &, int resolution = 1024);
struct VisualReference {
    VisionFeatures features;
    int grid_height, grid_width;
};
struct MultimodalPrompt {
    Tensor embeddings, positions;
    std::vector<Tensor> deepstack_deltas;
    std::vector<int> retained_indices, image_slots;
    Tensor retain(const Tensor &hidden) const;
};
// Input IDs contain exactly one image_pad per reference. Vision delimiters stay
// in the language conditioning; only expanded image embeddings are removed.
MultimodalPrompt assemble_prompt(const Tokens &, const Tensor &embedding_table,
                                const std::vector<VisualReference> &, int max_tokens = 32768);
std::string reference_prompt_template(const std::string &, size_t reference_count);
} // namespace tc::qwen21
