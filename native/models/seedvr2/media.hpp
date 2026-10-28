#pragma once
#include "model.hpp"

namespace tc::seedvr2 {
// NHWC float pixels; half-pixel centers, cubic coefficient -0.75,
// align_corners=false, no antialiasing. No uint8 resize/requantization.
Tensor resize_bicubic(const Tensor &, int height, int width);

struct NormalizedMediaInput {
    Tensor pixels; // Materialized FP32 RGB BCTHW in [-1,1].
    Tensor alpha;  // Images only: FP32 NHWC alpha in [0,1].
    bool alpha_is_owned = false;
};
// Video never retains resized source RGB after normalization. Images can
// retain the original RGBA through their alpha view (fast default), or copy
// alpha without arithmetic for the explicitly selected low-memory policy.
NormalizedMediaInput consume_resized_input(Tensor &unit, bool video, bool own_image_alpha = false);
} // namespace tc::seedvr2
