#pragma once
#include "model.hpp"

namespace tc::seedvr2 {
// NHWC float pixels; half-pixel centers, cubic coefficient -0.75,
// align_corners=false, no antialiasing. No uint8 resize/requantization.
Tensor resize_bicubic(const Tensor &, int height, int width);
} // namespace tc::seedvr2
