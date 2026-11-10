#pragma once

#include "../ltx_runtime/ltx_convrot_policy.h"
#include <stdexcept>

namespace tc::ltx_mlx {
// Hardware capability includes both M5+ TensorOps and the required OS support.
// Retain an explicit BF16 route for quality/performance comparisons.
inline bool configured_convrot_w8a8(bool hardware_supported) {
    char error[256] = {};
    const int enabled = ltx_convrot_w8a8_requested(hardware_supported, error, sizeof(error));
    if (enabled < 0) throw std::invalid_argument(error);
    return enabled != 0;
}
}
