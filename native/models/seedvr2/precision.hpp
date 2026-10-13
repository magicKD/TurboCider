#pragma once

namespace tc::seedvr2 {
inline constexpr const char *kStableRecipe = "seedvr2-stable-fp16-fp32-v1";
inline constexpr const char *kStablePrecision =
    "original_fp16_weights+fp16_dit_projections+fp32_dit_state+fp16_vae+fp32_euler";
} // namespace tc::seedvr2
