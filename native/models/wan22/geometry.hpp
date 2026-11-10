#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace tc::wan22 {

struct CanvasGeometry {
    int width, height;
    int left, top, right, bottom;
};

// H.264 export currently preserves exact dimensions only for even sizes.
// Wan's VAE and patch embedding require a multiple-of-32 model canvas. The
// transformer and VAE still use int for their flattened spatial/token sizes.
inline std::optional<CanvasGeometry> canvas_geometry(int width, int height, int frames) {
    if (width <= 0 || height <= 0 || width % 2 || height % 2 ||
        frames < 5 || frames > 97 || frames % 4 != 1) return std::nullopt;
    const uint64_t aligned_width = (uint64_t(width) + 31) / 32 * 32;
    const uint64_t aligned_height = (uint64_t(height) + 31) / 32 * 32;
    constexpr uint64_t int_max = std::numeric_limits<int>::max();
    if (aligned_width > int_max || aligned_height > int_max) return std::nullopt;
    const uint64_t spatial_tokens = (aligned_width / 32) * (aligned_height / 32);
    const uint64_t latent_frames = (frames - 1) / 4 + 1;
    // The VAE uses (H/16)*(W/16), four times spatial_tokens, as an int.
    if (spatial_tokens > int_max / std::max<uint64_t>(4, latent_frames)) return std::nullopt;
    const int extra_width = int(aligned_width) - width;
    const int extra_height = int(aligned_height) - height;
    const int left = extra_width / 2, top = extra_height / 2;
    return CanvasGeometry{int(aligned_width), int(aligned_height), left, top,
                          extra_width - left, extra_height - top};
}

} // namespace tc::wan22
