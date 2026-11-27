#pragma once
#include "../../core/common.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace tc::qwen21 {
inline constexpr int reference_alignment = 32;
inline constexpr int reference_grid_pixels = reference_alignment * reference_alignment;
inline constexpr int reference_latent_pixels = 16 * 16;
inline constexpr int maximum_reference_pixel_budget = 512 * 512;

inline int reference_grid_quota(int pixel_budget, int count, int index) {
    require(count > 0 && index >= 0 && index < count &&
            pixel_budget >= count * reference_grid_pixels &&
            pixel_budget <= maximum_reference_pixel_budget && pixel_budget % reference_grid_pixels == 0,
            "invalid Qwen21 shared reference pixel budget");
    const int cells = pixel_budget / reference_grid_pixels;
    return cells / count + (index < cells % count ? 1 : 0);
}

// Fit one reference on the 32-pixel grid. Prefer preserving the aspect ratio
// while using most of the available area. Each candidate respects the quota.
inline std::pair<int, int> fit_reference_pixels(int width, int height, int cells) {
    require(width > 0 && height > 0 && cells >= 1 && cells <= maximum_reference_pixel_budget / reference_grid_pixels,
            "invalid Qwen21 reference dimensions or pixel quota");
    const double ratio = double(width) / height;
    double best_score = std::numeric_limits<double>::infinity();
    int best_width = 1, best_height = 1, best_area = 1;
    for (int grid_width = 1; grid_width <= cells; ++grid_width) {
        const double ideal_height = std::clamp(grid_width / ratio, 1.0, double(cells / grid_width));
        for (int grid_height : {int(std::floor(ideal_height)), int(std::ceil(ideal_height))}) {
            const int area = grid_width * grid_height;
            const double aspect_error = std::abs(std::log(double(grid_width) / grid_height / ratio));
            const double score = aspect_error + 0.3 * (1.0 - double(area) / cells);
            if (score < best_score || (score == best_score && area > best_area)) {
                best_score = score;
                best_width = grid_width;
                best_height = grid_height;
                best_area = area;
            }
        }
    }
    return {best_width * reference_alignment, best_height * reference_alignment};
}
} // namespace tc::qwen21
