// Native adaptation of ByteDance-Seed/SeedVR's adaptive window geometry.
// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tc::seedvr2 {

std::array<int, 3> Window::shape() const {
    return {stop[0] - start[0], stop[1] - start[1], stop[2] - start[2]};
}
int Window::tokens() const {
    const auto s = shape();
    return s[0] * s[1] * s[2];
}
int padded_frames(int frames) {
    if (frames < 1) throw std::invalid_argument("SeedVR2 requires at least one frame");
    return frames == 1 ? 1 : 1 + 4 * ((frames + 2) / 4);
}
std::vector<Window> make_windows(int frames, int height, int width, bool shifted) {
    if (frames < 1 || frames > 1024 || height < 1 || width < 1 ||
        height > 512 || width > 512)
        throw std::invalid_argument("invalid SeedVR2 patched-latent dimensions");
    const auto scale = std::sqrt(3600.0 / (double(height) * width));
    const std::array<int, 3> dim{frames, height, width};
    const std::array<int, 3> size{
        (std::min(frames, 30) + 3) / 4,
        (int(std::nearbyint(height * scale)) + 2) / 3,
        (int(std::nearbyint(width * scale)) + 2) / 3};
    std::array<double, 3> shift{};
    std::array<int, 3> count{};
    for (int axis = 0; axis < 3; ++axis) {
        shift[axis] = shifted && size[axis] < dim[axis] ? .5 : 0.;
        count[axis] = shifted
            ? (shift[axis] ? int(std::ceil((dim[axis] - shift[axis]) / size[axis])) + 1 : 1)
            : (dim[axis] + size[axis] - 1) / size[axis];
    }
    std::vector<Window> result;
    for (int iw = 0; iw < count[2]; ++iw)
        for (int ih = 0; ih < count[1]; ++ih)
            for (int it = 0; it < count[0]; ++it) {
                Window window;
                const std::array<int, 3> index{it, ih, iw};
                for (int axis = 0; axis < 3; ++axis) {
                    window.start[axis] = std::max(0, int((index[axis] - shift[axis]) * size[axis]));
                    window.stop[axis] = std::min(dim[axis], int((index[axis] - shift[axis] + 1) * size[axis]));
                }
                const auto extent = window.shape();
                if (std::all_of(extent.begin(), extent.end(), [](int value) { return value > 0; }))
                    result.push_back(window);
            }
    return result;
}
std::vector<int32_t> window_indices(const Window &window, int height, int width) {
    std::vector<int32_t> result;
    result.reserve(window.tokens());
    for (int it = window.start[0]; it < window.stop[0]; ++it)
        for (int ih = window.start[1]; ih < window.stop[1]; ++ih)
            for (int iw = window.start[2]; iw < window.stop[2]; ++iw)
                result.push_back((it * height + ih) * width + iw);
    return result;
}

} // namespace tc::seedvr2
