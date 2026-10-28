#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace tc::seedvr2 {

struct Window {
    std::array<int, 3> start{}, stop{};
    std::array<int, 3> shape() const;
    int tokens() const;
};

// ByteDance's adaptive 720p window law, in patched-latent coordinates.
// Order is width, height, time; shifted boundaries truncate toward zero.
std::vector<Window> make_windows(int frames, int height, int width, bool shifted);
std::vector<int32_t> window_indices(const Window &, int height, int width);
int padded_frames(int frames);

} // namespace tc::seedvr2
