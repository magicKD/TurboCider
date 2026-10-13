#pragma once
#include "../backends/mlx.hpp"

namespace tc {
// Decode the complete declared CFR video into RGB [0,1] THWC pixels.
// Never silently truncate, synthesize frames, resample FPS, or discard a
// transform. Model-specific resizing remains separate from container decode.
Tensor load_video_tensor(const std::filesystem::path &, int expected_frames,
                         int expected_fps, const Event &, std::atomic<bool> &);
} // namespace tc
