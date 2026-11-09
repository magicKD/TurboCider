#pragma once

#include "../../backends/mlx.hpp"

#include <cstdint>
#include <limits>

namespace tc::ltx098::vae_padding {

// Match the decoder's original temporal edge replication followed by
// one-pixel spatial reflection. Also serves as the general fallback.
inline Tensor reference(const Tensor &input) {
    const int b = input.shape(0), f = input.shape(1), h = input.shape(2), w = input.shape(3), c = input.shape(4);
    auto first = mx::slice(input, {0, 0, 0, 0, 0}, {b, 1, h, w, c});
    auto last = mx::slice(input, {0, f - 1, 0, 0, 0}, {b, f, h, w, c});
    auto padded = mx::concatenate({first, input, last}, 1);
    auto top = mx::slice(padded, {0, 0, 1, 0, 0}, {b, f + 2, 2, w, c});
    auto bottom = mx::slice(padded, {0, 0, h - 2, 0, 0}, {b, f + 2, h - 1, w, c});
    padded = mx::concatenate({top, padded, bottom}, 2);
    auto left = mx::slice(padded, {0, 0, 0, 1, 0}, {b, f + 2, h + 2, 2, c});
    auto right = mx::slice(padded, {0, 0, 0, w - 2, 0}, {b, f + 2, h + 2, w - 1, c});
    return mx::concatenate({left, padded, right}, 3);
}

// One gather replaces three intermediate concatenations. It only copies
// BF16 bits: no arithmetic or precision change is applied to pixel values.
inline Tensor fused(const Tensor &input) {
    const int frames = input.shape(1), height = input.shape(2), width = input.shape(3), channels = input.shape(4);
    const size_t elements = size_t(frames + 2) * size_t(height + 2) * size_t(width + 2) * size_t(channels);
    static auto kernel = mx::fast::metal_kernel(
        "tc_ltx098_vae_reflect_edge", {"x"}, {"out"}, R"metal(
        uint group = thread_position_in_grid.x;
        if (group >= GROUPS) return;
        uint position = group * 8;
        uint channel = position % C;
        uint pixel = position / C;
        uint xp = pixel % (W + 2); pixel /= W + 2;
        uint yp = pixel % (H + 2); uint tp = pixel / (H + 2);
        uint t = tp == 0 ? 0 : (tp == F + 1 ? F - 1 : tp - 1);
        uint y = yp == 0 ? 1 : (yp == H + 1 ? H - 2 : yp - 1);
        uint xcoord = xp == 0 ? 1 : (xp == W + 1 ? W - 2 : xp - 1);
        size_t source = size_t(t) * x_strides[1] + size_t(y) * x_strides[2] +
                        size_t(xcoord) * x_strides[3] + size_t(channel) * x_strides[4];
        for (uint k = 0; k < 8; ++k) out[position + k] = x[source + size_t(k) * x_strides[4]];
        )metal", "", false);
    return kernel({input}, {{1, frames + 2, height + 2, width + 2, channels}}, {input.dtype()},
                  {int(elements / 8), 1, 1}, {256, 1, 1},
                  {{"T", input.dtype()}, {"F", frames}, {"H", height}, {"W", width},
                   {"C", channels}, {"GROUPS", int(elements / 8)}}, {}, false, {})[0];
}

// The MLX metadata supplied to the kernel describes execution-time strides.
// Unsupported layouts and shapes retain the previous implementation.
inline bool eligible(const Tensor &input) {
    if (input.ndim() != 5 || input.shape(0) != 1 || input.shape(1) < 1 || input.shape(1) > 121 ||
        input.shape(2) < 2 || input.shape(2) > 256 || input.shape(3) < 2 || input.shape(3) > 256 ||
        input.shape(4) < 8 || input.shape(4) > 2048 || input.shape(4) % 8 || input.dtype() != mx::bfloat16)
        return false;
    for (auto stride : input.strides()) if (stride < 0) return false;
    const size_t elements = size_t(input.shape(1) + 2) * size_t(input.shape(2) + 2) *
                            size_t(input.shape(3) + 2) * size_t(input.shape(4));
    return elements <= size_t(std::numeric_limits<uint32_t>::max());
}

inline Tensor apply(const Tensor &input) {
    return eligible(input) ? fused(input) : reference(input);
}

} // namespace tc::ltx098::vae_padding
