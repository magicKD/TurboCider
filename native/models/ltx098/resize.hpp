#pragma once

#include "../../backends/mlx.hpp"

#include <cstdint>
#include <limits>

namespace tc::ltx098 {

// PyTorch interpolate(mode="bilinear", align_corners=False) over BFHWC.
// The official multiscale pipeline resizes conditioning frames for each
// pass and resizes the decoded video back to the requested canvas.
inline Tensor resize_bilinear(const Tensor &input, int height, int width) {
    require(input.ndim() == 5 && input.shape(0) == 1 && input.shape(1) > 0 &&
                input.shape(2) > 0 && input.shape(3) > 0 && input.shape(4) == 3 &&
                (input.dtype() == mx::float32 || input.dtype() == mx::bfloat16) &&
                height > 0 && width > 0,
            "LTX 0.9.8 resize expects FP32/BF16 [1,F,H,W,3] pixels");
    for (auto stride : input.strides()) require(stride >= 0, "negative pixel stride is unsupported");
    const int frames = input.shape(1), source_h = input.shape(2), source_w = input.shape(3);
    if (source_h == height && source_w == width) return input;
    const size_t count = size_t(frames) * height * width * 3;
    require(count <= size_t(std::numeric_limits<int>::max()), "LTX 0.9.8 resized video is too large");
    static auto kernel = mx::fast::metal_kernel(
        "tc_ltx098_resize_bilinear", {"x"}, {"out"}, R"metal(
        uint i = thread_position_in_grid.x;
        if (i >= COUNT) return;
        uint c = i % 3; uint p = i / 3;
        uint ox = p % OW; p /= OW;
        uint oy = p % OH; uint t = p / OH;
        float fx = max((float(ox) + 0.5f) * float(IW) / float(OW) - 0.5f, 0.0f);
        float fy = max((float(oy) + 0.5f) * float(IH) / float(OH) - 0.5f, 0.0f);
        uint x0 = min(uint(fx), uint(IW - 1)), y0 = min(uint(fy), uint(IH - 1));
        uint x1 = min(x0 + 1, uint(IW - 1)), y1 = min(y0 + 1, uint(IH - 1));
        float ax = fx - float(x0), ay = fy - float(y0);
        size_t base = size_t(t) * x_strides[1] + size_t(c) * x_strides[4];
        float a = float(x[base + size_t(y0) * x_strides[2] + size_t(x0) * x_strides[3]]);
        float b = float(x[base + size_t(y0) * x_strides[2] + size_t(x1) * x_strides[3]]);
        float d = float(x[base + size_t(y1) * x_strides[2] + size_t(x0) * x_strides[3]]);
        float e = float(x[base + size_t(y1) * x_strides[2] + size_t(x1) * x_strides[3]]);
        out[i] = T((1.0f - ay) * ((1.0f - ax) * a + ax * b) +
                   ay * ((1.0f - ax) * d + ax * e));
        )metal", "", false);
    return kernel({input}, {{1, frames, height, width, 3}}, {input.dtype()},
                  {int(count), 1, 1}, {256, 1, 1},
                  {{"T", input.dtype()}, {"IH", source_h}, {"IW", source_w},
                   {"OH", height}, {"OW", width}, {"COUNT", int(count)}}, {}, false, {})[0];
}

} // namespace tc::ltx098
