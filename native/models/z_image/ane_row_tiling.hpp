#pragma once
#include "../../backends/mlx.hpp"
#include <algorithm>
#include <cstring>

namespace tc::z_image {
// Frozen FFNs operate independently on each token. Keep attention over the
// complete sequence, and tile only this projection when a caption outgrows
// its compiled bucket. Core ML reuses its output backing between calls, so
// copy each completed tile before submitting the next one.
template<class Predict>
Tensor predict_ane_rows(const Tensor &input, int capacity, Predict predict) {
    require(input.ndim() == 3 && input.shape(0) == 1 &&
                input.dtype() == mx::float16 && capacity > 0,
            "invalid Z-Image ANE row tiling input");
    const int rows = input.shape(1), channels = input.shape(2);
    require(rows > 0, "Z-Image ANE row tiling requires nonempty input");
    if (rows <= capacity) {
        auto padded = rows == capacity ? input : mx::concatenate(
            {input, mx::zeros({1, capacity - rows, channels}, mx::float16)}, 1);
        padded = mx::contiguous(padded);
        mx::eval(padded);
        return mx::slice(predict(padded), {0, 0, 0}, {1, rows, channels});
    }
    auto result = mx::zeros({1, rows, channels}, mx::float16);
    mx::eval(result);
    for (int offset = 0; offset < rows; offset += capacity) {
        const int count = std::min(capacity, rows - offset);
        auto tile = mx::slice(input, {0, offset, 0}, {1, offset + count, channels});
        if (count < capacity)
            tile = mx::concatenate(
                {tile, mx::zeros({1, capacity - count, channels}, mx::float16)}, 1);
        tile = mx::contiguous(tile);
        mx::eval(tile);
        const Tensor output = predict(tile);
        require(output.shape() == mx::Shape{1, capacity, channels} &&
                    output.dtype() == mx::float16 && output.flags().row_contiguous,
                "invalid Z-Image ANE tile output");
        mx::eval(output);
        std::memcpy(result.data<mx::float16_t>() + size_t(offset) * channels,
                    output.data<mx::float16_t>(), size_t(count) * channels * sizeof(mx::float16_t));
    }
    return result;
}
} // namespace tc::z_image
