#pragma once

#include "../../native/backends/mlx.hpp"

namespace tc::ltx098::rgb_cast_probe {

inline Tensor reference(const Tensor &pixels) {
    auto scaled = (mx::astype(pixels, mx::float32) + Tensor(1.f, mx::float32)) * Tensor(127.5f, mx::float32);
    return mx::astype(mx::contiguous(mx::clip(scaled, Tensor(0.f), Tensor(255.f))), mx::uint8);
}

inline Tensor fused(const Tensor &pixels) {
    static auto compiled = mx::compile([](const std::vector<Tensor> &values) {
        auto scaled = (mx::astype(values[0], mx::float32) + Tensor(1.f, mx::float32)) * Tensor(127.5f, mx::float32);
        auto rgb = mx::astype(mx::contiguous(mx::clip(scaled, Tensor(0.f), Tensor(255.f))), mx::uint8);
        return std::vector<Tensor>{rgb};
    }, true);
    return compiled({pixels})[0];
}

} // namespace tc::ltx098::rgb_cast_probe
