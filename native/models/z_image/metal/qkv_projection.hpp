#pragma once

#include "../../../backends/metal_qkv_projection.hpp"

#include <stdexcept>

namespace tc::z_metal {
namespace mx = mlx::core;

inline std::vector<mx::array> project_prepare_qkv(
    const mx::array &x, const mx::array &w, const mx::array &qw,
    const mx::array &kw, const mx::array &freqs) {
    if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(2) != 3840 ||
        w.shape() != mx::Shape{11520,3840} ||
        qw.shape() != mx::Shape{128} || kw.shape() != mx::Shape{128} ||
        freqs.shape() != mx::Shape{x.shape(1),64,2} ||
        x.dtype() != mx::bfloat16 || w.dtype() != x.dtype())
        throw std::invalid_argument("Z-Image fused QKV projection geometry/dtype mismatch");
    constexpr int bm = 48;
    static const auto kernel = metal_qkv::make_projection_kernel(
        "tc_z_mpp_qkv_prepare_48x128", {"x", "w", "qw", "kw", "freqs"}, "1e-5f", R"metal(
                float c = float(freqs[token*128+d]);
                float s = float(freqs[token*128+d+1]);
        )metal");
    return metal_qkv::run_projection(kernel, {x, w, qw, kw, freqs}, bm);
}

} // namespace tc::z_metal
