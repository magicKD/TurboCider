#pragma once

#include "../../../backends/metal_qkv_projection.hpp"

#include <stdexcept>

namespace tc::qwen21::metal {
namespace mx = mlx::core;

inline std::vector<mx::array> project_prepare_qkv(
    const mx::array &x, const mx::array &w, const mx::array &qw,
    const mx::array &kw, const mx::array &cos, const mx::array &sin,
    int bm = 48) {
    if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(2) != 4096 ||
        (bm != 16 && bm != 32 && bm != 48 && bm != 64) ||
        w.shape() != mx::Shape{12288, 4096} ||
        qw.shape() != mx::Shape{128} || kw.shape() != mx::Shape{128} ||
        cos.shape() != mx::Shape{x.shape(1), 64} || sin.shape() != cos.shape() ||
        x.dtype() != mx::bfloat16 || w.dtype() != x.dtype() ||
        qw.dtype() != x.dtype() || kw.dtype() != x.dtype() ||
        cos.dtype() != mx::float32 || sin.dtype() != cos.dtype())
        throw std::invalid_argument("Qwen21 fused QKV projection geometry/dtype mismatch");
    static const auto kernel = metal_qkv::make_projection_kernel(
        "tc_qwen21_mpp_qkv_prepare_screen", {"x", "w", "qw", "kw", "cos", "sin"}, "1e-6f", R"metal(
                float c = float(cos[token*64+d/2]);
                float s = float(sin[token*64+d/2]);
        )metal");
    return metal_qkv::run_projection(kernel, {x, w, qw, kw, cos, sin}, bm);
}
} // namespace tc::qwen21::metal
