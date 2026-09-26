#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::qwen21::metal {
namespace mx = mlx::core;

// One SIMD group handles one 128-channel Q/K head. Consume the token-major
// projection directly, and emit the head-major layout required by SDPA.
// Preserve the activation-dtype rounding between RMSNorm and RoPE.
inline std::vector<mx::array> prepare_qk(const mx::array &q, const mx::array &k,
                                         const mx::array &qw, const mx::array &kw,
                                         const mx::array &cos, const mx::array &sin,
                                         float epsilon) {
    if (q.shape() != k.shape() || q.ndim() != 3 || q.shape(0) != 1 || q.shape(2) != 4096 ||
        qw.shape() != mx::Shape{128} || kw.shape() != mx::Shape{128} ||
        cos.shape() != mx::Shape{q.shape(1), 64} || sin.shape() != cos.shape() ||
        q.dtype() != k.dtype() || q.dtype() != qw.dtype() || q.dtype() != kw.dtype() ||
        q.dtype() != mx::bfloat16 || cos.dtype() != mx::float32 ||
        sin.dtype() != mx::float32 || epsilon != 1e-6f)
        throw std::invalid_argument("Qwen21 fused Q/K norm/RoPE geometry or dtype mismatch");
    static auto kernel = mx::fast::metal_kernel(
        "tc_qwen21_qk_norm_rope", {"q", "k", "qw", "kw", "cos", "sin"},
        {"q_out", "k_out"}, R"metal(
        uint lane = thread_index_in_simdgroup;
        uint task = thread_position_in_grid.x / 32;
        if (task >= N * 32) return;
        uint token = task / 32;
        uint head = task % 32;
        uint src = token * 4096 + head * 128;
        uint dst = (head * N + token) * 128;
        float qx[4], kx[4];
        float qs = 0.0f, ks = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            uint d = lane * 4 + i;
            qx[i] = float(q[src + d]);
            kx[i] = float(k[src + d]);
            qs += qx[i] * qx[i];
            ks += kx[i] * kx[i];
        }
        float qi = metal::precise::rsqrt(simd_sum(qs) / 128.0f + 1e-6f);
        float ki = metal::precise::rsqrt(simd_sum(ks) / 128.0f + 1e-6f);
        for (uint i = 0; i < 4; ++i) {
            uint d = lane * 4 + i;
            qx[i] = float(T((qx[i] * qi) * float(qw[d])));
            kx[i] = float(T((kx[i] * ki) * float(kw[d])));
        }
        for (uint i = 0; i < 4; i += 2) {
            uint d = lane * 4 + i;
            float c = float(cos[token * 64 + d / 2]);
            float s = float(sin[token * 64 + d / 2]);
            q_out[dst + d] = T(qx[i] * c - qx[i + 1] * s);
            q_out[dst + d + 1] = T(qx[i + 1] * c + qx[i] * s);
            k_out[dst + d] = T(kx[i] * c - kx[i + 1] * s);
            k_out[dst + d + 1] = T(kx[i + 1] * c + kx[i] * s);
        }
        )metal");
    const int n = q.shape(1);
    const mx::Shape shape{1, 32, n, 128};
    return kernel({q, k, qw, kw, cos, sin}, {shape, shape}, {q.dtype(), k.dtype()},
                  {n * 32 * 32, 1, 1}, {128, 1, 1},
                  {{"T", q.dtype()}, {"N", n}}, {}, false, {});
}
} // namespace tc::qwen21::metal
